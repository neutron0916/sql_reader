import logging
import os
import re
import operator
import warnings
from typing import TypedDict, List, Dict, Any, Annotated

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def build_llm(temperature: float | None = None) -> ChatOpenAI:
    """建立 LLM（保留你的企業級 API 配置）"""
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "https://model-gateway.gclpgenaigw.gc.micron.com/api/v1")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "sk-dummy-key")
    model = os.getenv("OPENAI_COMPAT_MODEL", "gpt-5.2-codex")
    default_temperature = float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.1")) # 建議低溫以確保邏輯精準
    max_tokens_str = os.getenv("OPENAI_COMPAT_MAX_TOKENS", "")
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "HIGH")
    disable_ssl_verify = os.getenv("OPENAI_COMPAT_VERIFY_SSL", "false").lower() in {"0", "false", "no"}

    used_temperature = temperature if temperature is not None else default_temperature
    os.environ["NO_PROXY"] = os.getenv("NO_PROXY", "gc.micron.com")
    ssl_verify = not disable_ssl_verify
    
    http_client = httpx.Client(verify=ssl_verify)
    http_async_client = httpx.AsyncClient(verify=ssl_verify)
    is_gemini = "gemini" in model.lower()

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": used_temperature,
        "http_client": http_client,
        "http_async_client": http_async_client,
        "use_responses_api": False,
    }

    if max_tokens_str.strip(): kwargs["max_completion_tokens"] = int(max_tokens_str)
    if is_gemini and thinking_level: kwargs["extra_body"] = {"generation_config": {"thinking_config": {"thinking_level": thinking_level}}}

    return ChatOpenAI(**kwargs)

# ==========================================
# 🧠 1. 定義狀態與強型別輸出 (Pydantic)
# ==========================================
class SQLAnalysisState(TypedDict):
    raw_sql: str
    chunks: List[str]
    current_chunk_index: int       # 🌟 從最後一塊往前遞減 (Reverse Reading)
    
    # 動態狀態維護
    final_target: str              # 記錄最末端的最終表
    pending_list: List[str]        # 🌟 動態擴充的「待尋找來源」Temp Table 清單
    processed_list: List[str]      # 記錄已解析過的表，避免重疊 Chunk 造成重複解析
    current_depth: int
    step_count: int
    
    existing_kb: str
    report_steps: Annotated[List[str], operator.add] # 累積的報告段落
    parsed_result: Dict[str, Any]

class SourceTable(BaseModel):
    table_name: str = Field(description="來源表名 (例如 #Temp_B, ORD_HDR)")
    original_column: str = Field(description="對應的原始欄位名 (可為多個欄位或 *)")
    is_physical: bool = Field(description="是否為實體表 (判斷標準：非 # 開頭，且非 CTE，通常是底層實體表)")

class BPLogic(BaseModel):
    core_function: str = Field(description="核心功能：請用人類語言描述這張表在做什麼，例如：計算客戶去年的總消費額")
    key_transformation: str = Field(description="關鍵轉換：記錄 CASE WHEN, COALESCE 或聚合邏輯")
    field_meaning: str = Field(description="欄位意義：根據上下文推測此欄位的業務價值")

class ParsedLayer(BaseModel):
    target_table: str = Field(description="在此區塊中被產出/寫入的目標表名 (例如 #Temp_C 或最終報表)")
    target_column_alias: str = Field(description="在此層中關注的核心欄位名稱")
    sources: List[SourceTable] = Field(description="所有貢獻來源 (多對一 JOIN 或 UNION 必須列出所有來源表)")
    bp_logic: BPLogic
    is_fully_resolved: bool = Field(description="是否在此步驟完全定義了此表 (例如 CREATE TABLE, SELECT INTO)。若是 UPDATE 則設為 False。")

class KBTable(BaseModel):
    table_name: str = Field(description="實體表名")
    business_desc: str = Field(description="業務用途描述")
    granularity: str = Field(description="數據粒度 (Granularity)")
    remarks: str = Field(description="備註 (若與現有知識庫矛盾請標註)")

class ChunkParseResult(BaseModel):
    found_layers: List[ParsedLayer] = Field(description="在此 Chunk 中找到的所有資料寫入層。請務必按照 SQL 語句「由上往下」的出現順序排列！")
    kb_tables: List[KBTable] = Field(description="提煉出的實體表通用知識")
    patterns: List[str] = Field(description="記錄重複出現的公式或過濾習慣")

# ==========================================
# ⚙️ 2. 工具與 Nodes
# ==========================================
def chunk_mssql_sql(raw_sql: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """保留你的切塊邏輯：依標記或滑動視窗切塊"""
    lines = raw_sql.splitlines()
    marker_pattern = re.compile(r"^\s*---\s*CHUNK\s+BOUNDARY\s*---")
    
    if any(marker_pattern.match(l) for l in lines):
        chunks, current = [], []
        for line in lines:
            if marker_pattern.match(line):
                if current: chunks.append("\n".join(current))
                current = []
            else: current.append(line)
        if current: chunks.append("\n".join(current).strip())
        return [c.strip() for c in chunks if c.strip()]
        
    total = len(lines)
    chunks, start = [], 0
    while start < total:
        end = min(start + chunk_size, total)
        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text: chunks.append(chunk_text)
        if end >= total: break
        start = end - overlap
    return chunks

def init_node(state: SQLAnalysisState):
    """【Step 1】檔案切塊，並將指標設定在最後一塊 (Bottom-Up)"""
    logger.info("🟢 [NODE] init_node — 讀取 SQL 並切塊")
    raw_sql = state["raw_sql"]
    
    chunks = chunk_mssql_sql(raw_sql)
    logger.info(f"   📦 共切分為 {len(chunks)} 個區塊，將從 Chunk {len(chunks)} 開始「由下往上」逆向讀取！")
    
    # 建立或讀取 Knowledge_Base.md
    kb_path = "Knowledge_Base.md"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f: existing_kb = f.read()
    else:
        existing_kb = "Table Dictionary:\n| 實體表名 | 業務用途描述 | 數據粒度 (Granularity) | 備註 |\n| :--- | :--- | :--- | :--- |\n\nPattern Recognition:\n"
            
    # 初始化 Report.txt (清空舊檔案確保全新追溯)
    with open("Report.txt", "w", encoding="utf-8") as f:
        f.write("[Analysis Status]\n- 系統啟動中...\n---\n")

    return {
        "chunks": chunks,
        "current_chunk_index": len(chunks) - 1,  # 🌟 核心：指標指向最後一塊
        "final_target": "Unknown",
        "pending_list": [],                      # 一開始是空的，靠讀取慢慢產生
        "processed_list": [],
        "current_depth": 0,
        "step_count": 0,
        "report_steps": [],
        "existing_kb": existing_kb
    }

def analyze_backward_chunk_node(state: SQLAnalysisState):
    """【Step 2】解析 (Parse)：由後往前讀取 Chunk，動態推演 Pending List"""
    idx = state["current_chunk_index"]
    current_chunk = state["chunks"][idx]
    pending_list = state.get("pending_list", [])
    
    logger.info(f"\n{'='*60}\n🟢 [NODE] analyze_backward_chunk — 正在解析 Chunk {idx+1}/{len(state['chunks'])}")
    logger.info(f"   🔍 目前尋找來源的 Pending List: {pending_list if pending_list else '[] (正在抓取最底層 Final Target)'}")

    llm = build_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ChunkParseResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位頂尖的資料工程師。我們正在「由後往前 (Bottom-Up)」逆向閱讀 SQL。
        
        【任務規則與思考邏輯】：
        1. 找出這個區塊中「被寫入、被建立或被更新」的目標表 (Target Table)。
        2. 你的思考邏輯必須是：Target <--- #TempC <--- #TempB <--- Source Table。
        3. 【動態尋找】：目前的 Pending List 為 {pending_list}。
           - 若 Pending List 為空，請找出這個 Chunk 產出的「最終目標表 (Final Target)」，並解析它的來源。
           - 若 Pending List 不為空，請優先尋找這些 Temp Table 在此區塊中是否被建立或更新。
        4. 【多對一處理】：若表由多個表 JOIN 或 UNION 而成，必須將「所有來源表」加入 sources 清單。嚴禁漏掉！
        5. 將找到的轉換步驟，按照在 SQL 語句中「由上往下」的出現順序排列在 found_layers 中。
        6. 【知識對齊】：觸碰到實體表 (Physical Table，非 # 開頭) 時，優先參考並擴充知識庫：
        
        {existing_kb}
        """),
        ("user", "【當前 SQL 區塊】:\n{chunk}")
    ])
    
    logger.info("   ⏳ 呼叫 LLM 進行逆向血緣推演...")
    result: ChunkParseResult = structured_llm.invoke(prompt.format_messages(
        pending_list=pending_list,
        existing_kb=state["existing_kb"],
        chunk=current_chunk
    ))
    
    return {"parsed_result": result.model_dump()}

def update_node(state: SQLAnalysisState):
    """【Step 3】增量更新 (Update)：過濾結果、動態維護 Pending List，寫入實體檔案"""
    logger.info("🟢 [NODE] update_node — 狀態維護與檔案寫入")
    
    parsed = ChunkParseResult(**state.get("parsed_result", {}))
    pending_list = state.get("pending_list", []).copy()
    processed_list = state.get("processed_list", []).copy()
    final_target = state.get("final_target", "Unknown")
    step_count = state.get("step_count", 0)
    current_depth = state.get("current_depth", 0)
    
    new_report_steps = []
    
    # 🌟 核心：雖然我們是「由下往上」讀 Chunk，但 LLM 讀取單一 Chunk 內部是 Top-Down。
    # 因此我們必須在 Python 端反轉 parsed.layers，確保 Chunk 內部的 CTE 或連鎖更新能無縫溯源。
    parsed_layers = parsed.found_layers.copy()
    parsed_layers.reverse()
    
    for layer in parsed_layers:
        target = layer.target_table
        is_relevant = False
        
        # 情況 1：剛啟動，抓取最底層的 Final Target
        if final_target == "Unknown" and not pending_list:
            final_target = target
            is_relevant = True
            logger.info(f"   🎯 自動鎖定最終目標表 (Final Target): {final_target}")
        # 情況 2：命中 Pending List 裡的表，代表我們找到它的上游來源了！
        elif target.lower() in [p.lower() for p in pending_list]:
            is_relevant = True

        if is_relevant:
            # 層次遞歸：如果是 CREATE/INTO 這種完全解析的，將其移出佇列；若是 UPDATE 則保留繼續往上查。
            if target.lower() in [p.lower() for p in pending_list]:
                if layer.is_fully_resolved:
                    pending_list = [p for p in pending_list if p.lower() != target.lower()]
                    logger.info(f"   ✅ {target} 已完全解析，移出 Pending List")
                else:
                    logger.info(f"   ⚠️ {target} 僅為部分更新，保留在 Pending List 繼續往上追溯")
            
            # 防止切塊 Overlap 造成重複解析
            if layer.is_fully_resolved and target.lower() not in [p.lower() for p in processed_list]:
                processed_list.append(target.lower())
                
            step_count += 1
            current_depth += 1
            
            source_desc_list = []
            for src in layer.sources:
                source_desc_list.append(f"{src.table_name}.{src.original_column}")
                # 強制路徑完整：將未處理過的非實體來源表，通通推進 Pending List
                if not src.is_physical:
                    if src.table_name.lower() not in [p.lower() for p in processed_list] and \
                       src.table_name.lower() not in [p.lower() for p in pending_list] and \
                       src.table_name.lower() != target.lower(): # 避免自迴圈
                        pending_list.append(src.table_name)
                        logger.info(f"   ➕ 新增待追溯 Temp Table: {src.table_name}")
                        
            # 組合 Report Step (強制鎖死 Markdown 格式，不給 LLM 發揮)
            step_text = f"### [Step {step_count}: Layer Analysis]\n"
            step_text += f"- **Current Table**: {target}\n"
            step_text += f"- **Target Column Alias**: {layer.target_column_alias}\n"
            step_text += f"- **Source From**: {', '.join(source_desc_list)}\n"
            step_text += "- **BP Logic (業務轉譯)**: \n"
            step_text += f"    * 核心功能：{layer.bp_logic.core_function}\n"
            step_text += f"    * 關鍵轉換：{layer.bp_logic.key_transformation}\n"
            step_text += f"    * 欄位意義：{layer.bp_logic.field_meaning}\n"
            step_text += "---\n"
            
            new_report_steps.append(step_text)

    # ==========================
    # 寫入 Report.txt (短期記憶覆寫)
    # ==========================
    pending_str = f"[{', '.join(pending_list)}]"
    full_report_header = f"""[Analysis Status]
- Final Target: {final_target}
- Current Tracing Depth: Level {current_depth} (0 為最末端)
- Pending List: {pending_str} (目前還在追溯中、尚未找到來源的表)

---
"""
    # 結合 Annotated 自動累加的歷史 steps 寫入
    all_steps = "".join(state.get("report_steps", []) + new_report_steps)
    with open("Report.txt", "w", encoding="utf-8") as f:
        f.write(full_report_header + all_steps)
    
    # ==========================
    # 寫入 Knowledge_Base.md (長期資產擴充)
    # ==========================
    kb_lines = state["existing_kb"].splitlines()
    pattern_idx = next((i for i, line in enumerate(kb_lines) if "Pattern Recognition:" in line), -1)
    if pattern_idx == -1:
        kb_lines.append("\nPattern Recognition:")
        pattern_idx = len(kb_lines) - 1
        
    for kb in parsed.kb_tables:
        line = f"| {kb.table_name} | {kb.business_desc} | {kb.granularity} | {kb.remarks} |"
        if not any(kb.table_name in l for l in kb_lines[:pattern_idx]):
            kb_lines.insert(pattern_idx, line)
            pattern_idx += 1
            
    for pat in parsed.patterns:
        line = f"- {pat}"
        if line not in kb_lines: kb_lines.append(line)
            
    new_kb_content = "\n".join(kb_lines)
    with open("Knowledge_Base.md", "w", encoding="utf-8") as f:
        f.write(new_kb_content)
        
    logger.info(f"   ✅ Chunk 處理完畢，檔案已即時寫入。")

    return {
        "current_chunk_index": state["current_chunk_index"] - 1, # 🌟 推進迴圈：往上溯源
        "pending_list": pending_list,
        "processed_list": processed_list,
        "final_target": final_target,
        "step_count": step_count,
        "current_depth": current_depth,
        "existing_kb": new_kb_content,
        "report_steps": new_report_steps  # 交由 Annotated 進行自動陣列累加
    }

def router_check_continue(state: SQLAnalysisState) -> str:
    """【條件路由】判斷是否所有 Chunk (由下往上) 都讀完了"""
    idx = state["current_chunk_index"]
    
    if idx >= 0:
        logger.info(f"🔀 [ROUTER] 繼續往上溯源 Chunk {idx+1}")
        return "analyze_backward_chunk"
    
    logger.info("🏁 [ROUTER] 已讀取至檔案最頂端，血緣追溯完成！進入 END")
    return END

# ==========================================
# 🕸️ 3. 編譯 LangGraph 工作流
# ==========================================
def build_graph():
    workflow = StateGraph(SQLAnalysisState)
    
    workflow.add_node("init", init_node)
    workflow.add_node("analyze_backward_chunk", analyze_backward_chunk_node)
    workflow.add_node("update", update_node)
    
    workflow.add_edge(START, "init")
    workflow.add_edge("init", "analyze_backward_chunk")
    workflow.add_edge("analyze_backward_chunk", "update")
    
    # 迴圈控制：判斷是否繼續往上讀
    workflow.add_conditional_edges(
        "update",
        router_check_continue,
        {
            "analyze_backward_chunk": "analyze_backward_chunk",
            END: END
        }
    )
    
    return workflow.compile()

# ==========================================
# 🚀 4. 執行測試範例
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(script_dir, "ins_TechNodeFlow.sql")
    
    try:
        with open(sql_path, "r", encoding="utf-8") as f:
            raw_sql = f.read()
    except FileNotFoundError:
        print(f"⚠️ 找不到 {sql_path}，使用測試 SQL 模擬 Bottom-Up 溯源。")
        raw_sql = """
        -- 最上面的程式碼 (實體表撈取)
        SELECT ID, Amt INTO #TempA FROM Physical_T1;
        
        --- CHUNK BOUNDARY ---
        -- 中間的程式碼 (轉換，依賴 #TempA)
        SELECT a.ID, a.Amt, b.Name INTO #TempB FROM #TempA a JOIN Physical_T2 b ON a.ID = b.ID;
        
        --- CHUNK BOUNDARY ---
        -- 最下面的程式碼 (AI 會先讀這裡，自動鎖定 FinalTarget 並推演 #TempB 入 Queue)
        SELECT Name, SUM(Amt) AS Total INTO FinalTarget FROM #TempB GROUP BY Name;
        """

    initial_state = {"raw_sql": raw_sql}

    print("🚀 啟動 SQL-Chronos (逆向 Chunk 溯源模式)...")
    app = build_graph()
    app.invoke(initial_state)

    print("\n" + "="*60)
    print("🏁 全部追溯完成！")
    print("👉 你的 Pending List 與 Target 皆是由 AI 在逆推時動態捕捉出來的！")
    print("👉 請查看目錄下的 `Report.txt` 與 `Knowledge_Base.md`")
    print("="*60 + "\n")