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
    """建立 LLM（從 .env 讀取 OpenAI Compatible API 設定）。"""
    # --- 環境變數讀取 ---
    base_url = os.getenv(
        "OPENAI_COMPAT_BASE_URL",
        "https://model-gateway.gclpgenaigw.gc.micron.com/api/v1",
    )
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "sk-dummy-key")
    model = os.getenv("OPENAI_COMPAT_MODEL", "gpt-5.2-codex")
    default_temperature = float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.7"))
    max_tokens_str = os.getenv("OPENAI_COMPAT_MAX_TOKENS", "")
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "HIGH")
    disable_ssl_verify = os.getenv("OPENAI_COMPAT_VERIFY_SSL", "false").lower() in {
        "0",
        "false",
        "no",
    }
 
    # 允許呼叫端覆寫 temperature
    used_temperature = temperature if temperature is not None else default_temperature
 
    # --- NO_PROXY 設定（Micron 內網需要）---
    no_proxy = os.getenv("NO_PROXY", "gc.micron.com")
    os.environ["NO_PROXY"] = no_proxy
 
    # --- HTTP Client（同步 + 非同步都關閉 SSL 驗證）---
    ssl_verify = not disable_ssl_verify
    http_client = httpx.Client(verify=ssl_verify)
    http_async_client = httpx.AsyncClient(verify=ssl_verify)
 
    # --- 動態組裝 kwargs，避免傳入不支援的參數 ---
    is_gemini = "gemini" in model.lower()
 
    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": used_temperature,
        "http_client": http_client,
        "http_async_client": http_async_client,
        "use_responses_api": False,  # 強制走 /chat/completions，不走 /responses
    }
 
    # max_completion_tokens：僅在有設定值時才傳入（部分 gateway 不支援）
    if max_tokens_str.strip():
        kwargs["max_completion_tokens"] = int(max_tokens_str)
 
    # Gemini extra_body（thinking_config）：僅 Gemini 模型才注入
    if is_gemini and thinking_level:
        kwargs["extra_body"] = {
            "generation_config": {"thinking_config": {"thinking_level": thinking_level}}
        }
 
    logger.info(
        "[LLM] base_url=%s, model=%s, verify_ssl=%s, thinking_level=%s, temperature=%s, is_gemini=%s",
        base_url,
        model,
        not disable_ssl_verify,
        thinking_level if is_gemini else "N/A",
        used_temperature,
        is_gemini,
    )
 
    llm = ChatOpenAI(**kwargs)
    return llm
 
# ==========================================
# 🧠 1. 定義狀態記憶體與 Reducer (解決難點 1)
# ==========================================
# 自訂 Reducer：確保每次迴圈解析的新表，能「增量合併」到記憶體中，而非被下一個 Chunk 覆蓋
def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if a is None: a = {}
    if b is None: b = {}
    res = a.copy()
    res.update(b)
    return res
 
class SQLAnalysisState(TypedDict):
    raw_sql: str
    existing_kb: str
    chunks: List[str]
    current_chunk_index: int
   
    # 動態符號表 (Symbol Table)：JIT 注入上下文
    symbol_table: Annotated[Dict[str, Any], merge_dicts]
   
    # 雙向報告收集器：使用 operator.add 讓每次解析的故事自動 Append 到全局清單
    forward_story: Annotated[List[Dict[str, Any]], operator.add]
    backward_lineage: Annotated[List[Dict[str, Any]], operator.add]
   
    extracted_knowledge: str
    final_report: str
 
# ==========================================
# 🧱 2. Pydantic 強制結構化輸出 (解決難點 2 & 3)
# ==========================================
class ForwardStep(BaseModel):
    step_name: str = Field(description="步驟標題，例如: 建立有效庫存池 (#Temp_ActiveStock)")
    action_logic: str = Field(description="具體的 SQL 動作與業務邏輯描述")
 
class LineageEdge(BaseModel):
    target_column: str = Field(description="產出的目標欄位名稱")
    source_columns: str = Field(description="依賴的來源欄位 (例如 T_Stock.Qty)")
    transformation: str = Field(description="轉換邏輯 (例如 ISNULL, 相乘, SUM)")
 
class TempTableEntry(BaseModel):
    table_name: str = Field(description="Temp Table 名稱，例如 #Temp_ActiveStock")
    description: str = Field(description="用途與 Schema 摘要")
 
class ChunkAnalysisResult(BaseModel):
    new_temp_tables: List[TempTableEntry] = Field(
        description="此區塊新增的 Temp Table 清單。"
    )
    story: ForwardStep = Field(description="前向邏輯故事")
    lineages: List[LineageEdge] = Field(description="後向血緣追蹤")
 
class KnowledgeResult(BaseModel):
    yaml_knowledge: str = Field(description="提煉出的通用業務知識 (YAML 格式)")
 
# ==========================================
# ⚙️ 3. 實作 LangGraph 節點函數 (Nodes)
# ==========================================
def chunk_mssql_sql(raw_sql: str, mode: str = "auto", chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """MSSQL SQL 切塊，支援兩種模式。
 
    Args:
        raw_sql:    原始 SQL 文字
        mode:       切塊模式
                    - "marker"  : 依 --- CHUNK BOUNDARY --- 手動標記切割
                    - "auto"    : 自動偵測；若 SQL 中包含手動標記則走 marker 模式，
                                  否則走固定行數 (chunk_size) + 重疊 (overlap) 模式
        chunk_size: 固定行數模式下，每塊的核心行數 (預設 500)
        overlap:    固定行數模式下，前後重疊行數 (預設 100)
 
    Returns:
        List[str]: 切好的 SQL 區塊清單
    """
    lines = raw_sql.splitlines()
    marker_pattern = re.compile(r"^\s*---\s*CHUNK\s+BOUNDARY\s*---")
 
    # --- 決定實際使用的模式 ---
    has_markers = any(marker_pattern.match(l) for l in lines)
    if mode == "auto":
        effective_mode = "marker" if has_markers else "fixed"
    else:
        effective_mode = mode  # 允許外部強制指定 "marker" 或 "fixed"
 
    # ============================================================
    # 模式 1：依 --- CHUNK BOUNDARY --- 手動標記切割
    # ============================================================
    if effective_mode == "marker":
        chunks: List[str] = []
        current_chunk_lines: List[str] = []
 
        for line in lines:
            if marker_pattern.match(line):
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
            else:
                current_chunk_lines.append(line)
 
        # 收尾
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
 
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks
 
    # ============================================================
    # 模式 2：固定行數 + 前後重疊 (Sliding Window)
    # ============================================================
    total = len(lines)
    if total == 0:
        return []
 
    chunks = []
    start = 0
    step = chunk_size  # 每次往前推進的核心行數
 
    while start < total:
        end = min(start + chunk_size, total)
        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
 
        # 已到尾端，跳出
        if end >= total:
            break
 
        # 下一塊的起點 = 當前結束位置 - 重疊行數 (保證上下文連貫)
        start = end - overlap
 
    return chunks
 
def chunking_node(state: SQLAnalysisState):
    """【節點 1】SQL 切塊：自動偵測手動標記 or 固定 500 行 + 100 行重疊"""
    print("\n" + "="*60)
    print("🟢 [NODE] chunking_node — 開始切塊")
    print("="*60)
    raw_sql = state.get("raw_sql", "")
    print(f"   📄 raw_sql 長度: {len(raw_sql)} 字元")
    chunks = chunk_mssql_sql(raw_sql)  # mode="auto" 會自動判斷
    print(f"   📦 切塊完成：共切分為 {len(chunks)} 個區塊")
    for i, chunk in enumerate(chunks):
        line_count = chunk.count('\n') + 1
        preview = chunk[:80].replace('\n', ' ')
        print(f"   Chunk {i+1} ({line_count} 行): {preview}...")
    print("🔵 [NODE] chunking_node — 結束\n")
    return {"chunks": chunks, "current_chunk_index": 0}
 
def analyze_chunk_node(state: SQLAnalysisState):
    """【節點 2】動態增量分析：一次只看一塊，避免幻覺 (JIT)"""
    idx = state["current_chunk_index"]
    total = len(state["chunks"])
    print("\n" + "="*60)
    print(f"🟢 [NODE] analyze_chunk_node — 分析 Chunk {idx+1}/{total}")
    print("="*60)
    current_chunk = state["chunks"][idx]
    print(f"   📏 當前 Chunk 長度: {len(current_chunk)} 字元")
    print(f"   📋 已知 Symbol Table keys: {list(state.get('symbol_table', {}).keys())}")
   
    print("   ⏳ 建立 LLM (temperature=0.1)...")
    llm = build_llm(temperature=0.1)  # temperature=0.1 確保邏輯精準
    structured_llm = llm.with_structured_output(ChunkAnalysisResult)
    print("   ✅ LLM 建立完成")
   
    # 💡 核心機制：Prompt 中只給「現有底層知識」+「濃縮過的 Symbol Table」+「當前 Chunk」
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位頂尖資料工程師。請分析第 {index} 段 SQL。
       
        【現有通用知識庫 (Context)】:
        {kb}
       
        【已知的 Temp Tables 狀態記憶 (Symbol Table)】:
        {symbol_table}
       
        請提煉出該段 SQL 的新暫存表定義、前向故事與後向血緣。"""),
        ("user", "【當前 SQL 區塊】:\n{chunk}")
    ])
   
    print(f"   ⏳ 呼叫 LLM 分析 Chunk {idx+1}（這可能需要一段時間）...")
    result: ChunkAnalysisResult = structured_llm.invoke(prompt.format_messages(
        index=idx + 1,
        kb=state.get("existing_kb", ""),
        symbol_table=state.get("symbol_table", {}),
        chunk=current_chunk
    ))
    print(f"   ✅ LLM 回傳結果：")
    print(f"      - 新增 Temp Tables: {[t.table_name for t in result.new_temp_tables]}")
    print(f"      - 故事步驟: {result.story.step_name}")
    print(f"      - 血緣邊數: {len(result.lineages)}")
   
    # ---------- 印出完整 LLM 回傳內容 ----------
    print("\n   ╔══════════════════════════════════════════════════")
    print(f"   ║ 📦 [LLM Response] Chunk {idx+1} 完整內容")
    print("   ╠══════════════════════════════════════════════════")
    print("   ║ 🗂️  New Temp Tables:")
    for entry in result.new_temp_tables:
        print(f"   ║   • {entry.table_name}: {entry.description}")
    print("   ║")
    print(f"   ║ 📖 Forward Story:")
    print(f"   ║   步驟名稱: {result.story.step_name}")
    print(f"   ║   動作邏輯: {result.story.action_logic}")
    print("   ║")
    print(f"   ║ 🔗 Backward Lineage ({len(result.lineages)} edges):")
    for i, edge in enumerate(result.lineages, 1):
        print(f"   ║   {i}. {edge.target_column} ← {edge.source_columns}  [{edge.transformation}]")
    print("   ╚══════════════════════════════════════════════════\n")
   
    # 將 step_id 注入，供後續建立 Markdown 錨點跳轉使用
    story_dict = result.story.model_dump()
    story_dict["step_id"] = idx + 1
   
    lineage_list = [l.model_dump() for l in result.lineages]
    for l in lineage_list:
        l["step_id"] = idx + 1
 
    # 回傳更新狀態，LangGraph 的 Annotated 會自動把 Dict Merge，並把 List Append
    print(f"🔵 [NODE] analyze_chunk_node — Chunk {idx+1} 分析結束，推進到 index={idx+1}\n")
    # 將 List[TempTableEntry] 轉換為 Dict 以供 symbol_table merge
    new_tables_dict = {t.table_name: t.description for t in result.new_temp_tables}
    return {
        "current_chunk_index": idx + 1,       # 推進迴圈指標
        "symbol_table": new_tables_dict,
        "forward_story": [story_dict],
        "backward_lineage": lineage_list
    }
 
def router_check_continue(state: SQLAnalysisState) -> str:
    """【條件路由】判斷是否所有 Chunk 都已處理完畢"""
    idx = state["current_chunk_index"]
    total = len(state["chunks"])
    print(f"🔀 [ROUTER] current_chunk_index={idx}, total_chunks={total}", end=" ")
    if idx < total:
        print(f"→ 繼續分析下一個 Chunk (analyze_chunk)")
        return "analyze_chunk"
    print(f"→ 全部分析完成，進入知識萃取 (distill_knowledge)")
    return "distill_knowledge"
 
def distill_knowledge_node(state: SQLAnalysisState):
    """【節點 3】通用知識萃取：雙軌過濾器"""
    print("\n" + "="*60)
    print("🟢 [NODE] distill_knowledge_node — 開始知識萃取")
    print("="*60)
    print(f"   📊 累計故事數: {len(state.get('forward_story', []))}")
    print(f"   📊 累計血緣邊數: {len(state.get('backward_lineage', []))}")
    print(f"   📊 Symbol Table keys: {list(state.get('symbol_table', {}).keys())}")
   
    print("   ⏳ 建立 LLM (temperature=0.2)...")
    llm = build_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(KnowledgeResult)
    print("   ✅ LLM 建立完成")
   
    # LLM 此時不再看原始 SQL，而是看提煉出的故事與血緣，進行過濾
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位資料治理專家。請檢視這次 SQL 分析出來的邏輯與血緣。
        【任務】：
        1. 🔴 拋棄 Ad-hoc 邏輯：特定日期區間、針對特定部門、暫存表間的內部關聯。
        2. 🟢 提煉 Domain Knowledge：實體底層表之間的防呆極限 (如 ISNULL)、狀態碼意義、常駐關聯鍵。
        請嚴格以 YAML 格式輸出，方便存入 Data Dictionary。"""),
        ("user", "【分析故事】:\n{stories}\n\n【血緣追蹤】:\n{lineages}")
    ])
   
    print("   ⏳ 呼叫 LLM 進行知識萃取（這可能需要一段時間）...")
    result: KnowledgeResult = structured_llm.invoke(prompt.format_messages(
        stories=state.get("forward_story", []),
        lineages=state.get("backward_lineage", [])
    ))
    print(f"   ✅ 知識萃取完成，YAML 長度: {len(result.yaml_knowledge)} 字元")
   
    # ---------- 印出完整 LLM 回傳的 YAML 知識 ----------
    print("\n   ╔══════════════════════════════════════════════════")
    print("   ║ 🧠 [LLM Response] 萃取出的 YAML 知識")
    print("   ╠══════════════════════════════════════════════════")
    for line in result.yaml_knowledge.strip().splitlines():
        print(f"   ║ {line}")
    print("   ╚══════════════════════════════════════════════════\n")
   
    print("🔵 [NODE] distill_knowledge_node — 結束\n")
    return {"extracted_knowledge": result.yaml_knowledge}
 
def generate_report_node(state: SQLAnalysisState):
    """【節點 4】生成雙向整合報告：利用 Markdown 錨點無縫串聯"""
    print("\n" + "="*60)
    print("🟢 [NODE] generate_report_node — 開始產生報告")
    print("="*60)
    report = "# 📊 SQL-Chronos 雙向分析與知識庫報告\n\n"
   
    # 1. 後向血緣追溯矩陣 (查 Bug 逆推專用)
    report += "## 🔍 一、 後向血緣追溯矩陣 (Backward Lineage)\n"
    report += "| 目標欄位 (Target) | 來源欄位 (Source) | 轉換邏輯 | 參考步驟 (跳轉) |\n"
    report += "|:---|:---|:---|:---|\n"
    for lin in state.get("backward_lineage", []):
        step_id = lin["step_id"]
        # 建立 Markdown HTML 錨點超連結
        anchor_link = f"👉 [詳見 Step {step_id}](#step-{step_id})"
        report += f"| `{lin['target_column']}` | `{lin['source_columns']}` | {lin['transformation']} | {anchor_link} |\n"
 
    # 2. 前向邏輯演變 (業務邏輯對焦專用)
    report += "\n## 📖 二、 前向邏輯演變 (Forward Storytelling)\n"
    for story in state.get("forward_story", []):
        step_id = story["step_id"]
        # 定義接收跳轉的 HTML 錨點 id
        report += f"### <a id='step-{step_id}'></a>Step {step_id}: {story['step_name']}\n"
        report += f"- **動作與邏輯**: {story['action_logic']}\n\n"
       
    # 3. 提煉通用知識
    report += "## 🧠 三、 更新後的通用知識庫 (Knowledge Update)\n```yaml\n"
    report += state.get("extracted_knowledge", "").strip()
    report += "\n```\n"
   
    print(f"   📝 報告長度: {len(report)} 字元")
    print("🔵 [NODE] generate_report_node — 結束\n")
    return {"final_report": report}
 
# ==========================================
# 🕸️ 4. 編譯 LangGraph 工作流
# ==========================================
def build_graph():
    workflow = StateGraph(SQLAnalysisState)
   
    workflow.add_node("chunking", chunking_node)
    workflow.add_node("analyze_chunk", analyze_chunk_node)
    workflow.add_node("distill_knowledge", distill_knowledge_node)
    workflow.add_node("generate_report", generate_report_node)
   
    workflow.add_edge(START, "chunking")
    workflow.add_edge("chunking", "analyze_chunk")
   
    # 🌟 動態迴圈：只要 Chunk 沒讀完，就繞回 analyze_chunk 繼續讀下一段
    workflow.add_conditional_edges(
        "analyze_chunk",
        router_check_continue,
        {
            "analyze_chunk": "analyze_chunk",
            "distill_knowledge": "distill_knowledge"
        }
    )
   
    workflow.add_edge("distill_knowledge", "generate_report")
    workflow.add_edge("generate_report", END)
   
    return workflow.compile()
 
# ==========================================
# 🚀 5. 執行測試範例
# ==========================================
if __name__ == "__main__":
    # 檔案路徑 (與 agent.py 同目錄)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(script_dir, "ins_TechNodeFlow.sql")
    kb_path = os.path.join(script_dir, "knowledge_base.txt")
 
    # 讀取 SQL 與知識庫檔案
    with open(sql_path, "r", encoding="utf-8") as f:
        raw_sql = f.read()
    with open(kb_path, "r", encoding="utf-8") as f:
        existing_kb = f.read()
 
    print(f"📂 SQL 檔案: {sql_path}")
    print(f"📂 知識庫檔案: {kb_path}")
 
    # 1. 初始化 Graph
    app = build_graph()
 
    # 2. 定義初始 State
    initial_state = {
        "raw_sql": raw_sql,
        "existing_kb": existing_kb,
        "symbol_table": {},
        "forward_story": [],
        "backward_lineage": []
    }
 
    print("🚀 啟動 SQL-Chronos 引擎分析中...")
    print(f"   raw_sql 長度: {len(raw_sql)} 字元")
    print(f"   existing_kb 長度: {len(existing_kb)} 字元")
 
    # 3. 執行整個狀態機
    import time
    start_time = time.time()
    final_state = app.invoke(initial_state)
    elapsed = time.time() - start_time
 
    # 4. 印出完美的 Markdown 報告
    print("\n" + "="*60)
    print(f"🏁 全部完成！總耗時: {elapsed:.1f} 秒")
    print("="*60 + "\n")
    print(final_state["final_report"])