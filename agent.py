import logging
import os
import re
import copy
import warnings
from typing import TypedDict, List, Dict, Any

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
    """保留企業級設定，預設溫度設為 0 以確保血緣精準對齊"""
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "https://model-gateway.gclpgenaigw.gc.micron.com/api/v1")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "sk-dummy-key")
    model = os.getenv("OPENAI_COMPAT_MODEL", "gpt-5.2-codex")
    default_temperature = float(os.getenv("OPENAI_COMPAT_TEMPERATURE", "0.0"))
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
# 🧠 1. 定義狀態記憶體與 Pydantic
# ==========================================
class SQLAnalysisState(TypedDict):
    raw_sql: str
    chunks: List[str]
    current_chunk_index: int
    
    final_target: str
    pending_list: List[str]
    processed_list: List[str]
    
    # 🌟 核心：取代流水帳，這是一個全域「表與欄位關係」的 Graph 字典
    table_metadata: Dict[str, Any] 
    
    parsed_result: Dict[str, Any]
    final_report: str

class SourceColumn(BaseModel):
    table_name: str = Field(description="來源表名 (例如 #LP, ORD_HDR)")
    column_name: str = Field(description="來源欄位名")
    is_physical: bool = Field(description="是否為實體表 (非 # 開頭且非 CTE)")

class TargetColumn(BaseModel):
    column_name: str = Field(description="產出的目標欄位名稱")
    description: str = Field(description="欄位的業務意義與轉換邏輯 (des)")
    sources: List[SourceColumn] = Field(description="此欄位依賴的『直接』來源表與欄位")

class ParsedTable(BaseModel):
    table_name: str = Field(description="被建立或寫入的目標表名")
    table_description: str = Field(description="這張表的核心功能描述 (des)")
    is_fully_resolved: bool = Field(description="是否為完整建立(CREATE/INTO/CTE)。若是 UPDATE 則填 False")
    columns: List[TargetColumn] = Field(description="表內欄位定義與來源")

class ChunkParseResult(BaseModel):
    found_tables: List[ParsedTable] = Field(description="此 SQL 區塊中建立/更新的所有表")

# ==========================================
# ⚙️ 2. 解析與圖表建立 Nodes
# ==========================================
def chunk_mssql_sql(raw_sql: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
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
    logger.info("🟢 [NODE] init_node — 讀取 SQL 並切塊")
    chunks = chunk_mssql_sql(state["raw_sql"])
    return {
        "chunks": chunks,
        "current_chunk_index": len(chunks) - 1, # 指標指向最底層
        "final_target": "Unknown",
        "pending_list": [],
        "processed_list": [],
        "table_metadata": {}
    }

def analyze_backward_chunk_node(state: SQLAnalysisState):
    idx = state["current_chunk_index"]
    current_chunk = state["chunks"][idx]
    pending_list = state.get("pending_list", [])
    
    logger.info(f"\n{'='*60}\n🟢 [NODE] analyze_chunk — 解析 Chunk {idx+1}/{len(state['chunks'])}")
    logger.info(f"   🔍 待尋找來源之 Temp Tables: {pending_list}")

    llm = build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ChunkParseResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位頂尖的資料工程師。我們正在「由後往前 (Bottom-Up)」逆向閱讀 SQL。
        【任務規則】：
        1. 找出區塊中被建立/更新的目標表。若 Pending List {pending_list} 不為空，優先尋找它們。若為空，找出最終產出表。
        2. 針對目標表，精準解析【各個欄位】直接來自哪個來源表與來源欄位。
        3. 【多對一處理】：若欄位由多表 JOIN / 運算而成，必須將所有來源列入 sources。
        """),
        ("user", "【當前 SQL 區塊】:\n{chunk}")
    ])
    
    result: ChunkParseResult = structured_llm.invoke(prompt.format_messages(
        pending_list=pending_list, chunk=current_chunk
    ))
    return {"parsed_result": result.model_dump()}

def update_node(state: SQLAnalysisState):
    logger.info("🟢 [NODE] update_node — 將依賴關係寫入記憶體 Graph")
    
    parsed = ChunkParseResult(**state.get("parsed_result", {}))
    pending_list = state.get("pending_list", []).copy()
    processed_list = state.get("processed_list", []).copy()
    final_target = state.get("final_target", "Unknown")
    table_metadata = copy.deepcopy(state.get("table_metadata", {}))
    
    # 反轉以確保同 Chunk 內的 CTE/TEMP 先後順序正常
    parsed_tables = parsed.found_tables.copy()
    parsed_tables.reverse()
    
    for layer in parsed_tables:
        target = layer.table_name
        is_relevant = False
        
        if final_target == "Unknown" and not pending_list:
            final_target = target
            is_relevant = True
        elif target.lower() in [p.lower() for p in pending_list]:
            is_relevant = True

        if is_relevant:
            if layer.is_fully_resolved:
                pending_list = [p for p in pending_list if p.lower() != target.lower()]
                if target.lower() not in processed_list:
                    processed_list.append(target.lower())
                    
            if target not in table_metadata:
                table_metadata[target] = {
                    "description": layer.table_description,
                    "columns": {}
                }
                
            # 欄位解析與儲存 (包含 UPDATE 語句的欄位合併)
            for col in layer.columns:
                col_name = col.column_name
                sources_list = [{"table": s.table_name, "column": s.column_name} for s in col.sources]
                
                if col_name not in table_metadata[target]["columns"]:
                    table_metadata[target]["columns"][col_name] = {
                        "description": col.description,
                        "sources": sources_list
                    }
                else:
                    for s in sources_list:
                        if s not in table_metadata[target]["columns"][col_name]["sources"]:
                            table_metadata[target]["columns"][col_name]["sources"].append(s)
                            
                # 將未處理的來源表推入 Pending List
                for s in col.sources:
                    s_name = s.table_name
                    if not s.is_physical and s_name.lower() not in processed_list and s_name.lower() not in [p.lower() for p in pending_list] and s_name.lower() != target.lower():
                        pending_list.append(s_name)

    return {
        "current_chunk_index": state["current_chunk_index"] - 1,
        "pending_list": pending_list,
        "processed_list": processed_list,
        "final_target": final_target,
        "table_metadata": table_metadata
    }

def router_check_continue(state: SQLAnalysisState) -> str:
    if state["current_chunk_index"] >= 0:
        return "analyze_backward_chunk"
    logger.info("🏁 [ROUTER] 全文溯源完畢，進入報表渲染階段！")
    return "generate_report"

# ==========================================
# 📊 3. 終極報表生成 (遞迴 DFS 尋路演算法)
# ==========================================
def generate_report_node(state: SQLAnalysisState):
    logger.info("\n🟢 [NODE] generate_report_node — 展開血緣樹並產生格式化藍圖")
    
    metadata = state.get("table_metadata", {})
    final_target = state.get("final_target", "Unknown")
    
    # 🌟 核心：遞迴尋路演算法，負責自動串接 #LP ---> #CT ---> 底層table
    def get_source_path(table: str, col: str, visited=None) -> list:
        if visited is None: visited = set()
        node_key = f"{table}.{col}".lower()
        if node_key in visited: return ["[Loop Detected]"]
        visited.add(node_key)
        
        t_key = next((k for k in metadata if k.lower() == table.lower()), None)
        if not t_key: return []
        
        c_key = next((k for k in metadata[t_key]["columns"] if k.lower() == col.lower()), None)
        if not c_key: return []
        
        sources = metadata[t_key]["columns"][c_key]["sources"]
        paths = []
        for src in sources:
            src_tb = src["table"]
            src_col = src["column"]
            sub_paths = get_source_path(src_tb, src_col, visited.copy())
            
            # 若它還有來源，將其用 ---> 串接起來
            if sub_paths:
                for sp in sub_paths:
                    paths.append(f"{src_tb} ---> {sp}")
            else:
                paths.append(f"{src_tb}")
                
        return list(dict.fromkeys(paths))

    report = []
    
    # ------------------------------------
    # 區塊 1: Target Table (Markdown)
    # ------------------------------------
    report.append(f"# 🎯 SQL 血緣追溯報告")
    report.append("")
    report.append(f"## 目標表 (Target Table): `{final_target}`")
    report.append("")
    report.append("| 欄位名稱 | 業務描述 | 完整來源路徑 |")
    report.append("|----------|----------|-------------|")
    
    t_key = next((k for k in metadata if k.lower() == final_target.lower()), None)
    if t_key:
        for col_name, col_info in metadata[t_key]["columns"].items():
            desc = col_info["description"]
            paths = get_source_path(final_target, col_name)
            src_str = " , ".join(paths) if paths else "Unknown"
            report.append(f"| `{col_name}` | {desc} | {src_str} |")
    else:
        report.append("| - | (無欄位資訊) | - |")
        
    report.append("")
    
    # ------------------------------------
    # 區塊 2: Temp Table List (Markdown)
    # ------------------------------------
    report.append("## 🗂️ 中繼表清單 (Temp Tables)")
    report.append("")
    idx = 1
    # 反轉順序：原始 metadata 是由後往前分析的，報告改為由前往後呈現
    temp_tables = [(t_name, t_info) for t_name, t_info in metadata.items()
                   if t_name.lower() != final_target.lower()]
    temp_tables.reverse()
    
    for t_name, t_info in temp_tables:
            
        desc = t_info["description"]
        report.append(f"### {idx}. `{t_name}` — {desc}")
        report.append("")
        report.append("| 欄位名稱 | 業務描述 | 完整來源路徑 |")
        report.append("|----------|----------|-------------|")
        
        for col_name, col_info in t_info["columns"].items():
            c_desc = col_info["description"]
            paths = get_source_path(t_name, col_name)
            src_str = " , ".join(paths) if paths else "Unknown"
            report.append(f"| `{col_name}` | {c_desc} | {src_str} |")
            
        report.append("")
        idx += 1
    
    # ------------------------------------
    # 區塊 3: LLM 生成 Mermaid 流程圖
    # ------------------------------------
    report.append("## 📊 資料流程圖 (Data Lineage Flowchart)")
    report.append("")
    
    # 組裝依賴關係描述，交給 LLM 產生 Mermaid
    deps = {}
    for t_name, t_info in metadata.items():
        src_tables = set()
        for col_info in t_info["columns"].values():
            for src in col_info["sources"]:
                src_t = src["table"].upper()
                if src_t != t_name.upper():
                    src_tables.add(src_t)
        if src_tables:
            deps[t_name.upper()] = src_tables
    
    if deps:
        # 構建依賴描述文字（含表描述，讓 Mermaid 節點顯示說明）
        dep_lines = []
        for tgt, srcs in deps.items():
            for s in srcs:
                dep_lines.append(f"{s} --> {tgt}")
        dep_text = "\n".join(dep_lines)
        
        # 蒐集各表描述，傳給 LLM 讓節點帶說明
        table_descs = []
        for t_name, t_info in metadata.items():
            desc = t_info.get("description", "")
            if desc:
                table_descs.append(f"{t_name}: {desc}")
        desc_text = "\n".join(table_descs)
        
        logger.info("   🎨 呼叫 LLM 生成 Mermaid 流程圖...")
        llm = build_llm(temperature=0.0)
        mermaid_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位資料視覺化專家。根據以下的表依賴關係與表描述，生成一段 Mermaid flowchart 語法。
規則：
1. 使用 `graph LR`（從左至右）排列。
2. 實體表 (不以 # 開頭) 使用圓角矩形，節點標籤格式為 `(["表名\n描述"])`。
3. 暫存表 (以 # 開頭) 使用方框，節點標籤格式為 `["表名\n描述"]`。
4. 最終目標表用加粗雙框 `[["表名\n描述"]]` 標示。
5. 只需要回傳純 Mermaid 語法，不要加 ```mermaid 標記，不要加任何解釋文字。
6. 為暫存表節點使用不含 # 的 ID，例如 `LP["#LP\n篩選基礎資料"]`。
7. 若表沒有描述，則只寫表名即可。"""),
            ("user", """目標表: {target}

依賴關係:
{deps}

各表描述:
{descs}

請生成 Mermaid flowchart。""")
        ])
        
        mermaid_result = llm.invoke(mermaid_prompt.format_messages(
            target=final_target, deps=dep_text, descs=desc_text
        ))
        mermaid_code = mermaid_result.content.strip()
        
        # 清理 LLM 可能多加的 markdown 標記
        if mermaid_code.startswith("```"):
            lines = mermaid_code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            mermaid_code = "\n".join(lines)
        
        report.append("```mermaid")
        report.append(mermaid_code)
        report.append("```")
    else:
        report.append("*（無跨表依賴關係，無需流程圖）*")

    report.append("")
    report.append("---")
    report.append("*此報告由 SQL-Chronos 自動生成*")
    
    final_report = "\n".join(report)
    
    with open("Report.md", "w", encoding="utf-8") as f:
        f.write(final_report)
        
    logger.info("   ✅ 完美格式的 Report.md 產生完畢！")
    return {"final_report": final_report}

# ==========================================
# 🕸️ 4. 編譯 LangGraph 工作流
# ==========================================
def build_graph():
    workflow = StateGraph(SQLAnalysisState)
    
    workflow.add_node("init", init_node)
    workflow.add_node("analyze_backward_chunk", analyze_backward_chunk_node)
    workflow.add_node("update", update_node)
    workflow.add_node("generate_report", generate_report_node)
    
    workflow.add_edge(START, "init")
    workflow.add_edge("init", "analyze_backward_chunk")
    workflow.add_edge("analyze_backward_chunk", "update")
    
    workflow.add_conditional_edges(
        "update",
        router_check_continue,
        {
            "analyze_backward_chunk": "analyze_backward_chunk",
            "generate_report": "generate_report"
        }
    )
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()

# ==========================================
# 🚀 5. 執行入口
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(script_dir, "ins_TechNodeFlow.sql")
    
    try:
        with open(sql_path, "r", encoding="utf-8") as f:
            raw_sql = f.read()
    except FileNotFoundError:
        print(f"⚠️ 找不到 {sql_path}，使用測試 SQL 模擬。")
        raw_sql = """
        SELECT ID, Dept INTO #LP FROM Base_Dept;
        --- CHUNK BOUNDARY ---
        SELECT a.ID, a.Dept, b.Salary INTO #CT FROM #LP a JOIN Base_Salary b ON a.ID = b.ID;
        --- CHUNK BOUNDARY ---
        SELECT Dept, SUM(Salary) AS Total_Salary INTO Final_Report FROM #CT GROUP BY Dept;
        """

    initial_state = {"raw_sql": raw_sql}

    print("🚀 啟動 SQL-Chronos (遞迴圖論溯源模式)...")
    app = build_graph()
    result = app.invoke(initial_state)

    print("\n" + "="*60)
    print("🏁 全部追溯完成！以下為 Report.md 產出預覽：")
    print("="*60 + "\n")
    print(result["final_report"])