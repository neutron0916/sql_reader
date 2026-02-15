import os
import re
import operator
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

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

class ChunkAnalysisResult(BaseModel):
    new_temp_tables: Dict[str, str] = Field(
        description="此區塊新增的 Temp Table。Key為表名，Value為用途與Schema摘要。"
    )
    story: ForwardStep = Field(description="前向邏輯故事")
    lineages: List[LineageEdge] = Field(description="後向血緣追蹤")

class KnowledgeResult(BaseModel):
    yaml_knowledge: str = Field(description="提煉出的通用業務知識 (YAML 格式)")

# ==========================================
# ⚙️ 3. 實作 LangGraph 節點函數 (Nodes)
# ==========================================
def chunk_mssql_sql(raw_sql: str) -> List[str]:
    """智慧 MSSQL 語義切塊：自動識別 Temp Table 邊界，無需手動標記
    
    辨識的切割邊界 (優先級由高到低)：
      1. IF OBJECT_ID('tempdb..#xxx') IS NOT NULL DROP TABLE #xxx
      2. SELECT ... INTO #TempTable
      3. CREATE TABLE #TempTable
      4. GO 批次分隔符
      5. --- CHUNK BOUNDARY --- (向下相容手動標記)
    """
    lines = raw_sql.splitlines()
    
    # 定義 MSSQL 語義邊界的 Regex 模式
    boundary_patterns = [
        # 模式 1：IF OBJECT_ID('tempdb..#xxx') — 最常見的 Temp Table 防呆清理
        re.compile(r"^\s*IF\s+OBJECT_ID\s*\(", re.IGNORECASE),
        # 模式 2：SELECT ... INTO #TempTable (不在 IF 區塊內的獨立語句)
        re.compile(r"^\s*SELECT\s+", re.IGNORECASE),
        # 模式 3：CREATE TABLE #TempTable
        re.compile(r"^\s*CREATE\s+TABLE\s+#", re.IGNORECASE),
        # 模式 4：GO 批次分隔符
        re.compile(r"^\s*GO\s*$", re.IGNORECASE),
        # 模式 5：向下相容手動標記
        re.compile(r"^\s*---\s*CHUNK\s+BOUNDARY\s*---"),
    ]
    
    # 額外檢測：這一行是否包含 INTO # (用來判斷 SELECT INTO 區塊)
    into_temp_pattern = re.compile(r"\bINTO\s+#", re.IGNORECASE)
    
    chunks: List[str] = []
    current_chunk_lines: List[str] = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 跳過純空行 (不作為切割依據)
        if not stripped:
            if current_chunk_lines:
                current_chunk_lines.append(line)
            i += 1
            continue
        
        # 跳過 GO 分隔符，直接切塊
        if re.match(r"^\s*GO\s*$", stripped, re.IGNORECASE):
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
            i += 1
            continue
        
        # 跳過手動標記
        if re.match(r"^\s*---\s*CHUNK\s+BOUNDARY\s*---", stripped):
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
            i += 1
            continue
        
        # 檢測 IF OBJECT_ID 邊界
        if re.match(r"^\s*IF\s+OBJECT_ID\s*\(", stripped, re.IGNORECASE):
            # 把已累積的行作為上一個 chunk
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
            current_chunk_lines.append(line)
            i += 1
            continue
        
        # 檢測 CREATE TABLE # 邊界
        if re.match(r"^\s*CREATE\s+TABLE\s+#", stripped, re.IGNORECASE):
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
            current_chunk_lines.append(line)
            i += 1
            continue
        
        # 檢測 SELECT 開頭的語句
        if re.match(r"^\s*SELECT\s+", stripped, re.IGNORECASE):
            # 先往後看幾行，確認是否包含 INTO # (SELECT ... INTO #Temp)
            lookahead_text = stripped
            for j in range(i + 1, min(i + 10, len(lines))):
                lookahead_text += " " + lines[j].strip()
                if into_temp_pattern.search(lookahead_text):
                    break
            
            is_into_temp = into_temp_pattern.search(lookahead_text)
            
            if is_into_temp:
                # 這是一個 SELECT INTO #Temp 語句 → 新 chunk
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
            # 無論是否 INTO #，都繼續累積
            current_chunk_lines.append(line)
            i += 1
            continue
        
        # 一般行：繼續累積到當前 chunk
        current_chunk_lines.append(line)
        i += 1
    
    # 收尾：把最後累積的行加入
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    # 清理每個 chunk 的首尾空白
    chunks = [c.strip() for c in chunks if c.strip()]
    
    return chunks

def chunking_node(state: SQLAnalysisState):
    """【節點 1】智慧語義切塊：自動辨識 MSSQL 邊界 (IF OBJECT_ID / SELECT INTO # / CREATE TABLE # / GO)"""
    raw_sql = state.get("raw_sql", "")
    chunks = chunk_mssql_sql(raw_sql)
    print(f"📦 智慧切塊完成：共切分為 {len(chunks)} 個語義區塊")
    for i, chunk in enumerate(chunks):
        preview = chunk[:80].replace('\n', ' ')
        print(f"   Chunk {i+1}: {preview}...")
    return {"chunks": chunks, "current_chunk_index": 0}

def analyze_chunk_node(state: SQLAnalysisState):
    """【節點 2】動態增量分析：一次只看一塊，避免幻覺 (JIT)"""
    idx = state["current_chunk_index"]
    current_chunk = state["chunks"][idx]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # temperature=0 確保邏輯精準
    structured_llm = llm.with_structured_output(ChunkAnalysisResult)
    
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
    
    result: ChunkAnalysisResult = structured_llm.invoke(prompt.format_messages(
        index=idx + 1,
        kb=state.get("existing_kb", ""),
        symbol_table=state.get("symbol_table", {}),
        chunk=current_chunk
    ))
    
    # 將 step_id 注入，供後續建立 Markdown 錨點跳轉使用
    story_dict = result.story.model_dump()
    story_dict["step_id"] = idx + 1
    
    lineage_list = [l.model_dump() for l in result.lineages]
    for l in lineage_list:
        l["step_id"] = idx + 1

    # 回傳更新狀態，LangGraph 的 Annotated 會自動把 Dict Merge，並把 List Append
    return {
        "current_chunk_index": idx + 1,       # 推進迴圈指標
        "symbol_table": result.new_temp_tables,
        "forward_story": [story_dict],
        "backward_lineage": lineage_list
    }

def router_check_continue(state: SQLAnalysisState) -> str:
    """【條件路由】判斷是否所有 Chunk 都已處理完畢"""
    if state["current_chunk_index"] < len(state["chunks"]):
        return "analyze_chunk"
    return "distill_knowledge"

def distill_knowledge_node(state: SQLAnalysisState):
    """【節點 3】通用知識萃取：雙軌過濾器"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    structured_llm = llm.with_structured_output(KnowledgeResult)
    
    # LLM 此時不再看原始 SQL，而是看提煉出的故事與血緣，進行過濾
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位資料治理專家。請檢視這次 SQL 分析出來的邏輯與血緣。
        【任務】：
        1. 🔴 拋棄 Ad-hoc 邏輯：特定日期區間、針對特定部門、暫存表間的內部關聯。
        2. 🟢 提煉 Domain Knowledge：實體底層表之間的防呆極限 (如 ISNULL)、狀態碼意義、常駐關聯鍵。
        請嚴格以 YAML 格式輸出，方便存入 Data Dictionary。"""),
        ("user", "【分析故事】:\n{stories}\n\n【血緣追蹤】:\n{lineages}")
    ])
    
    result: KnowledgeResult = structured_llm.invoke(prompt.format_messages(
        stories=state.get("forward_story", []),
        lineages=state.get("backward_lineage", [])
    ))
    
    return {"extracted_knowledge": result.yaml_knowledge}

def generate_report_node(state: SQLAnalysisState):
    """【節點 4】生成雙向整合報告：利用 Markdown 錨點無縫串聯"""
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
    sql_path = os.path.join(script_dir, "input.sql")
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

    # 3. 執行整個狀態機
    final_state = app.invoke(initial_state)

    # 4. 印出完美的 Markdown 報告
    print("\n" + "="*60 + "\n")
    print(final_state["final_report"])
