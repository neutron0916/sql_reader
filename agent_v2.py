import os
import re
import json
import httpx
import sqlglot
from sqlglot import exp
from typing import Any, List, Dict, Set, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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
# 1. 結構化輸出模型 (Pydantic Models)
# ==========================================
class ColumnTransformation(BaseModel):
    target_column: str = Field(description="目標欄位名稱")
    source_expression: str = Field(description="來源 SQL 數學/邏輯表達式")
    business_meaning: str = Field(description="此公式在業務上的意義 (如: 計算稅後淨利)")

class DriftDetection(BaseModel):
    has_conflict: bool = Field(description="SQL 邏輯是否與知識庫描述發生衝突？")
    warning_message: Optional[str] = Field(None, description="若有衝突，說明原因給人類審查")

class AnalysisResult(BaseModel):
    business_logic: str = Field(description="此 SQL 區塊整體的隱含業務邏輯")
    data_quality_rules: List[str] = Field(description="潛在 DQ 規則 (如: DISTINCT 暗示需去重, LEFT JOIN 暗示 Null 風險)")
    transformations: List[ColumnTransformation] = Field(description="欄位級轉換邏輯")
    drift_check: DriftDetection

# ==========================================
# 2. 靈魂狀態機定義 (LangGraph State)
# ==========================================
class EngineState(TypedDict):
    chunks: List[str]                  
    current_idx: int                   
    pending_list: List[str]            
    resolved_tables: Set[str]          
    knowledge_base: Dict[str, str]     
    nodes: Dict[str, dict]
    edges: List[dict]
    drift_warnings: List[str]

# ==========================================
# 3. MSSQL 智能切割器 (The Chunker)
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

# ==========================================
# 4. Hybrid Parser: AST 確定性地基提取
# ==========================================
def extract_ast_ground_truth(sql_chunk: str) -> dict:
    info = {"target": None, "sources": set(), "expressions": {}}
    try:
        parsed = sqlglot.parse_one(sql_chunk, read="tsql")
        if not parsed: return info
            
        # 提取 目標表 (SELECT INTO, INSERT)
        for into in parsed.find_all(exp.Into):
            if isinstance(into.this, exp.Table):
                info["target"] = into.this.name.upper()
        if not info["target"] and isinstance(parsed, exp.Insert):
            info["target"] = parsed.this.name.upper()
            
        # 提取 來源表 (排除 Target 自己)
        for table in parsed.find_all(exp.Table):
            t_name = table.name.upper()
            if t_name and t_name != info["target"]:
                info["sources"].add(t_name)
                
        # 提取 欄位級公式
        for select in parsed.find_all(exp.Select):
            for proj in select.expressions:
                if isinstance(proj, exp.Alias):
                    info["expressions"][proj.alias] = proj.this.sql(dialect="tsql")
    except Exception:
        pass 
        
    info["sources"] = list(info["sources"])
    return info

# ==========================================
# 5. LangGraph 節點：逆向追蹤與語意引擎
# ==========================================
def node_analyze_backward(state: EngineState) -> dict:
    idx = state["current_idx"]
    chunk = state["chunks"][idx]
    
    pending = list(state["pending_list"])
    resolved = set(state["resolved_tables"])
    nodes = dict(state["nodes"])
    edges = list(state["edges"])
    drifts = list(state["drift_warnings"])
    
    ast_info = extract_ast_ground_truth(chunk)
    target = ast_info["target"] or f"UNKNOWN_TARGET_{idx}"
    
    # 【核心演算法剪枝】：判斷是否為「最終輸出目標」或「處於 Pending List 的依賴」
    is_final_chunk = (idx == len(state["chunks"]) - 1)
    
    if is_final_chunk or target in pending:
        print(f"🔍 [Bottom-Up 追蹤] 正在解析目標表: {target} (Chunk Index: {idx})")
        
        if target in pending: pending.remove(target)
        resolved.add(target)
        
        # 逆向生長：將未知依賴 (#Temp 或 CTE) 推入 Pending
        for src in ast_info["sources"]:
            if (src.startswith("#") or src.startswith("CTE")) and src not in resolved and src not in pending:
                pending.append(src)
                print(f"   ➔ 發現暫存上游依賴，推入 Pending List: {src}")
        
        # 呼叫 LLM 進行語意與漂移分析
        kb_context = state["knowledge_base"].get(target, "尚無此表的知識紀錄。")
        llm = build_llm(temperature=0).with_structured_output(AnalysisResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位精通企業級資料架構的 Data Architect。基於『AST 客觀結構』推斷業務邏輯。\n"
                       "【嚴格任務】：比對 SQL 邏輯與現有知識庫，若發現 WHERE 條件等業務邏輯發生衝突，務必將 has_conflict 設為 True 並發出警告！"),
            ("user", "Target Table: {target}\nAST 解析結果: {ast}\n\n原始 SQL: {sql}\n\n現有知識庫: {kb}")
        ])
        
        try:
            print(f"   🧠 AI 語意分析中...")
            chain = prompt | llm
            ai_res: AnalysisResult = chain.invoke({
                "target": target, "ast": json.dumps(ast_info, ensure_ascii=False), "sql": chunk, "kb": kb_context
            })
            
            # --- 構建節點與連線 ---
            nodes[target] = {
                "id": target, "label": target,
                "type": "Temp" if target.startswith("#") else "Physical",
                "business_logic": ai_res.business_logic,
                "dq_rules": ai_res.data_quality_rules
            }
            
            for src in ast_info["sources"]:
                if src not in nodes:
                    nodes[src] = {"id": src, "label": src, "type": "Physical", "business_logic": "Raw Source", "dq_rules": []}
                
                tooltip_html = "<div style='font-family: monospace; padding: 4px;'><b style='color:#60a5fa'>Transformations:</b><br/>"
                for t in ai_res.transformations:
                    tooltip_html += f"• <b>{t.target_column}</b> = <span style='color:#cbd5e1'>{t.source_expression}</span><br/><i style='color:#94a3b8; font-size: 11px;'>({t.business_meaning})</i><br/>"
                tooltip_html += "</div>"
                
                edges.append({
                    "from": src, "to": target,
                    "title": tooltip_html,
                    "transformations": [t.model_dump() for t in ai_res.transformations]
                })
                
            if ai_res.drift_check.has_conflict:
                drifts.append(f"**Table `{target}`**: {ai_res.drift_check.warning_message}")
                print(f"   ⚠️ 攔截到語意衝突 (Semantic Drift)！")
                
        except Exception as e:
            print(f"   ❌ LLM 分析失敗 ({target}): {e}")
            
    else:
        print(f"⏭️ [Bottom-Up 剪枝] 略過無關廢棄 Chunk (Target: {target}, Index: {idx})")

    return {
        "current_idx": idx - 1,
        "pending_list": pending,
        "resolved_tables": resolved,
        "nodes": nodes, "edges": edges, "drift_warnings": drifts
    }

def router_should_continue(state: EngineState) -> str:
    # 只要指標小於 0 結束。 
    # 【Early Stop】：如果 pending_list 空了，代表逆向追蹤的依賴網已經閉環，後面的廢棄程式碼可以直接跳過！
    if state["current_idx"] < 0 or (len(state["pending_list"]) == 0 and len(state["resolved_tables"]) > 0):
        return "generate"
    return "analyze"

# ==========================================
# 6. 生成標準化輸出與現代化儀表板
# ==========================================
def node_generate_outputs(state: EngineState) -> dict:
    nodes_data = list(state["nodes"].values())
    edges_data = state["edges"]
    
    with open("lineage_graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, indent=2, ensure_ascii=False)
        
    with open("semantic_drift_report.md", "w", encoding="utf-8") as f:
        f.write("# 🛡️ SQL Semantic & Knowledge Base Update Report\n\n")
        if state["drift_warnings"]:
            f.write("## ⚠️ Semantic Drift / Conflict Warnings\n> **System Notice:** 系統偵測到衝突，請在更新知識庫前進行人類審查：\n\n")
            for w in state["drift_warnings"]:
                f.write(f"- {w}\n")
        else:
            f.write("> ✅ 未偵測到業務邏輯衝突，可安全 Upsert。\n")

    # 使用 replace 注入 JSON，徹底避免 Python f-string 破壞 JS/CSS 大括號
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Enterprise SQL Lineage</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #network { width: 100%; height: 100vh; background-color: #0f172a; outline: none; }
            div.vis-tooltip { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 10px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.5); z-index: 10;}
        </style>
    </head>
    <body class="flex overflow-hidden font-sans text-slate-200">
        <div class="flex-grow relative h-screen">
            <div id="network"></div>
        </div>
        <div id="sidebar" class="w-96 bg-slate-800 border-l border-slate-700 p-6 absolute right-0 h-full transform transition-transform translate-x-full z-20 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
                <h2 id="sb-title" class="text-xl font-bold text-blue-400">Node Name</h2>
                <button onclick="closeSidebar()" class="text-3xl hover:text-red-400">&times;</button>
            </div>
            <h3 class="text-xs uppercase font-bold text-slate-500 mb-2">🧠 AI Business Logic</h3>
            <p id="sb-logic" class="text-sm bg-slate-900 p-4 rounded-lg mb-6 border border-slate-700 text-slate-300"></p>
            <h3 class="text-xs uppercase font-bold text-rose-500 mb-2">🛡️ Data Quality Rules</h3>
            <ul id="sb-dq" class="list-disc pl-5 text-sm space-y-2 text-rose-400 bg-rose-950/20 p-4 rounded-lg border border-rose-900/50"></ul>
        </div>
        <script>
            const nodesDict = __NODES_DICT__;
            const nodesData = __NODES_DATA__;
            const edgesData = __EDGES_DATA__;
            
            const nodes = new vis.DataSet(nodesData.map(n => ({id: n.id, label: n.label, group: n.type})));
            const edges = new vis.DataSet(edgesData);
            
            const options = {
                nodes: { shape: 'box', margin: 10, font: { color: '#fff', face: 'monospace' }, borderWidth: 2 },
                groups: {
                    Physical: { color: { background: '#1e293b', border: '#3b82f6' } },
                    Temp: { color: { background: '#334155', border: '#94a3b8' }, shapeProperties: { borderDashes: [5, 5] } }
                },
                edges: { color: '#64748b', arrows: 'to', smooth: { type: 'cubicBezier' }, width: 2 },
                interaction: { hover: true },
                layout: { hierarchical: { direction: 'LR', sortMethod: 'directed', levelSeparation: 250 } }
            };

            const network = new vis.Network(document.getElementById('network'), {nodes, edges}, options);

            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                    const info = nodesDict[params.nodes[0]] || {};
                    document.getElementById('sb-title').innerText = params.nodes[0];
                    document.getElementById('sb-logic').innerText = info.business_logic || "N/A";
                    document.getElementById('sb-dq').innerHTML = info.dq_rules ? info.dq_rules.map(r => `<li>${r}</li>`).join('') : '';
                    document.getElementById('sidebar').classList.remove('translate-x-full');
                } else { closeSidebar(); }
            });
            function closeSidebar() { document.getElementById('sidebar').classList.add('translate-x-full'); }
        </script>
    </body>
    </html>
    """
    html_content = html_template.replace("__NODES_DICT__", json.dumps(state["nodes"]))
    html_content = html_content.replace("__NODES_DATA__", json.dumps(nodes_data))
    html_content = html_content.replace("__EDGES_DATA__", json.dumps(edges_data))
    
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("\n📁 產出物已存檔：dashboard.html, semantic_drift_report.md, lineage_graph.json")
    return state

# ==========================================
# 7. 編譯與執行主入口 (Main Block)
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

    print("🚀 [系統啟動] 正在啟動企業級 SQL 語意與血緣分析引擎...")

    # 1. 執行 Chunking (切割)
    chunks = chunk_mssql_sql(raw_sql)
    print(f"📦 共切割出 {len(chunks)} 個 SQL 區塊。")

    # 2. 建立 LangGraph
    workflow = StateGraph(EngineState)
    workflow.add_node("analyze", node_analyze_backward)
    workflow.add_node("generate", node_generate_outputs)
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", router_should_continue, {"analyze": "analyze", "generate": "generate"})
    workflow.add_edge("generate", END)
    engine = workflow.compile()

    # 3. 初始化 LangGraph 狀態 (設定從最後一個 Chunk 開始 Bottom-Up)
    initial_state = {
        "chunks": chunks,
        "current_idx": len(chunks) - 1,
        "pending_list": [],
        "resolved_tables": set(),
        "knowledge_base": {},
        "nodes": {},
        "edges": [],
        "drift_warnings": []
    }

    # 4. 執行引擎
    print("\n🕸️ 開始執行 LangGraph 逆向狀態機演算法...")
    engine.invoke(initial_state)

    print("\n=======================================================")
    print(" 🎉 執行完畢！您可以直接在瀏覽器開啟 `dashboard.html`")
    print("=======================================================")