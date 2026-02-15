import json
import sqlglot
from sqlglot import exp
from typing import List, Dict, Set, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. 結構化輸出模型 (Pydantic Models)
# ==========================================
class ColumnTransformation(BaseModel):
    target_column: str = Field(description="目標欄位名稱")
    source_expression: str = Field(description="來源 SQL 數學/邏輯表達式")
    business_meaning: str = Field(description="此公式在業務上的意義 (如: 計算稅後淨利)")

class DriftDetection(BaseModel):
    has_conflict: bool = Field(description="SQL 過濾條件是否與知識庫描述發生衝突？")
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
    chunks: List[str]                  # 由 chunk_mssql_sql 切割好的 SQL 區塊
    current_idx: int                   # Bottom-Up 逆向追蹤指標
    pending_list: List[str]            # 【核心靈魂】：等待解析的 Temp/CTE 依賴清單
    resolved_tables: Set[str]          # 防無限迴圈的已解析清單
    knowledge_base: Dict[str, str]     # 既有知識庫 (Table -> Markdown)
    
    # 產出物 (Artifacts)
    nodes: Dict[str, dict]
    edges: List[dict]
    drift_warnings: List[str]

# ==========================================
# 3. Hybrid Parser: AST 確定性地基提取
# ==========================================
def extract_ast_ground_truth(sql_chunk: str) -> dict:
    """提取絕對客觀的語法結構，消除 LLM 幻覺"""
    info = {"target": None, "sources": set(), "expressions": {}}
    try:
        parsed = sqlglot.parse_one(sql_chunk, read="tsql")
        
        # 1. 提取 目標表 (SELECT INTO, INSERT)
        for into in parsed.find_all(exp.Into):
            if isinstance(into.this, exp.Table):
                info["target"] = into.this.name
        if not info["target"] and isinstance(parsed, exp.Insert):
            info["target"] = parsed.this.name
            
        # 2. 提取 來源表 (排除 Target 自己)
        for table in parsed.find_all(exp.Table):
            if table.name and table.name != info["target"]:
                info["sources"].add(table.name)
                
        # 3. 提取 欄位級公式 (e.g., Price * (1 - Rate))
        for select in parsed.find_all(exp.Select):
            for proj in select.expressions:
                if isinstance(proj, exp.Alias):
                    info["expressions"][proj.alias] = proj.this.sql(dialect="tsql")
                elif isinstance(proj, exp.Column):
                    info["expressions"][proj.name] = proj.sql(dialect="tsql")
    except Exception as e:
        pass # 容錯處理：極端方言降級
        
    info["sources"] = list(info["sources"])
    return info

# ==========================================
# 4. LangGraph 節點：逆向追蹤與語意引擎
# ==========================================
def node_analyze_backward(state: EngineState) -> dict:
    idx = state["current_idx"]
    chunk = state["chunks"][idx]
    pending = list(state["pending_list"])
    resolved = set(state["resolved_tables"])
    nodes, edges = dict(state["nodes"]), list(state["edges"])
    drifts = list(state["drift_warnings"])
    
    # 【混合解析 1】：取得 AST Ground Truth
    ast_info = extract_ast_ground_truth(chunk)
    target = ast_info["target"] or f"Unknown_Target_{idx}"
    
    # 【核心演算法剪枝】：只解析「最終目標表」或「在 Pending List 中的表」
    is_final_chunk = (idx == len(state["chunks"]) - 1)
    if is_final_chunk or target in pending:
        
        # 從 Pending List 消滅，並加入 Resolved
        if target in pending:
            pending.remove(target)
        resolved.add(target)
        
        # 將尚未解析的上游依賴推入 Pending List (逆向生長)
        for src in ast_info["sources"]:
            if (src.startswith("#") or src.upper().startswith("CTE")) and src not in resolved:
                if src not in pending:
                    pending.append(src)
        
        # 【混合解析 2】：呼叫 LLM 進行語意與漂移分析
        kb_context = state["knowledge_base"].get(target, "尚無此表的知識紀錄。")
        llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(AnalysisResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位精通企業級資料架構的 Data Architect。請基於提供的『AST 客觀結構』推斷隱含的業務邏輯與 DQ 規則。"
                       "【嚴格任務】：比對 SQL 邏輯與現有知識庫，若發現 WHERE 條件等業務邏輯與知識庫衝突，務必觸發 drift_check 警告！不要盲目覆蓋。"),
            ("user", "Target Table: {target}\nAST 解析結果: {ast}\n\n原始 SQL: {sql}\n\n現有知識庫: {kb}")
        ])
        
        ai_res: AnalysisResult = llm.invoke({
            "target": target, "ast": json.dumps(ast_info), "sql": chunk, "kb": kb_context
        })
        
        # --- 構建視覺化與 JSON 資料 ---
        nodes[target] = {
            "id": target, "label": target,
            "type": "Temp" if target.startswith("#") else "Physical",
            "business_logic": ai_res.business_logic,
            "dq_rules": ai_res.data_quality_rules
        }
        
        # 建立 Edge，並將「轉換公式」封裝為 HTML 供 Vis.js Hover 顯示
        for src in ast_info["sources"]:
            if src not in nodes: # 補齊 Source Node
                nodes[src] = {"id": src, "label": src, "type": "Physical", "business_logic": "Source Table", "dq_rules": []}
            
            tooltip_html = "<div style='font-family: monospace; padding: 4px;'><b style='color:#60a5fa'>Column Transformations:</b><br/>"
            for t in ai_res.transformations:
                tooltip_html += f"• <b>{t.target_column}</b> = <span style='color:#cbd5e1'>{t.source_expression}</span> <i style='color:#94a3b8'>({t.business_meaning})</i><br/>"
            tooltip_html += "</div>"
            
            edges.append({
                "from": src, "to": target,
                "title": tooltip_html, # Vis.js 原生支援 HTML Title Hover
                "transformations": [t.dict() for t in ai_res.transformations] # Machine Readable
            })
            
        # 收集 Semantic Drift
        if ai_res.drift_check.has_conflict:
            drifts.append(f"**Table `{target}`**: {ai_res.drift_check.warning_message}")
            
    return {
        "current_idx": idx - 1, # 繼續由後往前推進
        "pending_list": pending,
        "resolved_tables": resolved,
        "nodes": nodes, "edges": edges, "drift_warnings": drifts
    }

def router_should_continue(state: EngineState) -> str:
    # 當追蹤到最源頭 (idx < 0) 或 pending_list 清空時，結束解析進入產出
    if state["current_idx"] < 0 or (len(state["pending_list"]) == 0 and len(state["resolved_tables"]) > 0):
        return "generate"
    return "analyze"

# ==========================================
# 5. 生成標準化輸出與現代化儀表板
# ==========================================
def node_generate_outputs(state: EngineState) -> dict:
    nodes_data = list(state["nodes"].values())
    edges_data = state["edges"]
    
    # 1. 輸出 Machine Readable JSON (未來可整合 OpenLineage)
    lineage_export = {"nodes": nodes_data, "edges": edges_data}
    with open("lineage_graph.json", "w", encoding="utf-8") as f:
        json.dump(lineage_export, f, indent=2, ensure_ascii=False)
        
    # 2. 帶有衝突檢測的 Markdown (Zero-Trust)
    with open("semantic_drift_report.md", "w", encoding="utf-8") as f:
        f.write("# 🛡️ SQL Semantic & Knowledge Base Update Report\n\n")
        if state["drift_warnings"]:
            f.write("## ⚠️ Semantic Drift / Conflict Warnings\n> **System Notice:** 系統偵測到本次 SQL 邏輯與現有知識庫存在衝突。請在更新 `dictionary_knowledge.md` 前進行人類審查：\n\n")
            for w in state["drift_warnings"]:
                f.write(f"- {w}\n")
        else:
            f.write("> ✅ 未偵測到業務邏輯衝突，可安全 Upsert。\n")

    # 3. 現代化單頁互動 HTML (Vis.js + Tailwind)
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Enterprise SQL Lineage Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #network {{ width: 100%; height: 100vh; outline: none; background-color: #0f172a; }}
            div.vis-tooltip {{ background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; font-size: 13px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.5); }}
        </style>
    </head>
    <body class="flex overflow-hidden font-sans text-slate-200">
        
        <div class="flex-grow relative h-screen">
            <div class="absolute top-6 left-6 z-10 bg-slate-800 p-2 rounded-lg shadow-lg flex gap-2 border border-slate-700">
                <input type="text" id="searchInput" placeholder="Search Table/Column..." class="bg-slate-900 border border-slate-600 p-2 text-sm rounded outline-none focus:ring-2 focus:ring-blue-500 w-64 text-slate-200">
                <button onclick="searchNode()" class="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded text-sm transition-colors font-semibold">Search</button>
            </div>
            <div id="network"></div>
        </div>

        <div id="sidebar" class="w-96 bg-slate-800 border-l border-slate-700 shadow-2xl flex flex-col h-full transform transition-transform duration-300 translate-x-full absolute right-0 z-20">
            <div class="p-6 overflow-y-auto h-full">
                <div class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                    <div>
                        <h2 id="sb-title" class="text-2xl font-bold text-blue-400 break-all">Node Name</h2>
                        <span id="sb-type" class="inline-block mt-2 px-2 py-1 bg-slate-700 text-slate-300 text-xs font-semibold rounded uppercase tracking-wider">Type</span>
                    </div>
                    <button onclick="closeSidebar()" class="text-slate-500 hover:text-red-400 text-2xl transition-colors">&times;</button>
                </div>
                
                <h3 class="text-xs uppercase font-bold text-slate-500 mb-2 tracking-wider">🧠 AI Business Logic</h3>
                <p id="sb-logic" class="text-sm bg-slate-900 p-4 rounded-lg border border-slate-700 mb-6 leading-relaxed text-slate-300"></p>
                
                <h3 class="text-xs uppercase font-bold text-rose-500 mb-2 tracking-wider flex items-center gap-2">🛡️ Data Quality Rules</h3>
                <ul id="sb-dq" class="list-disc pl-5 text-sm space-y-2 text-rose-400 bg-rose-950/20 p-4 rounded-lg border border-rose-900/50"></ul>
            </div>
        </div>

        <script>
            const nodesDict = {json.dumps(state["nodes"])};
            const nodes = new vis.DataSet({json.dumps([{"id": n["id"], "label": n["label"], "group": n["type"]} for n in nodes_data])});
            const edges = new vis.DataSet({json.dumps(edges_data)});
            
            const options = {{
                nodes: {{ shape: 'box', margin: 14, font: {{ size: 14, color: '#f8fafc', face: 'monospace' }}, borderWidth: 2 }},
                groups: {{
                    Physical: {{ color: {{ background: '#1e293b', border: '#3b82f6' }} }},
                    Temp: {{ color: {{ background: '#334155', border: '#94a3b8' }}, shapeProperties: {{ borderDashes: [5, 5] }} }}
                }},
                edges: {{ color: '#64748b', smooth: {{ type: 'cubicBezier' }}, width: 2, arrows: 'to' }},
                interaction: {{ hover: true }},
                layout: {{ hierarchical: {{ direction: 'LR', sortMethod: 'directed', levelSeparation: 250 }} }}
            }};

            const network = new vis.Network(document.getElementById('network'), {{nodes, edges}}, options);

            // Node Click -> Open Sidebar
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    const info = nodesDict[nodeId] || {{}};
                    
                    document.getElementById('sb-title').innerText = nodeId;
                    document.getElementById('sb-type').innerText = info.type || 'Unknown';
                    document.getElementById('sb-logic').innerText = info.business_logic || "No logic found.";
                    
                    const dqList = document.getElementById('sb-dq');
                    dqList.innerHTML = info.dq_rules && info.dq_rules.length > 0 ? 
                        info.dq_rules.map(r => `<li>${{r}}</li>`).join('') : 
                        '<li class="text-emerald-500 list-none font-semibold">✅ No DQ risks detected.</li>';
                        
                    document.getElementById('sidebar').classList.remove('translate-x-full');
                }} else {{ closeSidebar(); }}
            }});

            function closeSidebar() {{ document.getElementById('sidebar').classList.add('translate-x-full'); }}
            
            function searchNode() {{
                const val = document.getElementById('searchInput').value.toLowerCase();
                const found = nodes.get().find(n => n.label.toLowerCase().includes(val));
                if(found) {{
                    network.selectNodes([found.id]);
                    network.focus(found.id, {{ scale: 1.2, animation: {{ duration: 500 }} }});
                    network.emit("click", {{ nodes: [found.id] }});
                }}
            }}
        </script>
    </body>
    </html>
    """
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    return state

# ==========================================
# 6. 編譯引擎
# ==========================================
workflow = StateGraph(EngineState)
workflow.add_node("analyze", node_analyze_backward)
workflow.add_node("generate", node_generate_outputs)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", router_should_continue, {"analyze": "analyze", "generate": "generate"})
workflow.add_edge("generate", END)

engine = workflow.compile()