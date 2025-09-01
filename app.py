#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time, json, math
import pandas as pd
from io import BytesIO
import plotly.express as px
from datetime import datetime
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Choose a stable, instruction-following model
MODEL_NAME = "gpt-4o-mini"   # great at structured JSON + low latency
LLM_TEMPERATURE = 0.1
SQL_TIMEOUT_SEC = 60
MAX_RETRIES_SQL_FIX = 2

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; }
.main-header { font-size: 2.2rem; color: #4a4a4a; font-weight: 700; margin-bottom: .25rem; }
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.0rem; border-radius: 12px; color: white; text-align: center; margin: .5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,.1);
}
.executive-summary {
    background-color: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 6px rgba(0,0,0,.05);
    margin-bottom: 1.2rem; border-left: 4px solid #764ba2;
}
.assistant-message { background-color: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #667eea; }
.small-muted { color:#6c757d; font-size:.85rem; }
.kpi-chip { background:#fff; border:1px solid #eee; border-radius:10px; padding:.35rem .6rem; margin-right:.4rem; display:inline-block;}
hr { border:none; border-top:1px solid #eaeaea; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# =========================
# KEYS & CONNECTIONS
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it to your app Secrets or .env")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource
def get_llm():
    return ChatOpenAI(model=MODEL_NAME, temperature=LLM_TEMPERATURE, request_timeout=60, max_retries=3)

llm = get_llm()

@st.cache_resource
def get_engine_and_schema():
    """Create engine and return schema string for the single allowed table."""
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
    except KeyError:
        st.error("❌ Missing DB_* secrets. Please add DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.")
        st.stop()

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    # Build schema doc from information_schema for the allowed table
    schema_rows = []
    q = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q)).fetchall()
        for c, t in rows:
            schema_rows.append(f'- "{c}" ({t})')
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(schema_rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

ORCHESTRATOR_PROMPT = """
You are a principal data analyst for a jewelry brand. PLAN the analysis and produce a STRICT JSON.
Given a natural language question and the database schema, output JSON with:
- "needs_clarification": boolean
- "clarifying_questions": array of strings (<=3 short questions) if needs_clarification
- "sql": a single **valid PostgreSQL SELECT** that:
   * Queries only voylla."voylla_design_ai"
   * Always includes WHERE "Sale Order Item Status" != 'CANCELLED'
   * If time period is vague, infer reasonable filters using "Date"
   * Use double-quotes for identifiers
- "analysis_plan": bullet-like array of 3-6 short steps for Python-side analysis
- "kpi_defs": array of KPIs to compute from the SQL result. Use keys from columns:
   * Revenue: SUM("Amount")
   * Units: SUM("Qty")
   * AOV: SUM("Amount")/NULLIF(SUM("Qty"),0)
   * MarginPct: (SUM("Amount") - SUM("Cost Price"*"Qty"))/NULLIF(SUM("Amount"),0)*100
   * Growth comparisons (YoY/MoM/QoQ) when time exists
- "viz": object like {"type":"bar|line|area|treemap","x":"col","y":"metric","color":"optional_col","title":"string"}
- "followups": array of 3-6 next questions that would drive decisions (short, executive-friendly)

Return ONLY minified JSON. No markdown, no commentary.
SCHEMA:
{schema}
QUESTION:
{q}
"""

SQL_REPAIR_PROMPT = """
You generated a SQL query that errored. Fix it.
RULES:
- Read-only SELECT only
- Only table voylla."voylla_design_ai"
- Always include WHERE "Sale Order Item Status" != 'CANCELLED'
- Use valid PostgreSQL and double-quoted identifiers
- Keep the original intent

Original SQL:
{sql}

Error:
{err}

Return ONLY the corrected SQL (no markdown).
"""

SUMMARY_PROMPT = """
You are an executive analyst. Using the user's question and the CSV preview of results,
write a crisp executive brief with:
- 4–7 bullet key findings (numbers in **bold**)
- 2–4 actionable recommendations (bullets)
- Call out best and worst performers if applicable

Keep it compact and businessy.

USER QUESTION:
{user_q}

RESULTS PREVIEW (CSV):
{preview_csv}
"""

def safe_json_loads(txt: str):
    """Try hard to parse JSON the model returns."""
    try:
        return json.loads(txt)
    except Exception:
        # heuristic: find first '{' to last '}' segment
        if '{' in txt and '}' in txt:
            candidate = txt[txt.find('{'): txt.rfind('}')+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        raise

def generate_orchestration(question: str) -> dict:
    prompt = ORCHESTRATOR_PROMPT.format(schema=schema_doc, q=question)
    raw = llm.invoke(prompt).content.strip()
    data = safe_json_loads(raw)
    # Guardrails
    if not isinstance(data, dict):
        raise ValueError("Invalid orchestration JSON.")
    if "sql" not in data or not isinstance(data.get("sql",""), str):
        raise ValueError("Orchestrator did not return SQL.")
    return data

def contains_dangerous(sql: str) -> bool:
    return bool(DANGEROUS.search(sql))

def run_sql(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)

def repair_sql(sql: str, err: str) -> str:
    prompt = SQL_REPAIR_PROMPT.format(sql=sql, err=err)
    fixed = llm.invoke(prompt).content.strip()
    # strip fences if any
    if fixed.startswith("```"):
        fixed = re.sub(r"^```[a-zA-Z0-9]*", "", fixed).strip()
        fixed = fixed[:-3] if fixed.endswith("```") else fixed
        fixed = fixed.strip()
    return fixed

def exec_sql_with_self_heal(sql: str) -> (pd.DataFrame, str):
    """Run SQL with self-healing on errors, returning df and final_sql used."""
    if contains_dangerous(sql):
        raise ValueError("Generated SQL contains non read-only keyword.")
    last_err = None
    current = sql
    for _ in range(MAX_RETRIES_SQL_FIX + 1):
        try:
            df = run_sql(current)
            return df, current
        except Exception as e:
            last_err = str(e)
            if _ < MAX_RETRIES_SQL_FIX:
                current = repair_sql(current, last_err)
                if contains_dangerous(current):
                    break
            else:
                break
    raise RuntimeError(f"SQL failed after retries. Last error: {last_err}")

def compute_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    # Try to compute universal KPIs when columns exist
    k = {}
    amt = df["Amount"] if "Amount" in df.columns else None
    qty = df["Qty"] if "Qty" in df.columns else None
    cost = df["Cost Price"] if "Cost Price" in df.columns else None

    if amt is not None:
        k["Revenue"] = float(np.nansum(amt))
    if qty is not None:
        k["Units"] = float(np.nansum(qty))
    if amt is not None and qty is not None:
        denom = np.nansum(qty)
        k["AOV"] = float(np.nansum(amt) / denom) if denom else None
    if amt is not None and cost is not None and qty is not None:
        rev = np.nansum(amt)
        gp = rev - np.nansum(cost * (qty if qty is not None else 0))
        k["MarginPct"] = float((gp / rev) * 100) if rev else None
    return k

def add_period_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        d = pd.to_datetime(df["Date"], errors="coerce")
        df["_year"] = d.dt.year
        df["_month"] = d.dt.to_period("M").astype(str)
        df["_qtr"] = d.dt.to_period("Q").astype(str)
    return df

def growth_table(df: pd.DataFrame, metric_col="Amount", period="month"):
    """Compute period growth for metric over time."""
    if df is None or df.empty or "Date" not in df.columns or metric_col not in df.columns:
        return None
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    if period == "month":
        d["_period"] = d["Date"].dt.to_period("M").astype(str)
    elif period == "quarter":
        d["_period"] = d["Date"].dt.to_period("Q").astype(str)
    elif period == "year":
        d["_period"] = d["Date"].dt.year.astype(str)
    else:
        return None
    g = d.groupby("_period", dropna=True)[metric_col].sum().reset_index()
    g["Prev"] = g[metric_col].shift(1)
    g["Growth_%"] = np.where(g["Prev"].fillna(0)==0, np.nan, (g[metric_col]-g["Prev"]) / g["Prev"] * 100.0)
    return g

def top_bottom(df: pd.DataFrame, group_col: str, metric_col: str, n=5):
    if df is None or df.empty or group_col not in df.columns or metric_col not in df.columns:
        return None, None
    g = df.groupby(group_col, dropna=False)[metric_col].sum().reset_index().sort_values(metric_col, ascending=False)
    return g.head(n), g.tail(n)

def detect_outliers(df: pd.DataFrame, metric_col: str):
    """Simple z-score to flag anomalies on metric_col."""
    if metric_col not in df.columns or df.empty:
        return None
    x = df[metric_col].astype(float)
    mu, sigma = np.nanmean(x), np.nanstd(x)
    if sigma == 0 or np.isnan(sigma):
        return None
    z = (x - mu) / sigma
    out = df.assign(_z=z)[np.abs(z) >= 2.0]
    return out if not out.empty else None

def driver_analysis(df: pd.DataFrame):
    """Aggregate revenue by design attributes to surface 'drivers'."""
    if df is None or df.empty:
        return {}
    amt_col = "Amount" if "Amount" in df.columns else None
    if amt_col is None:
        return {}
    attrs = [c for c in ["Design Style","Form","Look","Metal Color","Central Stone"] if c in df.columns]
    results = {}
    for a in attrs:
        agg = df.groupby(a, dropna=False)[amt_col].sum().reset_index().sort_values(amt_col, ascending=False)
        results[a] = agg.head(10)
    return results

def summarize_for_executives(df: pd.DataFrame, user_q: str) -> str:
    preview_csv = df.head(50).to_csv(index=False)
    prompt = SUMMARY_PROMPT.format(user_q=user_q, preview_csv=preview_csv)
    return llm.invoke(prompt).content.strip()

def render_viz(df: pd.DataFrame, viz: dict):
    if df is None or df.empty or not viz:
        return None
    vtype = viz.get("type","bar")
    x = viz.get("x")
    y = viz.get("y")
    color = viz.get("color")
    title = viz.get("title","")
    if not y or (x and x not in df.columns) or (y and y not in df.columns):
        return None
    try:
        work = df.copy()
        # compact top 20 for readability
        if x and y and len(work) > 20:
            # if numeric y, take nlargest by y
            if pd.api.types.is_numeric_dtype(work[y]):
                work = work.nlargest(20, y)
            else:
                work = work.head(20)
        if vtype == "line":
            fig = px.line(work, x=x if x in work.columns else None, y=y, color=color, title=title)
        elif vtype == "area":
            fig = px.area(work, x=x if x in work.columns else None, y=y, color=color, title=title)
        elif vtype == "treemap" and x and color:
            # treemap expects path; use [x,color] if both present
            fig = px.treemap(work, path=[x, color], values=y, title=title)
        else:
            fig = px.bar(work, x=x if x in work.columns else None, y=y, color=color, title=title)
        fig.update_layout(height=460, showlegend=True)
        return fig
    except Exception:
        return None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    with engine.connect() as conn:
        try:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM voylla."voylla_design_ai" 
                WHERE "Sale Order Item Status" != 'CANCELLED'
            """)).scalar()
            st.success(f"✅ Connected: {count:,} active records")
        except Exception as e:
            st.error(f"❌ Connection issue: {e}")

    st.markdown("---")
    st.header("💡 Executive Questions")
    presets = [
        "Channel-wise revenue & units this quarter; highlight top/bottom channels and MoM growth",
        "Top 20 products by revenue this month with margin and YoY trend",
        "Which design styles drive revenue and margin? Show top 10 and outliers",
        "Compare this year's monthly revenue vs last year; call out anomalies",
        "Which forms (Jhumka/Hoop/Stud etc.) over-index by channel?"
    ]
    for q in presets:
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state["auto_q"] = q
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state["last_df"] = None
            st.session_state["last_sql"] = ""
            st.session_state["last_orch"] = {}
            st.rerun()
    with col2:
        st.caption("Planner + Self-healing SQL • deep analysis")

# =========================
# SESSION
# =========================
if "chat" not in st.session_state: st.session_state.chat = []
if "auto_q" not in st.session_state: st.session_state.auto_q = None
if "last_df" not in st.session_state: st.session_state.last_df = None
if "last_sql" not in st.session_state: st.session_state.last_sql = ""
if "last_orch" not in st.session_state: st.session_state.last_orch = {}

# =========================
# HEADER
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence & Sales Analytics — planner + self-healing SQL + driver analysis")

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Always-visible input (keeps mobile keyboard)
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

# Handle follow-up button clicks (they set session_state.next_q)
if "next_q" in st.session_state and st.session_state.next_q:
    inp = st.session_state.next_q
    st.session_state.next_q = ""

def push_assistant(msg: str):
    st.session_state.chat.append({"role":"assistant","content":msg})
    with st.chat_message("assistant"):
        st.markdown(msg, unsafe_allow_html=True)

def push_user(msg: str):
    st.session_state.chat.append({"role":"user","content":msg})
    with st.chat_message("user"):
        st.markdown(msg)

if inp:
    push_user(inp)

    with st.spinner("Planning analysis & preparing SQL… ✨"):
        try:
            orch = generate_orchestration(inp)
        except Exception as e:
            push_assistant(f"<div class='assistant-message'>⚠️ Planner error: {e}</div>")
            orch = {}

    needs_clar = orch.get("needs_clarification") if isinstance(orch, dict) else False
    clar_qs = orch.get("clarifying_questions", []) if isinstance(orch, dict) else []

    # If planner says it needs clarification, ask succinctly but still proceed with a best-effort base SQL
    if needs_clar and clar_qs:
        push_assistant("<div class='assistant-message'><b>Quick clarification</b>: " + " • ".join(clar_qs) + "</div>")

    sql = (orch.get("sql") or "").strip() if isinstance(orch, dict) else ""
    if sql.startswith("```"):  # strip fences
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
        sql = sql.strip()

    if not sql:
        push_assistant("<div class='assistant-message'>⚠️ Could not produce SQL for your question.</div>")
    else:
        st.session_state["last_orch"] = orch
        try:
            df, final_sql = exec_sql_with_self_heal(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = final_sql
        except Exception as e:
            push_assistant(f"<div class='assistant-message'>⚠️ SQL failed: {e}</div>")
            df = pd.DataFrame()

        # === Python-side analytics ===
        if not df.empty:
            # add period helpers
            df = add_period_cols(df)

            # KPIs
            kpis = compute_kpis(df)

            # Growth tables if Date exists
            growth_m = growth_table(df, metric_col="Amount", period="month")
            growth_q = growth_table(df, metric_col="Amount", period="quarter")
            growth_y = growth_table(df, metric_col="Amount", period="year")

            # Top/bottom by Channel (if present)
            top_ch, bot_ch = (None, None)
            if "Channel" in df.columns and "Amount" in df.columns:
                top_ch, bot_ch = top_bottom(df, "Channel", "Amount", n=5)

            # Outliers on Amount
            outliers = detect_outliers(df, "Amount") if "Amount" in df.columns else None

            # Driver analysis by design attributes
            drivers = driver_analysis(df)

            # Executive summary
            summary = summarize_for_executives(df, inp)

            # Assistant message
            push_assistant(f"<div class='assistant-message'>{summary}</div>")

            # KPIs inline chips
            if kpis:
                chips = []
                if "Revenue" in kpis: chips.append(f"<span class='kpi-chip'>Revenue: <b>₹{kpis['Revenue']:,.0f}</b></span>")
                if "Units" in kpis: chips.append(f"<span class='kpi-chip'>Units: <b>{kpis['Units']:,.0f}</b></span>")
                if "AOV" in kpis and kpis['AOV'] is not None: chips.append(f"<span class='kpi-chip'>AOV: <b>₹{kpis['AOV']:,.0f}</b></span>")
                if "MarginPct" in kpis and kpis['MarginPct'] is not None: chips.append(f"<span class='kpi-chip'>Margin: <b>{kpis['MarginPct']:.1f}%</b></span>")
                st.markdown(" ".join(chips), unsafe_allow_html=True)

            # Show SQL (collapsible)
            with st.expander("View generated SQL"):
                st.code(st.session_state.last_sql, language="sql")

            # Data table
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            # Viz from planner
            viz = orch.get("viz", {})
            fig = render_viz(df, viz)
            if fig:
                st.subheader("📊 Visualization")
                st.plotly_chart(fig, use_container_width=True)

            # Growth and diagnostics
            diag_tabs = st.tabs(["📈 Periodic Growth", "🔎 Top/Bottom", "🚨 Outliers", "🧭 Drivers"])
            with diag_tabs[0]:
                if growth_m is not None:
                    st.markdown("**MoM Growth**")
                    st.dataframe(growth_m, use_container_width=True)
                if growth_q is not None:
                    st.markdown("**QoQ Growth**")
                    st.dataframe(growth_q, use_container_width=True)
                if growth_y is not None:
                    st.markdown("**YoY Growth**")
                    st.dataframe(growth_y, use_container_width=True)
            with diag_tabs[1]:
                if top_ch is not None:
                    st.markdown("**Top Channels (by Revenue)**")
                    st.dataframe(top_ch, use_container_width=True)
                if bot_ch is not None:
                    st.markdown("**Bottom Channels (by Revenue)**")
                    st.dataframe(bot_ch, use_container_width=True)
            with diag_tabs[2]:
                if outliers is not None:
                    st.markdown("**Anomalies (|z| ≥ 2) on Amount**")
                    st.dataframe(outliers, use_container_width=True)
                else:
                    st.caption("No strong outliers detected on Amount.")
            with diag_tabs[3]:
                if drivers:
                    for a, dfa in drivers.items():
                        st.markdown(f"**Top 10 by {a}**")
                        st.dataframe(dfa, use_container_width=True)
                else:
                    st.caption("Driver attributes not available in current result.")

            # Follow-up suggestions as buttons
            followups = orch.get("followups", [])[:6] if isinstance(orch, dict) else []
            if followups:
                st.markdown("----")
                st.subheader("Suggested Next Questions")
                cols = st.columns(min(3, len(followups)))
                for i, f in enumerate(followups):
                    with cols[i % len(cols)]:
                        if st.button(f"➡️ {f}", key=f"fu_{i}", use_container_width=True):
                            st.session_state.next_q = f
                            st.rerun()

            # Export pack
            st.markdown("---")
            st.subheader("📥 Export Results")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
                # KPIs
                kdf = pd.DataFrame([kpis]).T.reset_index()
                kdf.columns = ["KPI","Value"]
                kdf.to_excel(writer, index=False, sheet_name='KPIs')
                # Summary
                pd.DataFrame({"Executive Summary":[summary]}).to_excel(writer, index=False, sheet_name='Summary')
                # SQL + Viz
                meta = pd.DataFrame({
                    "Key":["SQL","VizSpec","Exported At"],
                    "Value":[st.session_state.last_sql, json.dumps(viz), datetime.now().strftime("%Y-%m-%d %H:%M")]
                })
                meta.to_excel(writer, index=False, sheet_name='Meta')
            output.seek(0)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "💾 Download Analysis Pack",
                data=output.getvalue(),
                file_name=f"voylla_analysis_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_pack"
            )

        else:
            push_assistant("<div class='assistant-message'>No rows returned for the current question. Try expanding the time range or changing the grouping.</div>")

# Footer tips
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:.9em;'>
💡 <b>Pro tips:</b> Ask for growth (YoY/MoM/QoQ), margins, drivers by style/form/look, and channel mix. Try:
“Compare Halo Essence vs other collections this quarter for revenue, margin and MoM growth; show top channels and outliers.”
</div>
""", unsafe_allow_html=True)
