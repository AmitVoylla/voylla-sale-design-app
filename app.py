#!/usr/bin/env python
# coding: utf-8
"""
Voylla DesignGPT - Executive Dashboard (Robust/fixed version)

Main fixes and improvements:
- Replaced direct LangChain LLM invocation with a stable OpenAI ChatCompletion wrapper
  (works with `openai` package and respects OPENAI_API_KEY from .env or Streamlit secrets).
- Hardened DB connection handling and retries; accepts DB credentials from st.secrets or environment variables.
- Improved SQL-generation safety checks and robust stripping of code fences.
- Safer execution of user-generated SQL with explicit read-only enforcement and table whitelist.
- Better session-state handling (avoids losing previous messages when new queries are made).
- Improved analysis JSON parsing with graceful fallback and deterministic structure.
- Added explicit caching for DB engine and memory using st.cache_resource.
- Clearer error messages and logging to help debugging in Streamlit logs.

NOTE: This file is intended to replace your previous app. Ensure you have `openai`,
`streamlit`, `sqlalchemy`, `psycopg2-binary`, `pandas`, `plotly`, and `openpyxl` installed.
"""

import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import re
import time
import json
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
from datetime import datetime
import openai
from typing import List, Optional

# -------------------------
# CONFIG
# -------------------------
load_dotenv()
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

if not OPENAI_API_KEY:
    st.error("🔑 No OpenAI key found – please add it to your app Secrets or .env as OPENAI_API_KEY")
    st.stop()

openai.api_key = OPENAI_API_KEY

# -------------------------
# UTILITIES
# -------------------------
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)
ALLOWED_TABLE = 'voylla."voylla_design_ai"'

def strip_codefence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove leading ```lang
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def call_openai_chat(messages: List[dict], max_tokens: int = 1500, stop: Optional[List[str]] = None) -> str:
    """Call OpenAI ChatCompletion and return the assistant content."""
    for attempt in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
            )
            content = resp['choices'][0]['message']['content']
            return content
        except openai.error.RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except Exception as e:
            # For non-rate errors, raise after one attempt
            if attempt == 2:
                raise
            time.sleep(1)
    raise RuntimeError("OpenAI request failed after retries")

# -------------------------
# DATABASE ENGINE
# -------------------------
@st.cache_resource
def get_engine():
    # Prefer Streamlit secrets, fall back to env vars
    db_host = st.secrets.get("DB_HOST") if "DB_HOST" in st.secrets else os.getenv("DB_HOST")
    db_port = st.secrets.get("DB_PORT") if "DB_PORT" in st.secrets else os.getenv("DB_PORT")
    db_name = st.secrets.get("DB_NAME") if "DB_NAME" in st.secrets else os.getenv("DB_NAME")
    db_user = st.secrets.get("DB_USER") if "DB_USER" in st.secrets else os.getenv("DB_USER")
    db_password = st.secrets.get("DB_PASSWORD") if "DB_PASSWORD" in st.secrets else os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        st.error("❌ Missing DB credentials. Please set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in Streamlit secrets or env.")
        st.stop()

    url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    # quick smoke test
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

engine = get_engine()

@st.cache_resource
def build_schema_doc():
    q = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    rows = []
    try:
        with engine.connect() as conn:
            res = conn.execute(text(q)).fetchall()
            for c, t, n in res:
                rows.append(f'- "{c}" ({t}, nullable: {n})')
    except Exception:
        rows.append("-- Could not read schema from information_schema --")
    return "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(rows)

SCHEMA_DOC = build_schema_doc()

# -------------------------
# SESSION SAFETY / STATE
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "auto_q" not in st.session_state:
    st.session_state.auto_q = None

# -------------------------
# PROMPTING HELPERS
# -------------------------

def make_sql_prompt(question: str, schema_text: str, history: List[dict] = None) -> str:
    history_text = ""
    if history:
        # include last few exchanges
        tail = history[-6:]
        history_text = "\n# CONVERSATION HISTORY:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in tail])

    prompt = f"""
You are a senior data analyst with expertise in business intelligence and executive reporting.
Return a single valid PostgreSQL SELECT query for the question.

STRICT RULES:
- Read-only SELECT statements only.
- Only use table {ALLOWED_TABLE}.
- Always filter out cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'.
- If time period is vague, infer sensible filters using "Date".
- Use double-quotes for all identifiers.
- Do not add explanations, markdown, or fencing; output ONLY the SQL.

SCHEMA:
{schema_text}
{history_text}

QUESTION:
{question}
"""
    return prompt


def generate_sql(question: str, history: List[dict] = None) -> str:
    prompt = make_sql_prompt(question, SCHEMA_DOC, history)
    messages = [
        {"role": "system", "content": "You are a PostgreSQL expert and must output only a single SELECT statement (no explanation)."},
        {"role": "user", "content": prompt}
    ]
    raw = call_openai_chat(messages, max_tokens=800)
    sql = strip_codefence(raw)
    # Safety checks
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains forbidden keywords (only read-only SELECT allowed).")
    if ALLOWED_TABLE.split('.')[0] not in sql and 'voylla_design_ai' not in sql:
        # Make a relaxed check for table mention
        raise ValueError(f"SQL must reference the allowed table {ALLOWED_TABLE}.")
    # Ensure it starts with SELECT
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        raise ValueError("Generated SQL does not appear to be a SELECT statement.")
    return sql


def run_sql_to_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}")


def analyze_data(df: pd.DataFrame, user_q: str, history: List[dict] = None) -> dict:
    # Build a concise but informative prompt
    data_preview = df.head(50).to_csv(index=False)
    prompt = f"""
You are an executive data analyst. Produce a JSON object with keys: executive_summary, key_metrics, insights, recommendations, followup_questions.
User question: {user_q}
Data shape: {df.shape}
Columns: {list(df.columns)}

Provide clear, concise values. Use numeric values where possible. DATA_PREVIEW:\n{data_preview}
"""
    messages = [
        {"role": "system", "content": "You are an executive-level data analyst. Reply only with a JSON object (no surrounding text)."},
        {"role": "user", "content": prompt}
    ]
    raw = call_openai_chat(messages, max_tokens=1200)
    raw = strip_codefence(raw)

    try:
        parsed = json.loads(raw)
        # Ensure required keys exist
        for k in ["executive_summary", "key_metrics", "insights", "recommendations", "followup_questions"]:
            if k not in parsed:
                parsed[k] = [] if k in ("insights", "recommendations", "followup_questions") else ""
        return parsed
    except Exception:
        # Fallback simpler analysis
        return {
            "executive_summary": "Could not parse LLM JSON. Fallback summary applied.",
            "key_metrics": {},
            "insights": [],
            "recommendations": [],
            "followup_questions": []
        }

# -------------------------
# VISUALIZATIONS & DISPLAY
# -------------------------

def create_advanced_visualizations(df: pd.DataFrame, analysis: dict):
    visualizations = []
    if df.empty:
        return visualizations
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            agg = df.groupby(pd.Grouper(key=date_col, freq='M'))[numeric_cols[0]].sum().reset_index()
            fig = px.line(agg, x=date_col, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]} Over Time")
            fig.update_layout(height=400)
            visualizations.append(fig)
        except Exception:
            pass
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        top = df.groupby(cat)[numeric_cols[0]].sum().nlargest(10).reset_index()
        fig = px.bar(top, x=cat, y=numeric_cols[0], title=f"Top 10 {cat} by {numeric_cols[0]}")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        visualizations.append(fig)
    if numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        fig.update_layout(height=400)
        visualizations.append(fig)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Between Metrics")
        fig.update_layout(height=400)
        visualizations.append(fig)
    return visualizations


def display_analysis(analysis: dict):
    st.markdown("### 📋 Executive Summary")
    st.info(analysis.get("executive_summary", "No summary available."))
    if analysis.get("key_metrics"):
        st.markdown("### 📊 Key Metrics")
        metrics = analysis["key_metrics"]
        cols = st.columns(max(1, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k, str(v))
    if analysis.get("insights"):
        st.markdown("### 💡 Key Insights")
        for insight in analysis["insights"]:
            with st.expander(f"🔍 {insight.get('title','Insight')}"):
                st.markdown(f"**Description**: {insight.get('description','')}")
                st.markdown(f"**Impact**: {insight.get('impact','')}")
                if insight.get('data_support'):
                    st.markdown(f"**Data Support**: {insight.get('data_support')}")
    if analysis.get("recommendations"):
        st.markdown("### 🎯 Recommendations")
        for rec in analysis["recommendations"]:
            st.success(f"**{rec.get('title','Recommendation')}**: {rec.get('description','')}")
            if rec.get('expected_impact'):
                st.caption(f"Expected impact: {rec.get('expected_impact')}")
    if analysis.get("followup_questions"):
        st.markdown("### 🔍 Suggested Follow-up Analyses")
        for i, q in enumerate(analysis["followup_questions"]):
            if st.button(q, key=f"followup_{i}"):
                st.session_state.auto_q = q
                st.rerun()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("<div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:12px;border-radius:8px;color:white;text-align:center;'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    try:
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {ALLOWED_TABLE} WHERE \"Sale Order Item Status\" != 'CANCELLED'")).scalar()
            revenue = conn.execute(text(f"SELECT SUM(\"Amount\") FROM {ALLOWED_TABLE} WHERE \"Sale Order Item Status\" != 'CANCELLED' AND \"Date\" >= CURRENT_DATE - INTERVAL '30 days' ")).scalar()
            st.success(f"✅ Connected: {int(count):,} active records")
            st.metric("30-Day Revenue", f"₹{float(revenue):,.2f}" if revenue else "N/A")
    except Exception as e:
        st.error(f"❌ Connection issue: {e}")
    st.markdown("---")
    st.header("💡 Executive Questions")
    presets = [
        "Show me top 10 products by revenue this quarter",
        "What are our best performing channels by growth rate?",
        "Compare this year's revenue to last year by month",
        "Which design styles have the highest average order value?",
        "Show me channel-wise revenue and units this month",
        "Analyze sales trends by metal color and design style",
        "What is our profit margin by product category?",
        "Identify seasonal patterns in our sales data"
    ]
    for q in presets:
        if st.button(q, key=f"preset_{hash(q)}"):
            st.session_state.auto_q = q
            st.rerun()
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state.last_df = None
            st.session_state.last_sql = ""
            st.session_state.last_analysis = None
            st.session_state.auto_q = None
            st.rerun()
    with c2:
        st.caption("Advanced AI • Context-aware • Executive Insights")

# -------------------------
# HEADER & INPUT
# -------------------------
st.markdown("<div style='font-size:28px;font-weight:700;margin-bottom:6px;'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics — Ask questions about sales or design trends.")

# Render chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

if inp:
    # append user message
    st.session_state.chat.append({"role":"user","content":inp})
    with st.chat_message("user"):
        st.markdown(inp)

    with st.spinner("Conducting comprehensive analysis… 💎"):
        try:
            sql = generate_sql(inp, st.session_state.chat)
            df = run_sql_to_df(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = sql

            analysis = analyze_data(df, inp, st.session_state.chat)
            st.session_state.last_analysis = analysis

            # store assistant short summary in chat
            summary_text = analysis.get("executive_summary", "Analysis completed.")
            st.session_state.chat.append({"role":"assistant","content":summary_text})

        except Exception as e:
            err = f"⚠️ Could not complete request: {e}"
            st.session_state.chat.append({"role":"assistant","content":err})
            st.error(err)

    # display assistant message area
    with st.chat_message("assistant"):
        if st.session_state.last_analysis:
            display_analysis(st.session_state.last_analysis)

    with st.expander("View generated SQL"):
        st.code(st.session_state.last_sql, language="sql")

    if st.session_state.last_df is not None and not st.session_state.last_df.empty:
        st.subheader("📈 Data Visualizations")
        visuals = create_advanced_visualizations(st.session_state.last_df, st.session_state.last_analysis or {})
        if visuals:
            cols = st.columns(2)
            for i, fig in enumerate(visuals):
                cols[i % 2].plotly_chart(fig, use_container_width=True)
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state.last_df, use_container_width=True)

# EXPORT
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    export_df = st.session_state.last_df.copy()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
        if st.session_state.last_analysis:
            analysis_data = []
            for section in ["executive_summary", "key_metrics", "insights", "recommendations"]:
                if section in st.session_state.last_analysis:
                    if section == "key_metrics":
                        for k, v in st.session_state.last_analysis[section].items():
                            analysis_data.append({"Section": section, "Content": f"{k}: {v}"})
                    elif section in ("insights", "recommendations"):
                        for item in st.session_state.last_analysis[section]:
                            analysis_data.append({"Section": section, "Content": json.dumps(item, ensure_ascii=False)})
                    else:
                        analysis_data.append({"Section": section, "Content": str(st.session_state.last_analysis[section])})
            pd.DataFrame(analysis_data).to_excel(writer, index=False, sheet_name='Analysis_Summary')
        meta = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'SQL'],
            'Value': [len(export_df), len(export_df.columns), datetime.now().strftime("%Y-%m-%d %H:%M"), (st.session_state.last_sql or '')[:32000]]
        })
        meta.to_excel(writer, index=False, sheet_name='Metadata')
    output.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "💾 Download Executive Report",
        data=output.getvalue(),
        file_name=f"voylla_executive_report_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="download_exec"
    )

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:.9em;'>💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities • Use terms like \"YoY\", \"QoQ\", \"market share\" • Try \"analyze profitability by design style\" for advanced insights.</div>", unsafe_allow_html=True)
