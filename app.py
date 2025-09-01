#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configuration
MODEL_NAME = "gpt-4"  # More capable model for complex analysis
LLM_TEMPERATURE = 0.1

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
.assistant-message { 
    background-color: #f8f9fa; 
    border-radius: 12px; 
    padding: 1rem; 
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.insight-card {
    background-color: #fff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #ffd700;
}
.recommendation-card {
    background-color: #f0f8ff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #4682b4;
}
.stButton button {
    width: 100%;
}
.css-1d391kg {padding: 1.5rem;}
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
    return ChatOpenAI(model=MODEL_NAME, temperature=LLM_TEMPERATURE, request_timeout=120, max_retries=3)

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
    # smoke test
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    # Build schema doc from information_schema for the allowed table
    schema_rows = []
    q = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q)).fetchall()
        for c, t, n in rows:
            schema_rows.append(f'- "{c}" ({t}, nullable: {n})')
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(schema_rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# MEMORY & CONVERSATION MANAGEMENT
# =========================
@st.cache_resource
def get_memory():
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")

memory = get_memory()

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

def make_sql_prompt(question: str, schema_text: str, history: list = None) -> str:
    history_text = ""
    if history:
        history_text = "\n# CONVERSATION HISTORY:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    
    return f"""
You are a senior data analyst with expertise in business intelligence and executive reporting. 
Return a single **valid PostgreSQL** SELECT query for the question.

STRICT RULES:
- Read-only SELECT statements only.
- Only use table voylla."voylla_design_ai".
- Always filter out cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'.
- If time period is vague (e.g., "this quarter", "last 6 months"), infer sensible filters using "Date".
- Use double-quotes for all identifiers.
- Do not add explanations, markdown, or fencing; output ONLY the SQL.

SCHEMA:
{schema_text}
{history_text}

QUESTION:
{question}
"""

def generate_sql(question: str, history: list = None) -> str:
    prompt = make_sql_prompt(question, schema_doc, history)
    sql = llm.invoke(prompt).content.strip()
    # strip possible codefences if the model adds them
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
        sql = sql.strip()
    # safety
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains a non read-only keyword.")
    if "voylla_design_ai" not in sql:
        raise ValueError("SQL must reference voylla.\"voylla_design_ai\".")
    return sql

def run_sql_to_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}")

def analyze_data(df: pd.DataFrame, user_q: str, history: list = None) -> dict:
    # Create a comprehensive analysis with multiple sections
    analysis_prompt = f"""
You are an executive data analyst at Voylla, a jewelry company. 
Analyze the provided data to answer the user's question and provide actionable insights.

USER QUESTION: {user_q}

DATA PREVIEW (first 20 rows):
{df.head(20).to_csv(index=False)}

DATA STRUCTURE:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}

ANALYSIS REQUIREMENTS:
1. Provide a comprehensive executive summary with key findings
2. Identify top 3-5 insights with supporting data
3. Highlight trends, patterns, and anomalies
4. Compare performance metrics (YoY, MoM, QoQ) if relevant
5. Provide 3-5 actionable recommendations
6. Suggest 2-3 follow-up questions for deeper analysis

Format your response as JSON with the following structure:
{{
  "executive_summary": "Brief overview of key findings",
  "key_metrics": {{
    "metric1": "value1",
    "metric2": "value2"
  }},
  "insights": [
    {{
      "title": "Insight title",
      "description": "Detailed explanation",
      "impact": "Business impact",
      "data_support": "Supporting numbers or stats"
    }}
  ],
  "recommendations": [
    {{
      "title": "Recommendation title",
      "description": "Detailed recommendation",
      "expected_impact": "Expected business impact"
    }}
  ],
  "followup_questions": [
    "Question 1",
    "Question 2"
  ]
}}
"""
    try:
        response = llm.invoke(analysis_prompt).content.strip()
        return json.loads(response)
    except:
        # Fallback if JSON parsing fails
        return {
            "executive_summary": "Comprehensive analysis of the data with key business insights.",
            "key_metrics": {},
            "insights": [],
            "recommendations": [],
            "followup_questions": []
        }

def create_advanced_visualizations(df: pd.DataFrame, analysis: dict):
    """Create multiple visualizations based on data and analysis"""
    visualizations = []
    
    if df.empty:
        return visualizations
    
    # 1. Time series visualization if date column exists
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        try:
            # Try to convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            # Aggregate by time period
            time_df = df.groupby(pd.Grouper(key=date_col, freq='M'))[numeric_cols[0]].sum().reset_index()
            fig = px.line(time_df, x=date_col, y=numeric_cols[0], 
                         title=f"Trend of {numeric_cols[0]} Over Time")
            fig.update_layout(height=400)
            visualizations.append(fig)
        except:
            pass
    
    # 2. Top performers visualization
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        # Get top 10 categories by first numeric column
        top_df = df.groupby(cat_col)[numeric_cols[0]].sum().nlargest(10).reset_index()
        fig = px.bar(top_df, x=cat_col, y=numeric_cols[0], 
                    title=f"Top 10 {cat_col} by {numeric_cols[0]}")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        visualizations.append(fig)
    
    # 3. Distribution visualization
    if len(numeric_cols) > 0:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        fig.update_layout(height=400)
        visualizations.append(fig)
    
    # 4. Correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       title="Correlation Between Metrics")
        fig.update_layout(height=400)
        visualizations.append(fig)
    
    return visualizations

def display_analysis(analysis: dict):
    """Display the analysis in a structured way"""
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    st.info(analysis.get("executive_summary", "No summary available."))
    
    # Key Metrics
    if analysis.get("key_metrics"):
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(len(analysis["key_metrics"]))
        for i, (metric, value) in enumerate(analysis["key_metrics"].items()):
            cols[i].metric(metric, value)
    
    # Insights
    if analysis.get("insights"):
        st.markdown("### 💡 Key Insights")
        for insight in analysis["insights"]:
            with st.expander(f"🔍 {insight.get('title', 'Insight')}"):
                st.markdown(f"**Description**: {insight.get('description', '')}")
                st.markdown(f"**Impact**: {insight.get('impact', '')}")
                if insight.get('data_support'):
                    st.markdown(f"**Data Support**: {insight.get('data_support')}")
    
    # Recommendations
    if analysis.get("recommendations"):
        st.markdown("### 🎯 Recommendations")
        for rec in analysis["recommendations"]:
            st.success(f"**{rec.get('title', 'Recommendation')}**: {rec.get('description', '')}")
            if rec.get('expected_impact'):
                st.caption(f"Expected impact: {rec.get('expected_impact')}")
    
    # Follow-up Questions
    if analysis.get("followup_questions"):
        st.markdown("### 🔍 Suggested Follow-up Analyses")
        for i, question in enumerate(analysis["followup_questions"]):
            if st.button(question, key=f"followup_{i}"):
                st.session_state.auto_q = question
                st.rerun()

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
            revenue = conn.execute(text("""
                SELECT SUM("Amount") FROM voylla."voylla_design_ai" 
                WHERE "Sale Order Item Status" != 'CANCELLED'
                AND "Date" >= CURRENT_DATE - INTERVAL '30 days'
            """)).scalar()
            st.success(f"✅ Connected: {count:,} active records")
            st.metric("30-Day Revenue", f"₹{revenue:,.2f}" if revenue else "N/A")
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
        if st.button(f"• {q}", key=f"preset_{hash(q)}"):
            st.session_state["auto_q"] = q
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state["last_df"] = None
            st.session_state["last_sql"] = ""
            st.session_state["last_analysis"] = None
            memory.clear()
            st.rerun()
    with col2:
        st.caption("Advanced AI • Context-aware • Executive Insights")

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state: st.session_state.chat = []
if "auto_q" not in st.session_state: st.session_state.auto_q = None
if "last_df" not in st.session_state: st.session_state.last_df = None
if "last_sql" not in st.session_state: st.session_state.last_sql = ""
if "last_analysis" not in st.session_state: st.session_state.last_analysis = None

# =========================
# HEADER
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics — Advanced analysis with follow-up capabilities")

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Always-visible input (keeps mobile keyboard)
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")
if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

if inp:
    st.session_state.chat.append({"role":"user","content":inp})
    with st.chat_message("user"):
        st.markdown(inp)

    with st.spinner("Conducting comprehensive analysis… 💎"):
        try:
            # Generate and execute SQL
            sql = generate_sql(inp, st.session_state.chat)
            df = run_sql_to_df(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = sql
            
            # Perform advanced analysis
            analysis = analyze_data(df, inp, st.session_state.chat)
            st.session_state.last_analysis = analysis
            
            # Add to memory
            memory.save_context({"input": inp}, {"output": f"Analysis completed. Key findings: {analysis.get('executive_summary', '')}"})
            
        except Exception as e:
            error_msg = f"⚠️ Could not complete request: {e}"
            st.session_state.chat.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
            st.stop()

    # Assistant message with analysis
    with st.chat_message("assistant"):
        if st.session_state.last_analysis:
            display_analysis(st.session_state.last_analysis)
            # Add to chat history
            summary_text = st.session_state.last_analysis.get("executive_summary", "Analysis completed.")
            st.session_state.chat.append({"role": "assistant", "content": summary_text})

    # Show SQL (collapsible)
    with st.expander("View generated SQL"):
        st.code(st.session_state.last_sql, language="sql")

    # Data table & visualizations
    if not df.empty:
        st.subheader("📈 Data Visualizations")
        visuals = create_advanced_visualizations(df, st.session_state.last_analysis)
        
        if visuals:
            # Display visualizations in a grid
            cols = st.columns(2)
            for i, fig in enumerate(visuals):
                cols[i % 2].plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Data Preview")
        st.dataframe(df, use_container_width=True)

# Export
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    export_df = st.session_state.last_df.copy()
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
        
        # Add analysis summary
        if st.session_state.last_analysis:
            analysis_data = []
            for section in ["executive_summary", "key_metrics", "insights", "recommendations"]:
                if section in st.session_state.last_analysis:
                    if section == "key_metrics":
                        for k, v in st.session_state.last_analysis[section].items():
                            analysis_data.append({"Section": section, "Content": f"{k}: {v}"})
                    elif section == "insights" or section == "recommendations":
                        for item in st.session_state.last_analysis[section]:
                            analysis_data.append({"Section": section, "Content": str(item)})
                    else:
                        analysis_data.append({"Section": section, "Content": st.session_state.last_analysis[section]})
            
            pd.DataFrame(analysis_data).to_excel(writer, index=False, sheet_name='Analysis_Summary')
        
        meta = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'SQL'],
            'Value': [len(export_df), len(export_df.columns),
                      datetime.now().strftime("%Y-%m-%d %H:%M"),
                      st.session_state.last_sql[:32000]]
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
st.markdown("""
<div style='text-align:center;color:#666;font-size:.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" •
Try "analyze profitability by design style" or "show me seasonal trends" for advanced insights.
</div>
""", unsafe_allow_html=True)
