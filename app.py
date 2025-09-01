#!/usr/bin/env python
# coding: utf-8
import streamlit as st
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os, re, time
import pandas as pd
from io import BytesIO
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Voylla DesignGPT - Executive Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, stable model for deterministic SQL text output
MODEL_NAME = "gpt-4o-mini"   # Fixed model name
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
    background-color: #f8f9fa; border-radius: 12px; padding: 1rem; 
    border-left: 4px solid #667eea; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# KEYS & CONNECTIONS
# =========================
load_dotenv()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
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
        # Try secrets first, then environment variables
        try:
            db_host = st.secrets["DB_HOST"]
            db_port = st.secrets["DB_PORT"]
            db_name = st.secrets["DB_NAME"]
            db_user = st.secrets["DB_USER"]
            db_password = st.secrets["DB_PASSWORD"]
        except:
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            
        if not all([db_host, db_port, db_name, db_user, db_password]):
            raise ValueError("Missing database configuration")
            
    except Exception as e:
        st.error("❌ Missing DB configuration. Please add DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.")
        st.stop()

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True, pool_recycle=3600, pool_size=5, max_overflow=10
    )
    
    # Smoke test
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

    # Build schema doc from information_schema for the allowed table
    schema_rows = []
    q = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='voylla' AND table_name='voylla_design_ai'
        ORDER BY ordinal_position
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(q)).fetchall()
            for c, t in rows:
                schema_rows.append(f'- "{c}" ({t})')
    except Exception as e:
        st.error(f"❌ Could not fetch schema: {e}")
        st.stop()
    
    if not schema_rows:
        st.error("❌ Table voylla.voylla_design_ai not found or empty")
        st.stop()
        
    schema_string = "Table: voylla.\"voylla_design_ai\" (read-only)\n" + "\n".join(schema_rows)
    return engine, schema_string

engine, schema_doc = get_engine_and_schema()

# =========================
# HELPERS
# =========================
DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I)

def get_conversation_context():
    """Build enhanced conversation context with better formatting."""
    if not st.session_state.chat_history:
        return ""
    
    # Get last 6 exchanges for better context
    recent_history = st.session_state.chat_history[-6:]
    
    context_parts = ["### Conversation History:"]
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]  # Limit length
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

def should_show_chart(question: str, df: pd.DataFrame) -> bool:
    """Determine if a chart would be appropriate for the question and data."""
    if df.empty or len(df) < 2:
        return False
    
    # Keywords that suggest visualization would be helpful
    chart_keywords = [
        'trend', 'compare', 'comparison', 'growth', 'over time', 'by month', 'by year',
        'top', 'ranking', 'performance', 'distribution', 'analysis', 'breakdown',
        'versus', 'vs', 'chart', 'graph', 'visualize', 'show me'
    ]
    
    # Keywords that suggest no chart needed
    no_chart_keywords = [
        'count', 'total', 'sum', 'average', 'mean', 'specific', 'exact',
        'list all', 'show all', 'details', 'information about'
    ]
    
    question_lower = question.lower()
    
    # If explicitly asking for no visualization
    for keyword in no_chart_keywords:
        if keyword in question_lower and not any(ck in question_lower for ck in chart_keywords):
            return False
    
    # If asking for visualization or comparative analysis
    for keyword in chart_keywords:
        if keyword in question_lower:
            return True
    
    # Check data characteristics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Need at least one numeric column for meaningful charts
    if not numeric_cols:
        return False
    
    # If we have good categorical data and the result isn't too large
    if categorical_cols and len(df) <= 50:
        return True
    
    return False

def should_show_table(question: str, df: pd.DataFrame) -> bool:
    """Determine if showing the full table would be appropriate."""
    if df.empty:
        return False
    
    # Always show table for small results
    if len(df) <= 10:
        return True
    
    # Keywords that suggest detailed tabular data is needed
    table_keywords = [
        'list', 'show all', 'details', 'breakdown', 'individual', 'each',
        'specific', 'exact', 'complete', 'full'
    ]
    
    # Keywords that suggest summary is enough
    summary_keywords = [
        'total', 'sum', 'count', 'average', 'mean', 'trend', 'growth',
        'top 5', 'top 10', 'summary', 'overview'
    ]
    
    question_lower = question.lower()
    
    # If asking for summary-level info, limit table size
    for keyword in summary_keywords:
        if keyword in question_lower:
            return len(df) <= 20
    
    # If asking for detailed info, show more
    for keyword in table_keywords:
        if keyword in question_lower:
            return len(df) <= 100
    
    # Default: show table if reasonable size
    return len(df) <= 25

def make_sql_prompt(question: str, schema_text: str) -> str:
    conversation_context = get_conversation_context()
    
    return f"""
You are a senior data analyst. Return a single **valid PostgreSQL** SELECT query for the question.

STRICT RULES:
- Read-only SELECT statements only.
- Only use table voylla."voylla_design_ai".
- Always filter out cancelled items: WHERE "Sale Order Item Status" != 'CANCELLED'.
- If time period is vague (e.g., "this quarter", "last 6 months"), infer sensible filters using "Date".
- Use double-quotes for all identifiers.
- Do not add explanations, markdown, or fencing; output ONLY the SQL.
- Limit results to reasonable sizes (use LIMIT when appropriate).

{conversation_context}

SCHEMA:
{schema_text}

QUESTION:
{question}
"""

def generate_sql(question: str) -> str:
    prompt = make_sql_prompt(question, schema_doc)
    response = llm.invoke(prompt)
    sql = response.content.strip()
    
    # Strip possible codefences if the model adds them
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z0-9]*", "", sql).strip()
        sql = sql[:-3] if sql.endswith("```") else sql
        sql = sql.strip()
    
    # Safety checks
    if DANGEROUS.search(sql):
        raise ValueError("Generated SQL contains a non read-only keyword.")
    if "voylla_design_ai" not in sql:
        raise ValueError("SQL must reference voylla.\"voylla_design_ai\".")
    
    return sql

def run_sql_to_df(sql: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(sql, conn)
    except Exception as e:
        raise RuntimeError(f"SQL execution error: {e}")

def summarize_for_executives(df: pd.DataFrame, user_q: str) -> str:
    """Generate executive summary of the results."""
    if df.empty:
        return "No data found for the requested analysis."
    
    # Keep token-light by downsampling preview
    preview_csv = df.head(50).to_csv(index=False)
    conversation_context = get_conversation_context()
    
    prompt = f"""
You are Voylla DesignGPT Executive Edition, an expert analytics assistant for Voylla jewelry data analysis.

{conversation_context}

# EXECUTIVE REPORTING GUIDELINES
- Focus on business insights, not just data
- Highlight trends, opportunities, and risks
- Compare performance metrics when possible
- Use clear, concise language appropriate for executives
- Provide actionable recommendations when relevant
- Format responses with proper markdown for readability

# KEY BUSINESS METRICS
- Revenue: SUM("Amount")
- Units: SUM("Qty") 
- Average Order Value: Revenue/Units
- Profit Margin: (Revenue - Cost)/Revenue * 100

USER QUESTION: {user_q}

DATA PREVIEW (CSV):
{preview_csv}

Provide a clear executive summary focusing on key insights and business implications.
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Analysis completed. Found {len(df)} records. Please review the data table below for detailed results."

def auto_chart(df: pd.DataFrame, question: str):
    """Generate appropriate chart based on data and question context."""
    if df.empty or not should_show_chart(question, df):
        return None
    
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not num_cols:
        return None
    
    try:
        # Choose the most appropriate chart type
        if date_cols and len(df) > 1:
            # Time series chart
            fig = px.line(df, x=date_cols[0], y=num_cols[0], 
                         title=f"{num_cols[0]} Over Time")
        elif cat_cols and len(df) <= 20:
            # Bar chart for categorical data
            work_df = df.copy()
            if len(work_df) > 15:
                work_df = work_df.nlargest(15, num_cols[0])
            fig = px.bar(work_df, x=cat_cols[0], y=num_cols[0], 
                        title=f"{num_cols[0]} by {cat_cols[0]}")
            fig.update_xaxis(tickangle=45)
        elif len(num_cols) >= 2 and len(df) <= 100:
            # Scatter plot for correlation
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                           title=f"{num_cols[1]} vs {num_cols[0]}")
        else:
            return None
        
        fig.update_layout(height=400, showlegend=False)
        return fig
        
    except Exception as e:
        return None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<div class='metric-card'>📊 Executive Dashboard</div>", unsafe_allow_html=True)
    
    # Connection status
    try:
        with engine.connect() as conn:
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
        "Show me top 10 products by revenue this quarter",
        "What are our best performing channels by growth rate?", 
        "Compare this year's revenue to last year by month",
        "Which design styles have the highest average order value?",
        "Show me channel-wise revenue and units this month"
    ]
    
    for i, q in enumerate(presets):
        if st.button(f"• {q}", key=f"preset_{i}"):
            st.session_state["auto_q"] = q

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            for key in ['chat', 'chat_history', 'last_df', 'last_sql', 'auto_q']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        st.caption("Agent-free • stable")

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state: 
    st.session_state.chat = []
if "auto_q" not in st.session_state: 
    st.session_state.auto_q = None
if "last_df" not in st.session_state: 
    st.session_state.last_df = None
if "last_sql" not in st.session_state: 
    st.session_state.last_sql = ""

# =========================
# MAIN INTERFACE
# =========================
st.markdown("<div class='main-header'>Voylla DesignGPT Executive Dashboard</div>", unsafe_allow_html=True)
st.caption("AI-Powered Design Intelligence and Sales Analytics")

# Render chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle input
inp = st.chat_input("Ask an executive question about sales or design trends…", key="chat_box")

if st.session_state.auto_q:
    inp = st.session_state.auto_q
    st.session_state.auto_q = None

if inp:
    # Add to chat history
    st.session_state.chat.append({"role": "user", "content": inp})
    st.session_state.chat_history.append({"role": "user", "content": inp})
    
    with st.chat_message("user"):
        st.markdown(inp)
    
    with st.spinner("Analyzing your request... 💎"):
        try:
            # Generate and execute SQL
            sql = generate_sql(inp)
            df = run_sql_to_df(sql)
            st.session_state.last_df = df
            st.session_state.last_sql = sql
            
            # Generate executive summary
            summary = summarize_for_executives(df, inp)
            
        except Exception as e:
            summary = f"⚠️ Could not complete request: {str(e)}"
            df = pd.DataFrame()
    
    # Display assistant response
    with st.chat_message("assistant"):
        if summary:
            st.markdown(f"<div class='assistant-message'>{summary}</div>", unsafe_allow_html=True)
    
    # Add assistant response to chat history
    if summary:
        st.session_state.chat.append({"role": "assistant", "content": summary})
        st.session_state.chat_history.append({"role": "assistant", "content": summary})
    
    # Show SQL (collapsible)
    if st.session_state.last_sql:
        with st.expander("🔍 View Generated SQL"):
            st.code(st.session_state.last_sql, language="sql")
    
    # Show results conditionally
    if not df.empty:
        # Show table only if appropriate
        if should_show_table(inp, df):
            st.subheader("📋 Data Results")
            if len(df) > 100:
                st.info(f"Showing first 100 rows of {len(df)} total results")
                st.dataframe(df.head(100), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        # Show chart only if appropriate  
        fig = auto_chart(df, inp)
        if fig:
            st.subheader("📊 Data Visualization")
            st.plotly_chart(fig, use_container_width=True)

# Export functionality
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    export_df = st.session_state.last_df.copy()
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Executive_Report')
            
            # Add metadata sheet
            meta = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Export Date', 'SQL Query'],
                'Value': [
                    len(export_df), 
                    len(export_df.columns),
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    st.session_state.last_sql[:1000]  # Truncate long queries
                ]
            })
            meta.to_excel(writer, index=False, sheet_name='Summary')
        
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
        
    except Exception as e:
        st.error(f"Export failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:.9em;'>
💡 <b>Executive Tips:</b> Ask about trends, comparisons, performance metrics, and growth opportunities •
Use terms like "YoY", "QoQ", "market share", "trending", "best performing" •
Say "show me channel-wise performance this quarter" to start.
</div>
""", unsafe_allow_html=True)
