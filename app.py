#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from io import BytesIO
import re
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- Enhanced spinner messages -----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as file:
        lines = [line.strip().split('. ', 1)[1] for line in file if '. ' in line]
else:
    lines = [
        "Crunching the numbers with a sparkle ✨",
        "Polishing your insights… 💎",
        "Setting the stones in your report…",
        "Crafting your jewelry analytics… 💍",
        "Mining data gems for you… ⛏️",
        "Designing your perfect answer… ✨",
    ]

# ---------- Enhanced secrets handling ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM with error recovery ----------
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

llm = get_llm()

# ---------- Enhanced DB CONNECTION with caching ----------
@st.cache_resource
def get_database_connection():
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        db = SQLDatabase(
            engine,
            include_tables=["voylla_design_ai"],
            schema="voylla"
        )
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

db = get_database_connection()

# ---------- Enhanced markdown parser ----------
def markdown_to_dataframe(markdown_text: str):
    """Parse a markdown table into a DataFrame with better error handling."""
    if not markdown_text or '|' not in markdown_text:
        return None
        
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    header_idx = None
    
    for i in range(len(lines) - 1):
        if '|' in lines[i]:
            sep = lines[i + 1] if i + 1 < len(lines) else ""
            if re.match(r'^[\s\|\-:]+$', sep) and sep.count('-') >= 2:
                header_idx = i
                break
    
    if header_idx is None:
        return None
    
    table_lines = [row for row in lines[header_idx:] if '|' in row and not re.match(r'^[\s\|\-:]+$', row)]
    if len(table_lines) < 2:
        return None
    
    # Normalize pipes and handle edge cases
    normalized = []
    for row in table_lines:
        r = row.strip()
        if not r.startswith('|'):
            r = '|' + r
        if not r.endswith('|'):
            r += '|'
        normalized.append(r)
    
    # Ensure consistent column count
    if not normalized:
        return None
        
    header_cols = len(normalized[0].split('|')) - 2
    cleaned_rows = [r for r in normalized if (len(r.split('|')) - 2) == header_cols]
    
    from io import StringIO
    try:
        df = pd.read_csv(
            StringIO("\n".join(cleaned_rows)),
            sep=r'\s*\|\s*',
            engine='python',
            skipinitialspace=True
        )
        df = df.dropna(how='all', axis=1)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        
        # Clean up common data issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        st.error(f"Table parsing error: {str(e)}")
        return None

# ---------- Enhanced conversation context ----------
def get_conversation_context():
    """Build enhanced conversation context with better formatting."""
    if not st.session_state.chat_history:
        return ""
    
    # Get last 8 exchanges for better context
    recent_history = st.session_state.chat_history[-8:]
    
    context_parts = []
    for i, msg in enumerate(recent_history):
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:600]  # Slightly longer context
        
        # Add timing context for recent messages
        if i >= len(recent_history) - 2:
            context_parts.append(f"[Recent] {role}: {content}")
        else:
            context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

# ---------- Chart generation helper ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    """Create appropriate charts based on DataFrame structure."""
    if df is None or df.empty:
        return None
    
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    # Auto-detect best chart type
    if chart_type == "auto":
        if len(df) <= 20 and len(categorical_cols) >= 1:
            chart_type = "bar"
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
        else:
            chart_type = "line"
    
    try:
        if chart_type == "bar" and len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}")
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        elif chart_type == "line" and len(numeric_cols) >= 1:
            fig = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}")
        else:
            return None
            
        fig.update_layout(height=400)
        return fig
    except Exception:
        return None

# ---------- Enhanced UI ----------
st.set_page_config(
    page_title="Voylla DesignGPT",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling
st.markdown("""
<style>
.stApp {
    background-color: #fcf1ed;
    color: #000;
    font-weight: 500
}
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] {
    color:#000 !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.success-indicator {
    color: #28a745;
    font-weight: bold;
}
.warning-indicator {
    color: #ffc107;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- Enhanced Sidebar ----------
with st.sidebar:
    st.header("📊 Connection Status")
    st.success("✅ Design table: voylla_design_ai")
    
    # Add connection health check
    try:
        result = db.run("SELECT COUNT(*) as total_records FROM voylla.\"voylla_design_ai\" LIMIT 1")
        if result:
            record_count = result.strip().split('\n')[-1].strip('[]()').split(',')[0] if result else "Unknown"
            st.info(f"📈 Total records: {record_count}")
    except:
        st.warning("⚠️ Connection check failed")
    
    st.header("💡 Enhanced Sample Questions")
    
    # Categorized sample questions
    with st.expander("📈 Sales Performance", expanded=True):
        sample_questions = [
            "Best selling success combination last 90 days",
            "Top 20 SKUs by revenue this month", 
            "Which Design Style performs best on Myntra?",
            "Revenue trends by month for last 6 months"
        ]
        for q in sample_questions:
            if st.button(f"• {q}", key=f"perf_{q}"):
                st.session_state.auto_question = q
    
    with st.expander("🎨 Design Analysis"):
        design_questions = [
            "Trend of Form × Metal Color by qty",
            "Success combinations for Wedding/Festive look",
            "Most popular Central Stone by category",
            "Contemporary vs Traditional design performance"
        ]
        for q in design_questions:
            if st.button(f"• {q}", key=f"design_{q}"):
                st.session_state.auto_question = q
    
    with st.expander("📊 Channel Analysis"):
        channel_questions = [
            "Performance comparison across all channels",
            "Best performing designs on each platform",
            "Channel-wise AOV analysis"
        ]
        for q in channel_questions:
            if st.button(f"• {q}", key=f"channel_{q}"):
                st.session_state.auto_question = q
    
    st.header("🔧 Context Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        show_charts = st.checkbox("📊 Auto Charts", value=False)
    
    # Memory usage indicator (with safe check)
    if "chat_history" in st.session_state:
        memory_size = len(st.session_state.chat_history)
        if memory_size > 0:
            st.caption(f"💭 Chat history: {memory_size} messages")

# ---------- Main Title ----------
st.title("💬 Voylla DesignGPT")
st.caption("Ask anything about design traits, sales performance, and success combinations.")

# Add quick stats if available
try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">📊 Real-time Analytics</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🎨 Design Intelligence</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">💎 Sales Insights</div>', unsafe_allow_html=True)
except:
    pass

# ---------- Initialize session state FIRST ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- Enhanced Memory / Agent ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=12  # Increased for better context
    )

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=10,  # Allow more iterations for complex queries
        early_stopping_method="generate"
    )

# ---------- Render chat history ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Handle auto-generated questions ----------
if "auto_question" in st.session_state and st.session_state.auto_question:
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None
else:
    user_input = st.chat_input("Ask about sales or design trends…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- ENHANCED SYSTEM PROMPT ----------
    conversation_context = get_conversation_context()
    
    # Better follow-up detection
    follow_up_indicators = [
        "show me", "can you", "what about", "how about", "also", "and", 
        "but", "however", "additionally", "similarly", "compare", "versus",
        "breakdown", "details", "more", "expand", "drill down"
    ]
    is_follow_up = any(indicator in user_input.lower() for indicator in follow_up_indicators) or len(st.session_state.chat_history) > 1

    # Enhanced prompt with better context awareness
    prompt = f"""
You are Voylla DesignGPT, an expert SQL/analytics assistant for Voylla jewelry data analysis.

# CONVERSATION CONTEXT
{"🔄 FOLLOW-UP DETECTED: This appears to be a follow-up question. Build upon previous analysis. " if is_follow_up else ""}

Recent conversation:
{conversation_context}

# ENHANCED SAFETY PROTOCOLS
- **STRICTLY READ-ONLY**: Never run DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE
- **Complete Results**: Return ALL rows unless user explicitly requests LIMIT
- **PostgreSQL Syntax**: Always double-quote column names, use proper PostgreSQL functions
- **Error Recovery**: If query fails, suggest simpler alternatives

# DATABASE SCHEMA: voylla."voylla_design_ai"

## KEY COLUMNS FOR ANALYSIS
### Transaction & Business Data
- **"Date"** (timestamp) — Transaction date (DD/MM/YYYY HH:MM)
- **"Channel"** (text) — Sales platform (Cloudtail, FLIPKART, MYNTRA, NYKAA, etc.)
- **"Sale Order Item Status"** (text) — Order status (DISPATCHED, DELIVERED, CANCELLED)
- **"Qty"** (integer) — Units sold
- **"Amount"** (numeric) — Revenue (Qty × price)
- **"MRP"** (numeric) — Maximum Retail Price
- **"Cost Price"** (numeric) — Unit cost
- **"Discount"** (numeric) — Discount applied

### Product Identification
- **"EAN"** (text) — Product barcode
- **"Product Code"** (text) — SKU identifier
- **"Collection"** (text) — Product collection
- **"Category"** (text) — Main category (Earrings, Necklaces, Rings)
- **"Sub-Category"** (text) — Subcategory (Studs, Hoops, Chandbali)

### Design Intelligence Attributes
- **"Design Style"** (text) — Aesthetic (Tribal, Contemporary, Traditional/Ethnic, Minimalist)
- **"Form"** (text) — Shape (Triangle, Stud, Hoop, Jhumka, Ear Cuff)
- **"Metal Color"** (text) — Finish (Antique Silver, Yellow Gold, Rose Gold, Silver, Antique Gold, Oxidized Black)
- **"Look"** (text) — Occasion/vibe (Oxidized, Everyday, Festive, Party, Wedding)
- **"Craft Style"** (text) — Technique (Handcrafted, etc.)
- **"Central Stone"** (text) — Primary gemstone
- **"Surrounding Layout"** (text) — Stone arrangement
- **"Stone Setting"** (text) — Mounting style
- **"Style Motif"** (text) — Design theme (Geometric, Floral, Abstract)

# MANDATORY FILTERS
- **Always exclude cancelled**: WHERE "Sale Order Item Status" <> 'CANCELLED'
- **Date handling**: If no date specified, use ALL available data
- **Channel analysis**: Group by "Channel" when comparing platforms

# ENHANCED METRICS DEFINITIONS
- **Total Quantity**: SUM("Qty")
- **Total Revenue**: SUM("Amount")
- **Average Order Value (AOV)**: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- **Success Score**: Rank by both quantity and revenue
- **Profit Margin**: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100

# INTELLIGENT QUERY PATTERNS
## For Trending Analysis:
```sql
SELECT 
    EXTRACT(YEAR FROM "Date"::date) as year,
    EXTRACT(MONTH FROM "Date"::date) as month,
    "Design Style",
    SUM("Qty") as total_qty,
    SUM("Amount") as total_revenue
FROM voylla."voylla_design_ai"
WHERE "Sale Order Item Status" <> 'CANCELLED'
GROUP BY year, month, "Design Style"
ORDER BY year DESC, month DESC, total_qty DESC;
```

## For Success Combinations:
```sql
SELECT 
    "Design Style", "Form", "Metal Color", "Look",
    COUNT(*) as transactions,
    SUM("Qty") as total_qty,
    SUM("Amount") as total_revenue,
    ROUND(SUM("Amount") / NULLIF(SUM("Qty"), 0), 2) as aov
FROM voylla."voylla_design_ai"
WHERE "Sale Order Item Status" <> 'CANCELLED'
GROUP BY "Design Style", "Form", "Metal Color", "Look"
HAVING SUM("Qty") >= 10
ORDER BY total_qty DESC, total_revenue DESC
LIMIT 15;
```

# ADVANCED FOLLOW-UP HANDLING
- **Expansion requests**: "show more details" → Add more columns or increase LIMIT
- **Filtering requests**: "what about gold items" → Add WHERE "Metal Color" LIKE '%Gold%'
- **Comparison requests**: "compare X vs Y" → Use CASE statements or UNION
- **Time analysis**: "trends over time" → Add date grouping
- **Reference previous results**: Start with "Building on our previous analysis..."

# ENHANCED OUTPUT FORMATTING
- **Always** return results as clean markdown tables
- **Include context**: "Based on X records from Y date range..."
- **Highlight insights**: Bold key findings in the explanation
- **Suggest follow-ups**: End with "Would you like me to analyze [specific aspect]?"

# CURRENT ANALYSIS REQUEST
{user_input}

Remember: Be conversational, insightful, and always build upon previous context when applicable.
"""

    # ---------- Enhanced execution with better error handling ----------
    random_template = random.choice(lines)
    
    with st.spinner(random_template.strip()):
        try:
            response = st.session_state.agent_executor.run(prompt)
            st.session_state.last_query_result = response
            
        except ValueError as e:
            raw = str(e)
            if "Could not parse LLM output:" in raw:
                response = raw.split("Could not parse LLM output:")[-1].strip()
            else:
                response = f"I encountered a parsing issue. Let me try a different approach: {raw}"
                
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                response = "⏱️ I'm experiencing high demand. Please wait a moment and try again."
            elif "connection" in error_msg.lower():
                response = "🔌 Database connection issue. Please check your connection and try again."
            else:
                response = f"I apologize, but I encountered an error: {error_msg}. Could you please rephrase your question?"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)

    # ---------- Enhanced table parsing and visualization ----------
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res
        
        # Auto-generate charts if enabled
        if 'show_charts' in locals() and show_charts and len(df_res) > 1:
            chart = create_chart_from_dataframe(df_res)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

# ---------- Enhanced download functionality ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        MAX_EXPORT_ROWS = st.selectbox("Export rows:", [100, 500, 1000, "All"], index=1)
        if MAX_EXPORT_ROWS == "All":
            export_df = st.session_state.last_df.copy()
        else:
            export_df = st.session_state.last_df.iloc[:MAX_EXPORT_ROWS].copy()
    
    with col2:
        # Show data summary
        st.caption(f"📋 {len(export_df)} rows × {len(export_df.columns)} columns")
    
    with col3:
        # Enhanced download options
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='VoyllaData')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        st.download_button(
            "📥 Download Excel",
            data=output.getvalue(),
            file_name=f"voylla_insights_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_button_enhanced"
        )

# ---------- Footer with tips ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <b>Pro Tips:</b> Ask follow-up questions like "show me more details" or "what about gold items?" • 
Try questions about trends, comparisons, and specific time periods • 
Use natural language - I understand context!
</div>
""", unsafe_allow_html=True)
