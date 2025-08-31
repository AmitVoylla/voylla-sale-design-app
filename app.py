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
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Spinner messages for professional tone -----------
SPINNER_MESSAGES = [
    "Analyzing jewelry trends with precision…",
    "Generating strategic insights…",
    "Processing your data query…",
    "Crafting actionable analytics…",
    "Building executive summary…",
    "Polishing your report…",
]

# ---------- Secrets handling with validation ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API key not found. Please configure it in environment variables.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM with retry mechanism ----------
@st.cache_resource
def get_llm():
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=3)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        logger.error(f"LLM initialization error: {str(e)}")
        st.stop()

llm = get_llm()

# ---------- Database connection with robust error handling ----------
@st.cache_resource(show_spinner=False)
def get_database_connection():
    try:
        db_config = {
            "host": st.secrets.get("DB_HOST", os.getenv("DB_HOST")),
            "port": st.secrets.get("DB_PORT", os.getenv("DB_PORT")),
            "name": st.secrets.get("DB_NAME", os.getenv("DB_NAME")),
            "user": st.secrets.get("DB_USER", os.getenv("DB_USER")),
            "password": st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD")),
        }
        if not all(db_config.values()):
            st.error("Database configuration incomplete. Check secrets/env variables.")
            st.stop()

        engine = create_engine(
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}",
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        
        db = SQLDatabase(
            engine,
            include_tables=["voylla_design_ai"],
            schema="voylla"
        )
        # Test connection
        db.run("SELECT 1")
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        logger.error(f"Database connection error: {str(e)}")
        st.stop()

db = get_database_connection()

# ---------- Markdown to DataFrame parser with validation ----------
def markdown_to_dataframe(markdown_text: str):
    try:
        if not markdown_text or '|' not in markdown_text:
            return None
        
        lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        header_idx = None
        
        for i in range(len(lines) - 1):
            if '|' in lines[i] and re.match(r'^[\s\|\-:]+$', lines[i + 1]):
                header_idx = i
                break
        
        if header_idx is None:
            return None
        
        table_lines = [row for row in lines[header_idx:] if '|' in row and not re.match(r'^[\s\|\-:]+$', row)]
        if len(table_lines) < 2:
            return None
        
        normalized = [f"|{row.strip()}|" if not row.startswith('|') else row for row in table_lines]
        header_cols = len(normalized[0].split('|')) - 2
        cleaned_rows = [r for r in normalized if len(r.split('|')) - 2 == header_cols]
        
        if not cleaned_rows:
            return None
        
        df = pd.read_csv(
            BytesIO("\n".join(cleaned_rows).encode()),
            sep=r'\s*\|\s*',
            engine='python',
            skipinitialspace=True
        )
        df = df.dropna(how='all', axis=1)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        logger.error(f"Table parsing error: {str(e)}")
        return None

# ---------- Chart generation with enhanced visuals ----------
def create_chart_from_dataframe(df, chart_type="auto"):
    try:
        if df is None or df.empty:
            return None
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        if chart_type == "auto":
            if len(df) <= 20 and categorical_cols:
                chart_type = "bar"
            elif len(numeric_cols) >= 2:
                chart_type = "scatter"
            else:
                chart_type = "line"
        
        fig = None
        if chart_type == "bar" and categorical_cols and numeric_cols:
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                        color_discrete_sequence=['#764ba2'])
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                           color_discrete_sequence=['#667eea'])
        elif chart_type == "line" and numeric_cols:
            fig = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}",
                         color_discrete_sequence=['#764ba2'])
        
        if fig:
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#333'),
                title_font=dict(size=16, color='#333')
            )
            return fig
        return None
    except Exception as e:
        logger.error(f"Chart creation error: {str(e)}")
        return None

# ---------- Professional UI configuration ----------
st.set_page_config(
    page_title="Voylla DesignGPT",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling for executive presentation
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
    color: #333;
}
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] {
    color: #333 !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.success-indicator {
    color: #28a745;
    font-weight: 600;
}
.warning-indicator {
    color: #dc3545;
    font-weight: 600;
}
.stButton>button {
    background-color: #764ba2;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar with executive focus ----------
with st.sidebar:
    st.header("📊 System Status")
    try:
        result = db.run("SELECT COUNT(*) as total_records FROM voylla.\"voylla_design_ai\" LIMIT 1")
        record_count = int(result.strip('[]()').split(',')[0]) if result else "Unknown"
        st.markdown(f'<div class="success-indicator">✅ Connected: {record_count} records</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="warning-indicator">⚠️ Database connection issue</div>', unsafe_allow_html=True)
    
    st.header("💡 Strategic Questions")
    sample_questions = [
        ("📈 Sales Insights", [
            "Top 10 best-selling designs this quarter",
            "Revenue trends by channel (last 6 months)",
            "Highest margin products by category",
            "AOV comparison across platforms"
        ]),
        ("🎨 Design Strategy", [
            "Top-performing design combinations",
            "Trending metal colors for festive season",
            "Popular styles by customer segment",
            "Traditional vs. Contemporary performance"
        ]),
        ("📊 Market Analysis", [
            "Channel-wise sales distribution",
            "Best-performing SKUs on Myntra",
            "Customer retention by collection",
            "Seasonal demand patterns"
        ])
    ]
    
    for category, questions in sample_questions:
        with st.expander(category, expanded=category == "📈 Sales Insights"):
            for q in questions:
                if st.button(f"• {q}", key=f"sample_{q}"):
                    st.session_state.auto_question = q
    
    st.header("⚙️ Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    with col2:
        show_charts = st.checkbox("📊 Show Visuals", value=True)

# ---------- Main interface ----------
st.title("💎 Voylla DesignGPT")
st.markdown("**Strategic insights for jewelry sales and design trends**")

# Executive summary metrics
try:
    col1, col2, col3 = st.columns(3)
    with col1:
        total_revenue = db.run('SELECT SUM("Amount") FROM voylla."voylla_design_ai" WHERE "Sale Order Item Status" <> \'CANCELLED\'')
        st.markdown(f'<div class="metric-card">💰 Total Revenue<br><b>{float(total_revenue.strip("[]()")):,.2f}</b></div>', unsafe_allow_html=True)
    with col2:
        total_qty = db.run('SELECT SUM("Qty") FROM voylla."voylla_design_ai" WHERE "Sale Order Item Status" <> \'CANCELLED\'')
        st.markdown(f'<div class="metric-card">📦 Units Sold<br><b>{int(total_qty.strip("[]()")):,}</b></div>', unsafe_allow_html=True)
    with col3:
        aov = db.run('SELECT SUM("Amount")/NULLIF(SUM("Qty"),0) FROM voylla."voylla_design_ai" WHERE "Sale Order Item Status" <> \'CANCELLED\'')
        st.markdown(f'<div class="metric-card">🛒 Avg. Order Value<br><b>{float(aov.strip("[]()")):,.2f}</b></div>', unsafe_allow_html=True)
except:
    st.warning("Unable to load summary metrics.")

# ---------- Session state initialization ----------
for key in ["chat_history", "last_df", "last_query_result", "auto_question"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

# ---------- Agent and memory setup ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=15
    )

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=15,
        early_stopping_method="generate"
    )

# ---------- Render chat history ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Handle input --------
if "auto_question" in st.session_state and st.session_state.auto_question:
    user_input = st.session_state.auto_question
    st.session_state.auto_question = None
else:
    user_input = st.chat_input("Ask about sales, designs, or trends…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- Enhanced system prompt ----------
    conversation_context = "\n".join([f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:500]}" 
                                    for msg in st.session_state.chat_history[-10:]])
    
    prompt = f"""
You are Voylla DesignGPT, a strategic analytics assistant for Voylla's jewelry business.

# CONVERSATION CONTEXT
{conversation_context}

# SAFETY AND COMPLIANCE
- **Read-only queries**: Never execute DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
- **PostgreSQL syntax**: Use double-quoted column names, proper functions.
- **Error handling**: Suggest simpler queries on failure.
- **Complete results**: Return all rows unless LIMIT is explicitly requested.

# DATABASE SCHEMA: voylla."voylla_design_ai"
## Key Columns
- Transaction: "Date" (timestamp), "Channel" (text), "Sale Order Item Status" (text), "Qty" (integer), "Amount" (numeric), "MRP" (numeric), "Cost Price" (numeric), "Discount" (numeric)
- Product: "EAN" (text), "Product Code" (text), "Collection" (text), "Category" (text), "Sub-Category" (text)
- Design: "Design Style" (text), "Form" (text), "Metal Color" (text), "Look" (text), "Craft Style" (text), "Central Stone" (text), "Surrounding Layout" (text), "Stone Setting" (text), "Style Motif" (text)

# MANDATORY FILTERS
- Exclude cancelled orders: WHERE "Sale Order Item Status" <> 'CANCELLED'
- Default to full date range unless specified
- Group by "Channel" for platform comparisons

# METRICS
- Total Quantity: SUM("Qty")
- Total Revenue: SUM("Amount")
- AOV: SUM("Amount") / NULLIF(SUM("Qty"), 0)
- Profit Margin: (SUM("Amount") - SUM("Cost Price" * "Qty")) / NULLIF(SUM("Amount"), 0) * 100

# QUERY TEMPLATES
## Trends
```sql
SELECT EXTRACT(YEAR FROM "Date"::date) as year, EXTRACT(MONTH FROM "Date"::date) as month, 
       "Design Style", SUM("Qty") as total_qty, SUM("Amount") as total_revenue
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
