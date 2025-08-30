#!/usr/bin/env python
# coding: utf-8
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from io import BytesIO
import re
import random
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
MAX_ITERATIONS = 8
QUERY_TIMEOUT = 30
MAX_EXPORT_ROWS = 1000

# ---------- Templates ----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
DEFAULT_TEMPLATES = [
    "Analyzing jewelry trends with precision ✨",
    "Discovering design insights... 💎", 
    "Crafting your analytics report...",
    "Mining design data treasures...",
    "Polishing your sales insights... 💍"
]

def load_spinner_templates():
    """Load spinner templates from file or use defaults"""
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r", encoding="utf-8") as file:
                lines = [line.strip().split('. ', 1)[1] for line in file if '. ' in line]
                return lines if lines else DEFAULT_TEMPLATES
        except Exception:
            return DEFAULT_TEMPLATES
    return DEFAULT_TEMPLATES

# ---------- Database Connection ----------
@st.cache_resource
def get_database_connection():
    """Create cached database connection"""
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Database connection
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"] 
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
        
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        db = SQLDatabase(
            engine,
            include_tables=["voylla_design_ai"],
            schema="voylla"
        )
        
        return db, api_key
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# ---------- LLM Setup ----------
@st.cache_resource
def get_llm():
    """Create cached LLM instance"""
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        request_timeout=QUERY_TIMEOUT
    )

# ---------- Utility Functions ----------
def clean_response(response: str) -> str:
    """Clean and format the agent response"""
    lines = response.split('\n')
    cleaned_lines = []
    skip_line = False
    
    for line in lines:
        if any(marker in line.lower() for marker in ['action:', 'thought:', 'observation:', 'final answer:']):
            if 'final answer:' in line.lower():
                skip_line = False
                continue
            skip_line = True
            continue
        
        if not skip_line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def markdown_to_dataframe(markdown_text: str) -> Optional[pd.DataFrame]:
    """Parse a markdown table into a DataFrame with better error handling"""
    try:
        lines = markdown_text.splitlines()
        header_idx = None
        
        for i in range(len(lines) - 1):
            if '|' in lines[i]:
                sep = lines[i + 1]
                if re.match(r'^[\s\|\-:]+$', sep) and sep.count('-') >= 2:
                    header_idx = i
                    break
        
        if header_idx is None:
            return None
        
        table_lines = [row for row in lines[header_idx:] if '|' in row and not re.match(r'^[\s\|\-:]+$', row)]
        if len(table_lines) < 2:
            return None
        
        normalized = []
        for row in table_lines:
            r = row.strip()
            if not r.startswith('|'):
                r = '|' + r
            if not r.endswith('|'):
                r += '|'
            normalized.append(r)
        
        if len(normalized) < 2:
            return None
            
        header_cols = len(normalized[0].split('|')) - 2
        cleaned_rows = [r for r in normalized if (len(r.split('|')) - 2) == header_cols]
        
        if len(cleaned_rows) < 2:
            return None
        
        from io import StringIO
        df = pd.read_csv(
            StringIO("\n".join(cleaned_rows)),
            sep=r'\s*\|\s*',
            engine='python',
            skipinitialspace=True
        )
        
        df = df.dropna(how='all', axis=1)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        df = df.dropna(how='all')
        
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"Error parsing markdown table: {str(e)}")
        return None

def get_conversation_context() -> str:
    """Build conversation context from chat history"""
    if not st.session_state.chat_history:
        return ""
    
    recent_history = st.session_state.chat_history[-4:]
    
    context_parts = []
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

def is_follow_up_question(user_input: str) -> bool:
    """Detect if this is a follow-up question"""
    follow_up_indicators = [
        "show me", "can you", "what about", "how about", "also", "and", 
        "but", "however", "additionally", "similarly", "more details",
        "expand", "break down", "drill down", "compare"
    ]
    return any(indicator in user_input.lower() for indicator in follow_up_indicators)

# ---------- UI Setup ----------
st.set_page_config(
    page_title="Voylla DesignGPT",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Improved styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fcf1ed 0%, #faf7f5 100%);
    color: #2c2c2c;
    font-weight: 500;
}
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] {
    color: #2c2c2c !important;
}
.success-metric {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.error-message {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- Initialize Session States FIRST ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=6
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_df" not in st.session_state:
    st.session_state.last_df = None

# FIX 1: Use unique keys for each button and prevent multiple clicks
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------- Database and Agent Setup ----------
try:
    db, api_key = get_database_connection()
except Exception as e:
    st.error(f"❌ Connection failed: {str(e)}")
    st.stop()

if "agent_executor" not in st.session_state:
    llm = get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=MAX_ITERATIONS
    )

# ---------- Sidebar WITH KEYBOARD FIXES ----------
with st.sidebar:
    st.header("📊 Connection Status")
    st.success("✅ Connected to voylla_design_ai")
    
    st.header("💡 Quick Insights")
    
    # FIX 2: Use session state to track button clicks without immediate rerun
    # Create columns to prevent button spacing issues
    button_col = st.container()
    
    with button_col:
        # FIX 3: Use unique keys and check if button was clicked
        if st.button("🔥 Top 10 Best Sellers", key="btn_bestsellers", use_container_width=True):
            st.session_state.pending_query = "Show me the top 10 best selling products by quantity in the last 90 days"
            st.session_state.button_clicked = True
        
        if st.button("💰 Revenue Leaders", key="btn_revenue", use_container_width=True):
            st.session_state.pending_query = "What are the top 10 products by revenue this year?"
            st.session_state.button_clicked = True
        
        if st.button("📈 Trending Combinations", key="btn_trending", use_container_width=True):
            st.session_state.pending_query = "Show me trending Form and Metal Color combinations by quantity"
            st.session_state.button_clicked = True
        
        if st.button("🏪 Channel Performance", key="btn_channel", use_container_width=True):
            st.session_state.pending_query = "Compare sales performance across all channels"
            st.session_state.button_clicked = True
    
    st.header("🎯 Sample Questions")
    st.markdown("""
    • Best selling trait combos last 90 days
    • Top 20 SKUs by revenue this month  
    • Trend of Form × Metal Color by qty
    • Which Design Style performs best on Myntra?
    • Success combinations for Wedding/Festive look
    • What's trending in Traditional vs Contemporary?
    • Performance of Oxidized vs Gold finishes
    """)
    
    st.header("🔧 Controls")
    # FIX 4: Use callback pattern for clear button
    if st.button("🗑️ Clear Chat History", key="btn_clear", use_container_width=True):
        # Clear everything but don't rerun immediately
        for key in ['chat_history', 'last_df', 'memory', 'button_clicked', 'pending_query']:
            if key in st.session_state:
                if key == 'chat_history':
                    st.session_state[key] = []
                elif key == 'memory':
                    st.session_state[key].clear()
                else:
                    st.session_state[key] = None
        # Use experimental_rerun instead of rerun for better keyboard handling
        st.rerun()

# ---------- Main Interface ----------
st.title("💬 Voylla DesignGPT")
st.caption("🔍 Ask anything about design traits, sales performance, and success combinations")

# ---------- Display Chat History ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Handle Input with Keyboard Fix ----------
# FIX 5: Process pending queries from buttons first
if st.session_state.button_clicked and st.session_state.pending_query:
    user_input = st.session_state.pending_query
    # Reset the flags
    st.session_state.button_clicked = False
    st.session_state.pending_query = None
else:
    # FIX 6: Use form to prevent premature submission and preserve keyboard
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask about sales trends, design performance, or success combinations...",
            key="chat_input",
            placeholder="Type your question here..."
        )
        
        # FIX 7: Create two columns for submit options
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
        with col2:
            # This helps maintain focus
            st.caption("Press Enter or click Send to ask your question")
        
        # Only process if form was submitted and there's input
        if not submitted or not user_input.strip():
            user_input = None

# ---------- Process Input with Better State Management ----------
if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Build enhanced prompt with context
    conversation_context = get_conversation_context()
    is_follow_up = is_follow_up_question(user_input)
    
    enhanced_prompt = f"""
You are Voylla DesignGPT, an expert SQL analytics assistant for jewelry sales data.

# CONVERSATION CONTEXT
{"This is a follow-up question. " if is_follow_up else ""}Recent conversation:
{conversation_context}

# CRITICAL SAFETY RULES
- **ONLY** read-only SELECT queries allowed
- **NEVER** use: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE
- Always exclude cancelled orders: "Sale Order Item Status" <> 'CANCELLED'
- **Always** double-quote column names in PostgreSQL

# SCHEMA: voylla."voylla_design_ai"
Key columns:
- Identifiers: "EAN", "Product Code", "Collection"  
- Sales: "Date", "Channel", "Qty", "Amount", "MRP"
- Design: "Category", "Sub-Category", "Look", "Design Style", "Form", "Metal Color"
- Craft: "Craft Style", "Central Stone", "Stone Setting", "Style Motif"

# QUERY STRATEGY
1. Keep queries SIMPLE and focused
2. Use LIMIT 100 for large result sets unless user asks for more
3. For trends: GROUP BY month/quarter + design attributes
4. For combinations: GROUP BY 2-3 key traits maximum
5. Always ORDER BY key metrics (Qty, Amount) DESC

# RESPONSE FORMAT
- Provide brief context about what you found
- Return results as a clean markdown table
- If query fails, suggest a simpler alternative
- For follow-ups, reference previous analysis

# USER QUESTION
{user_input}
"""
    
    # Execute with better error handling
    templates = load_spinner_templates()
    spinner_text = random.choice(templates)
    
    # FIX 8: Use empty container to preserve layout during processing
    response_container = st.empty()
    
    with st.spinner(spinner_text):
        try:
            with st.status("Processing query...", expanded=False) as status:
                status.write("🔍 Analyzing your question...")
                
                response = st.session_state.agent_executor.run(enhanced_prompt)
                status.write("✅ Query completed successfully")
                response = clean_response(response)
                
        except ValueError as e:
            error_msg = str(e)
            if "Could not parse LLM output:" in error_msg:
                response = error_msg.split("Could not parse LLM output:")[-1].strip()
            else:
                response = f"I encountered a parsing error. Let me try a simpler approach to your question: {error_msg}"
        
        except Exception as e:
            error_details = str(e)
            logger.error(f"Agent execution error: {error_details}")
            
            if "iteration limit" in error_details.lower():
                response = """I apologize - your query was quite complex and hit our processing limit. 
                
Let me suggest a simpler approach:
- Try asking for specific time periods (e.g., "last 30 days")
- Focus on one main design attribute at a time
- Ask for "top 10" or "top 20" results initially

Would you like me to try a simplified version of your question?"""
            
            elif "timeout" in error_details.lower():
                response = "The query took too long to process. Try asking for a smaller date range or more specific filters."
            
            else:
                response = f"I encountered an issue processing your request. Could you try rephrasing your question or being more specific? Error: {error_details}"
    
    # Add assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # FIX 9: Display response in a way that doesn't trigger unnecessary reruns
    with response_container.container():
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # Try to extract and display any data table
            df_result = markdown_to_dataframe(response)
            if df_result is not None and not df_result.empty:
                st.session_state.last_df = df_result
                
                # Show quick stats
                if len(df_result) > 5:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Results Found", len(df_result))
                    with col2:
                        if 'Amount' in df_result.columns:
                            total_revenue = df_result['Amount'].sum() if pd.api.types.is_numeric_dtype(df_result['Amount']) else "N/A"
                            st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}" if total_revenue != "N/A" else "N/A")
                    with col3:
                        if 'Qty' in df_result.columns:
                            total_qty = df_result['Qty'].sum() if pd.api.types.is_numeric_dtype(df_result['Qty']) else "N/A"
                            st.metric("📦 Total Quantity", f"{total_qty:,.0f}" if total_qty != "N/A" else "N/A")

# ---------- Download Feature with Keyboard Preservation ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    # FIX 10: Separate download section to prevent interference
    st.markdown("---")
    download_col1, download_col2 = st.columns([1, 4])
    
    with download_col1:
        export_df = st.session_state.last_df.iloc[:MAX_EXPORT_ROWS].copy()
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Design_Insights')
            
            rows_exported = len(export_df)
            total_rows = len(st.session_state.last_df)
            
            st.download_button(
                f"📥 Download Excel ({rows_exported} rows)",
                data=output.getvalue(),
                file_name=f"voylla_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=f"Downloads first {MAX_EXPORT_ROWS} rows. Total results: {total_rows}",
                key="download_btn"  # FIX 11: Unique key for download button
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    with download_col2:
        if len(st.session_state.last_df) > MAX_EXPORT_ROWS:
            st.info(f"📋 Showing data preview. Full dataset has {len(st.session_state.last_df)} rows.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
💎 Voylla DesignGPT • Powered by Advanced Analytics • 
<span style='color: #d4af37;'>Crafting Insights from Design Data</span>
</div>
""", unsafe_allow_html=True)
