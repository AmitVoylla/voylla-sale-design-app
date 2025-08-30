#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory  # Changed to WindowMemory
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from io import BytesIO
import re
import random

# ---------- Optional: spinner copy -----------
TEMPLATES_FILE = "voylla_about_templates_attractive.txt"
if os.path.exists(TEMPLATES_FILE):
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as file:
        lines = [line.strip().split('. ', 1)[1] for line in file if '. ' in line]
else:
    lines = [
        "Crunching the numbers with a sparkle ✨",
        "Polishing your insights… 💎",
        "Setting the stones in your report…",
    ]

random_template = random.choice(lines)

# ---------- Secrets / API ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 No OpenAI key found – please add it in your app's Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)  # Better model for context

# ---------- DB CONNECTION ----------
db_host = st.secrets["DB_HOST"]
db_port = st.secrets["DB_PORT"]
db_name = st.secrets["DB_NAME"]
db_user = st.secrets["DB_USER"]
db_password = st.secrets["DB_PASSWORD"]

engine = create_engine(
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

db = SQLDatabase(
    engine,
    include_tables=["voylla_design_ai"],
    schema="voylla"
)

# ---------- Helpers ----------
def markdown_to_dataframe(markdown_text: str):
    """Parse a markdown table into a DataFrame."""
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
    
    table_lines = [row for row in lines[header_idx:] if '|' in row]
    if len(table_lines) < 2:
        return None
    
    # normalize pipes
    normalized = []
    for row in table_lines:
        r = row.strip()
        if not r.startswith('|'):
            r = '|' + r
        if not r.endswith('|'):
            r += '|'
        normalized.append(r)
    
    # consistent col count
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
        return df
    except Exception:
        return None

def get_conversation_context():
    """Build conversation context from chat history for better follow-ups."""
    if not st.session_state.chat_history:
        return ""
    
    # Get last 6 exchanges (3 user + 3 assistant) for context
    recent_history = st.session_state.chat_history[-6:]
    
    context_parts = []
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500]  # Truncate long responses
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

# ---------- UI ----------
st.set_page_config(
    page_title="Voylla DesignGPT",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Light theme + particles (optional eye candy)
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 Connected")
    st.success("Design table: voylla_design_ai")
    
    st.header("💡 Sample Questions")
    st.write("• Best selling trait combos last 90 days")
    st.write("• Top 20 SKUs by revenue this month")
    st.write("• Trend of Form × Metal Color by qty")
    st.write("• Which Design Style performs best on Myntra?")
    st.write("• Success combinations for Wedding/Festive look")
    
    # Add context controls
    st.header("🔧 Context Settings")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

st.title("💬 Voylla DesignGPT")
st.caption("Ask anything about design traits, sales performance, and success combinations.")

# ---------- Memory / Agent (Improved) ----------
if "memory" not in st.session_state:
    # Use WindowMemory to keep recent conversations
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10  # Keep last 10 exchanges
    )

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory,
        max_iterations=5  # Prevent infinite loops
        # ,early_stopping_method="generate"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_df" not in st.session_state:
    st.session_state.last_df = None

if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# ---------- Render history ----------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Input ----------
user_input = st.chat_input("Ask about sales or design trends…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------- IMPROVED SYSTEM PROMPT WITH CONTEXT ----------
    conversation_context = get_conversation_context()
    
    # Check if this is a follow-up question
    follow_up_indicators = ["show me", "can you", "what about", "how about", "also", "and", "but", "however", "additionally", "similarly"]
    is_follow_up = any(indicator in user_input.lower() for indicator in follow_up_indicators) or len(st.session_state.chat_history) > 1

    prompt = f"""
You are Voylla DesignGPT, an expert SQL/analytics assistant for Voylla jewelry data.

# CONVERSATION CONTEXT
{"This appears to be a follow-up question. " if is_follow_up else ""}Here's our recent conversation:
{conversation_context}

# SAFETY / GUARANTEES
- **Never** run mutating queries. Disallow: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
- Read-only analytical SELECTs only.
- Return complete tables (no truncation) unless the user explicitly asks for a LIMIT.
- Use **PostgreSQL** syntax and always **double-quote** column names.

# DATABASE SCHEMA
Table: voylla."voylla_design_ai"

## COLUMN DEFINITIONS & DATA TYPES

### IDENTIFIERS & METADATA
- **"EAN"** (text) — Product barcode/identifier (e.g., 8.90512E+12)
- **"Product Code"** (text) — Voylla internal SKU (e.g., VMJAI41936)
- **"Collection"** (text) — Product collection name (e.g., "Indigo Affair")
- **"Image Url"** (text) — Product image URL

### TRANSACTION DATA
- **"Date"** (timestamp) — Order/transaction date (format: DD/MM/YYYY HH:MM)
- **"Channel"** (text) — Sales channel (e.g., "Cloudtail-VRP", "FLIPKART_MH", "MYNTRA SOR", "NYKAA_FASHION_53GBAPL")
- **"Type"** (text) — Transaction type (appears to be "Online" for e-commerce)
- **"Sale Order Item Status"** (text) — Order status (DISPATCHED, DELIVERED, CANCELLED, etc.)

### PRODUCT CLASSIFICATION
- **"Category"** (text) — Main product category (e.g., "Earrings", "Necklaces", "Rings")
- **"Sub-Category"** (text) — Product subcategory (e.g., "Studs", "Hoops", "Chandbali")
- **"Look"** (text) — Style occasion/vibe (e.g., "Oxidized", "Everyday", "Festive", "Party", "Wedding")

### DESIGN ATTRIBUTES
- **"Design Style"** (text) — Overall design aesthetic (e.g., "Tribal", "Contemporary", "Traditional/Ethnic", "Minimalist")
- **"Form"** (text) — Physical shape/structure (e.g., "Triangle", "Stud", "Hoop", "Jhumka", "Ear Cuff")
- **"Metal Color"** (text) — Metal finish (e.g., "Antique Silver", "Yellow Gold", "Rose Gold", "Silver", "Antique Gold", "Oxidized Black")
- **"Craft Style"** (text) — Manufacturing/craft technique (e.g., "Handcrafted", may include up to 2 labels joined by ' | ')
- **"Central Stone"** (text) — Primary gemstone/material (can be empty)
- **"Surrounding Layout"** (text) — Stone arrangement pattern (can be empty)
- **"Stone Setting"** (text) — How stones are mounted (e.g., "Enamel Panel", "Prong Setting", "Mixed")
- **"Style Motif"** (text) — Design theme/pattern (e.g., "Geometric", "Floral", "Abstract")

### FINANCIAL DATA
- **"Qty"** (integer) — Quantity sold in this transaction
- **"Amount"** (numeric) — Total revenue for this line item (Qty × selling price)
- **"MRP"** (numeric) — Maximum Retail Price per unit
- **"Cost Price"** (numeric) — Cost per unit to Voylla
- **"Discount"** (numeric) — Discount amount or rate applied

### SYSTEM DATA
- **"Last_updated_at"** (timestamp) — Record last modification time

# GLOBAL FILTERS (APPLY BY DEFAULT)
- Exclude cancelled: "Sale Order Item Status" <> 'CANCELLED'
- If user asks for "online" vs "offline", infer using "Channel". Otherwise, include all channels.
- If date range unspecified, use **all available dates**.

# METRICS DEFINITIONS
- **Quantity sold**: SUM("Qty")
- **Revenue**: SUM("Amount") 
- **AOV** (if asked): SUM("Amount") / NULLIF(SUM("Qty"),0)
- **Discount rate** (if asked): average of "Discount" or ratio as described by user — ask back if ambiguous
- **Success combination** = any combination of traits that maximizes **Qty** and/or **Amount**

**Primary traits for analysis**: ("Design Style","Form","Metal Color","Craft Style","Central Stone","Surrounding Layout","Stone Setting","Style Motif","Look")

# FOLLOW-UP HANDLING
- If the user says "show me more details" or similar, expand on the previous analysis
- If they ask for "top 10" after showing top 5, modify the previous query  
- If they want to "filter by X" or "what about Y", apply additional filters to previous context
- If they ask comparative questions like "what about gold vs silver", create comparisons
- Reference previous results when relevant: "Based on the previous analysis showing..."

# QUERY PATTERNS
- For **trending designs**: GROUP BY date bucket (e.g., month) and Design Style
- For **success combinations**: GROUP BY 3–5 traits (avoid overly wide groups); order by SUM("Qty") DESC then SUM("Amount") DESC
- For **channel breakdown**: include "Channel" in SELECT/GROUP BY
- For **top SKUs**: group or filter by "Product Code"
- Always **quote** column names and fully qualify the table as voylla."voylla_design_ai"

# OUTPUT RULES
- Return results as a markdown table with **all** rows (unless user asks a LIMIT)
- If the user asks for a chart, return a small aggregated table that can be easily charted by Streamlit later
- If a question is ambiguous, make a **reasonable assumption** and state it briefly above the table
- For follow-up questions, acknowledge the previous context: "Building on the previous analysis..." or "Expanding on those results..."

# For Non-Database Questions:
- Respond naturally and helpfully
- Handle greetings, casual chat, and general knowledge  
- Be conversational and friendly

# CURRENT USER QUESTION
{user_input}
"""

    # ---------- Execute ----------
    with st.spinner(random_template.strip()):
        try:
            response = st.session_state.agent_executor.run(prompt)
            
            # Store the result for potential follow-ups
            st.session_state.last_query_result = response
            
        except ValueError as e:
            raw = str(e)
            response = raw.split("Could not parse LLM output:")[-1].strip(" ") if "Could not parse LLM output:" in raw else raw
        except Exception as e:
            response = f"I apologize, but I encountered an error: {str(e)}. Could you please rephrase your question?"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)

    # try to parse result table for download
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res

# # ---------- Download button ----------
# if st.session_state.last_df is not None and not st.session_state.last_df.empty:
#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='openpyxl') as writer:
#         st.session_state.last_df.to_excel(writer, index=False)
    
#     st.download_button(
#         "📥 Download Excel",
#         data=output.getvalue(),
#         file_name="design_insights.xlsx",
#         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         key="download_button_design"
#     )
# ---------- Download button ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    MAX_EXPORT_ROWS = 500  # <-- set whatever you want
    export_df = st.session_state.last_df.iloc[:MAX_EXPORT_ROWS].copy()

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False)
    
    st.download_button(
        "📥 Download Excel",
        data=output.getvalue(),
        file_name="design_insights.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_button_design"
    )
