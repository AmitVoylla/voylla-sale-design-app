#!/usr/bin/env python
# coding: utf-8
# S1L1lO6I3O65Ta
# In[ ]:

#!/usr/bin/env python
# coding: utf-8
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
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
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ---------- DB CONNECTION ----------
# keep your RDS string or use your env var
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
.stApp { background-color: #fcf1ed; color: #000; font-weight: 500 }
.stChatMessage .element-container div[data-testid="stMarkdownContainer"] { color:#000 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 Connected")
    st.success("Design table: `voylla_design_ai`")
    st.header("💡 Sample Questions")
    st.write("• Best selling trait combos last 90 days")
    st.write("• Top 20 SKUs by revenue this month")
    st.write("• Trend of Form × Metal Color by qty")
    st.write("• Which Design Style performs best on Myntra?")
    st.write("• Success combinations for Wedding/Festive look")

st.title("💬 Voylla DesignGPT")
st.caption("Ask anything about design traits, sales performance, and success combinations.")

# ---------- Memory / Agent ----------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "agent_executor" not in st.session_state:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    st.session_state.agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.memory
    )

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_df" not in st.session_state: st.session_state.last_df = None

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

    # ---------- SYSTEM PROMPT FOR THIS APP ----------
    prompt = f"""
You are Voylla DesignGPT, an expert SQL/analytics assistant for Voylla.

# SAFETY / GUARANTEES
- **Never** run mutating queries. Disallow: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
- Read-only analytical SELECTs only.
- Return complete tables (no truncation) unless the user explicitly asks for a LIMIT.
- Use **PostgreSQL** syntax and always **double-quote** column names.

# DATABASE
Only table: voylla."voylla_design_ai"

Available columns (quoted names):
- "EAN" (text/number)
- "Date" (timestamp) — use for time filters
- "Channel" (text)
- "Type" (text)
- "Product Code" (text)
- "Collection" (text)
- "Image Url" (text)
- "Category" (text)
- "Sub-Category" (text)
- "Look" (text)  -- e.g., Everyday, Festive, Party, Wedding (standardised)
- "Discount" (numeric) -- selling price after discount or just discount amount; don't assume margin
- "Qty" (integer)
- "Amount" (numeric) -- revenue for the line
- "MRP" (numeric)
- "Cost Price" (numeric)
- "Sale Order Item Status" (text)
- "Design Style" (text) -- e.g., Contemporary, Traditional/Ethnic...
- "Form" (text) -- e.g., Stud, Hoop, Jhumka, Ear Cuff...
- "Metal Color" (text) -- Yellow Gold, Rose Gold, Silver, Antique Silver, Antique Gold, Oxidized Black
- "Craft Style" (text) -- up to 2 labels joined by ' | ' from a fixed vocab
- "Central Stone" (text)
- "Surrounding Layout" (text)
- "Stone Setting" (text)
- "Style Motif" (text)
- "Last_updated_at" (timestamp)

# GLOBAL FILTERS (APPLY BY DEFAULT)
- Exclude cancelled: "Sale Order Item Status" <> 'CANCELLED'
- If user asks for "online" vs "offline", infer using "Channel". Otherwise, include all channels.
- If date range unspecified, use **all available dates**.

# METRICS DEFINITIONS
- Quantity sold: SUM("Qty")
- Revenue: SUM("Amount")
- AOV (if asked): SUM("Amount") / NULLIF(SUM("Qty"),0)
- Discount rate (if asked): average of "Discount" or ratio as described by user — ask back if ambiguous.
- Success combination = any combination of traits that maximizes **Qty** and/or **Amount**.
  Typical trait set to analyze:
  ("Design Style","Form","Metal Color","Craft Style","Central Stone","Surrounding Layout","Stone Setting","Style Motif","Look")

# QUERY PATTERNS
- For **trending designs**: GROUP BY date bucket (e.g., month) and Design Style.
- For **success combinations**: GROUP BY 3–5 traits (avoid overly wide groups); order by SUM("Qty") DESC then SUM("Amount") DESC.
- For **channel breakdown**: include "Channel" in SELECT/GROUP BY.
- For **top SKUs**: group or filter by "Product Code".
- Always **quote** column names and fully qualify the table as voylla."voylla_design_ai".

# OUTPUT RULES
- Return results as a markdown table with **all** rows (unless user asks a LIMIT).
- If the user asks for a chart, return a small aggregated table that can be easily charted by Streamlit later.
- If a question is ambiguous, make a **reasonable assumption** and state it briefly above the table.

# For Non-Database Questions:
- Respond naturally and helpfully
- Handle greetings, casual chat, and general knowledge
- Be conversational and friendly

# USER QUESTION
{user_input}
"""

    # ---------- Execute ----------
    with st.spinner(random_template.strip()):
        try:
            response = st.session_state.agent_executor.invoke({"input": user_input})["output"]
        except ValueError as e:
            raw = str(e)
            response = raw.split("Could not parse LLM output:")[-1].strip(" `") if "Could not parse LLM output:" in raw else raw

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # try to parse result table for download
    df_res = markdown_to_dataframe(response)
    if df_res is not None and not df_res.empty:
        st.session_state.last_df = df_res

# ---------- Download button ----------
if st.session_state.last_df is not None and not st.session_state.last_df.empty:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.last_df.to_excel(writer, index=False)
    st.download_button(
        "📥 Download Excel",
        data=output.getvalue(),
        file_name="design_insights.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_button_design"
    )
