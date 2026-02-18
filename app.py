import os
import datetime as dt
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Union, Dict, Annotated, TypedDict
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# Technical Analysis Imports
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD
from ta.volume import volume_weighted_average_price

from fpdf import FPDF
import io

# Load .env if you have one
load_dotenv()

# --- 1. PROMPT DEFINITION ---
FUNDAMENTAL_ANALYST_PROMPT = """
You are a fundamental analyst specializing in evaluating company (whose symbol is {company}) performance based on stock prices, technical indicators, and financial metrics. Your task is to provide a comprehensive summary of the fundamental analysis for a given stock.

You have access to the following tools:
1. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like RSI, MACD, Drawdown and VWAP.
2. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.

### Your Task:
1. **Input Stock Symbol**: Use the provided stock symbol to query the tools and gather the relevant information.
2. **Analyze Data**: Evaluate the results from the tools and identify potential resistance, key trends, strengths, or concerns.
3. **Provide Summary**: Write a concise, well-structured summary that highlights:
    - Recent stock price movements, trends and potential resistance.
    - Key insights from technical indicators (e.g., whether the stock is overbought or oversold).
    - Financial health and performance based on financial metrics.

### Constraints:
- Use only the data provided by the tools.
- Avoid speculative language; focus on observable data and trends.
- If any tool fails to provide data, clearly state that in your summary.

### Output Format:
RRespond in the following format, using clear headings and double line breaks between sections:

**Stock**: <Stock Symbol>

**Price Analysis**: <Detailed analysis of stock price trends>

**Technical Analysis**: <Detailed time series Analysis from ALL technical indicators>

**Financial Analysis**: <Detailed analysis from financial metrics>

**Final Summary**: <Full Conclusion based on the above analyses>

**Asked Question Answer**: <Answer based on the details and analysis above>

Ensure that your response is objective, concise, and actionable.
"""


def generate_pdf(text, ticker):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Financial Analysis Report: {ticker}", ln=True, align='C')
    pdf.ln(10) # Line break
    
    # Body Text
    pdf.set_font("Arial", size=10)
    # This handles special characters and multi-line wrapping
    pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    
    # Return as bytes for Streamlit
    return pdf.output(dest='S').encode('latin-1')


# --- 2. TOOL DEFINITIONS ---
@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicators."""
    try:
        # Download data (72 weeks for accurate indicators)
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=72),
            end=dt.datetime.now(),
            interval='1d'
        )
        if data.empty:
            return f"No data found for {ticker}."
            
        df = data.copy()
        # Clean multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        indicators = {}
        # Calculations
        rsi_series = RSIIndicator(df['Close'], window=14).rsi().tail(12)
        indicators["RSI"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in rsi_series.dropna().to_dict().items()}
        
        macd_obj = MACD(df['Close'])
        macd_val = macd_obj.macd().tail(12)
        indicators["MACD"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in macd_val.to_dict().items()}

        vwap_val = volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume']).tail(12)
        indicators["VWAP"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in vwap_val.to_dict().items()}

        # RETURN ONLY LAST 15 DAYS to stay under Groq 12k Token Limit
        recent_prices = df.tail(15).reset_index()
        recent_prices['Date'] = recent_prices['Date'].dt.strftime('%Y-%m-%d')
        
        return {
            'recent_stock_prices': recent_prices.to_dict(orient='records'), 
            'indicators': indicators,
            'summary': f"Current price: {df['Close'].iloc[-1]:.2f}"
        }
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins')
        }
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. GRAPH SETUP ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    stock: str

tools = [get_stock_prices, get_financial_metrics]

def create_app(api_key: str):
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=api_key)
    llm_with_tool = llm.bind_tools(tools)

    def fundamental_analyst(state: State):
        sys_msg = SystemMessage(content=FUNDAMENTAL_ANALYST_PROMPT.format(company=state['stock']))
        messages = [sys_msg] + state['messages']
        return {'messages': [llm_with_tool.invoke(messages)]}

    builder = StateGraph(State)
    builder.add_node('analyst', fundamental_analyst)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'analyst')
    builder.add_conditional_edges('analyst', tools_condition)
    builder.add_edge('tools', 'analyst')
    return builder.compile()

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI Stock Analyst", page_icon="📈")
st.title("🤖 AI Fundamental Analyst")
st.caption("Using LangGraph + Groq Llama 3.3")

secret_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("Configuration")
    if secret_key:
        # If the key is already in Secrets, don't make the user type it
        st.success("✅ API Key loaded from Secrets")
        api_key = secret_key
    else:
        # If no secret is found, show the input box
        api_key = st.text_input("Enter Groq API Key", type="password")
        st.info("Get your key at [console.groq.com](https://console.groq.com)")

# --- Side-by-Side Layout from your Screenshot ---
col1, col2 = st.columns([4, 1]) 

with col1:
    raw_ticker = st.text_input("Enter Ticker", value="ONGC").upper().strip()

with col2:
    # Use HTML to align the checkbox perfectly with the text box
    st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
    is_indian = st.checkbox("Indian (.NS)", value=True)

# 1. Apply the .NS Logic
ticker = raw_ticker
if is_indian and not raw_ticker.endswith(".NS"):
    ticker = f"{raw_ticker}.NS"

# 2. Status Caption (Matches your screenshot)
st.caption(f"Currently searching NSE India: **{ticker}**")

# 3. Question Box & Button
question = st.text_area("What would you like to know?", value="Should I buy this stock?")
analyze_btn = st.button("Analyze Stock")

if st.button("Analyze Stock"):
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
    else:
        try:
            with st.spinner("Analyzing..."):
                app = create_app(api_key)
                result = app.invoke({"messages": [HumanMessage(content=question)], "stock": ticker})
                report_text = result["messages"][-1].content
                
                # 1. Display on Screen
                st.subheader(f"Analysis Report for {ticker}")
                st.markdown(report_text)
                
                # 2. Create the Download Button
                pdf_bytes = generate_pdf(report_text, ticker)
                st.download_button(
                    label="📩 Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{ticker}_Analysis_{dt.date.today()}.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")


