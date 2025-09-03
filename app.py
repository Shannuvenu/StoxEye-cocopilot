import os, time
import streamlit as st
from dotenv import load_dotenv
from finance import fetch_quote, format_change, trend_meter
from research import research
from citations import build_citation_block, build_titled_block

# Optional LLM synthesis
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

load_dotenv()
st.set_page_config(page_title="StoxEye Copilot Web", page_icon="💹")

st.title("💹 StoxEye Copilot — Research Chat")
st.caption("Indian market context • Live quote • Trend meter • News & filings scan • Cited answers • Optional AI synthesis")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Symbol detection (ignore normal words; index aliases) ----------
INDEX_ALIASES = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}
STOPWORDS = {
    "WHAT","IS","THE","A","AN","OF","ON","FOR","IN","ABOUT","TELL","ME","VIEW",
    "AFTER","RESULTS","TODAY","NEAR","TERM","SHORT","LONG","AND","OR","PLEASE",
    "STOCK","MARKET","PRICE","OUTLOOK","UPDATE","NEWS","VS","COMPARE"
}
def probable_symbol(token: str) -> str | None:
    t = token.upper().strip(",.!?:;()[]{}")
    if t in INDEX_ALIASES:
        return INDEX_ALIASES[t]
    if t.isalpha() and 2 <= len(t) <= 6 and t not in STOPWORDS:
        return t
    return None

# ---------- Synthesis ----------
def synthesize_answer(question: str, quote: dict | None, research_data: dict, trend: dict | None) -> str:
    profile = research_data.get("profile") or ""
    news_items = research_data.get("news") or []
    filings = research_data.get("filings") or []
    sources_text = "\n".join([f"- {n.get('title','')} ({n.get('source','')})" for n in (news_items[:3] + filings[:2])])

    if not (OPENAI_OK and os.getenv("OPENAI_API_KEY")):
        price_line = ""
        if quote and quote.get("price") is not None:
            price_line = f"Current {quote['symbol']} price: ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])}"
        trend_line = ""
        if trend and trend.get("label"):
            tm = trend
            trend_line = f"Trend: **{tm['label']}** (RSI14: {tm['rsi14'] and round(tm['rsi14'],1)}, SMA20/50/200: {tm['sma20'] and round(tm['sma20'],1)}/{tm['sma50'] and round(tm['sma50'],1)}/{tm['sma200'] and round(tm['sma200'],1)})"
        bullets = [
            price_line,
            trend_line,
            f"Profile: {profile}" if profile else "",
            "Latest headlines suggest: watch earnings, guidance, and sector moves." if news_items else "No major headlines found just now.",
            "Check corporate announcements for official updates." if filings else "",
            "Note: Research info only, not investment advice."
        ]
        return "\n".join([b for b in bullets if b])

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sys_prompt = (
        "You are StoxEye Copilot, a factual, concise Indian-market research assistant. "
        "Use the provided quote, trend meter, profile and headlines/filings. "
        "Return: 1) Snapshot, 2) What matters now, 3) Risks, 4) One-liner takeaway. "
        "Keep it under 160 words. Do not invent facts. No price targets."
    )
    user_context = f"""
Question: {question}

QUOTE:
{quote}

TREND:
{trend}

PROFILE (Wikipedia):
{profile}

HEADLINES/FILINGS (top):
{sources_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user_context}
        ]
    )
    return resp.choices[0].message.content.strip()

# ---------- Answer flow ----------
def answer(user_text: str):
    tokens = [t.strip() for t in user_text.split()]
    main_symbol = None
    for t in tokens:
        cand = probable_symbol(t)
        if cand:
            main_symbol = cand
            break

    quote = fetch_quote(main_symbol) if main_symbol else None
    tm = trend_meter(main_symbol) if main_symbol else None
    rs = research(main_symbol or user_text)

    body = synthesize_answer(user_text, quote, rs, tm)

    # Citations
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.get("timestamp", time.time())))
    news_block = build_titled_block("News", rs.get("news", []))
    filings_block = build_titled_block("Corporate Announcements (best-effort)", rs.get("filings", []))

    # Render
    if quote and quote.get("price") is not None:
        st.markdown(
            f"**{quote['symbol']}** · ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])}  "
            f"· {quote.get('exchange','')} · {quote.get('currency','')} · _{quote['time']}_"
        )
        if tm and tm.get("label"):
            st.markdown(
                f"**Trend Meter:** {tm['label']}  ·  RSI14: `{tm['rsi14'] and round(tm['rsi14'],1)}`  ·  "
                f"SMA20/50/200: `{tm['sma20'] and round(tm['sma20'],1)}` / `{tm['sma50'] and round(tm['sma50'],1)}` / `{tm['sma200'] and round(tm['sma200'],1)}`  \n"
                f"_{tm.get('note','')}_"
            )
        st.divider()

    st.markdown(body)
    st.divider()
    st.markdown(f"**Sources & Timestamps**  \n_Data scan:_ `{ts}`")
    st.markdown(news_block)
    st.markdown(filings_block)
    st.caption("Research only. Not investment advice.")

# ---------- Chat UI ----------
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask about a stock, index, or sector…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer(prompt)
