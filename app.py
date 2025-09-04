import os, re, time
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv, find_dotenv, dotenv_values

from finance import fetch_quote, format_change, trend_meter
from research import research

# ---------------- OpenAI availability flag ----------------
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# ---------------- Load .env (robust) ----------------
DOTENV_PATH = find_dotenv(filename=".env", usecwd=True)
_ = load_dotenv(DOTENV_PATH, override=True)
HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))

# ---------------- Page must be configured FIRST ----------------
st.set_page_config(page_title="StoxEye Copilot Web", page_icon="💹")
st.title("💹 StoxEye Copilot — Research Chat")
st.caption("Indian market context • Live quote • Trend meter • Fresh multi-source synthesis • Publisher credits (no links)")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Scan Settings")
    freshness = st.radio(
        "News window",
        options=[("Today", 1), ("Last 3 days", 3), ("Last 7 days", 7)],
        index=0,
        format_func=lambda opt: opt[0],
    )[1]
    max_articles = st.slider("Articles to fuse", 1, 8, 5, 1)
    max_chars = st.slider("Chars per article", 400, 2400, 1200, 200)

    # Status
    st.info(f"AI synthesis: {'ON ✅' if (OPENAI_OK and HAS_KEY) else 'OFF ❌'}")
    if not HAS_KEY:
        st.warning("No OPENAI_API_KEY found in .env. Using extractive summaries.", icon="⚠️")

    st.divider()
    st.caption("Developed by **venugAAdu**")

# ---------------- Symbol detection ----------------
INDEX_ALIASES = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK"}
STOPWORDS = {
    "WHAT","WHY","HOW","IS","THE","A","AN","OF","ON","FOR","IN","ABOUT","TELL","ME","VIEW",
    "AFTER","RESULTS","TODAY","NEAR","TERM","SHORT","LONG","AND","OR","PLEASE",
    "STOCK","MARKET","PRICE","OUTLOOK","UPDATE","NEWS","VS","COMPARE","HOLD","HOLDING"
}

def probable_symbol(token: str) -> str | None:
    t = token.upper().strip(",.!?:;()[]{}")
    if t in INDEX_ALIASES:
        return INDEX_ALIASES[t]
    if all(ch.isalnum() or ch == "." for ch in t) and 2 <= len(t) <= 10 and t not in STOPWORDS:
        return t
    return None

def extract_symbol_from_text(text: str) -> str | None:
    for raw in text.split():
        cand = probable_symbol(raw)
        if cand:
            return cand
    return None

# ---------------- Extractive fallback summarizer ----------------
KEY_POS = {"rise","rises","up","gain","gains","surge","higher","jump","beats","profit","upgrade","buy","bullish","strong"}
KEY_NEG = {"fall","falls","down","drop","drops","lower","plunge","miss","loss","downgrade","sell","bearish","weak"}

def _sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+|•|\u2022|\u25CF", text)
    return [s.strip() for s in parts if len(s.strip()) > 30]

def _score_sentence(s: str) -> int:
    t = s.lower()
    pos = sum(1 for k in KEY_POS if k in t)
    neg = sum(1 for k in KEY_NEG if k in t)
    nums = 1 if re.search(r"\d+(\.\d+)?%|\d{2,}", s) else 0
    return pos + neg + nums

def summarize_from_articles(question: str, articles: List[Dict], max_sents: int = 4) -> str:
    corpus: List[str] = []
    for a in articles:
        title = a.get("title", "")
        if title:
            corpus.append(title)
        snip = a.get("snippet", "") or ""
        corpus.extend(_sentences(snip)[:4])
    ranked = sorted(set(corpus), key=_score_sentence, reverse=True)
    points = ranked[:max_sents]
    if not points:
        return "Couldn’t extract specifics from today’s coverage. Check earnings, filings and sector cues."
    bullets = "\n".join([f"- {p}" for p in points])
    return f"**Direct answer (extractive):**\n{bullets}\n\n**Takeaway:** Movement likely ties to the factors above; monitor filings/guidance."

# ---------------- Synthesis (LLM if available, else extractive) ----------------
def synthesize_answer(question: str, quote: dict | None, rs: dict, trend: dict | None) -> str:
    profile = rs.get("profile") or ""
    articles = rs.get("articles") or []

    # Fallback path
    if not (OPENAI_OK and HAS_KEY):
        price_line = ""
        if quote and quote.get("price") is not None:
            price_line = f"Snapshot: {quote['symbol']} ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])}"
        trend_line = ""
        if trend and trend.get("label"):
            tm = trend
            rsi = tm["rsi14"]; s20, s50, s200 = tm["sma20"], tm["sma50"], tm["sma200"]
            trend_line = f"Trend: {tm['label']} (RSI14: {rsi and round(rsi,1)}, SMA20/50/200: {s20 and round(s20,1)}/{s50 and round(s50,1)}/{s200 and round(s200,1)})"
        fused = summarize_from_articles(question, articles, max_sents=4)
        parts = [price_line, trend_line, fused, "Note: Research info only; not investment advice."]
        return "\n\n".join([p for p in parts if p])

    # LLM path
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        evidence_chunks = []
        for a in articles[:5]:
            src = a.get("source", "")
            title = a.get("title", "")
            snip = (a.get("snippet", "") or "")[:900]
            evidence_chunks.append(f"[{src}] {title}\n{snip}")
        evidence_text = "\n\n---\n".join(evidence_chunks)

        sys_prompt = (
            "You are StoxEye Copilot, an Indian-markets research assistant. "
            "Answer the user's EXACT question by synthesizing the provided multi-source evidence. "
            "Be concise and factual. Structure: 1) Direct answer, 2) Key evidence (cite publishers by name only), "
            "3) Risks/unknowns, 4) One-liner takeaway. Do NOT invent facts or price targets."
        )
        user_context = f"""
USER_QUESTION:
{question}

QUOTE_SNAPSHOT:
{quote}

TREND_METER:
{trend}

PROFILE:
{profile}

EVIDENCE_SNIPPETS (publisher, title, excerpt):
{evidence_text}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_context},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        fused = summarize_from_articles(question, articles, max_sents=4)
        return f"_LLM unavailable. Using extractive summary._\n\n{fused}"

# ---------------- Answer flow ----------------
def answer(user_text: str):
    symbol = extract_symbol_from_text(user_text)

    quote = fetch_quote(symbol) if symbol else None
    tm = trend_meter(symbol) if symbol else None
    rs = research(
        user_text,
        days=freshness,
        prefer_symbol=symbol,
        max_articles=max_articles,
        max_chars=max_chars,
    )

    # Publishers list (names only, deduped)
    pubs: List[str] = []
    for a in rs.get("articles", []):
        s = (a.get("source") or "").strip()
        if s and s not in pubs:
            pubs.append(s)
    publishers_str = ", ".join(pubs[:8]) if pubs else "—"

    body = synthesize_answer(user_text, quote, rs, tm)
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.get("timestamp", time.time())))

    # Header (quote + trend)
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
    st.markdown(f"**Data scan:** `{ts}`")
    st.markdown(f"**Publishers used:** {publishers_str}")
    st.caption("Research only. Not investment advice. • Built with ❤️ by **venugAAdu**")

# ---------------- Chat UI ----------------
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
