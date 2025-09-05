# ============================
# StoxEye Copilot Web — app.py
# ============================

# ---- Standard libs ----
import os, re, time
from typing import List, Dict

# ---- Streamlit must be FIRST call ----
import streamlit as st
if not st.session_state.get("_page_config_set", False):
    st.set_page_config(
        page_title="StoxEye Copilot Web",
        page_icon="💹",
        layout="wide",
    )
    st.session_state["_page_config_set"] = True

# ---- After set_page_config: other imports ----
from dotenv import load_dotenv, find_dotenv, dotenv_values

# Your local modules
from finance import fetch_quote, format_change, trend_meter
from research import research

# ============================
# Secrets / Keys (safe + robust)
# ============================

# Load .env locally (safe on cloud too)
load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

def get_openai_key() -> str:
    """
    Local dev: .env (OPENAI_API_KEY)
    Streamlit Cloud: Secrets (OPENAI_API_KEY)
    Never crash if secrets missing.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        return ""

OPENAI_KEY = get_openai_key()

OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = bool(OPENAI_KEY)
except Exception:
    OPENAI_OK = False


# ============================
# UI — Styles (dark + blue)
# ============================
DARK_CSS = """
<style>
:root {
  --sx-bg: #0b0f14;
  --sx-elev: #111827;
  --sx-card: #0f172a;
  --sx-text: #e5e7eb;
  --sx-sub: #93c5fd;
  --sx-accent: #3b82f6;
  --sx-muted: #64748b;
  --sx-good: #22c55e;
  --sx-bad:  #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--sx-bg);
}

h1,h2,h3,h4,h5,h6, p, span, label, div, code, kbd {
  color: var(--sx-text) !important;
}

.block-container { padding-top: 2.2rem; }

.sx-hero {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.3px;
  display: flex; align-items: center; gap: 0.6rem;
}

.sx-subtitle {
  color: var(--sx-muted) !important;
  font-size: 0.95rem;
  margin-top: 2px;
}

.sx-card {
  background: var(--sx-card);
  border: 1px solid #1f2937;
  border-radius: 14px;
  padding: 16px 18px;
}

.sx-elev {
  background: var(--sx-elev);
  border-radius: 12px;
  padding: 10px 14px;
}

.sx-badge {
  background: rgba(59,130,246,.12);
  color: var(--sx-sub);
  border: 1px solid rgba(59,130,246,.4);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  display: inline-flex; gap: 6px; align-items: center;
}

.sx-ok { color: var(--sx-good) !important; }
.sx-warn { color: #f59e0b !important; }
.sx-bad { color: var(--sx-bad) !important; }

hr { border-color: #1f2937 !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ============================
# Symbol detection
# ============================
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


# ============================
# Extractive fallback summarizer
# ============================
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
        return "Couldn’t extract specifics from the latest coverage. Check earnings, filings and sector cues."
    bullets = "\n".join([f"- {p}" for p in points])
    return f"**Direct answer (extractive):**\n{bullets}\n\n**Takeaway:** Movement likely ties to the factors above; monitor filings/guidance."


# ============================
# Synthesis (LLM if available)
# ============================
def synthesize_answer(question: str, quote: dict | None, rs: dict, trend: dict | None, llm_enabled: bool) -> str:
    # If user turned OFF or no key available — use extractive
    if not (llm_enabled and OPENAI_OK):
        profile = rs.get("profile") or ""
        articles = rs.get("articles") or []
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
        client = OpenAI(api_key=OPENAI_KEY)
        articles = rs.get("articles") or []
        profile = rs.get("profile") or ""

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
        fused = summarize_from_articles(question, rs.get("articles") or [], max_sents=4)
        return f"_LLM unavailable. Using extractive summary._\n\n{fused}"


# ============================
# Header
# ============================
st.markdown(
    """
<div class="sx-hero">💹 StoxEye Copilot — Research Chat</div>
<div class="sx-subtitle">Indian market context • Live quote • Trend meter • Fresh multi-source synthesis • Publisher credits (no links)</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ============================
# Controls (center, minimal)
# ============================
col1, col2, col3 = st.columns([0.9, 0.1, 0.9])
with col1:
    llm_enabled = st.toggle("AI synthesis", value=True, help="Turn OFF to force extractive summary only.")
with col3:
    with st.expander("Scan settings (optional)", expanded=False):
        freshness = st.radio(
            "News window",
            options=[("Today", 1), ("Last 3 days", 3), ("Last 7 days", 7)],
            index=0,
            horizontal=True,
            format_func=lambda x: x[0],
        )[1]
        max_articles = st.slider("Articles to fuse", 1, 8, 5, 1)
        max_chars = st.slider("Chars per article", 400, 2400, 1200, 200)

# Small status badge
status = "ON ✅" if (llm_enabled and OPENAI_OK) else "OFF ❌"
st.markdown(f'<div class="sx-badge">AI synthesis: <b>{status}</b></div>', unsafe_allow_html=True)
if llm_enabled and not OPENAI_OK:
    st.caption("No OPENAI_API_KEY found in .env / Secrets. Using extractive summaries.")

st.write("")


# ============================
# Chat loop
# ============================
if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask about a stock, index, or sector…")
def publishers_used(articles: List[Dict]) -> str:
    pubs: List[str] = []
    for a in articles or []:
        s = (a.get("source") or "").strip()
        if s and s not in pubs:
            pubs.append(s)
    return ", ".join(pubs[:8]) if pubs else "—"

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

    body = synthesize_answer(user_text, quote, rs, tm, llm_enabled)
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.get("timestamp", time.time())))
    pubs_str = publishers_used(rs.get("articles"))

    # Header strip
    header_bits = []
    if quote and quote.get("price") is not None:
        header_bits.append(
            f"**{quote['symbol']}** · ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])} · "
            f"{quote.get('exchange','')} · {quote.get('currency','')} · _{quote['time']}_"
        )
    if tm and tm.get("label"):
        header_bits.append(
            f"**Trend:** {tm['label']} · RSI14: `{tm['rsi14'] and round(tm['rsi14'],1)}` · "
            f"SMA20/50/200: `{tm['sma20'] and round(tm['sma20'],1)}`/"
            f"`{tm['sma50'] and round(tm['sma50'],1)}`/"
            f"`{tm['sma200'] and round(tm['sma200'],1)}`"
        )

    if header_bits:
        st.markdown("  \n".join(header_bits))
        st.divider()

    st.markdown(body)
    st.divider()
    st.markdown(f"**Data scan:** `{ts}`")
    st.markdown(f"**Publishers used:** {pubs_str}")
    st.caption("Research only. Not investment advice. • Built with ❤️ by **venugAAdu**")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer(prompt)
