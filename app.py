# ============================
# StoxEye Copilot — app.py
# ============================

# ---- Standard libs ----
import os, re, time
from typing import List, Dict

# ---- Streamlit must be FIRST call ----
import streamlit as st
if not st.session_state.get("_page_config_set", False):
    st.set_page_config(
        page_title="StoxEye Copilot",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.session_state["_page_config_set"] = True

# ---- After set_page_config: other imports ----
from dotenv import load_dotenv, find_dotenv
# Your local modules
from finance import fetch_quote, format_change, trend_meter
from research import research

# ============================
# Secrets / Keys (safe + robust)
# ============================
load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

def get_openai_key() -> str:
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
# Theme / CSS
# ============================
CSS = """
<style>
:root {
  --sx-bg: #070B11;
  --sx-surface: #0C121A;
  --sx-card: rgba(17, 24, 39, .7);
  --sx-stroke: #1f2937;
  --sx-text: #E6EAF2;
  --sx-dim: #97A3B6;
  --sx-blue: #3B82F6;
  --sx-blue-2: #60A5FA;
  --sx-green: #22c55e;
  --sx-red: #ef4444;
  --sx-amber: #f59e0b;
}
html, body, [data-testid="stAppViewContainer"] { background: radial-gradient(1200px 1200px at 10% -10%, rgba(59,130,246,.12), transparent 40%), var(--sx-bg); }
.block-container { padding-top: 1.1rem; max-width: 1150px; }
* { color: var(--sx-text); }
.sx-hero { font-size: 2.2rem; font-weight: 900; display:flex; align-items:center; gap:.6rem; }
.sx-sub { color: var(--sx-dim); margin-top:.25rem; }
.sx-divider { height:1px; background: linear-gradient(90deg, transparent, #1f2937, transparent); margin:.75rem 0 1rem; }
.sx-badge { background:rgba(59,130,246,.15); border:1px solid rgba(59,130,246,.35); padding:.35rem .6rem; border-radius:999px; font-size:.8rem; color:#b9d6ff; display:inline-flex; gap:.4rem; align-items:center; }
.sx-card { background: var(--sx-card); backdrop-filter: blur(10px); border: 1px solid var(--sx-stroke); border-radius: 16px; padding: 16px 18px; box-shadow: 0 8px 30px rgba(0,0,0,.25); }
a { color: var(--sx-blue-2); text-decoration: none; }
a:hover { opacity:.9; text-decoration: underline 0.05em; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================
# Symbol detection + Sector inference
# ============================
INDEX_ALIASES = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK",
    "NIFTYAUTO": "^CNXAUTO", "AUTO": "^CNXAUTO", "AUTOS": "^CNXAUTO", "AUTOMOBILE": "^CNXAUTO",
    "IT": "^CNXIT", "NIFTYIT": "^CNXIT", "TECH": "^CNXIT",
    "PHARMA": "^CNXPHARMA", "NIFTYPHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "NIFTYFMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL", "NIFTYMETAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY", "NIFTYENERGY": "^CNXENERGY",
    "FINANCE": "^CNXFINANCE", "FINANCIAL": "^CNXFINANCE", "NIFTYFIN": "^CNXFINANCE",
}
SECTOR_KEYWORDS = {
    "AUTO": "^CNXAUTO", "AUTOS": "^CNXAUTO", "AUTOMOBILE": "^CNXAUTO", "AUTOMOBILES": "^CNXAUTO",
    "IT": "^CNXIT", "TECH": "^CNXIT", "TECHNOLOGY": "^CNXIT",
    "PHARMA": "^CNXPHARMA", "HEALTHCARE": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "CONSUMER": "^CNXFMCG",
    "METAL": "^CNXMETAL", "METALS": "^CNXMETAL",
    "ENERGY": "^CNXENERGY", "OIL": "^CNXENERGY", "GAS": "^CNXENERGY",
    "FINANCE": "^CNXFINANCE", "FINANCIAL": "^CNXFINANCE",
    "BANK": "^NSEBANK", "BANKS": "^NSEBANK", "BANKING": "^NSEBANK",
}
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

def infer_sector_symbol(text: str) -> str | None:
    t = text.upper()
    for kw, sym in SECTOR_KEYWORDS.items():
        if kw in t:
            return sym
    return None

def extract_symbol_from_text(text: str) -> str | None:
    for raw in text.split():
        cand = probable_symbol(raw)
        if cand:
            return cand
    return infer_sector_symbol(text)

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
    if not (llm_enabled and OPENAI_OK):
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
        return f"LLM unavailable. Using extractive summary.\n\n{fused}"

# ============================
# Session State (engagement)
# ============================
st.session_state.setdefault("history", [])
st.session_state.setdefault("saved", [])      # list of {"q":..., "ts":...}
st.session_state.setdefault("last", [])       # last 10 questions
st.session_state.setdefault("focus", False)   # focus mode flag

def push_last(q: str):
    if not q: return
    st.session_state.last = ([q] + [x for x in st.session_state.last if x != q])[:10]

def save_query(q: str):
    if not q: return
    st.session_state.saved = ([{"q": q, "ts": time.time()}] +
                              [x for x in st.session_state.saved if x["q"] != q])[:25]
    st.toast("Saved ⭐", icon="⭐")

# ============================
# Header
# ============================
left, right = st.columns([0.82, 0.18])
with left:
    st.markdown('<div class="sx-hero">💹 StoxEye Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sx-sub">Indian market context • Live quote • Trend meter • Multi-source synthesis • Publisher credits</div>', unsafe_allow_html=True)
with right:
    st.toggle("Focus mode", key="focus", help="Hide chrome for a distraction-free view.")
    if st.session_state.focus:
        # hide top toolbar
        st.markdown("<style>.st-emotion-cache-1dp5vir{display:none;}</style>", unsafe_allow_html=True)

st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)

# ============================
# 📰 Daily Brief (auto-refresh + button)
# ============================
from streamlit import runtime

refresh_opt = st.selectbox("Daily Brief refresh", ["Off", "30s", "1m", "5m"], index=0, help="Auto-refresh cadence")
interval_map = {"Off": 0, "30s": 30_000, "1m": 60_000, "5m": 300_000}
interval_ms = interval_map[refresh_opt]
if interval_ms:
    st.autorefresh = st.experimental_rerun  # fallback alias on very old versions
    try:
        st.experimental_autorefresh(interval=interval_ms, key="brief_auto")
    except Exception:
        # Streamlit < 1.25 fallback: do nothing; manual refresh available
        pass

col_b1, col_b2 = st.columns([0.85, 0.15])
with col_b1:
    st.markdown("### 📰 Daily Brief")
with col_b2:
    if st.button("Refresh now", key="btn_refresh_brief"):
        st.experimental_rerun()

def build_daily_brief() -> Dict:
    # Light query for NIFTY/SENSEX snapshot
    q = "Give me a compact Indian market brief for today (NIFTY, SENSEX, key movers)."
    rs = research(q, days=1, prefer_symbol="^NSEI", max_articles=4, max_chars=900)
    articles = rs.get("articles") or []
    items = []
    for a in articles:
        src = (a.get("source") or "").strip()
        ttl = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if ttl:
            if url:
                items.append(f"- [{ttl}]({url}) · {src}")
            else:
                items.append(f"- {ttl} · {src}")
    return {"when": time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.get("timestamp", time.time()))),
            "items": items[:6]}

brief = build_daily_brief()
with st.container():
    st.markdown("\n".join(brief["items"]) or "_No fresh headlines captured._")
    st.caption(f"Updated: `{brief['when']}`")
st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)

# ============================
# Quick Chips + Controls
# ============================
colA, colB = st.columns([0.7, 0.3])

with colA:
    st.markdown("**Quick look**")
    cc = st.container()
    chip_cols = cc.columns(8)
    chips = [
        ("NIFTY", "NIFTY today?"),
        ("SENSEX", "SENSEX today?"),
        ("BANKNIFTY", "Why is Bank Nifty moving?"),
        ("IT", "Why is IT sector up/down?"),
        ("AUTO", "What’s driving autos?"),
        ("PHARMA", "Why pharma stocks?"),
        ("FMCG", "FMCG momentum today?"),
        ("METAL", "Metal sector sentiment?"),
    ]
    for i,(lab, q) in enumerate(chips):
        if chip_cols[i].button(lab, key=f"chip_{lab}", use_container_width=True):
            st.session_state["_prefill"] = q
            st.rerun()

with colB:
    llm_enabled = st.toggle("AI synthesis", value=True, help="Turn OFF to force extractive summary only.")
    status = "ON ✅" if (llm_enabled and OPENAI_OK) else "OFF ❌"
    st.markdown(f'<div class="sx-badge">Model: <b>{status}</b></div>', unsafe_allow_html=True)
    if llm_enabled and not OPENAI_OK:
        st.caption("No OPENAI_API_KEY found in .env / Secrets. Using extractive summaries.")

with st.expander("Scan settings", expanded=False):
    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        freshness = st.radio(
            "News window",
            options=[("Today", 1), ("Last 3 days", 3), ("Last 7 days", 7)],
            index=0, horizontal=True, format_func=lambda x: x[0],
        )[1]
    with colS2:
        max_articles = st.slider("Articles to fuse", 1, 8, 5, 1)
    with colS3:
        max_chars = st.slider("Chars per article", 400, 2400, 1200, 200)

st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)

# defaults
if "freshness" not in st.session_state:
    st.session_state.freshness = 1
st.session_state.freshness = locals().get("freshness", st.session_state.freshness)
st.session_state.max_articles = locals().get("max_articles", 5)
st.session_state.max_chars = locals().get("max_chars", 1200)

# ============================
# Chat + helpers
# ============================
def dedupe_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def publishers_used(articles: List[Dict]) -> str:
    pubs: List[str] = []
    for a in articles or []:
        s = (a.get("source") or "").strip()
        if s and s not in pubs:
            pubs.append(s)
    return ", ".join(pubs[:10]) if pubs else "—"

def publishers_links(articles: List[Dict]) -> str:
    items = []
    for a in articles or []:
        src = (a.get("source") or "").strip()
        ttl = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not ttl and src: ttl = src
        if ttl:
            if url:
                items.append(f"- [{ttl}]({url})")
            else:
                items.append(f"- {ttl}")
    if not items:
        return "_No source links captured for this query._"
    return "\n".join(dedupe_keep_order(items))[:3000]

# render prior convo
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prefill = st.session_state.pop("_prefill", "")
prompt = st.chat_input("Ask about a stock, index, or sector…", key="chat_inp", max_chars=300)
if prefill and not prompt:
    prompt = prefill

def answer(user_text: str):
    symbol = extract_symbol_from_text(user_text)

    quote = fetch_quote(symbol) if symbol else None
    tm = trend_meter(symbol) if symbol else None
    rs = research(
        user_text,
        days=st.session_state.freshness,
        prefer_symbol=symbol,
        max_articles=st.session_state.max_articles,
        max_chars=st.session_state.max_chars,
    )

    body = synthesize_answer(user_text, quote, rs, tm, llm_enabled)
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(rs.get("timestamp", time.time())))
    pubs_str = publishers_used(rs.get("articles"))
    pubs_md = publishers_links(rs.get("articles"))

    header_bits = []
    if quote and quote.get("price") is not None:
        header_bits.append(
            f"**{quote['symbol']}** · ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])} · "
            f"{quote.get('exchange','')} · {quote.get('currency','')} · {quote['time']}"
        )
    if tm and tm.get("label"):
        header_bits.append(
            f"*Trend:* {tm['label']} · RSI14: {tm['rsi14'] and round(tm['rsi14'],1)} · "
            f"SMA20/50/200: {tm['sma20'] and round(tm['sma20'],1)}/"
            f"{tm['sma50'] and round(tm['sma50'],1)}/"
            f"{tm['sma200'] and round(tm['sma200'],1)}"
        )

    with st.chat_message("assistant"):
        if header_bits:
            st.markdown("  \n".join(header_bits))
            st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)
        st.markdown(body)
        st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)

        meta_col1, meta_col2 = st.columns([0.55, 0.45])
        with meta_col1:
            st.markdown(f"**Data scan:** `{ts}`  ·  **Publishers:** {pubs_str}")
            st.caption("Research only. Not investment advice. • Built with ❤ by *venugAAdu*")
        with meta_col2:
            with st.expander("Open sources (links)"):
                st.markdown(pubs_md)

    push_last(user_text)

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    answer(prompt)

# Right rail: Saved / Recent
colLists = st.columns([0.74, 0.26])[1]
with colLists:
    st.markdown("### ⭐ Saved")
    if not st.session_state.saved:
        st.caption("No saved searches yet.")
    else:
        for item in st.session_state.saved:
            if st.button(item["q"], key=f"sv_{item['q']}", use_container_width=True):
                st.session_state["_prefill"] = item["q"]; st.rerun()

    st.markdown("### 🕘 Recent")
    if not st.session_state.last:
        st.caption("No recent queries.")
    else:
        for q in st.session_state.last:
            c1, c2 = st.columns([0.85, 0.15])
            if c1.button(q, key=f"last_{q}", use_container_width=True):
                st.session_state["_prefill"] = q; st.rerun()
            if c2.button("⭐", key=f"save_{q}"):
                save_query(q)

st.markdown('<div class="sx-divider"></div>', unsafe_allow_html=True)
st.markdown('Tip: **/** or **k** to focus chat • **f** Focus mode • **r** refresh Daily Brief • **1-8** Quick look chips • Save any query with ⭐')

# ============================
# Keyboard shortcuts (vanilla JS)
# ============================
st.markdown(
    """
<script>
document.addEventListener('keydown', (e) => {
  const isTyping = ['INPUT','TEXTAREA'].includes(document.activeElement.tagName);
  if (isTyping && e.key !== 'Escape') return;

  // focus chat
  if (e.key === '/' || e.key.toLowerCase() === 'k') {
    const ta = document.querySelector('[data-testid="stChatInput"] textarea');
    if (ta) { e.preventDefault(); ta.focus(); }
  }

  // toggle focus mode
  if (e.key.toLowerCase() === 'f') {
    const toggles = Array.from(document.querySelectorAll('label')).filter(el => el.innerText.includes('Focus mode'));
    if (toggles.length) { e.preventDefault(); toggles[0].click(); }
  }

  // refresh daily brief
  if (e.key.toLowerCase() === 'r') {
    const btns = Array.from(document.querySelectorAll('button')).filter(el => el.innerText.trim() === 'Refresh now');
    if (btns.length) { e.preventDefault(); btns[0].click(); }
  }

  // quick chips 1..8
  if (e.key >= '1' && e.key <= '8') {
    const n = parseInt(e.key, 10) - 1;
    const btns = Array.from(document.querySelectorAll('button')).filter(el => ['NIFTY','SENSEX','BANKNIFTY','IT','AUTO','PHARMA','FMCG','METAL'].includes(el.innerText.trim()));
    if (btns[n]) { e.preventDefault(); btns[n].click(); }
  }
});
</script>
""",
    unsafe_allow_html=True,
)
