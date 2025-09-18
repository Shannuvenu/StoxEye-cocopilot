# ============================
# StoxEye Copilot — app.py
# ============================

# ---- Std libs ----
import os, re, time, datetime
from typing import List, Dict

# ---- Streamlit must be first ----
import streamlit as st
st.set_page_config(page_title="StoxEye Copilot", page_icon="💹", layout="wide")

# ---- Other imports AFTER page config ----
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

# Optional external libs (guarded)
try:
    import feedparser
except Exception:
    feedparser = None

# Your local modules (assumed present)
# Make sure finance.py & research.py are in the same folder
from finance import fetch_quote, format_change, trend_meter
from research import research


# ============================
# Helpers: OpenAI key (optional)
# ============================
def get_openai_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
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
# Styles (dark + blue)
# ============================
DARK_CSS = """
<style>
:root {
  --sx-bg: #0b0f14;
  --sx-card: #0f172a;
  --sx-elev: #111827;
  --sx-text: #e5e7eb;
  --sx-muted: #94a3b8;
  --sx-accent: #3b82f6;
}

/* Backgrounds */
html, body, [data-testid="stAppViewContainer"] { background: var(--sx-bg); }

/* Make sure there’s room above the first element */
[data-testid="stAppViewContainer"] > .main { padding-top: 28px !important; }
.block-container { padding-top: 28px !important; }

/* Typography */
h1,h2,h3,h4,h5,h6, p, span, label, div, code, kbd { color: var(--sx-text) !important; }

/* Header styling */
.sx-hero {
  margin-top: 4px;              /* small gap so it's never clipped */
  line-height: 1.15;            /* tighter line height so emoji + text align nicely */
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: .3px;
  display: flex; gap: .6rem; align-items: center;
}
.sx-emoji { display:inline-block; transform: translateY(2px); }  /* nudge emoji down */

/* Sub, cards, badge */
.sx-sub   { color: var(--sx-muted) !important; margin-bottom: .5rem; }
.sx-card  { background: var(--sx-card); border: 1px solid #1f2937; border-radius: 12px; padding: 12px 14px; }
.sx-badge { background: rgba(59,130,246,.12); color: #93c5fd; border: 1px solid rgba(59,130,246,.4);
            border-radius: 999px; padding: 4px 10px; font-size: 12px; display: inline-flex; gap: 6px; align-items: center; }

hr { border-color: #1f2937 !important; }
a, a:visited { color: #93c5fd; text-decoration: none; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ============================
# Symbol detection (simple & robust)
# ============================
INDEX_ALIASES = {"NIFTY":"^NSEI","NIFTY50":"^NSEI","SENSEX":"^BSESN","BANKNIFTY":"^NSEBANK"}
STOPWORDS = {
    "WHAT","WHY","HOW","IS","THE","A","AN","OF","ON","FOR","IN","ABOUT","TELL","ME","VIEW",
    "AFTER","RESULTS","TODAY","NEAR","TERM","SHORT","LONG","AND","OR","PLEASE",
    "STOCK","MARKET","PRICE","OUTLOOK","UPDATE","NEWS","VS","COMPARE","HOLD","HOLDING"
}
def probable_symbol(token: str) -> str | None:
    t = token.upper().strip(",.!?:;()[]{}")
    if t in INDEX_ALIASES: return INDEX_ALIASES[t]
    if all(ch.isalnum() or ch=="." for ch in t) and 2 <= len(t) <= 10 and t not in STOPWORDS:
        return t
    return None
def extract_symbol_from_text(text: str) -> str | None:
    for raw in text.split():
        cand = probable_symbol(raw)
        if cand: return cand
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
        title = a.get("title","")
        if title: corpus.append(title)
        snip = a.get("snippet","") or ""
        corpus.extend(_sentences(snip)[:4])
    ranked = sorted(set(corpus), key=_score_sentence, reverse=True)
    points = ranked[:max_sents]
    if not points:
        return "Couldn’t extract specifics from the latest coverage. Check earnings, filings and sector cues."
    bullets = "\n".join([f"- {p}" for p in points])
    return f"**Direct answer (extractive):**\n{bullets}\n\n**Takeaway:** Movement likely ties to the factors above; monitor filings/guidance."


# ============================
# Synthesis (OpenAI if available)
# ============================
def synthesize_answer(question: str, quote: dict | None, rs: dict, trend: dict | None, llm_enabled: bool) -> str:
    # Fallback path
    if not (llm_enabled and OPENAI_OK):
        price_line = ""
        if quote and quote.get("price") is not None:
            price_line = f"Snapshot: {quote['symbol']} ₹{quote['price']:.2f} {format_change(quote['change'], quote['pct'])}"
        trend_line = ""
        if trend and trend.get("label"):
            tm = trend
            rsi = tm["rsi14"]; s20, s50, s200 = tm["sma20"], tm["sma50"], tm["sma200"]
            trend_line = f"Trend: {tm['label']} (RSI14: {rsi and round(rsi,1)}, SMA20/50/200: {s20 and round(s20,1)}/{s50 and round(s50,1)}/{s200 and round(s200,1)})"
        fused = summarize_from_articles(question, rs.get("articles") or [], max_sents=4)
        parts = [price_line, trend_line, fused, "Note: Research info only; not investment advice."]
        return "\n\n".join([p for p in parts if p])

    # LLM path
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        articles = rs.get("articles") or []
        profile = rs.get("profile") or ""
        evidence_chunks = []
        for a in articles[:5]:
            src = a.get("source","")
            title = a.get("title","")
            snip = (a.get("snippet","") or "")[:900]
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
                {"role":"system","content":sys_prompt},
                {"role":"user","content":user_context},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        fused = summarize_from_articles(question, rs.get("articles") or [], max_sents=4)
        return f"LLM unavailable. Using extractive summary.\n\n{fused}"


# ============================
# UI Header
# ============================
st.markdown('<div class="sx-hero"><span class="sx-emoji">💹</span> StoxEye Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sx-sub">Indian market context • Live quote • Trend meter • Multi-source synthesis • Publisher credits</div>',
    unsafe_allow_html=True
)


# ============================
# NAVIGATION — solid tabs
# ============================
tab_research, tab_brief, tab_alerts = st.tabs(["🔎 Research", "🗞️ Daily Brief", "🔔 Alerts"])


# ============================
# TAB 1 — Research
# ============================
with tab_research:
    st.markdown('<div class="sx-badge">Model: {}</div>'.format("ON ✅" if OPENAI_OK else "OFF ❌"), unsafe_allow_html=True)

    left, right = st.columns([0.65, 0.35])
    with right:
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
            llm_enabled = st.toggle("AI synthesis", value=True, help="Turn OFF to force extractive summary only.")
    with left:
        q = st.chat_input("Ask about a stock, index, or sector…")
        if "history" not in st.session_state:
            st.session_state.history = []

        for role, msg in st.session_state.history:
            with st.chat_message(role):
                st.markdown(msg)

        def publishers_used(articles: List[Dict]) -> str:
            pubs, links = [], []
            for a in articles or []:
                s = (a.get("source") or "").strip()
                u = (a.get("url") or "").strip()
                if s and s not in pubs:
                    pubs.append(s)
                    if u: links.append(f"[{s}]({u})")
            return ", ".join(pubs[:8]) if pubs else "—", " • ".join(links[:10]) if links else ""

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
            pubs_str, pub_links = publishers_used(rs.get("articles"))

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
                    f"SMA20/50/200: `{tm['sma20'] and round(tm['sma20'],1)}` / "
                    f"`{tm['sma50'] and round(tm['sma50'],1)}` / "
                    f"`{tm['sma200'] and round(tm['sma200'],1)}`"
                )
            if header_bits:
                st.markdown("  \n".join(header_bits))
                st.divider()

            st.markdown(body)
            st.divider()
            st.markdown(f"**Data scan:** `{ts}`   •   **Publishers:** {pubs_str}")
            with st.expander("Open sources (links)"):
                if pub_links:
                    st.markdown(pub_links)
                else:
                    st.info("No source links captured for this query.")
            st.caption("Research only. Not investment advice. • Built with ❤ by *venugAAdu*")

        if q:
            st.session_state.history.append(("user", q))
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                answer(q)


# ============================
# TAB 2 — Daily Brief
# ============================
with tab_brief:
    st.subheader("Daily Brief")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        refresh_choice = st.selectbox(
            "Daily Brief refresh",
            ["Off", "Every 5 min", "Every 15 min", "Every 30 min", "Hourly"],
            index=0,
            help="Auto-refresh pulls latest headlines periodically."
        )
    with colB:
        if st.button("Refresh now"):
            st.rerun()

    # Simple feeds (India markets flavored). You can tweak/add more.
    FEEDS = [
        ("Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol News",    "https://www.moneycontrol.com/rss/latestnews.xml"),
        ("Business Standard",    "https://www.business-standard.com/rss/latest.rss"),
        ("Mint Markets",         "https://www.livemint.com/rss/markets"),
    ]

    # Auto refresh control
    auto_map = {"Off": 0, "Every 5 min": 300, "Every 15 min": 900, "Every 30 min": 1800, "Hourly": 3600}
    wait = auto_map.get(refresh_choice, 0)
    if wait and "last_refresh" in st.session_state:
        if time.time() - st.session_state["last_refresh"] > wait:
            st.session_state["last_refresh"] = time.time()
            st.rerun()
    elif wait and "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()

    # Render brief
    if not feedparser:
        st.warning("Feedparser not installed. Add `feedparser` to requirements.txt for Daily Brief.", icon="⚠️")
    else:
        items = []
        for name, url in FEEDS:
            try:
                d = feedparser.parse(url)
                for e in d.get("entries", [])[:4]:
                    title = e.get("title", "").strip()
                    link = e.get("link", "").strip()
                    if title and link:
                        items.append((title, link, name))
            except Exception:
                pass

        if not items:
            st.info("No headlines fetched right now.")
        else:
            # List
            for t, lnk, src in items[:20]:
                st.markdown(f"- {t} · _{src}_  —  [{'link'}]({lnk})")
            st.caption("Updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

# ============================
# TAB 3 — Alerts (placeholder)
# ============================
with tab_alerts:
    st.subheader("Alerts")
    st.info("Price/RSI/MA cross alerts can be added here later. For now, this is a placeholder.")
