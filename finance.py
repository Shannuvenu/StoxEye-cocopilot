from __future__ import annotations
import datetime as dt
import time
from typing import Optional, Dict
import yfinance as yf

# -------- Safe live quote (rate-limit tolerant) --------
def fetch_quote(symbol: str | None) -> dict | None:
    if not symbol:
        return None
    s = (symbol or "").upper().strip()
    price = prev = change = pct = None
    exch, curr = "", "INR"
    for attempt in range(2):  # one retry
        try:
            t = yf.Ticker(s)
            info = t.fast_info
            price = float(info["last_price"]) if info.get("last_price") else None
            prev = float(info["previous_close"]) if info.get("previous_close") else None
            change = (price - prev) if (price is not None and prev is not None) else None
            pct = ((change / prev) * 100) if (change is not None and prev and prev != 0) else None
            exch = info.get("exchange", "")
            curr = info.get("currency", "INR") or "INR"
            break
        except Exception:
            if attempt == 0:
                time.sleep(1.2)
                continue
            # give up; return Nones
    return {
        "symbol": s,
        "price": price,
        "prev_close": prev,
        "change": change,
        "pct": pct,
        "currency": curr,
        "exchange": exch,
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

def format_change(change: float | None, pct: float | None) -> str:
    if change is None or pct is None:
        return ""
    arrow = "▲" if change >= 0 else "▼"
    return f"{arrow} {change:.2f} ({pct:.2f}%)"

# -------- Basic technicals for a quick “Trend Meter” --------
def _rsi(series, period: int = 14) -> Optional[float]:
    try:
        import pandas as pd
        delta = pd.Series(series).diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        if len(gains) < period or len(losses) < period:
            return None
        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception:
        return None

def _sma(series, window: int) -> Optional[float]:
    try:
        import pandas as pd
        if len(series) < window:
            return None
        return float(pd.Series(series).rolling(window=window).mean().iloc[-1])
    except Exception:
        return None

def trend_meter(symbol: str | None, period: str = "6mo") -> Dict:
    out = {
        "symbol": (symbol or "").upper(),
        "close": None,
        "rsi14": None,
        "sma20": None,
        "sma50": None,
        "sma200": None,
        "label": "Neutral",
        "note": "Insufficient data",
    }
    try:
        if not symbol:
            out["note"] = "No symbol"
            return out
        df = yf.download(symbol, period=period, interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            out["note"] = "No history"
            return out
        closes = df["Close"].dropna().tolist()
        out["close"] = float(closes[-1])
        out["rsi14"] = _rsi(closes, 14)
        out["sma20"] = _sma(closes, 20)
        out["sma50"] = _sma(closes, 50)
        out["sma200"] = _sma(closes, 200)

        c, s50, s200, r = out["close"], out["sma50"], out["sma200"], out["rsi14"]
        if all(x is not None for x in [c, s50, s200]):
            if c > s50 > s200 and (r is None or r >= 55):
                out["label"] = "Bullish"; out["note"] = "Above 50 & 200 SMA; momentum OK"
            elif c < s50 < s200 and (r is None or r <= 45):
                out["label"] = "Bearish"; out["note"] = "Below 50 & 200 SMA; weak momentum"
            else:
                out["label"] = "Neutral"; out["note"] = "Mixed signals / range"
        return out
    except Exception:
        return out
