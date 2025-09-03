from __future__ import annotations
import datetime as dt
import time
from typing import Optional, Dict
import yfinance as yf

# ---------- SAFE QUOTE (rate-limit tolerant) ----------
def fetch_quote(symbol: str) -> dict:
    s = (symbol or "").upper().strip()
    for attempt in range(2):  # brief retry once
        try:
            t = yf.Ticker(s)
            info = t.fast_info
            price = float(info["last_price"]) if info.get("last_price") else None
            prev = float(info["previous_close"]) if info.get("previous_close") else None
            change = (price - prev) if (price is not None and prev is not None) else None
            pct = ((change / prev) * 100) if (change is not None and prev and prev != 0) else None
            exch = info.get("exchange", "")
            curr = info.get("currency", "INR")
            break
        except Exception:
            if attempt == 0:
                time.sleep(1.2)
                continue
            price = prev = change = pct = None
            exch, curr = "", "INR"

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

# ---------- TECHNICALS ----------
def _rsi(series, period: int = 14) -> Optional[float]:
    try:
        import numpy as np
        import pandas as pd
        if len(series) < period + 1:
            return None
        delta = pd.Series(series).diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]
        if avg_gain is None or avg_loss is None or avg_loss == 0:
            return None if avg_gain is None else 100.0
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

def trend_meter(symbol: str, period: str = "6mo") -> Dict:
    """
    Returns a dict with RSI(14), SMA20/50/200 and a simple label: Bullish / Bearish / Neutral.
    """
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

        c, s20, s50, s200, r = out["close"], out["sma20"], out["sma50"], out["sma200"], out["rsi14"]

        # Simple regime logic
        if all(x is not None for x in [c, s50, s200, r]):
            if c > s50 > s200 and (r is None or r >= 55):
                out["label"] = "Bullish"
                out["note"] = "Price above 50 & 200 SMA; momentum ok"
            elif c < s50 < s200 and (r is None or r <= 45):
                out["label"] = "Bearish"
                out["note"] = "Price below 50 & 200 SMA; weak momentum"
            else:
                out["label"] = "Neutral"
                out["note"] = "Mixed signals / range"

        return out
    except Exception:
        return out
