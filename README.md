# StoxEye Copilot — Web (Streamlit)

Research-grade Q&A for Indian markets: live quote, news scan, Wikipedia profile, concise answer, and **citations**. Optional OpenAI key for synthesis.

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv; . .venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env   # add your OPENAI_API_KEY= (optional)
streamlit run app.py
```
Open http://localhost:8501

## Ask examples
- `What is INFY outlook after results?`
- `Compare HDFC Bank vs ICICI Bank near term`
- `NIFTY IT view`
- `Is TCS bullish near-term?`

## Notes
- Quotes via `yfinance` (best-effort). Do not use for trading.
- News via Google News RSS; profile via Wikipedia.
- If no OpenAI key, the app returns a rule-based concise summary.
- Research only. Not investment advice.
