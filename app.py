import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")
st.title("📊 AI 포트폴리오 매니저 Pro")

# ---------------------------
# 입력
# ---------------------------
tickers_input = st.text_input("종목코드 (쉼표 구분)", "AAPL, MSFT, NVDA")
buy_prices_input = st.text_input("평단가", "150, 300, 400")
quantities_input = st.text_input("수량", "10, 5, 3")

def safe_float_list(s: str):
    out = []
    for item in s.split(","):
        try:
            out.append(float(item.strip()))
        except:
            out.append(0.0)
    return out

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
buy_prices = safe_float_list(buy_prices_input)
quantities = safe_float_list(quantities_input)

# 길이 보정
max_len = max(len(tickers), len(buy_prices), len(quantities))
while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices) < max_len:
    buy_prices.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)

# ---------------------------
# 리스크/점수
# ---------------------------
def calculate_risk(vol: float) -> str:
    if vol < 0.02:
        return "낮음"
    elif vol < 0.05:
        return "보통"
    else:
        return "높음"

def calculate_ai_score(trend: float, vol: float, mom: float) -> int:
    score = (trend * 200) + ((1 - vol) * 30) + (mom * 100) + 20
    return int(np.clip(score, 0, 100))

# ---------------------------
# 처리
# ---------------------------
for i in range(max_len):
    ticker = tickers[i]
    if ticker == "":
        continue

    try:
        data = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        continue

    if data is None or len(data) == 0:
        st.warning(f"{ticker} 데이터 없음")
        continue

    if "Close" not in data.columns:
        st.warning(f"{ticker} 종가(Close) 없음")
        continue

    # ✅ Close를 항상 1차원 Series로 정규화
    close = data["Close"]
    if isinstance(close, pd.DataFrame):   # (가끔
