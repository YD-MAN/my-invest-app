import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="내 자산 기준 투자 판단 시스템", layout="centered")
st.title("📊 내 자산 기준 투자 판단 시스템")

# =========================
# 자산 정의
# =========================
ASSETS = {
    "NASDAQ": "^IXIC",
    "S&P500": "^GSPC",
    "비트코인": "BTC-USD",
    "삼성전자": "005930.KS",
    "엔비디아": "NVDA",
    "테슬라": "TSLA"
}

asset_name = st.selectbox("자산 선택", list(ASSETS.keys()))
ticker = ASSETS[asset_name]

# =========================
# 데이터 로드
# =========================
data = yf.download(ticker, period="6mo", progress=False)

if data.empty or len(data) < 60:
    st.error("데이터가 충분하지 않습니다.")
    st.stop()

# =========================
# 이동평균
# =========================
data["MA20"] = data["Close"].rolling(20).mean()
data["MA60"] = data["Close"].rolling(60).mean()

# =========================
# RSI 계산 (완전 안정 버전)
# =========================
close = data["Close"].squeeze()
delta = close.diff()

gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# =========================
# NaN 제거
# =========================
data = data.dropna()

if len(data) < 21:
    st.error("지표 계산 후 데이터가 부족합니다.")
    st.stop()

# =========================
# 최근 값
# =========================
latest_ma20 = data["MA20"].iloc[-1]
latest_ma60 = data["MA60"].iloc[-1]
latest_rsi = data["RSI"].iloc[-1]

# =========================
# 최근 1개월 수익률
# =========================
month_return = (data["Close"].iloc[-1] / data["Close"].iloc[-21] - 1) * 100
month_return = float(month_return)

# =========================
# 판단 로직
# =========================
if latest_ma20 > latest_ma60 and latest_rsi < 70:
    decision = "✅ 매수"
elif latest_ma20 > latest_ma60:
    decision = "⏸ 대기"
else:
    decision = "❌ 매도"

# =========================
# 출력
# =========================
st.subheader(f"📌 오늘의 판단: {decision}")
st.metric("최근 1개월 수익률", f"{month_return:.2f}%")
st.metric("RSI", f"{latest_rsi:.1f}")

st.line_chart(data[["Close", "MA20", "MA60"]])
