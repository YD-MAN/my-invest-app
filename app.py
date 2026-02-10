import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="내 자산 투자 판단", layout="centered")
st.title("📊 내 자산 기준 투자 판단 시스템")

assets = {
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "비트코인": "BTC-USD",
    "삼성전자": "005930.KS",
    "엔비디아": "NVDA",
    "테슬라": "TSLA",
    "넷플릭스": "NFLX"
}

asset_name = st.selectbox("자산 선택", list(assets.keys()))
ticker = assets[asset_name]

data = yf.download(ticker, period="6mo")

data["MA20"] = data["Close"].rolling(20).mean()
data["MA60"] = data["Close"].rolling(60).mean()

close = data["Close"].squeeze()

# ===== 데이터 안전성 체크 =====
if data.empty or len(data) < 60:
    st.warning("데이터가 충분하지 않습니다.")
    st.stop()

close = data["Close"].squeeze()

# ===== RSI 계산 (안정 버전) =====
delta = close.diff()

gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

data["RSI"] = rsi

# ===== NaN 제거 =====
data = data.dropna()

# ===== 최신 데이터 =====
latest_close = data["Close"].iloc[-1]
latest_rsi = data["RSI"].iloc[-1]

# ===== 최근 1개월 수익률 =====
month_return = (data["Close"].iloc[-1] / data["Close"].iloc[-21] - 1) * 100

if data["MA20"].iloc[-1] > data["MA60"].iloc[-1] and latest_rsi < 70:
    decision = "✅ 매수"
elif data["MA20"].iloc[-1] > data["MA60"].iloc[-1]:
    decision = "⏸ 대기"
else:
    decision = "❌ 매도"



st.subheader(f"📌 오늘의 판단: {decision}")
st.metric("최근 1개월 수익률", f"{month_return:.2f}%")
st.metric("RSI", f"{latest['RSI']:.1f}")
st.line_chart(data[["Close", "MA20", "MA60"]])
