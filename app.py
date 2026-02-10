import streamlit as st
import yfinance as yf
import pandas as pd

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
    "엔비디아": "NVDA",
    "테슬라": "TSLA",
}

asset_name = st.selectbox("자산 선택", list(ASSETS.keys()))
ticker = ASSETS[asset_name]

# =========================
# 데이터 로드 (가장 안전한 방식)
# =========================
raw = yf.download(
    ticker,
    period="6mo",
    auto_adjust=True,
    progress=False,
    group_by="column"
)

# 데이터 없으면 중단
if raw.empty:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

# =========================
# 컬럼 정규화 (이게 핵심)
# =========================
# Close 컬럼이 있는 경우만 사용
if "Close" not in raw.columns:
    st.error("가격 데이터가 없습니다.")
    st.stop()

data = raw[["Close"]].copy()
data = data.dropna()

# 데이터 길이 체크
if len(data) < 30:
    st.error("분석에 필요한 데이터가 부족합니다.")
    st.stop()

# =========================
# 최근 수익률 계산
# =========================
recent_return = (data["Close"].iloc[-1] / data["Close"].iloc[-21] - 1) * 100
recent_return = float(recent_return)

# =========================
# 단순 추세 판단
# =========================
ma_short = data["Close"].rolling(5).mean().iloc[-1]
ma_long = data["Close"].rolling(20).mean().iloc[-1]

if ma_short > ma_long:
    decision = "✅ 상승 추세 (관심)"
else:
    decision = "⚠️ 약세 / 횡보"

# =========================
# 출력
# =========================
st.subheader(f"📌 오늘의 판단: {decision}")
st.metric("최근 1개월 수익률", f"{recent_return:.2f}%")

# =========================
# 차트 (절대 안 깨지는 구조)
# =========================
st.line_chart(data)
