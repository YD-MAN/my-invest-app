import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI 포트폴리오 매니저", layout="wide")

st.title("📊 AI 포트폴리오 매니저 (완전 안정화 버전)")

# 자동 새로고침 (Streamlit 기본 방식)
if "refresh" not in st.session_state:
    st.session_state.refresh = 0

st.session_state.refresh += 1

# ----------------------------
# 종목 입력
# ----------------------------

st.header("📌 종목 입력")

ticker_input = st.text_area(
    "종목코드 입력 (쉼표로 구분, 국내주식은 .KS 또는 .KQ 붙이기)",
    "AAPL, MSFT, 005930.KS"
)

avg_price_input = st.text_area(
    "평단가 입력 (위 종목 순서대로, 쉼표로 구분)",
    "150, 300, 70000"
)

tickers = [t.strip() for t in ticker_input.split(",")]
avg_prices = [float(p.strip()) for p in avg_price_input.split(",")]

# ----------------------------
# 실시간 데이터 수집
# ----------------------------

data = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True)

if len(tickers) == 1:
    data = pd.DataFrame({tickers[0]: data["Close"]})
else:
    data = data["Close"]

latest_prices = data.iloc[-1]

# ----------------------------
# 포트폴리오 계산
# ----------------------------

portfolio = pd.DataFrame({
    "Ticker": tickers,
    "평단가": avg_prices,
    "현재가": latest_prices.values
})

portfolio["수익률(%)"] = ((portfolio["현재가"] - portfolio["평단가"]) / portfolio["평단가"]) * 100

portfolio["평가금액"] = portfolio["현재가"]
total_value = portfolio["평가금액"].sum()
portfolio["비중(%)"] = (portfolio["평가금액"] / total_value) * 100

st.subheader("💰 수익 현황")
st.dataframe(portfolio.round(2), use_container_width=True)

# ----------------------------
# 📊 종목별 비중 파이차트
# ----------------------------

fig_pie = px.pie(
    portfolio,
    names="Ticker",
    values="비중(%)",
    title="📊 종목별 비중"
)

st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------
# 📈 월별 자산 추이
# ----------------------------

monthly = data.resample("M").last()
portfolio_trend = monthly.sum(axis=1)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=portfolio_trend.index,
    y=portfolio_trend.values,
    mode="lines",
    name="총 자산"
))

fig_trend.update_layout(title="📈 월별 자산 추이")
st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------------
# 📊 이동평균 매수 판단 + AI 점수
# ----------------------------

st.header("🤖 AI 분석 결과")

analysis_results = []

for ticker in tickers:
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    close = df["Close"]

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    score = 0

    # 1️⃣ 이동평균 매수 시그널
    if ma5.iloc[-1] > ma20.iloc[-1]:
        signal = "📈 단기 상승 (매수 유리)"
        score += 40
    else:
        signal = "📉 하락 추세 (관망)"
        score -= 20

    # 2️⃣ 최근 30일 추세
    change_30 = ((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]) * 100
    if change_30 > 5:
        trend_warn = "✅ 상승 추세"
        score += 30
    elif change_30 < -5:
        trend_warn = "⚠ 하락 경고"
        score -= 30
    else:
        trend_warn = "➖ 횡보"

    # 3️⃣ 변동성 기반 점수
    volatility = close.pct_change().std() * 100
    if volatility < 2:
        score += 30
    else:
        score -= 10

    analysis_results.append([ticker, signal, trend_warn, score])

analysis_df = pd.DataFrame(
    analysis_results,
    columns=["Ticker", "이동평균 판단", "30일 추세", "AI 점수"]
)

st.dataframe(analysis_df, use_container_width=True)

# ----------------------------
# 📱 요약 카드 모드
# ----------------------------

st.header("📲 홈 요약 카드")

col1, col2, col3 = st.columns(3)

col1.metric("총 자산 가치", f"{total_value:,.0f}")
col2.metric("평균 수익률", f"{portfolio['수익률(%)'].mean():.2f}%")
col3.metric("최고 AI 점수", f"{analysis_df['AI 점수'].max()}")

st.caption("※ 1분마다 새로고침하면 실시간 데이터 반영됩니다.")
