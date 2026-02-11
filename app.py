import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI 포트폴리오 매니저", layout="wide")

st.title("📊 AI 포트폴리오 매니저 (완전 방어형 최종판)")

# -------------------------
# 입력 영역
# -------------------------
st.header("📌 종목 입력")

tickers_input = st.text_input(
    "종목코드 (쉼표 구분, 국내 .KS)",
    "005930.KS, NVDA, QQQ"
)

avg_prices_input = st.text_input(
    "평단가 (쉼표 구분)",
    "64000, 450, 400"
)

qty_input = st.text_input(
    "보유 수량 (쉼표 구분)",
    "10, 5, 8"
)

tickers = [t.strip() for t in tickers_input.split(",")]
avg_prices = [float(x.strip()) for x in avg_prices_input.split(",")]
quantities = [float(x.strip()) for x in qty_input.split(",")]

portfolio_data = []

# -------------------------
# 종목별 안전 다운로드
# -------------------------
for i, ticker in enumerate(tickers):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)

        if df.empty or len(df) < 30:
            continue

        close = df["Close"]
        current_price = float(close.iloc[-1])

        avg_price = avg_prices[i]
        qty = quantities[i]

        value = current_price * qty
        profit = (current_price - avg_price) * qty
        return_pct = ((current_price - avg_price) / avg_price) * 100

        # 이동평균
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()

        # 매수/매도 판단
        if ma5.iloc[-1] > ma20.iloc[-1]:
            signal = "📈 매수 우위"
            score = 70
        else:
            signal = "📉 매도/관망"
            score = 40

        # 30일 추세
        trend_30 = ((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]) * 100

        if trend_30 > 5:
            score += 15
        elif trend_30 < -5:
            score -= 15

        # 변동성
        volatility = close.pct_change().std() * 100
        if volatility < 2:
            score += 10

        # 단순 예측 (회귀)
        X = np.arange(len(close)).reshape(-1, 1)
        y = close.values
        model = LinearRegression().fit(X, y)
        future = model.predict([[len(close) + 5]])[0]

        portfolio_data.append([
            ticker, avg_price, current_price, qty,
            value, profit, return_pct,
            signal, trend_30, score, future
        ])

    except:
        continue

# -------------------------
# 데이터프레임 구성
# -------------------------
columns = [
    "Ticker", "평단가", "현재가", "보유수량",
    "총평가금액", "평가손익", "수익률(%)",
    "매매신호", "30일추세(%)", "AI점수", "5일예측가"
]

portfolio = pd.DataFrame(portfolio_data, columns=columns)

if portfolio.empty:
    st.error("데이터를 불러올 수 없습니다. 종목코드를 확인하세요.")
    st.stop()

total_asset = portfolio["총평가금액"].sum()
portfolio["비중(%)"] = portfolio["총평가금액"] / total_asset * 100

# -------------------------
# 📊 현재 자산 현황
# -------------------------
st.header("💰 현재 자산 현황")

col1, col2, col3 = st.columns(3)

col1.metric("총 자산", f"{total_asset:,.0f} 원")
col2.metric("총 손익", f"{portfolio['평가손익'].sum():,.0f} 원")
col3.metric("평균 수익률", f"{portfolio['수익률(%)'].mean():.2f}%")

st.dataframe(portfolio.round(2), use_container_width=True)

# -------------------------
# 📈 월별 추이
# -------------------------
st.header("📈 자산 추이")

trend_total = []

for ticker in tickers:
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
        trend_total.append(df["Close"] * quantities[tickers.index(ticker)])
    except:
        continue

combined = pd.concat(trend_total, axis=1).sum(axis=1)
st.line_chart(combined)

# -------------------------
# 🤖 AI 요약
# -------------------------
st.header("🤖 AI 종합 판단")

best_stock = portfolio.sort_values("AI점수", ascending=False).iloc[0]
worst_stock = portfolio.sort_values("AI점수").iloc[0]

st.success(f"📌 매수 우선 검토: {best_stock['Ticker']} (점수 {best_stock['AI점수']})")
st.warning(f"⚠ 리스크 관리 필요: {worst_stock['Ticker']} (점수 {worst_stock['AI점수']})")
