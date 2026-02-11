import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 AI 포트폴리오 매니저 Pro")

# ----------------------
# 입력 영역
# ----------------------

with st.expander("📌 포트폴리오 입력", expanded=True):
    tickers = st.text_input("종목코드", "AAPL, MSFT, NVDA")
    buy_prices = st.text_input("평단가", "150, 300, 400")
    quantities = st.text_input("수량", "10, 5, 3")

tickers = [t.strip() for t in tickers.split(",")]
buy_prices = list(map(float, buy_prices.split(",")))
quantities = list(map(int, quantities.split(",")))

# ----------------------
# 함수 정의
# ----------------------

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

results = []
total_value = 0
total_score = 0

for ticker, buy_price, qty in zip(tickers, buy_prices, quantities):

    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty:
        continue

    current_price = df['Close'].iloc[-1]
    ma5 = df['Close'].rolling(5).mean().iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma60 = df['Close'].rolling(60).mean().iloc[-1]

    rsi = calculate_rsi(df['Close']).iloc[-1]
    return_30 = (df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1) * 100
    volatility = df['Close'].pct_change().std()

    # MDD
    rolling_max = df['Close'].cummax()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    mdd = drawdown.min()

    # 점수 계산
    trend_strength = (ma5 - ma60) / ma60
    trend_score = min(max(trend_strength * 200, 0), 25)

    momentum_score = min(max(return_30 * 1.5, 0), 20)

    rsi_score = 15 - abs(rsi - 50) * 0.3
    rsi_score = max(min(rsi_score, 15), 0)

    vol_score = max(20 - volatility * 400, 0)

    mdd_score = max(20 + (mdd * 200), 0)

    ai_score = trend_score + momentum_score + rsi_score + vol_score + mdd_score
    ai_score = min(max(ai_score, 0), 100)

    value = current_price * qty
    total_value += value
    total_score += ai_score

    results.append({
        "종목": ticker,
        "현재가": round(current_price, 2),
        "AI 점수": round(ai_score, 1),
        "30일 수익률(%)": round(return_30, 2),
        "보유가치": round(value, 2)
    })

df_result = pd.DataFrame(results)

# ----------------------
# KPI 카드
# ----------------------

col1, col2, col3 = st.columns(3)

col1.metric("총 자산 가치", f"${round(total_value,2)}")
col2.metric("평균 AI 점수", f"{round(total_score/len(df_result),1) if len(df_result)>0 else 0}")
col3.metric("보유 종목 수", len(df_result))

st.dataframe(df_result, use_container_width=True)

# ----------------------
# 리밸런싱
# ----------------------

if total_score > 0:
    st.subheader("🔄 자동 리밸런싱 추천")

    df_result["목표 비중"] = df_result["AI 점수"] / total_score
    df_result["목표 금액"] = df_result["목표 비중"] * total_value
    df_result["조정 필요 금액"] = df_result["목표 금액"] - df_result["보유가치"]

    st.dataframe(df_result, use_container_width=True)

# ----------------------
# 차트
# ----------------------

if len(tickers) > 0:
    chart_ticker = tickers[0]
    df_chart = yf.download(chart_ticker, period="6mo", progress=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].rolling(20).mean(), name="MA20"))
    fig.update_layout(height=350)

    st.plotly_chart(fig, use_container_width=True)
