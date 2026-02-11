import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time

st.set_page_config(page_title="AI 포트폴리오 매니저", layout="centered")
st.title("📊 AI 포트폴리오 매니저 (실시간 자동 갱신)")

# -----------------------------
# 자동 새로고침 (60초)
# -----------------------------
st.experimental_autorefresh(interval=60000, key="refresh")

# -----------------------------
# 세션 상태
# -----------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# -----------------------------
# 초기화
# -----------------------------
if st.button("🧹 전체 초기화"):
    st.session_state.portfolio = []
    st.success("포트폴리오 초기화 완료")

st.markdown("---")

# -----------------------------
# 종목 입력 (평단 기준)
# -----------------------------
st.subheader("➕ 종목 입력 (실시간 가격 자동 반영)")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("종목명", placeholder="삼성전자")
    code = st.text_input("종목코드", placeholder="005930.KS 또는 AAPL")

with col2:
    qty = st.number_input("보유 수량", min_value=0.0, step=1.0)
    avg_price = st.number_input("평단가", min_value=0, step=100)

if st.button("종목 추가"):
    st.session_state.portfolio.append({
        "종목명": name,
        "종목코드": code,
        "보유수량": qty,
        "평단가": avg_price
    })
    st.success("종목 추가 완료")

# -----------------------------
# 실시간 주가 조회 함수
# -----------------------------
@st.cache_data(ttl=60)
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return round(data["Close"].iloc[-1], 2)
        else:
            return None
    except:
        return None

# -----------------------------
# 포트폴리오 분석
# -----------------------------
if st.session_state.portfolio:
    df = pd.DataFrame(st.session_state.portfolio)

    current_prices = []
    profit_rates = []
    signals = []

    for i, row in df.iterrows():
        price = get_current_price(row["종목코드"])
        current_prices.append(price)

        if price and row["평단가"] > 0:
            profit = (price - row["평단가"]) / row["평단가"] * 100
        else:
            profit = 0

        profit_rates.append(round(profit, 2))

        # 매매 신호
        if profit <= -10:
            signal = "🔵 추가매수 고려"
        elif profit >= 20:
            signal = "🔴 분할매도 고려"
        else:
            signal = "🟡 보유"

        signals.append(signal)

    df["현재가"] = current_prices
    df["수익률(%)"] = profit_rates
    df["매매신호"] = signals
    df["평가금액"] = df["보유수량"] * df["현재가"]

    st.markdown("---")
    st.subheader("📋 실시간 수익 현황 & 매매 판단")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # 수익률 그래프
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 종목별 수익률")

    fig, ax = plt.subplots()
    ax.bar(df["종목명"], df["수익률(%)"])
    ax.axhline(0)
    ax.set_ylabel("수익률 (%)")
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # 전체 판단
    # -----------------------------
    avg_return = df["수익률(%)"].mean()

    st.markdown("---")
    st.subheader("🧠 포트폴리오 종합 판단")

    if avg_return >= 15:
        st.success("전체 수익 구간. 일부 분할매도 고려 가능.")
    elif avg_return <= -10:
        st.warning("손실 구간. 리밸런싱 또는 추가매수 검토.")
    else:
        st.info("중립 구간. 추이 관찰 권장.")
