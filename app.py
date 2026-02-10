import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="통합 포트폴리오 관리", layout="wide")

st.title("📊 통합 포트폴리오 관리 시스템")
st.caption("국내주식 · 해외주식 · ETF · 코인 통합")

# =========================
# 1. 포트폴리오 입력
# =========================
st.subheader("📌 포트폴리오 입력")

default_df = pd.DataFrame({
    "ticker": ["QQQ", "SPY", "005930.KS", "BTC-USD"],
    "avg_price": [380, 450, 72000, 52000],
    "quantity": [10, 5, 10, 0.2]
})

portfolio_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True
)

# =========================
# 2. 분석 실행
# =========================
if st.button("📈 포트폴리오 분석 실행"):

    total_cost = 0.0
    total_value = 0.0
    valid_assets = []

    st.divider()
    st.subheader("📊 종목별 분석 결과")

    for _, row in portfolio_df.iterrows():
        ticker = str(row["ticker"]).strip()
        avg_price = float(row["avg_price"])
        qty = float(row["quantity"])

        if ticker == "" or qty <= 0:
            continue

        data = yf.download(ticker, period="6mo", progress=False)

        if data.empty:
            st.error(f"❌ {ticker} 데이터 로드 실패")
            continue

        latest_price = data["Close"].iloc[-1]
        cost = avg_price * qty
        value = latest_price * qty
        pnl = (value - cost) / cost * 100

        total_cost += cost
        total_value += value
        valid_assets.append(ticker)

        st.success(
            f"✔ {ticker} | 현재가 {latest_price:.2f} | 수익률 {pnl:.2f}%"
        )

    st.divider()

    # =========================
    # 3. 포트폴리오 요약
    # =========================
    st.subheader("📌 포트폴리오 요약")

    if total_cost == 0:
        st.warning("유효한 종목 데이터가 없어 포트폴리오를 계산할 수 없습니다.")
    else:
        portfolio_pnl = (total_value - total_cost) / total_cost * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("총 투자금", f"{total_cost:,.0f}")
        col2.metric("현재 가치", f"{total_value:,.0f}")
        col3.metric("총 수익률", f"{portfolio_pnl:.2f}%")

        # =========================
        # 4. 비상 경보 (간단 버전)
        # =========================
        if portfolio_pnl < -15:
            st.error("🚨 포트폴리오 비상 단계 (손실 -15% 초과)")
