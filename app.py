import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI 포트폴리오 매니저", layout="centered")
st.title("📊 AI 포트폴리오 매니저")

# -----------------------------
# Session State
# -----------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# -----------------------------
# 초기화
# -----------------------------
if st.button("🧹 전체 초기화"):
    st.session_state.portfolio = []
    st.success("포트폴리오가 초기화되었습니다.")

st.markdown("---")

# -----------------------------
# 자산 입력 (평단가 기준)
# -----------------------------
st.subheader("➕ 종목 입력 (평단가 기준)")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("종목명", placeholder="삼성전자")
    code = st.text_input("종목코드", placeholder="005930")
    asset_type = st.selectbox("자산군", ["주식", "ETF"])

with col2:
    qty = st.number_input("보유 수량", min_value=0.0, step=1.0)
    avg_price = st.number_input("평단가", min_value=0, step=100)
    current_price = st.number_input("현재가", min_value=0, step=100)

if st.button("종목 추가"):
    profit_rate = (
        (current_price - avg_price) / avg_price * 100
        if avg_price > 0 else 0
    )

    if profit_rate <= -10:
        signal = "🔵 추가매수 고려"
    elif profit_rate >= 20:
        signal = "🔴 분할매도 고려"
    else:
        signal = "🟡 보유"

    st.session_state.portfolio.append({
        "종목명": name,
        "종목코드": code,
        "자산군": asset_type,
        "보유수량": qty,
        "평단가": avg_price,
        "현재가": current_price,
        "평가금액": qty * current_price,
        "수익률(%)": round(profit_rate, 2),
        "매매신호": signal
    })

    st.success("종목이 추가되었습니다.")

# -----------------------------
# 포트폴리오 분석
# -----------------------------
if st.session_state.portfolio:
    df = pd.DataFrame(st.session_state.portfolio)

    st.markdown("---")
    st.subheader("📋 종목별 수익 현황 & 매매 신호")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # 수익률 분포
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
    # 요약 판단
    # -----------------------------
    avg_return = df["수익률(%)"].mean()

    st.markdown("---")
    st.subheader("🧠 포트폴리오 종합 판단")

    if avg_return >= 15:
        st.success("전체적으로 수익 구간입니다. 리스크 관리(분할매도)를 고려하세요.")
    elif avg_return <= -10:
        st.warning("손실 구간입니다. 추가매수 또는 리밸런싱 검토가 필요합니다.")
    else:
        st.info("중립 구간입니다. 추이를 관찰하세요.")
