import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="AI 포트폴리오 매니저",
    layout="centered"
)

st.title("📊 AI 포트폴리오 매니저")

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if "asset_history" not in st.session_state:
    st.session_state.asset_history = []

# -----------------------------
# 전체 초기화
# -----------------------------
if st.button("🧹 전체 초기화 (처음부터 다시 입력)"):
    st.session_state.portfolio = []
    st.session_state.asset_history = []
    st.success("모든 데이터가 초기화되었습니다.")

st.markdown("---")

# -----------------------------
# 자산 입력
# -----------------------------
st.subheader("➕ 자산 입력")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("종목명", placeholder="삼성전자 / 현금")
    code = st.text_input("종목코드", placeholder="005930 / CASH")

with col2:
    asset_type = st.selectbox("자산군", ["주식", "ETF", "현금"])
    qty = st.number_input("보유 수량", min_value=0.0, step=1.0)
    price = st.number_input("현재가 (원)", min_value=0, step=100)

if st.button("자산 추가"):
    st.session_state.portfolio.append({
        "종목명": name,
        "종목코드": code,
        "자산군": asset_type,
        "보유수량": qty,
        "현재가": price,
        "평가금액": qty * price
    })
    st.success("자산이 추가되었습니다.")

# -----------------------------
# 포트폴리오 테이블
# -----------------------------
if st.session_state.portfolio:
    df = pd.DataFrame(st.session_state.portfolio)

    st.markdown("---")
    st.subheader("📋 보유 자산 현황")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # 자산군별 요약 (파이차트 + 테이블)
    # -----------------------------
    asset_summary = (
        df.groupby("자산군")["평가금액"]
        .sum()
        .reset_index()
    )

    st.markdown("---")
    st.subheader("📊 자산군별 비중")

    fig1, ax1 = plt.subplots()
    ax1.pie(
        asset_summary["평가금액"],
        labels=asset_summary["자산군"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.axis("equal")

    st.pyplot(fig1)
    st.table(asset_summary)

# -----------------------------
# 월별 자산 기록
# -----------------------------
st.markdown("---")
st.subheader("📈 월별 자산 기록")

month = st.date_input("기준 월")
total_value = st.number_input(
    "해당 월 총자산 (원)",
    min_value=0,
    step=100000
)

if st.button("월별 자산 저장"):
    st.session_state.asset_history.append({
        "월": month.strftime("%Y-%m"),
        "총자산": total_value
    })
    st.success("월별 자산이 저장되었습니다.")

# -----------------------------
# 월별 자산 추이 그래프
# -----------------------------
if st.session_state.asset_history:
    hist_df = pd.DataFrame(st.session_state.asset_history)

    hist_df = (
        hist_df.groupby("월")["총자산"]
        .max()
        .reset_index()
        .sort_values("월")
    )

    st.markdown("---")
    st.subheader("📈 월별 자산 추이")

    fig2, ax2 = plt.subplots()
    ax2.plot(
        hist_df["월"],
        hist_df["총자산"],
        marker="o"
    )
    ax2.set_xlabel("월")
    ax2.set_ylabel("총자산 (원)")
    ax2.grid(True)

    st.pyplot(fig2)
