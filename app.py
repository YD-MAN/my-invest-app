import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 기본 설정 (모바일 최적화)
# ===============================
st.set_page_config(
    page_title="📲 AI 포트폴리오",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📲 AI 포트폴리오 매니저")

# ===============================
# 1. 사이드바 입력
# ===============================
st.sidebar.header("➕ 자산 추가")

ticker = st.sidebar.text_input("종목 코드 (현금: CASH)")
qty = st.sidebar.number_input("수량", min_value=0.0, value=1.0)
avg_price = st.sidebar.number_input("평균 단가", min_value=0.0, value=0.0)
asset_type = st.sidebar.selectbox("자산군", ["주식", "ETF", "현금"])

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if st.sidebar.button("➕ 추가"):
    st.session_state.portfolio.append({
        "Ticker": ticker.upper(),
        "Qty": qty,
        "AvgPrice": avg_price,
        "Asset": asset_type
    })

if not st.session_state.portfolio:
    st.info("사이드바에서 자산을 추가해 주세요.")
    st.stop()

# ===============================
# 2. 현재 평가
# ===============================
rows = []
total_value = 0

for item in st.session_state.portfolio:
    try:
        if item["Asset"] == "현금":
            latest_price = 1
        else:
            data = yf.download(item["Ticker"], period="5d", progress=False)
            if data.empty:
                continue
            latest_price = float(data["Close"].iloc[-1])

        value = latest_price * item["Qty"]
        total_value += value

        rows.append({
            "종목": item["Ticker"],
            "자산군": item["Asset"],
            "평가금액": round(value, 2)
        })
    except:
        continue

df = pd.DataFrame(rows)

# ===============================
# 3. 자산군 요약
# ===============================
asset_summary = df.groupby("자산군")["평가금액"].sum().reset_index()
asset_summary["비중(%)"] = round(asset_summary["평가금액"] / total_value * 100, 2)

stock_ratio = asset_summary.loc[asset_summary["자산군"]=="주식","비중(%)"].values[0] if "주식" in asset_summary["자산군"].values else 0
etf_ratio   = asset_summary.loc[asset_summary["자산군"]=="ETF","비중(%)"].values[0] if "ETF" in asset_summary["자산군"].values else 0
cash_ratio  = asset_summary.loc[asset_summary["자산군"]=="현금","비중(%)"].values[0] if "현금" in asset_summary["자산군"].values else 0

# ===============================
# 4. 🧠 투자 성향 판별
# ===============================
max_concentration = df["평가금액"].max() / df["평가금액"].sum()

if stock_ratio >= 65 or max_concentration >= 40:
    investor_type = "🔥 공격형"
    comment = "수익 기회는 크지만 변동성 관리가 필요합니다."
elif cash_ratio >= 20 or etf_ratio >= 60:
    investor_type = "🧊 보수형"
    comment = "안정성은 높지만 수익 잠재력이 제한적입니다."
else:
    investor_type = "⚖️ 중립형"
    comment = "균형 잡힌 포트폴리오입니다."

# ===============================
# 5. 📲 홈 화면 요약 카드
# ===============================
st.subheader("📲 홈 요약")

c1, c2, c3, c4 = st.columns(4)

c1.metric("총 자산", f"{total_value:,.0f}원")
c2.metric("주식 비중", f"{stock_ratio:.1f}%")
c3.metric("투자 성향", investor_type)
c4.metric("AI 코멘트", comment)

st.markdown("---")

# ===============================
# 6. 자산군 파이차트
# ===============================
st.subheader("🏦 자산군 비중")

fig, ax = plt.subplots(figsize=(4,6))
ax.pie(
    asset_summary["평가금액"],
    labels=asset_summary["자산군"],
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
st.pyplot(fig)

# ===============================
# 7. 🤖 AI 리밸런싱 제안
# ===============================
st.subheader("🤖 AI 리밸런싱 제안")

TARGET = {"주식":50, "ETF":40, "현금":10}

for _, row in asset_summary.iterrows():
    target = TARGET[row["자산군"]]
    diff = row["비중(%)"] - target

    if diff > 5:
        st.warning(
            f"⚠ {row['자산군']} 비중 과다 "
            f"(현재 {row['비중(%)']}% / 목표 {target}%)"
        )
    elif diff < -5:
        st.info(
            f"ℹ {row['자산군']} 비중 부족 "
            f"(현재 {row['비중(%)']}% / 목표 {target}%)"
        )

# ===============================
# 8. 상세 테이블
# ===============================
st.subheader("📋 상세 포트폴리오")

if st.checkbox("📱 모바일 요약 테이블", value=True):
    st.dataframe(
        df[["종목", "자산군", "평가금액"]],
        use_container_width=True
    )
else:
    st.dataframe(df, use_container_width=True)
