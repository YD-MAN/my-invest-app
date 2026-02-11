import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 포트폴리오 관리", layout="wide")
st.title("📊 포트폴리오 관리 대시보드")

# ===============================
# 1. 입력 영역
# ===============================
st.sidebar.header("📌 종목 입력")

ticker = st.sidebar.text_input("종목 코드 (예: AAPL, 005930.KS)")
qty = st.sidebar.number_input("보유 수량", min_value=1, value=1)
avg_price = st.sidebar.number_input("평균 매수가", min_value=0.0, value=0.0)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if st.sidebar.button("➕ 종목 추가"):
    st.session_state.portfolio.append({
        "Ticker": ticker.upper(),
        "Qty": qty,
        "AvgPrice": avg_price
    })

# ===============================
# 2. 현재 포트폴리오 계산
# ===============================
rows = []
total_value = 0

for item in st.session_state.portfolio:
    try:
        data = yf.download(item["Ticker"], period="5d", progress=False)
        if data.empty:
            continue

        latest_price = float(data["Close"].iloc[-1])
        value = latest_price * item["Qty"]
        cost = item["AvgPrice"] * item["Qty"]
        pnl = float((value - cost) / cost * 100) if cost > 0 else 0

        total_value += value

        rows.append({
            "종목": item["Ticker"],
            "수량": item["Qty"],
            "평균단가": round(item["AvgPrice"], 2),
            "현재가": round(latest_price, 2),
            "평가금액": round(value, 2),
            "수익률(%)": round(pnl, 2)
        })
    except:
        continue

df = pd.DataFrame(rows)

# ===============================
# 3. 포트폴리오 테이블
# ===============================
st.subheader("📋 포트폴리오 현황")

if df.empty:
    st.info("종목을 추가해 주세요.")
    st.stop()

df["비중(%)"] = round(df["평가금액"] / df["평가금액"].sum() * 100, 2)
st.dataframe(df, use_container_width=True)

# ===============================
# 4. 종목별 비중 파이차트
# ===============================
st.subheader("📊 종목별 비중")

fig1, ax1 = plt.subplots()
ax1.pie(
    df["평가금액"],
    labels=df["종목"],
    autopct="%1.1f%%",
    startangle=90
)
ax1.axis("equal")
st.pyplot(fig1)

# ===============================
# 5. 📈 자산 추이 그래프
# ===============================
st.subheader("📈 자산 추이 (최근 3개월)")

portfolio_history = pd.DataFrame()

for item in st.session_state.portfolio:
    try:
        hist = yf.download(item["Ticker"], period="3mo", progress=False)
        if hist.empty:
            continue
        hist_value = hist["Close"] * item["Qty"]
        portfolio_history[item["Ticker"]] = hist_value
    except:
        continue

if not portfolio_history.empty:
    portfolio_history["Total"] = portfolio_history.sum(axis=1)

    fig2, ax2 = plt.subplots()
    ax2.plot(portfolio_history.index, portfolio_history["Total"])
    ax2.set_ylabel("총 자산 가치")
    ax2.set_xlabel("날짜")
    ax2.set_title("포트폴리오 자산 추이")

    st.pyplot(fig2)

# ===============================
# 6. 🔄 자동 리밸런싱 제안
# ===============================
st.subheader("🔄 자동 리밸런싱 제안")

TARGET_WEIGHT = 100 / len(df)

for _, row in df.iterrows():
    diff = row["비중(%)"] - TARGET_WEIGHT

    if diff > 5:
        reduce_amount = row["평가금액"] * diff / 100
        st.warning(
            f"⚠ {row['종목']} 비중 과다\n"
            f"- 현재 비중: {row['비중(%)']}%\n"
            f"- 목표 비중: {TARGET_WEIGHT:.1f}%\n"
            f"- 약 {reduce_amount:,.0f}원 축소 고려"
        )

# ===============================
# 7. 요약
# ===============================
st.subheader("💰 요약")
st.metric("총 평가금액", f"{total_value:,.0f} 원")
