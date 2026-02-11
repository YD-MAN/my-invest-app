import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🤖 AI 포트폴리오", layout="wide")
st.title("🤖 AI 기반 포트폴리오 관리")

# ===============================
# 1. 사이드바 입력
# ===============================
st.sidebar.header("📌 자산 추가")

ticker = st.sidebar.text_input("종목 코드 (현금은 CASH)")
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
st.subheader("📋 포트폴리오")
st.dataframe(df, use_container_width=True)

# ===============================
# 3. 자산군 비중
# ===============================
asset_summary = df.groupby("자산군")["평가금액"].sum().reset_index()
asset_summary["비중(%)"] = round(asset_summary["평가금액"] / total_value * 100, 2)

st.subheader("🏦 자산군 비중")
st.dataframe(asset_summary, use_container_width=True)

# ===============================
# 4. 🤖 AI 리밸런싱 엔진
# ===============================
st.subheader("🤖 AI 리밸런싱 전략 제안")

TARGET = {"주식": 50, "ETF": 40, "현금": 10}

for _, row in asset_summary.iterrows():
    target = TARGET[row["자산군"]]
    diff = row["비중(%)"] - target

    if diff > 5:
        reduce = total_value * diff / 100
        st.warning(
            f"⚠ {row['자산군']} 비중 과다\n"
            f"- 현재 {row['비중(%)']}% / 목표 {target}%\n"
            f"- 약 {reduce:,.0f}원 축소 고려"
        )

    elif diff < -5:
        add = total_value * abs(diff) / 100
        st.info(
            f"ℹ {row['자산군']} 비중 부족\n"
            f"- 현재 {row['비중(%)']}% / 목표 {target}%\n"
            f"- 약 {add:,.0f}원 추가 고려"
        )

# ===============================
# 5. 종목 집중도 검사
# ===============================
st.subheader("🔍 종목 집중도 분석")

for asset in df["자산군"].unique():
    subset = df[df["자산군"] == asset]
    if subset["평가금액"].max() / subset["평가금액"].sum() > 0.4:
        st.error(f"❗ {asset} 자산군 내 특정 종목 비중 과도")
