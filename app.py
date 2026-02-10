import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import requests
from datetime import datetime

# =============================
# 🔐 비상 메시지 설정
# =============================
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except:
        pass


# =============================
# 📥 데이터 로딩
# =============================
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="1y")
    df.dropna(inplace=True)
    return df


# =============================
# 📊 지표 계산
# =============================
def calculate_indicators(df):
    df["MA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    df["MA60"] = SMAIndicator(df["Close"], 60).sma_indicator()
    df["MA120"] = SMAIndicator(df["Close"], 120).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()
    return df


# =============================
# 🚨 종목 Emergency 판단
# =============================
def emergency_check(df):
    latest = df.iloc[-1]

    trend_break = (
        latest["Close"] < latest["MA120"] and
        latest["MA20"] < latest["MA60"] < latest["MA120"]
    )

    volatility_shock = latest["Volatility"] > df["Volatility"].quantile(0.9)

    if trend_break and volatility_shock:
        return True, "장기 추세 붕괴 + 변동성 쇼크"
    elif trend_break:
        return True, "장기 추세 붕괴"
    else:
        return False, "장기 투자 전제 유지"


# =============================
# 🧮 Streamlit UI
# =============================
st.set_page_config(page_title="장기 투자 포트폴리오 매니저", layout="wide")

st.title("📊 장기 투자 포트폴리오 매니저")
st.caption("위험할 때만 알려주는 비상 경보 시스템")

# -----------------------------
# 📋 포트폴리오 입력
# -----------------------------
st.subheader("📋 포트폴리오 테이블")

portfolio_df = st.data_editor(
    pd.DataFrame({
        "ticker": ["QQQ", "SPY", "BTC-USD"],
        "avg_price": [380.0, 450.0, 52000.0],
        "quantity": [10, 5, 0.2]
    }),
    num_rows="dynamic",
    use_container_width=True
)

# -----------------------------
# 🔄 분석 실행
# -----------------------------
if st.button("🔍 포트폴리오 분석 실행"):

    results = []
    emergency_count = 0
    total_value = 0
    total_cost = 0

    for _, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        avg_price = row["avg_price"]
        qty = row["quantity"]

        try:
            df = calculate_indicators(load_data(ticker))
            current_price = df["Close"].iloc[-1]

            pnl_pct = (current_price - avg_price) / avg_price * 100
            value = current_price * qty
            cost = avg_price * qty

            emergency, reason = emergency_check(df)
            if emergency:
                emergency_count += 1

            results.append({
                "종목": ticker,
                "현재가": round(current_price, 2),
                "수익률(%)": round(pnl_pct, 2),
                "평가금액": int(value),
                "상태": "🔴 Emergency" if emergency else "🟢 Normal",
                "판단": reason
            })

            total_value += value
            total_cost += cost

        except Exception as e:
            st.error(f"{ticker} 데이터 오류")

    result_df = pd.DataFrame(results)

    # -----------------------------
    # 📊 포트폴리오 상태 판단
    # -----------------------------
    portfolio_pnl = (total_value - total_cost) / total_cost * 100
    emergency_ratio = emergency_count / len(result_df)

    portfolio_emergency = (
        emergency_ratio >= 0.4 or
        (portfolio_pnl <= -10 and emergency_count > 0)
    )

    # -----------------------------
    # 🚨 포트폴리오 비상 알림
    # -----------------------------
    if portfolio_emergency:
        st.error("🚨 포트폴리오 비상 상태")
        alert_msg = f"""
🚨 포트폴리오 비상 경보
시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Emergency 종목 비율: {emergency_ratio:.0%}
포트폴리오 수익률: {portfolio_pnl:.2f}%
"""
        send_telegram_alert(alert_msg)
    else:
        st.success("🟢 포트폴리오 정상 상태")

    # -----------------------------
    # 📈 결과 출력
    # -----------------------------
    st.subheader("📈 분석 결과")
    st.dataframe(result_df, use_container_width=True)

    st.markdown("### 💰 포트폴리오 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 평가금액", f"{int(total_value):,}")
    col2.metric("총 수익률", f"{portfolio_pnl:.2f}%")
    col3.metric("Emergency 종목 수", emergency_count)
