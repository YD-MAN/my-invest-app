import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import requests
from email.mime.text import MIMEText

st.set_page_config(page_title="장기 투자 비상 경보 시스템", layout="wide")
st.title("🚨 장기 투자 관리 & 비상 경보 시스템")

# =====================
# 알림 함수
# =====================
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = st.secrets["EMAIL_USER"]
        msg["To"] = st.secrets["EMAIL_USER"]

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASSWORD"])
        server.send_message(msg)
        server.quit()
    except:
        pass


def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{st.secrets['TELEGRAM_TOKEN']}/sendMessage"
        data = {
            "chat_id": st.secrets["TELEGRAM_CHAT_ID"],
            "text": message
        }
        requests.post(url, data=data)
    except:
        pass


# =====================
# 데이터 로딩
# =====================
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="2y", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Close" not in df.columns:
        df["Close"] = df["Adj Close"]

    return df.dropna()


# =====================
# 지표 계산
# =====================
def calculate_indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()


# =====================
# Emergency 판단
# =====================
def emergency_check(df):
    latest = df.iloc[-1]
    recent_vol = df["Close"].pct_change().rolling(3).std().iloc[-1]
    past_vol = df["Close"].pct_change().rolling(60).std().iloc[-1]

    structure_break = (
        latest["Close"] < latest["MA120"]
        and latest["MA20"] < latest["MA60"] < latest["MA120"]
    )

    volatility_shock = recent_vol > past_vol * 2

    if structure_break and volatility_shock:
        return True, "장기 추세 붕괴 + 변동성 쇼크"
    return False, None


# =====================
# 사용자 입력
# =====================
st.sidebar.header("📥 보유 종목 입력")

ticker = st.sidebar.text_input("티커 (예: QQQ, BTC-USD)")
avg_price = st.sidebar.number_input("평균 매입가", min_value=0.0)
quantity = st.sidebar.number_input("보유 수량", min_value=0.0)

if ticker:
    df = calculate_indicators(load_data(ticker))
    current_price = df["Close"].iloc[-1]
    pnl = (current_price - avg_price) / avg_price * 100

    st.metric("현재가", f"{current_price:,.2f}")
    st.metric("수익률", f"{pnl:.2f}%")

    # Emergency 판단
    emergency, reason = emergency_check(df)

    st.line_chart(df.set_index("Date")[["Close", "MA20", "MA60", "MA120"]])

    if emergency:
        st.error(f"🚨 비상 경보 발생: {reason}")

        message = f"""
🚨 EMERGENCY ALERT 🚨
종목: {ticker}
사유: {reason}
현재가: {current_price:,.2f}
수익률: {pnl:.2f}%

계좌 보호를 위한 방어적 대응을 권고합니다.
"""

        send_email("🚨 투자 비상 경보", message)
        send_telegram(message)

    else:
        st.success("🟢 장기 투자 전제 유지 중 (비상 신호 없음)")
