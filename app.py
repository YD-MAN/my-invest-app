import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI 포트폴리오 매니저 Pro",
    layout="wide"
)

# ---------------------------
# 🎨 프리미엄 다크 스타일
# ---------------------------
st.markdown("""
<style>
body { background-color: #0F1115; color: #F2F2F2; }
.card {
    background-color: #181C22;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.up { color: #E10600; font-weight: bold; font-size:22px; }
.down { color: #1F4E9E; font-weight: bold; font-size:22px; }
.flat { color: #8A8F98; font-weight: bold; font-size:22px; }
.score { font-size:18px; font-weight:600; color:#FFD700; }
.risk-low { color:#4CAF50; font-weight:600;}
.risk-mid { color:#FFA500; font-weight:600;}
.risk-high { color:#FF5252; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI 포트폴리오 매니저 Pro")

# ---------------------------
# 📥 입력 영역
# ---------------------------
tickers_input = st.text_input("종목코드 (쉼표로 구분)", "AAPL, MSFT, NVDA")
buy_prices_input = st.text_input("평단가", "150, 300, 400")
quantities_input = st.text_input("수량", "10, 5, 3")

tickers = [t.strip().upper() for t in tickers_input.split(",")]
buy_prices = [float(x.strip()) for x in buy_prices_input.split(",")]
quantities = [float(x.strip()) for x in quantities_input.split(",")]

# 길이 체크
if not (len(tickers) == len(buy_prices) == len(quantities)):
    st.error("입력 데이터 길이가 맞지 않습니다.")
    st.stop()

# ---------------------------
# 📊 리스크 계산 함수
# ---------------------------
def calculate_risk(volatility):
    if volatility < 0.02:
        return "낮음", "risk-low"
    elif volatility < 0.05:
        return "보통", "risk-mid"
    else:
        return "높음", "risk-high"

# ---------------------------
# 📈 AI 점수 계산
# ---------------------------
def calculate_ai_score(trend_strength, volatility, momentum):
    trend_score = np.clip(trend_strength * 200, 0, 25)
    vol_score = np.clip((1 - volatility) * 30, 0, 30)
    momentum_score = np.clip(momentum * 100, 0, 25)
    diversification_score = 20
    total = trend_score + vol_score + momentum_score + diversification_score
    return round(np.clip(total, 0, 100))

# ---------------------------
# 🔍 데이터 처리
# ---------------------------
for i, ticker in enumerate(tickers):

    data = yf.download(ticker, period="3mo", progress=False)

    if data.empty:
        st.warning(f"{ticker} 데이터 불러오기 실패")
        continue

    close = data["Close"].dropna()

    if close.empty:
        continue

    current_price = float(close.iloc[-1])
    buy_price = buy_prices[i]
    quantity = quantities[i]

    change_pct = ((current_price - buy_price) / buy_price) * 100

    # 상승/하락 표시
    if change_pct > 0:
        change_html = f"<span class='up'>▲ +{change_pct:.2f}%</span>"
    elif change_pct < 0:
        change_html = f"<span class='down'>▼ {change_pct:.2f}%</span>"
    else:
        change_html = f"<span class='flat'>0.00%</span>"

    # 변동성
    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if not returns.empty else 0

    # 추세 강도
    ma_short = close.rolling(20).mean().iloc[-1]
    ma_long = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma_short
    trend_strength = (ma_short - ma_long) / ma_long if ma_long != 0 else 0

    # 모멘텀
    momentum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

    ai_score = calculate_ai_score(trend_strength, volatility, momentum)

    risk_label, risk_class = calculate_risk(volatility)

    # ---------------------------
    # 📱 모바일 카드 UI
    # ---------------------------
    st.markdown(f"""
    <div class="card">
        <h3>{ticker}</h3>
        <div style="font-size:16px; color:#A0A7B5;">
            현재가: ${current_price:.2f}
        </div>
        <div>{change_html}</div>
        <div class="score">AI 점수: {ai_score}점</div>
        <div class="{risk_class}">리스크: {risk_label}</div>
    </div>
    """, unsafe_allow_html=True)
