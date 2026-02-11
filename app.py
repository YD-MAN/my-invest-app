import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")

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
tickers_input = st.text_input("종목코드 (쉼표 구분)", "AAPL, MSFT, NVDA")
buy_prices_input = st.text_input("평단가", "150, 300, 400")
quantities_input = st.text_input("수량", "10, 5, 3")

# ---------------------------
# 🔒 안전한 입력 파싱 함수
# ---------------------------
def safe_float_list(input_str):
    result = []
    for item in input_str.split(","):
        item = item.strip()
        try:
            result.append(float(item))
        except:
            result.append(0.0)
    return result

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
buy_prices = safe_float_list(buy_prices_input)
quantities = safe_float_list(quantities_input)

# 길이 보정 (짧으면 0으로 채움)
max_len = max(len(tickers), len(buy_prices), len(quantities))

while len(buy_prices) < max_len:
    buy_prices.append(0.0)

while len(quantities) < max_len:
    quantities.append(0.0)

while len(tickers) < max_len:
    tickers.append("")

# ---------------------------
# 📊 리스크 계산
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
    return int(np.clip(total, 0, 100))

# ---------------------------
# 🔍 데이터 처리
# ---------------------------
for i in range(max_len):

    ticker = tickers[i]
    if ticker == "":
        continue

    try:
        data = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
    except:
        st.warning(f"{ticker} 다운로드 실패")
        continue

    if data is None or data.empty:
        st.warning(f"{ticker} 데이터 없음")
        continue

    # MultiIndex 방지
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        st.warning(f"{ticker} 종가 없음")
        continue

    close = data["Close"].dropna()

    if len(close) < 5:
        st.warning(f"{ticker} 데이터 부족")
        continue

    current_price = float(close.iloc[-1])
    buy_price = float(buy_prices[i])
    quantity = float(quantities[i])

    if buy_price == 0:
        change_pct = 0
    else:
        change_pct = ((current_price - buy_price) / buy_price) * 100

    # 상승/하락 표시
    if change_pct > 0:
        change_html = f"<span class='up'>▲ +{change_pct:.2f}%</span>"
    elif change_pct < 0:
        change_html = f"<span class='down'>▼ {change_pct:.2f}%</span>"
    else:
        change_html = f"<span class='flat'>0.00%</span>"

    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 0 else 0.0

    ma_short = close.rolling(20).mean().iloc[-1]
    if len(close) >= 60:
        ma_long = close.rolling(60).mean().iloc[-1]
    else:
        ma_long = ma_short

    if ma_long == 0:
        trend_strength = 0
    else:
        trend_strength = (ma_short - ma_long) / ma_long

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
