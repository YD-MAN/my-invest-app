import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 AI 포트폴리오 매니저 (안정화 리빌드 버전)")

# -------------------------
# 입력
# -------------------------
tickers_input = st.text_input("종목코드 (.KS 포함)", 
"005930.KS, NFLX, 360750.KS, QQQ, SCHD, 458730.KS, VT, NVDA, TSLA")

avg_input = st.text_input("평단가", 
"64260, 58385, 21380, 774567, 36992, 12360, 186885, 189419, 386170")

qty_input = st.text_input("보유 수량", 
"64, 4, 445, 9, 156, 152, 8, 5, 2")

tickers = [t.strip() for t in tickers_input.split(",")]
avg_prices = [float(x.strip()) for x in avg_input.split(",")]
quantities = [float(x.strip()) for x in qty_input.split(",")]

portfolio_rows = []
error_list = []

# -------------------------
# 종목별 처리
# -------------------------
for i, ticker in enumerate(tickers):

    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

    if df.empty or len(df) < 40:
        error_list.append(ticker)
        continue

    close = df["Close"].dropna()
    current_price = float(close.iloc[-1])

    avg_price = avg_prices[i]
    qty = quantities[i]

    value = current_price * qty
    profit = (current_price - avg_price) * qty
    return_pct = (current_price - avg_price) / avg_price * 100

    # 기술지표
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    trend30 = (close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100
    volatility = close.pct_change().std() * 100

    score = 50
    signal = "관망"

    if ma5.iloc[-1] > ma20.iloc[-1]:
        score += 20
        signal = "📈 매수 우위"
    else:
        score -= 10

    if trend30 > 5:
        score += 15
    elif trend30 < -5:
        score -= 15

    if volatility < 2:
        score += 10

    # -------------------------
    # LSTM (상위 2개 종목만 실행)
    # -------------------------
    pred_price = None

    if i < 2:  # 과부하 방지
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(close.values.reshape(-1,1))

            X, y = [], []
            lookback = 20

            for j in range(lookback, len(scaled)):
                X.append(scaled[j-lookback:j])
                y.append(scaled[j])

            X, y = np.array(X), np.array(y)

            model = Sequential()
            model.add(LSTM(32, input_shape=(X.shape[1],1)))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer="adam")
            model.fit(X, y, epochs=3, batch_size=16, verbose=0)

            last_seq = scaled[-lookback:]
            last_seq = np.reshape(last_seq,(1,lookback,1))
            pred_scaled = model.predict(last_seq, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        except:
            pred_price = None

    # fallback 선형회귀
    if pred_price is None:
        X_lr = np.arange(len(close)).reshape(-1,1)
        y_lr = close.values
        lr = LinearRegression().fit(X_lr,y_lr)
        pred_price = lr.predict([[len(close)+5]])[0]

    portfolio_rows.append([
        ticker, avg_price, current_price, qty,
        value, profit, return_pct,
        signal, trend30, score, pred_price
    ])

# -------------------------
# 오류 종목 표시
# -------------------------
if error_list:
    st.warning(f"⚠ 데이터 부족 또는 오류 종목: {', '.join(error_list)}")

if not portfolio_rows:
    st.error("모든 종목 데이터 로딩 실패")
    st.stop()

cols = [
    "Ticker","평단가","현재가","보유수량",
    "총평가금액","평가손익","수익률(%)",
    "매매신호","30일추세(%)","AI점수","예측가"
]

portfolio = pd.DataFrame(portfolio_rows, columns=cols)

total_asset = portfolio["총평가금액"].sum()
portfolio["비중(%)"] = portfolio["총평가금액"]/total_asset*100

# -------------------------
# 자산 현황
# -------------------------
st.header("💰 현재 자산")

col1,col2,col3 = st.columns(3)
col1.metric("총 자산", f"{total_asset:,.0f}")
col2.metric("총 손익", f"{portfolio['평가손익'].sum():,.0f}")
col3.metric("평균 수익률", f"{portfolio['수익률(%)'].mean():.2f}%")

st.dataframe(portfolio.round(2), use_container_width=True)

# -------------------------
# 리밸런싱
# -------------------------
st.header("🔁 자동 리밸런싱")

score_sum = portfolio["AI점수"].sum()
portfolio["목표비중(%)"] = portfolio["AI점수"]/score_sum*100
portfolio["목표금액"] = total_asset * portfolio["목표비중(%)"]/100
portfolio["조정수량"] = (portfolio["목표금액"] - portfolio["총평가금액"]) / portfolio["현재가"]

st.dataframe(portfolio[["Ticker","비중(%)","목표비중(%)","조정수량"]].round(2),
use_container_width=True)

# -------------------------
# 종합 판단
# -------------------------
best = portfolio.sort_values("AI점수",ascending=False).iloc[0]
worst = portfolio.sort_values("AI점수").iloc[0]

st.success(f"📌 매수 우선: {best['Ticker']} (점수 {best['AI점수']})")
st.warning(f"⚠ 비중 축소 고려: {worst['Ticker']} (점수 {worst['AI점수']})")
