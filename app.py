# app.py
# =========================================================
# AI 포트폴리오 매니저 Pro (최종)
# - URL + localStorage 하이브리드 입력 저장/복원(완전 자동)
# - 원화 평단가 입력 + 국내/해외 통화 표시 + 원화 손익
# - 환율 안정화 + 수동 환율 오버라이드
# - RandomForest 상승확률 + 뉴스 감성(ETF/지수 키워드 대체 + NewsAPI 보강)
# - 한국어 감성(Transformer) 가능할 때 자동 적용
# - 포트폴리오 합산(총 평가금액/손익/비중/손익 차트) - Streamlit 내장 차트(추가 설치 無)
# =========================================================

import json
import re
from urllib.parse import quote_plus, unquote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ localStorage용(없어도 앱이 죽지 않게 try/except)
HAS_JS_EVAL = False
try:
    from streamlit_js_eval import streamlit_js_eval
    HAS_JS_EVAL = True
except Exception:
    HAS_JS_EVAL = False

# ---------------------------
# Streamlit 기본
# ---------------------------
st.set_page_config(page_title="AI 포트폴리오 매니저 Pro", layout="wide")
st.title("📊 AI 포트폴리오 매니저 Pro")

# =========================================================
# 0) URL + localStorage 하이브리드 저장/복원
# =========================================================

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, 005930.KS, BTC-KRW"
DEFAULT_BUYPRICES = "150000, 300000, 400000, 70000, 50000000"
DEFAULT_QTYS = "10, 5, 3, 10, 0.01"

LS_KEY = "ai_portfolio_manager_pro_v1"  # localStorage key (버전 바꾸면 새로 저장됨)


def get_qp() -> dict:
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    return st.experimental_get_query_params()


def set_qp(**kwargs):
    safe = {k: v for k, v in kwargs.items() if v is not None}
    if hasattr(st, "query_params"):
        st.query_params.clear()
        for k, v in safe.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**safe)


def qp_get_str(qp: dict, key: str, default: str) -> str:
    if key not in qp:
        return default
    v = qp[key]
    if isinstance(v, list):
        v = v[0] if len(v) else default
    if v is None or v == "":
        return default
    try:
        return unquote_plus(str(v))
    except Exception:
        return str(v)


def enc(s: str) -> str:
    return quote_plus(s)


def mark_dirty():
    st.session_state["__qp_dirty__"] = True


def request_reset():
    st.session_state["__reset_requested__"] = True


def request_apply_loaded_settings(payload: dict):
    st.session_state["__loaded_settings__"] = payload
    st.session_state["__apply_loaded_settings__"] = True


def load_from_localstorage() -> dict:
    """
    localStorage에서 JSON 문자열을 읽어 dict로 반환.
    streamlit-js-eval이 없으면 {} 반환.
    """
    if not HAS_JS_EVAL:
        return {}
    try:
        raw = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{LS_KEY}')",
            want_output=True,
            key="LS_GET",
        )
        if raw is None:
            return {}
        if isinstance(raw, str) and raw.strip() == "":
            return {}
        # raw가 JSON 문자열일 수 있음
        try:
            return json.loads(raw)
        except Exception:
            return {}
    except Exception:
        return {}


def save_to_localstorage(payload: dict):
    """
    localStorage에 JSON 저장.
    streamlit-js-eval이 없으면 아무 것도 안 함.
    """
    if not HAS_JS_EVAL:
        return
    try:
        # JS 문자열 안전하게 넣기 위해 JSON을 다시 dumps
        s = json.dumps(payload, ensure_ascii=False)
        # JS 문자열 리터럴로 안전하게 넣기(따옴표 이스케이프)
        s_js = s.replace("\\", "\\\\").replace("'", "\\'")
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('{LS_KEY}', '{s_js}')",
            want_output=False,
            key="LS_SET",
        )
    except Exception:
        pass


# 플래그 초기화
if "__qp_dirty__" not in st.session_state:
    st.session_state["__qp_dirty__"] = False
if "__reset_requested__" not in st.session_state:
    st.session_state["__reset_requested__"] = False
if "__apply_loaded_settings__" not in st.session_state:
    st.session_state["__apply_loaded_settings__"] = False
if "__loaded_settings__" not in st.session_state:
    st.session_state["__loaded_settings__"] = {}
if "__ls_bootstrapped__" not in st.session_state:
    st.session_state["__ls_bootstrapped__"] = False

# 1) URL 읽기
qp = get_qp()
url_has_any = ("tickers" in qp) or ("buy" in qp) or ("qty" in qp)

init_tickers = qp_get_str(qp, "tickers", DEFAULT_TICKERS)
init_buy = qp_get_str(qp, "buy", DEFAULT_BUYPRICES)
init_qty = qp_get_str(qp, "qty", DEFAULT_QTYS)

# 2) URL이 비어있거나 홈스크린에서 유실된 경우 → localStorage에서 복원(최초 1회만)
#    - URL이 이미 있으면 URL을 우선(공유 링크/북마크 일관성)
if (not url_has_any) and (not st.session_state["__ls_bootstrapped__"]):
    ls_data = load_from_localstorage()
    if isinstance(ls_data, dict) and ls_data.get("tickers_input") and ls_data.get("buy_prices_input") and ls_data.get("quantities_input"):
        init_tickers = str(ls_data["tickers_input"])
        init_buy = str(ls_data["buy_prices_input"])
        init_qty = str(ls_data["quantities_input"])
        # URL도 같이 맞춰주면 이후 공유/새로고침 일관성이 좋아짐
        set_qp(tickers=enc(init_tickers), buy=enc(init_buy), qty=enc(init_qty))
    st.session_state["__ls_bootstrapped__"] = True

# 입력 state 초기값(최초 1회)
if "tickers_input" not in st.session_state:
    st.session_state["tickers_input"] = init_tickers
if "buy_prices_input" not in st.session_state:
    st.session_state["buy_prices_input"] = init_buy
if "quantities_input" not in st.session_state:
    st.session_state["quantities_input"] = init_qty

# Reset(콜백 밖)
if st.session_state["__reset_requested__"]:
    for k in ["tickers_input", "buy_prices_input", "quantities_input"]:
        if k in st.session_state:
            del st.session_state[k]

    # URL reset
    set_qp(tickers=enc(DEFAULT_TICKERS), buy=enc(DEFAULT_BUYPRICES), qty=enc(DEFAULT_QTYS))

    # localStorage reset도 같이
    save_to_localstorage(
        {
            "tickers_input": DEFAULT_TICKERS,
            "buy_prices_input": DEFAULT_BUYPRICES,
            "quantities_input": DEFAULT_QTYS,
        }
    )

    st.session_state["__qp_dirty__"] = False
    st.session_state["__reset_requested__"] = False
    st.rerun()

# JSON 업로드 적용(콜백 밖)
if st.session_state["__apply_loaded_settings__"]:
    loaded = st.session_state.get("__loaded_settings__", {}) or {}
    t = str(loaded.get("tickers_input", DEFAULT_TICKERS))
    b = str(loaded.get("buy_prices_input", DEFAULT_BUYPRICES))
    q = str(loaded.get("quantities_input", DEFAULT_QTYS))

    for k in ["tickers_input", "buy_prices_input", "quantities_input"]:
        if k in st.session_state:
            del st.session_state[k]

    set_qp(tickers=enc(t), buy=enc(b), qty=enc(q))
    save_to_localstorage({"tickers_input": t, "buy_prices_input": b, "quantities_input": q})

    st.session_state["__apply_loaded_settings__"] = False
    st.session_state["__loaded_settings__"] = {}
    st.rerun()

# ---------------------------
# 입력 UI
# ---------------------------
st.subheader("입력 (URL + localStorage 자동 저장)")
col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

with col1:
    st.text_input("종목코드 (쉼표 구분)", key="tickers_input", on_change=mark_dirty)

with col2:
    st.text_input("평단가 (원화 기준, 쉼표 구분)", key="buy_prices_input", on_change=mark_dirty)

with col3:
    st.text_input("수량 (쉼표 구분)", key="quantities_input", on_change=mark_dirty)

with col4:
    st.write("")
    st.write("")
    st.button("Reset", on_click=request_reset)

# URL 자동 저장(dirty일 때만)
if st.session_state["__qp_dirty__"]:
    desired_t = enc(st.session_state["tickers_input"])
    desired_b = enc(st.session_state["buy_prices_input"])
    desired_q = enc(st.session_state["quantities_input"])

    current = get_qp()
    cur_t = current.get("tickers", "")
    cur_b = current.get("buy", "")
    cur_q = current.get("qty", "")

    if isinstance(cur_t, list):
        cur_t = cur_t[0] if len(cur_t) else ""
    if isinstance(cur_b, list):
        cur_b = cur_b[0] if len(cur_b) else ""
    if isinstance(cur_q, list):
        cur_q = cur_q[0] if len(cur_q) else ""

    if (cur_t != desired_t) or (cur_b != desired_b) or (cur_q != desired_q):
        set_qp(tickers=desired_t, buy=desired_b, qty=desired_q)

    st.session_state["__qp_dirty__"] = False

# ✅ localStorage는 매번(혹은 값 변경 직후) 저장해도 부담이 작아서 항상 최신으로 맞춤
save_to_localstorage(
    {
        "tickers_input": st.session_state["tickers_input"],
        "buy_prices_input": st.session_state["buy_prices_input"],
        "quantities_input": st.session_state["quantities_input"],
    }
)

st.caption("✅ URL과 브라우저(localStorage)에 동시에 저장됩니다. 모바일 홈스크린(북마크)에서도 입력 유지가 훨씬 안정적입니다.")

# =========================================================
# 1) 파싱
# =========================================================

tickers_input = st.session_state["tickers_input"]
buy_prices_input = st.session_state["buy_prices_input"]
quantities_input = st.session_state["quantities_input"]


def safe_float_list(s: str) -> list[float]:
    out: list[float] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            out.append(0.0)
            continue
        try:
            out.append(float(item))
        except Exception:
            out.append(0.0)
    return out


tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
buy_prices_krw = safe_float_list(buy_prices_input)
quantities = safe_float_list(quantities_input)

max_len = max(len(tickers), len(buy_prices_krw), len(quantities))
while len(tickers) < max_len:
    tickers.append("")
while len(buy_prices_krw) < max_len:
    buy_prices_krw.append(0.0)
while len(quantities) < max_len:
    quantities.append(0.0)

# =========================================================
# 2) 유틸
# =========================================================

def extract_close_series(data: pd.DataFrame) -> pd.Series | None:
    if data is None or len(data) == 0:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return None
        close_part = data.xs("Close", axis=1, level=0, drop_level=True)
        if isinstance(close_part, pd.DataFrame):
            if close_part.shape[1] == 0:
                return None
            close = close_part.iloc[:, 0]
        else:
            close = close_part
    else:
        if "Close" not in data.columns:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            close = close.iloc[:, 0]

    if not isinstance(close, pd.Series):
        try:
            close = close.squeeze()
        except Exception:
            return None
        if not isinstance(close, pd.Series):
            return None

    close = close.dropna()
    if len(close) == 0:
        return None
    return close


def is_korea_ticker(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.endswith(".KO")


def is_crypto_ticker(ticker: str) -> bool:
    return "-" in ticker and (ticker.endswith("-USD") or ticker.endswith("-KRW"))


def fmt_krw(x: float) -> str:
    return f"₩{x:,.0f}"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"

# =========================================================
# 3) 환율(USD/KRW) 안정화 + 수동 오버라이드
# =========================================================

@st.cache_data(ttl=60 * 30)
def try_fetch_usdkrw() -> tuple[float, str]:
    try:
        fx = yf.download("KRW=X", period="1mo", progress=False)
        c = extract_close_series(fx)
        if c is not None and len(c) > 0:
            v = float(c.iloc[-1])
            if v > 0:
                return v, "KRW=X(download)"
    except Exception:
        pass

    try:
        fx = yf.Ticker("KRW=X").history(period="1mo")
        c = extract_close_series(fx)
        if c is not None and len(c) > 0:
            v = float(c.iloc[-1])
            if v > 0:
                return v, "KRW=X(history)"
    except Exception:
        pass

    try:
        fx = yf.download("USDKRW=X", period="1mo", progress=False)
        c = extract_close_series(fx)
        if c is not None and len(c) > 0:
            v = float(c.iloc[-1])
            if v > 0:
                return v, "USDKRW=X(download)"
    except Exception:
        pass

    return 0.0, "fail"


usdkrw_auto, usdkrw_src = try_fetch_usdkrw()

with st.sidebar:
    st.header("💱 환율 설정")
    manual_fx = st.number_input(
        "USD/KRW 수동 입력(0이면 자동)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="자동 환율이 실패하거나 0이면 직접 입력하세요.",
    )

if manual_fx > 0:
    usdkrw = float(manual_fx)
    usdkrw_src_final = "manual"
else:
    usdkrw = float(usdkrw_auto)
    usdkrw_src_final = usdkrw_src

if usdkrw == 0.0:
    st.warning("USD/KRW 환율을 가져오지 못했습니다. 해외 종목 환산 KRW가 0으로 표시될 수 있어요. (사이드바 수동 환율 입력 가능)")
else:
    st.info(f"USD/KRW 환율: {usdkrw:,.2f}  (source: {usdkrw_src_final})")

# =========================================================
# 4) 기술지표 + 점수
# =========================================================

def calculate_risk(vol: float) -> str:
    if vol < 0.02:
        return "낮음"
    if vol < 0.05:
        return "보통"
    return "높음"


def base_ai_score(trend: float, vol: float, mom: float) -> float:
    score = (trend * 200.0) + ((1.0 - vol) * 30.0) + (mom * 100.0) + 20.0
    return float(score)


def upgraded_ai_score(trend: float, vol: float, mom: float, prob_up: float, senti: float) -> int:
    rf_component = (prob_up - 0.5) * 50.0
    news_component = senti * 10.0
    s = base_ai_score(trend, vol, mom) + rf_component + news_component
    return int(np.clip(s, 0, 100))


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# =========================================================
# 5) RandomForest 상승확률
# =========================================================

@st.cache_data(ttl=60 * 60)
def rf_up_probability(ticker: str) -> tuple[float, dict]:
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
    except Exception as e:
        return 0.0, {"status": f"download_fail: {e}"}

    close = extract_close_series(df)
    if close is None or len(close) < 140:
        return 0.0, {"status": "not_enough_data"}

    s = close.copy()
    ret1 = s.pct_change()
    vol10 = ret1.rolling(10).std()
    vol20 = ret1.rolling(20).std()
    ma5 = s.rolling(5).mean()
    ma20 = s.rolling(20).mean()
    ma60 = s.rolling(60).mean()

    feat = pd.DataFrame(
        {
            "ret1": ret1,
            "vol10": vol10,
            "vol20": vol20,
            "ma_gap_5_20": (ma5 - ma20) / ma20.replace(0, np.nan),
            "ma_gap_20_60": (ma20 - ma60) / ma60.replace(0, np.nan),
            "rsi14": rsi(s, 14),
            "mom20": (s - s.shift(20)) / s.shift(20).replace(0, np.nan),
        }
    ).dropna()

    y = (s.shift(-1).loc[feat.index] > s.loc[feat.index]).astype(int)

    X = feat.iloc[:-1]
    y = y.iloc[:-1]
    X_last = feat.iloc[[-1]]

    if len(X) < 120:
        return 0.0, {"status": "not_enough_train_rows"}

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, _ = list(tscv.split(X))[-1]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    prob_up = float(model.predict_proba(X_last)[0, 1])
    return prob_up, {"status": "ok", "train_rows": int(len(X_train))}

# =========================================================
# 6) 뉴스 감성: ETF/지수 키워드 대체 + NewsAPI + 한국어(가능 시)
# =========================================================

analyzer = SentimentIntensityAnalyzer()

with st.sidebar:
    st.header("📰 뉴스/감성 옵션")
    ENABLE_NEWSAPI = st.toggle("NewsAPI 보강 사용", value=True)
    ENABLE_KO_TRANSFORMER = st.toggle("한국어 감성(Transformer) 사용", value=True)
    st.caption("한국어 감성은 무거울 수 있어요. 느리면 끄세요.")

KOREAN_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


def get_newsapi_key() -> str:
    try:
        return st.secrets.get("NEWSAPI_KEY", "")
    except Exception:
        return ""


NEWSAPI_KEY = get_newsapi_key()


@st.cache_data(ttl=60 * 60)
def guess_quote_type(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        qt = (info.get("quoteType") or "").upper()
        return qt if qt else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def is_etf_or_index(ticker: str) -> bool:
    if ticker.startswith("^"):
        return True
    qt = guess_quote_type(ticker)
    return qt in {"ETF", "INDEX", "MUTUALFUND"}


@st.cache_data(ttl=60 * 30)
def get_keyword_for_news(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        name = str(name).strip()
        if name:
            return name
    except Exception:
        pass
    return ticker[1:] if ticker.startswith("^") else ticker


@st.cache_data(ttl=60 * 20)
def newsapi_everything(query: str, language: str | None, page_size: int) -> tuple[list[str], dict]:
    if not (ENABLE_NEWSAPI and NEWSAPI_KEY):
        if ENABLE_NEWSAPI and not NEWSAPI_KEY:
            return [], {"status": "no_newsapi_key"}
        return [], {"status": "newsapi_disabled"}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    if language:
        params["language"] = language

    try:
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get("status") != "ok":
            return [], {"status": "newsapi_error", "message": data.get("message", "")}

        titles = []
        for a in data.get("articles", [])[:page_size]:
            t = (a.get("title") or "").strip()
            if t:
                titles.append(t)

        if not titles:
            return [], {"status": "newsapi_no_titles"}
        return titles, {"status": "ok", "count": len(titles)}
    except Exception as e:
        return [], {"status": "newsapi_fail", "error": str(e)}


def detect_language_simple(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


@st.cache_resource
def get_ko_sentiment_pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=KOREAN_SENTIMENT_MODEL)


def stars_to_compound(label: str) -> float:
    m = re.search(r"(\d)", label)
    if not m:
        return 0.0
    stars = int(m.group(1))
    return (stars - 3) / 2.0


def sentiment_compound_from_titles(titles: list[str]) -> tuple[float, str]:
    if not titles:
        return 0.0, "중립"

    titles = titles[:10]
    joined = " ".join(titles[:3])
    lang = detect_language_simple(joined)

    if lang == "en":
        vals = [float(analyzer.polarity_scores(t).get("compound", 0.0)) for t in titles]
        comp = float(np.mean(vals)) if vals else 0.0
    else:
        if not ENABLE_KO_TRANSFORMER:
            comp = 0.0
        else:
            try:
                pipe = get_ko_sentiment_pipe()
                preds = pipe(titles)
                vals = [stars_to_compound(str(p.get("label", ""))) for p in preds]
                comp = float(np.mean(vals)) if vals else 0.0
            except Exception:
                comp = 0.0

    if comp > 0.05:
        lab = "긍정"
    elif comp < -0.05:
        lab = "부정"
    else:
        lab = "중립"
    return comp, lab


@st.cache_data(ttl=60 * 20)
def news_sentiment_score(ticker: str, max_items: int = 12) -> tuple[float, list[str], dict]:
    try:
        news = getattr(yf.Ticker(ticker), "news", None)
    except Exception:
        news = None

    titles = []
    if news:
        for item in news[:max_items]:
            title = (item.get("title") or "").strip()
            if title:
                titles.append(title)

    if len(titles) == 0:
        kw = get_keyword_for_news(ticker)

        if is_etf_or_index(ticker):
            t_en, dbg_en = newsapi_everything(kw, "en", max_items)
            if t_en:
                comp, lab = sentiment_compound_from_titles(t_en)
                return comp, t_en, {"status": "fallback_newsapi_etf_index", "keyword": kw, "lang": "en", "label": lab, **dbg_en}

            t_ko, dbg_ko = newsapi_everything(kw, "ko", max_items)
            if t_ko:
                comp, lab = sentiment_compound_from_titles(t_ko)
                return comp, t_ko, {"status": "fallback_newsapi_etf_index", "keyword": kw, "lang": "ko", "label": lab, **dbg_ko}

            return 0.0, [], {"status": "no_news_after_fallback", "keyword": kw, "quoteType": guess_quote_type(ticker)}

        USE_NEWSAPI_FOR_STOCKS_TOO = True
        if USE_NEWSAPI_FOR_STOCKS_TOO:
            t_en, dbg_en = newsapi_everything(kw, "en", max_items)
            if t_en:
                comp, lab = sentiment_compound_from_titles(t_en)
                return comp, t_en, {"status": "fallback_newsapi_stock", "keyword": kw, "lang": "en", "label": lab, **dbg_en}

            t_ko, dbg_ko = newsapi_everything(kw, "ko", max_items)
            if t_ko:
                comp, lab = sentiment_compound_from_titles(t_ko)
                return comp, t_ko, {"status": "fallback_newsapi_stock", "keyword": kw, "lang": "ko", "label": lab, **dbg_ko}

            return 0.0, [], {"status": "no_news_after_fallback", "keyword": kw, "quoteType": guess_quote_type(ticker)}

        return 0.0, [], {"status": "no_valid_titles"}

    comp, lab = sentiment_compound_from_titles(titles)
    return comp, titles, {"status": "ok", "source": "yfinance", "label": lab, "count": len(titles)}

# =========================================================
# 7) 분석 + 포트폴리오 합산/차트
# =========================================================

st.divider()
st.subheader("분석 결과")

rows = []

for i in range(max_len):
    ticker = tickers[i]
    if ticker == "":
        continue

    buy_price_krw = float(buy_prices_krw[i])
    qty = float(quantities[i])

    try:
        data = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        continue

    close = extract_close_series(data)
    if close is None or len(close) < 5:
        st.warning(f"{ticker} 데이터 부족/종가 없음")
        continue

    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if len(returns) > 0 else 0.0

    ma20_val = close.rolling(20).mean().iloc[-1]
    ma60_val = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20_val
    ma20 = float(ma20_val) if pd.notna(ma20_val) else np.nan
    ma60 = float(ma60_val) if pd.notna(ma60_val) else np.nan
    trend = 0.0 if (np.isnan(ma20) or np.isnan(ma60) or ma60 == 0.0) else (ma20 - ma60) / ma60

    current_native = float(close.iloc[-1])
    first_native = float(close.iloc[0])
    momentum = 0.0 if first_native == 0.0 else (current_native - first_native) / first_native

    risk = calculate_risk(volatility)

    if is_korea_ticker(ticker) or (is_crypto_ticker(ticker) and ticker.endswith("-KRW")):
        current_krw = current_native
        currency_label = "KRW"
        fx_line = ""
        price_line = f"현재가: {fmt_krw(current_krw)}"
    else:
        current_usd = current_native
        current_krw = current_usd * usdkrw if usdkrw > 0 else 0.0
        currency_label = "USD→KRW"
        fx_line = f"환율(USD/KRW): {usdkrw:,.2f}  (source: {usdkrw_src_final})" if usdkrw > 0 else "환율(USD/KRW): 없음(수동 입력 필요)"
        price_line = f"현재가: {fmt_usd(current_usd)}  (환산 {fmt_krw(current_krw)})"

    change_pct = 0.0 if buy_price_krw == 0.0 else ((current_krw - buy_price_krw) / buy_price_krw) * 100.0
    eval_krw = current_krw * qty
    pnl_krw = (current_krw - buy_price_krw) * qty if buy_price_krw != 0.0 else 0.0

    color = "red" if change_pct > 0 else ("blue" if change_pct < 0 else "gray")
    arrow = "▲" if change_pct > 0 else ("▼" if change_pct < 0 else "")

    prob_up, rf_debug = rf_up_probability(ticker)
    senti, titles, news_debug = news_sentiment_score(ticker)
    ai_score = upgraded_ai_score(trend, volatility, momentum, prob_up, senti)

    prob_pct = prob_up * 100.0
    senti_label = "긍정" if senti > 0.05 else ("부정" if senti < -0.05 else "중립")

    news_line = ""
    if titles:
        news_line = f"- 뉴스 감성(제목 기반): **{senti_label} ({senti:+.2f})**\n"

    st.markdown(
        f"""
---
### {ticker}  <span style="font-size:14px; color:#666;">[{currency_label}]</span>

{price_line}  
{fx_line}  
평단가(원화 입력): {fmt_krw(buy_price_krw)} / 수량: {qty:g}  
평가금액(원화): {fmt_krw(eval_krw)}  
평가손익(원화): {fmt_krw(pnl_krw)}  

<span style='color:{color}; font-size:20px; font-weight:bold;'>
{arrow} 수익률(원화 기준): {change_pct:.2f}%
</span>  

**AI 점수(업그레이드): {ai_score}점**  
- RF 상승확률(다음 거래일): **{prob_pct:.1f}%**
{news_line}- 리스크(변동성): **{risk}**
""",
        unsafe_allow_html=True,
    )

    with st.expander(f"🔎 {ticker} 상세(모델/뉴스) 보기"):
        st.write("**RandomForest 상태**:", rf_debug)
        if titles:
            st.write("**뉴스 상태**:", news_debug)
            st.write("**최근 뉴스 제목(일부)**")
            for t in titles[:10]:
                st.write(f"- {t}")

    rows.append(
        {
            "Ticker": ticker,
            "Currency": currency_label,
            "Qty": qty,
            "Buy_KRW": buy_price_krw,
            "Now_KRW": current_krw,
            "Eval_KRW": eval_krw,
            "PnL_KRW": pnl_krw,
            "Return_%": change_pct,
            "AI_Score": ai_score,
            "RF_Up_%": prob_pct,
        }
    )

st.divider()
st.subheader("📌 포트폴리오 합산")

if len(rows) == 0:
    st.info("표시할 포트폴리오 데이터가 없습니다. 종목 코드를 확인해 주세요.")
else:
    df = pd.DataFrame(rows)

    total_eval = float(df["Eval_KRW"].sum())
    total_pnl = float(df["PnL_KRW"].sum())
    total_buy = total_eval - total_pnl
    total_return = (total_pnl / total_buy) * 100.0 if total_buy != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("총 평가금액(원화)", fmt_krw(total_eval))
    c2.metric("총 평가손익(원화)", fmt_krw(total_pnl))
    c3.metric("총 수익률(원화 기준)", f"{total_return:.2f}%")

    show_cols = ["Ticker", "Currency", "Qty", "Buy_KRW", "Now_KRW", "Eval_KRW", "PnL_KRW", "Return_%", "AI_Score", "RF_Up_%"]
    st.dataframe(df[show_cols].sort_values("Eval_KRW", ascending=False), use_container_width=True)

    st.subheader("비중 차트 (평가금액 기준)")
    w = df[["Ticker", "Eval_KRW"]].copy()
    w = w[w["Eval_KRW"] > 0].sort_values("Eval_KRW", ascending=False)
    if len(w) == 0:
        st.info("비중 차트를 그릴 데이터가 없습니다(평가금액이 0 이하).")
    else:
        st.bar_chart(w.set_index("Ticker"), height=320)

    st.subheader("손익 차트 (원화)")
    p = df[["Ticker", "PnL_KRW"]].copy().set_index("Ticker")
    st.bar_chart(p, height=320)

# =========================================================
# 8) 설정 JSON 저장/불러오기
# =========================================================
with st.expander("💾 설정 JSON 저장/불러오기 (URL이 길어질 때 추천)"):
    current_settings = {
        "tickers_input": st.session_state["tickers_input"],
        "buy_prices_input": st.session_state["buy_prices_input"],
        "quantities_input": st.session_state["quantities_input"],
    }
    settings_json = json.dumps(current_settings, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇️ 현재 설정 JSON 다운로드",
        data=settings_json.encode("utf-8"),
        file_name="portfolio_settings.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("⬆️ 설정 JSON 업로드", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.loads(uploaded.read().decode("utf-8"))
            request_apply_loaded_settings(loaded)
            st.success("설정을 불러왔습니다. 적용을 위해 화면을 갱신합니다.")
        except Exception as e:
            st.error(f"설정 파일을 읽는 중 오류: {e}")
