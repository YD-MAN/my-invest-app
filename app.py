import streamlit as st
from urllib.parse import quote_plus, unquote_plus

# ---------------------------
# 기본값
# ---------------------------
DEFAULT_TICKERS = "AAPL, MSFT, NVDA, 005930.KS, BTC-KRW"
DEFAULT_BUYPRICES = "150000, 300000, 400000, 70000, 50000000"
DEFAULT_QTYS = "10, 5, 3, 10, 0.01"

# ---------------------------
# Query Params 호환 래퍼
# ---------------------------
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

# ---------------------------
# 초기값: URL -> 기본값
# (위젯 만들기 "이전"에만 session_state 기본값 세팅)
# ---------------------------
qp = get_qp()
init_tickers = qp_get_str(qp, "tickers", DEFAULT_TICKERS)
init_buy = qp_get_str(qp, "buy", DEFAULT_BUYPRICES)
init_qty = qp_get_str(qp, "qty", DEFAULT_QTYS)

if "tickers_input" not in st.session_state:
    st.session_state["tickers_input"] = init_tickers
if "buy_prices_input" not in st.session_state:
    st.session_state["buy_prices_input"] = init_buy
if "quantities_input" not in st.session_state:
    st.session_state["quantities_input"] = init_qty
if "__qp_dirty__" not in st.session_state:
    st.session_state["__qp_dirty__"] = False

# ---------------------------
# 콜백: 입력 변경 시 URL 저장 플래그
# ---------------------------
def mark_dirty():
    st.session_state["__qp_dirty__"] = True

# ---------------------------
# Reset: 위젯 key를 직접 덮어쓰지 말고 삭제 후 재시작
# ---------------------------
def do_reset():
    for k in ["tickers_input", "buy_prices_input", "quantities_input"]:
        if k in st.session_state:
            del st.session_state[k]

    set_qp(
        tickers=enc(DEFAULT_TICKERS),
        buy=enc(DEFAULT_BUYPRICES),
        qty=enc(DEFAULT_QTYS),
    )
    st.rerun()

st.subheader("입력 (자동 URL 저장)")

col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

with col1:
    st.text_input(
        "종목코드 (쉼표 구분)",
        key="tickers_input",
        on_change=mark_dirty,
    )

with col2:
    st.text_input(
        "평단가 (원화 기준, 쉼표 구분)",
        key="buy_prices_input",
        on_change=mark_dirty,
    )

with col3:
    st.text_input(
        "수량 (쉼표 구분)",
        key="quantities_input",
        on_change=mark_dirty,
    )

with col4:
    st.write("")
    st.write("")
    st.button("Reset", on_click=do_reset)

# ---------------------------
# ✅ 자동 URL 저장 (다를 때만 set_qp 호출)
# ---------------------------
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

st.caption("✅ 입력을 바꾸면 자동으로 URL에 저장됩니다. 새로고침/재접속/공유해도 동일 입력 유지")
