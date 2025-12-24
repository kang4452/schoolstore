import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from simulation.simulator import run_simulation, PRODUCTS, EVENTS, DAYS, INITIAL_CASH

st.set_page_config(page_title="학교 매점 모의투자", layout="centered")

# =========================
# 스타일 (토스 느낌: 심플, 카드형)
# =========================
st.markdown(
    '''
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 6rem; max-width: 460px;}
      h1, h2, h3 {letter-spacing: -0.2px;}
      .muted {color:#6b7280; font-size: 0.9rem;}
      .card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 14px 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,.06);
      }
      .pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(0,0,0,0.04);
        font-size: 12px;
        color:#374151;
      }
      /* Streamlit 버튼을 카드처럼 */
      div.stButton > button {
        width: 100%;
        border-radius: 18px !important;
        padding: 14px 14px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        background: #ffffff !important;
        box-shadow: 0 8px 20px rgba(0,0,0,.06) !important;
        text-align: left !important;
        font-weight: 800 !important;
      }
      div.stButton > button:hover {border: 1px solid rgba(0,0,0,0.12) !important;}
      .bottom-bar {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: rgba(245,246,248,.75);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0,0,0,0.08);
        padding: 10px 16px 14px;
        z-index: 9999;
      }
      .bottom-inner {max-width: 460px; margin: 0 auto; display:flex; gap:10px;}
      .buy-btn button {background:#111 !important; color:#fff !important; border: none !important;}
      .sell-btn button {background:#fff !important; color:#111 !important;}
      .kpi {display:flex; gap:10px;}
      .kpi .box {flex:1; background:#fff; border: 1px solid rgba(0,0,0,0.06); border-radius: 18px; padding: 12px 14px; box-shadow: 0 8px 20px rgba(0,0,0,.06);}
      .kpi .label {color:#6b7280; font-size:12px; margin-bottom:6px;}
      .kpi .value {font-size:16px; font-weight:900;}
    </style>
    ''',
    unsafe_allow_html=True,
)

# =========================
# 세션 상태 초기화
# =========================
if "initialized" not in st.session_state:
    df = run_simulation()
    st.session_state.df = df
    st.session_state.day = 1
    st.session_state.cash = float(INITIAL_CASH)
    st.session_state.holdings = {p: 0 for p in PRODUCTS}
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.initialized = True

df: pd.DataFrame = st.session_state.df
day: int = st.session_state.day
cash: float = st.session_state.cash
holdings = st.session_state.holdings
history = st.session_state.history

def fmt_money(x: float) -> str:
    return f"{x:,.0f}원"

def get_price_series(product: str) -> pd.Series:
    s = df[df["product"] == product].sort_values("day")["price_end"].astype(float)
    # 길이 보정 (혹시 결측이 있어도 안전)
    s = s.reset_index(drop=True)
    return s

def draw_month_chart(prices: np.ndarray):
    # 상승=빨강, 하락=파랑 (구간별 LineCollection)
    x = np.arange(1, len(prices) + 1)
    y = prices.astype(float)

    # segments: (N-1)개의 선분
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    up = (y[1:] - y[:-1]) >= 0
    colors = np.where(up, "#ef4444", "#3b82f6").tolist()

    lc = LineCollection(segments, colors=colors, linewidths=3, capstyle="round")

    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=120)
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    pad = max(1.0, (y.max() - y.min()) * 0.08)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    # 아주 연한 그리드
    ax.grid(True, linewidth=0.7, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#6b7280")
    ax.set_xticks([1, 10, 20, 30] if len(prices) >= 30 else np.linspace(1, len(prices), 4).astype(int))
    ax.set_yticks([])

    # 마지막 점 표시
    ax.scatter([x[-1]], [y[-1]], s=35, edgecolors="rgba(17,17,17,0.25)", linewidths=1, facecolors="white", zorder=5)

    fig.tight_layout(pad=1)
    return fig

# =========================
# 상단
# =========================
st.title("매점 모의주식")
st.markdown('<div class="muted">상품을 클릭하면 1개월 가격 그래프와 매수/매도 버튼이 나와요</div>', unsafe_allow_html=True)

# 오늘 가격(현재가) 맵
today_rows = df[df["day"] == day]
price_map = {row["product"]: float(row["price_end"]) for _, row in today_rows.iterrows()}

portfolio_value = sum(holdings[p] * price_map.get(p, 0.0) for p in PRODUCTS)
total_value = cash + portfolio_value

# KPI 카드
st.markdown(
    f'''
    <div class="kpi" style="margin-top: 10px;">
      <div class="box">
        <div class="label">현금</div>
        <div class="value">{fmt_money(cash)}</div>
      </div>
      <div class="box">
        <div class="label">총 자산</div>
        <div class="value">{fmt_money(total_value)}</div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# 날짜/이벤트 (작게)
event = EVENTS.get(day, {"code": "일반일", "title": "일반일", "desc": ""})
st.markdown(
    f'<div style="margin-top:10px;"><span class="pill">DAY {day}/{DAYS}</span> <span class="muted" style="margin-left:8px;">{event["title"]}</span></div>',
    unsafe_allow_html=True
)

# 다음날 / 초기화
c1, c2 = st.columns(2)
with c1:
    if st.button("다음날로 이동 ➜", use_container_width=True):
        if day < DAYS:
            st.session_state.day += 1
        else:
            st.warning("마지막 날입니다. 초기화해서 다시 시작할 수 있어요.")
with c2:
    if st.button("초기화", use_container_width=True):
        st.session_state.df = run_simulation()
        st.session_state.day = 1
        st.session_state.cash = float(INITIAL_CASH)
        st.session_state.holdings = {p: 0 for p in PRODUCTS}
        st.session_state.history = []
        st.session_state.selected_product = None
        st.success("초기화 완료!")

st.write("")  # spacing

# =========================
# 상품 리스트 (클릭)
# =========================
st.subheader("상품")

# 리스트에서 현재가/전일 대비를 계산해서 버튼 라벨에 함께 보여주기
def day_price(product: str, d: int) -> float:
    row = df[(df["product"] == product) & (df["day"] == d)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["price_end"])

for product in PRODUCTS.keys():
    cur = day_price(product, day)
    prev = day_price(product, max(1, day - 1))
    diff = cur - prev if np.isfinite(cur) and np.isfinite(prev) else 0.0
    arrow = "▲" if diff >= 0 else "▼"
    diff_abs = abs(diff)

    label = f"{product}\n현재가  {fmt_money(cur)}   {arrow} {fmt_money(diff_abs)}"
    if st.button(label, key=f"select_{product}"):
        st.session_state.selected_product = product

selected = st.session_state.selected_product

# =========================
# 선택한 상품 상세 (그래프 + 매수/매도)
# =========================
if selected:
    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    series = get_price_series(selected)
    prices = series.to_numpy()

    # 현재가/한달 등락
    last = float(prices[min(day, len(prices)) - 1])  # 현재 day까지 기준
    first = float(prices[0])
    month_diff = last - first
    up_month = month_diff >= 0
    month_arrow = "▲" if up_month else "▼"
    month_color = "#ef4444" if up_month else "#3b82f6"

    st.markdown(
        f'''
        <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px;">
          <div>
            <div style="font-weight:900; font-size:16px;">{selected}</div>
            <div class="muted" style="margin-top:4px;">최근 1개월 가격 변동</div>
          </div>
          <div style="text-align:right;">
            <div style="font-weight:900; font-size:16px;">{fmt_money(last)}</div>
            <div style="margin-top:4px; font-size:12px; font-weight:800; color:{month_color};">
              {month_arrow} {fmt_money(abs(month_diff))} ({(month_diff/first*100 if first else 0):+.1f}%)
            </div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    fig = draw_month_chart(prices[: max(2, min(day, len(prices)))])  # 최소 2개는 그리기
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="muted">상승 구간은 <b style="color:#ef4444;">빨강</b>, 하락 구간은 <b style="color:#3b82f6;">파랑</b>으로 표시돼요.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # 주문 영역 (하단 바 느낌)
    # =========================
    st.markdown('<div class="bottom-bar"><div class="bottom-inner">', unsafe_allow_html=True)

    # 수량 입력은 바깥(고정 바 위로) 노출되는 게 Streamlit에선 어려워서,
    # 바깥에 간단히 두고 버튼은 바에 두는 형태로 구성
    qty = st.number_input("수량", min_value=1, max_value=1000, value=1, step=1, key="trade_qty")

    # 버튼 두 개를 fixed bar 안에서 보여주기 위해 columns 사용
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="buy-btn">', unsafe_allow_html=True)
        if st.button("매수", use_container_width=True, key="buy_btn_main"):
            price = day_price(selected, day)
            cost = price * qty
            if st.session_state.cash >= cost:
                st.session_state.cash -= cost
                st.session_state.holdings[selected] += int(qty)
                st.session_state.history.append(
                    {"day": day, "product": selected, "side": "매수", "qty": int(qty), "price": float(price), "amount": float(cost)}
                )
                st.success(f"{selected} {int(qty)}개 매수 완료!")
            else:
                st.error("현금이 부족합니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="sell-btn">', unsafe_allow_html=True)
        if st.button("매도", use_container_width=True, key="sell_btn_main"):
            price = day_price(selected, day)
            if st.session_state.holdings[selected] >= qty:
                revenue = price * qty
                st.session_state.cash += revenue
                st.session_state.holdings[selected] -= int(qty)
                st.session_state.history.append(
                    {"day": day, "product": selected, "side": "매도", "qty": int(qty), "price": float(price), "amount": float(revenue)}
                )
                st.success(f"{selected} {int(qty)}개 매도 완료!")
            else:
                st.error("보유수량이 부족합니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# 보유/주문 내역 (아래쪽에 작게)
# =========================
st.write("")
st.subheader("내 보유")
hold_df = pd.DataFrame(
    [{"상품": p, "보유수량": int(holdings[p]), "평가금액": float(holdings[p] * price_map.get(p, 0.0))} for p in PRODUCTS]
)
st.dataframe(hold_df, use_container_width=True, hide_index=True)

st.subheader("주문 내역")
if history:
    hist_df = pd.DataFrame(history)[["day", "product", "side", "qty", "price", "amount"]].sort_values(["day"]).reset_index(drop=True)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
else:
    st.info("아직 주문한 내역이 없습니다.")
