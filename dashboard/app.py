"""US Inflation Tracker — complete dashboard."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import requests
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
API_BASE   = "http://localhost:8000"

MODEL_NAMES = {
    "linear":   "Linear Regression",
    "xgboost":  "XGBoost",
    "arima":    "ARIMA",
    "lstm":     "LSTM Neural Net",
    "ensemble": "Ensemble (Stacking)",
}

FEATURE_LABELS = {
    "cpi_lag1": "CPI — 1 month ago", "cpi_lag3": "CPI — 3 months ago",
    "cpi_lag6": "CPI — 6 months ago", "cpi_lag12": "CPI — 1 year ago",
    "cpi_roll_mean3": "3-Month CPI Average", "cpi_roll_mean12": "12-Month CPI Average",
    "cpi_roll_std3": "CPI Volatility (3 mo)", "cpi_roll_std12": "CPI Volatility (12 mo)",
    "m2": "Money Supply (M2)", "m2_mom": "Money Supply Change (MoM)",
    "m2_yoy": "Money Supply Change (YoY)", "fed_funds": "Fed Interest Rate",
    "fed_funds_mom": "Interest Rate Change (MoM)", "fed_funds_yoy": "Interest Rate Change (YoY)",
    "unemployment": "Unemployment Rate", "unemployment_mom": "Unemployment Change (MoM)",
    "unemployment_yoy": "Unemployment Change (YoY)",
    "oil_wti": "Oil Price (WTI)", "oil_wti_yoy": "Oil Price Change (YoY)",
    "ppi": "Producer Price Index", "ppi_yoy": "Producer Price Change (YoY)",
    "ppi_mom": "Producer Price Change (MoM)",
    "gdp_growth": "GDP Growth", "gdp_growth_lag1": "GDP Growth (prev. quarter)",
    "rate_diff": "Rate Momentum", "m2_rate_interact": "Money × Rate Interaction",
    "post_covid": "Post-COVID Period",
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inflation Tracker", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.stApp { background: #F4F6F9; }
.block-container { padding: 0 2.5rem 4rem 2.5rem !important; max-width: 1280px !important; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 0 1.4rem 0;
    border-bottom: 1px solid #E2E5EA;
    margin-bottom: 2rem;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-dot  { width: 9px; height: 9px; background: #2563EB; border-radius: 50%; }
.topbar-title{ font-size: 1.05rem; font-weight: 700; color: #0F172A; letter-spacing: -0.01em; }
.topbar-tag  {
    font-size: .72rem; font-weight: 500; color: #6B7280;
    background: #F1F5F9; border: 1px solid #E2E8F0;
    padding: 3px 10px; border-radius: 6px;
}

.kpi-strip {
    background: #fff; border: 1px solid #E2E5EA;
    border-radius: 12px; display: flex; margin-bottom: 1.5rem;
}
.kpi-item { flex: 1; padding: 1.4rem 1.8rem; border-right: 1px solid #E2E5EA; }
.kpi-item:last-child { border-right: none; }
.kpi-label { font-size: .68rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: .09em; color: #9CA3AF; margin-bottom: .45rem; }
.kpi-number{ font-size: 2rem; font-weight: 700; color: #0F172A;
             letter-spacing: -.03em; line-height: 1; }
.kpi-sub   { font-size: .75rem; color: #9CA3AF; margin-top: .4rem; }
.kpi-pill  {
    display: inline-flex; align-items: center; gap: 3px;
    font-size: .7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 99px; margin-top: .35rem;
}
.pill-red  { background: #FEE2E2; color: #B91C1C; }
.pill-amber{ background: #FEF3C7; color: #92400E; }
.pill-green{ background: #D1FAE5; color: #065F46; }
.pill-blue { background: #DBEAFE; color: #1E40AF; }
.pill-gray { background: #F3F4F6; color: #4B5563; }

.sec-head { margin: 0 0 .15rem 0; font-size: .92rem; font-weight: 700;
            color: #0F172A; letter-spacing: -.01em; }
.sec-desc { font-size: .78rem; color: #9CA3AF; margin-bottom: 1rem; }

.panel {
    background: #fff; border: 1px solid #E2E5EA;
    border-radius: 12px; padding: 1.5rem 1.75rem;
}

.result-wrap {
    border: 1px solid #BFDBFE; background: #EFF6FF;
    border-radius: 10px; padding: 1.4rem 1.5rem;
    text-align: center; margin-top: .8rem;
}
.result-label{ font-size: .7rem; font-weight: 600; text-transform: uppercase;
               letter-spacing: .08em; color: #3B82F6; margin-bottom: .3rem; }
.result-value{ font-size: 3rem; font-weight: 800; color: #1E40AF;
               letter-spacing: -.04em; line-height: 1; }
.result-unit { font-size: .8rem; color: #60A5FA; margin-top: .25rem; }
.result-note { font-size: .78rem; color: #1E40AF; background: rgba(255,255,255,.7);
               border-radius: 6px; padding: .45rem .75rem; margin-top: .7rem; }

.model-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: .7rem 1rem; border-radius: 8px; margin-bottom: .4rem;
}
.model-row:last-child { margin-bottom: 0; }
.model-row-hl    { background: #EFF6FF; }
.model-row-plain { background: #F9FAFB; }
.model-row-name  { font-size: .85rem; font-weight: 600; color: #0F172A; }
.model-row-badge {
    font-size: .62rem; font-weight: 700; text-transform: uppercase;
    background: #2563EB; color: #fff;
    padding: 1px 7px; border-radius: 99px; margin-left: 6px;
}
.model-row-stat  { font-size: .8rem; color: #6B7280; text-align: right; }
.model-row-stat b{ color: #111827; }

.vdiv { height: 1px; background: #E2E5EA; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    p = PROC_DIR / "features.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None

@st.cache_resource
def load_bundle(name):
    p = MODELS_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_feat_cols():
    p = MODELS_DIR / "feature_columns.json"
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data
def load_metrics():
    p = MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_shap_summary():
    p = MODELS_DIR / "shap_summary.json"
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data
def load_shap_values():
    p_vals  = MODELS_DIR / "shap_values.npy"
    p_dates = MODELS_DIR / "shap_dates.json"
    if not p_vals.exists() or not p_dates.exists():
        return None, None
    return np.load(str(p_vals)), json.loads(p_dates.read_text())

@st.cache_data
def load_regime_periods():
    p = MODELS_DIR / "regime_periods.json"
    return json.loads(p.read_text()) if p.exists() else []

def best_model():
    p = MODELS_DIR / "best_model.txt"
    return p.read_text().strip() if p.exists() else "xgboost"

def get_preds(df, model_name):
    cols = [c for c in load_feat_cols() if c in df.columns]
    X = df[cols]
    if model_name == "linear":
        b = load_bundle("linear")
        return b["model"].predict(b["scaler"].transform(X)) if b else None
    if model_name == "xgboost":
        b = load_bundle("xgboost")
        return b["model"].predict(X) if b else None
    if model_name == "arima":
        b = load_bundle("arima")
        return b["model"].predict_in_sample() if b else None
    if model_name == "ensemble":
        b = load_bundle("ensemble")
        if b is None:
            return None
        pred_lin = b["linear_bundle"]["model"].predict(b["linear_bundle"]["scaler"].transform(X))
        pred_xgb = b["xgb_bundle"]["model"].predict(X)
        meta_X   = np.column_stack([pred_lin, pred_xgb])
        return b["meta_model"].predict(meta_X)
    return None

def rate_color(v):
    if v >= 5:   return "#DC2626"
    if v >= 3:   return "#D97706"
    return "#16A34A"

def rate_pill(v):
    if v >= 5:   return "pill-red",  "High"
    if v >= 3:   return "pill-amber","Elevated"
    if v >= 1.5: return "pill-green","Healthy"
    return "pill-blue", "Low"


# ── Load & validate ───────────────────────────────────────────────────────────
df_all = load_data()
if df_all is None:
    st.error("Run `python -m src.ingest` and `python -m src.features` first.")
    st.stop()


# ── Controls (sidebar) ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    avail = [p.stem for p in MODELS_DIR.glob("*.joblib") if p.stem in MODEL_NAMES]
    bm = best_model()
    model_sel = st.selectbox("Model", avail,
                             index=avail.index(bm) if bm in avail else 0,
                             format_func=lambda x: MODEL_NAMES.get(x, x))
    yr_min, yr_max = st.slider("Year range", 1990, 2025, (2000, 2025))
    show_band  = st.toggle("Uncertainty band", value=True)
    show_split = st.toggle("Train/test split line", value=True)


# ── Filtered view ─────────────────────────────────────────────────────────────
df = df_all.loc[str(yr_min):str(yr_max)]
preds = get_preds(df, model_sel)

latest_actual = float(df["inflation_rate"].iloc[-1])
latest_date   = df.index[-1].strftime("%b %Y")
latest_pred   = float(preds[-1]) if preds is not None else None
mae = float(np.mean(np.abs(df["inflation_rate"].values - preds))) if preds is not None else None

if preds is not None and len(preds) >= 6:
    trend = float(np.mean(preds[-3:])) - float(np.mean(preds[-6:-3]))
else:
    trend = None


# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-dot"></div>
    <span class="topbar-title">US Inflation Tracker</span>
    <span class="topbar-tag">FRED · BLS · 1990–2025</span>
  </div>
  <span class="topbar-tag">Model: {MODEL_NAMES.get(model_sel, model_sel)} &nbsp;·&nbsp; {yr_min}–{yr_max}</span>
</div>
""", unsafe_allow_html=True)


# ── KPI strip ─────────────────────────────────────────────────────────────────
pill_cls, pill_word = rate_pill(latest_actual)
color = rate_color(latest_actual)

pred_html = (
    f'<div class="kpi-number" style="color:{rate_color(latest_pred)}">'
    f'{latest_pred:.1f}<span style="font-size:1rem;font-weight:500;color:#9CA3AF">%</span></div>'
    if latest_pred is not None
    else '<div class="kpi-number" style="color:#D1D5DB">—</div>'
)

mae_html = (
    f'<div class="kpi-number">±{mae:.2f}'
    f'<span style="font-size:1rem;font-weight:500;color:#9CA3AF">pp</span></div>'
    if mae is not None
    else '<div class="kpi-number" style="color:#D1D5DB">—</div>'
)

if trend is None:
    trend_html = '<div class="kpi-number" style="color:#D1D5DB">—</div>'
elif trend > 0.05:
    trend_html = f'<div class="kpi-number">↑ Rising</div><span class="kpi-pill pill-red">+{trend:.2f}pp</span>'
elif trend < -0.05:
    trend_html = f'<div class="kpi-number">↓ Falling</div><span class="kpi-pill pill-green">{trend:.2f}pp</span>'
else:
    trend_html = '<div class="kpi-number">→ Stable</div><span class="kpi-pill pill-gray">flat</span>'

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-item">
    <div class="kpi-label">Current Inflation</div>
    <div class="kpi-number" style="color:{color}">{latest_actual:.1f}<span style="font-size:1rem;font-weight:500;color:#9CA3AF">%</span></div>
    <span class="kpi-pill {pill_cls}">{pill_word}</span>
    <div class="kpi-sub">Year-over-year · {latest_date}</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-label">Model Forecast</div>
    {pred_html}
    <div class="kpi-sub">{MODEL_NAMES.get(model_sel, model_sel)}</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-label">Average Error</div>
    {mae_html}
    <div class="kpi-sub">Percentage points off on average</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-label">6-Month Trend</div>
    {trend_html}
    <div class="kpi-sub">Based on recent predictions</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    f'<p class="sec-head">Inflation History</p>'
    f'<p class="sec-desc">Actual year-over-year CPI change vs '
    f'{MODEL_NAMES.get(model_sel, model_sel)} predictions, {yr_min}–{yr_max}</p>',
    unsafe_allow_html=True,
)

fig = go.Figure()

# Regime shading — HMM high-inflation periods (red tint)
regime_periods = load_regime_periods()
for period in regime_periods:
    if period["regime"] != 1:
        continue
    p_start = pd.Timestamp(period["start"])
    p_end   = pd.Timestamp(period["end"])
    # Only draw if the period overlaps with the visible year range
    view_start = pd.Timestamp(f"{yr_min}-01-01")
    view_end   = pd.Timestamp(f"{yr_max}-12-31")
    if p_end < view_start or p_start > view_end:
        continue
    fig.add_vrect(
        x0=max(p_start, view_start).isoformat(),
        x1=min(p_end,   view_end).isoformat(),
        fillcolor="rgba(220,38,38,0.07)", line_width=0,
        annotation_text="High inflation regime",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color="#DC2626",
    )

# 2% target
fig.add_hline(y=2, line_dash="dot", line_color="#CBD5E1", line_width=1,
              annotation_text="2% Fed Target", annotation_position="right",
              annotation_font_size=10, annotation_font_color="#94A3B8")

# Train/test split
if show_split and yr_min <= 2019 <= yr_max:
    fig.add_vline(x="2019-01-01", line_color="#E2E8F0", line_width=1.5, line_dash="dash")
    fig.add_annotation(x="2019-03-01",
                       y=df["inflation_rate"].max() * .92,
                       text="Train  |  Test",
                       showarrow=False, font=dict(size=10, color="#CBD5E1"), xanchor="left")

# Uncertainty band
if show_band and preds is not None:
    resid    = df["inflation_rate"].values - preds
    roll_std = pd.Series(resid).rolling(12, min_periods=3).std().bfill().values
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(preds + roll_std) + list((preds - roll_std)[::-1]),
        fill="toself", fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty range", hoverinfo="skip", showlegend=True,
    ))

# Prediction line
if preds is not None:
    fig.add_trace(go.Scatter(
        x=df.index, y=preds,
        name=f"{MODEL_NAMES.get(model_sel,'Model')} prediction",
        line=dict(color="#93C5FD", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}  Predicted: <b>%{y:.2f}%</b><extra></extra>",
    ))

# Actual line
fig.add_trace(go.Scatter(
    x=df.index, y=df["inflation_rate"],
    name="Actual inflation",
    line=dict(color="#1D4ED8", width=2),
    hovertemplate="%{x|%b %Y}  Actual: <b>%{y:.2f}%</b><extra></extra>",
))

# COVID peak annotation
if yr_min <= 2022 <= yr_max:
    peak_idx = df["inflation_rate"].idxmax()
    fig.add_annotation(
        x=peak_idx, y=float(df["inflation_rate"].max()),
        text=f"Peak: {float(df['inflation_rate'].max()):.1f}%",
        showarrow=True, arrowhead=0, arrowcolor="#CBD5E1", arrowwidth=1,
        ax=55, ay=-30, font=dict(size=10, color="#6B7280"),
        bgcolor="#fff", bordercolor="#E2E8F0", borderwidth=1, borderpad=4,
    )

fig.update_layout(
    template="plotly_white", height=360,
    margin=dict(l=0, r=60, t=10, b=0), hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0,
                font=dict(size=11, color="#6B7280"), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#9CA3AF"), showline=False),
    yaxis=dict(ticksuffix="%", tickfont=dict(size=11, color="#9CA3AF"),
               gridcolor="#F1F5F9", zeroline=False, showline=False),
    font=dict(family="Inter, system-ui, sans-serif"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)


# ── Bottom two columns — forecast tool + feature importance ───────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="sec-head">What-If Forecast</p><p class="sec-desc">Adjust the inputs and see what inflation the model would predict</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        unemp  = c1.slider("Unemployment (%)",      1.0,  15.0,  3.7,  0.1)
        ffrate = c2.slider("Fed Interest Rate (%)", 0.0,  20.0,  5.25, 0.25)
        oil    = c1.slider("Oil Price ($/barrel)",  20.0, 150.0, 85.0, 1.0)
        gdp    = c2.slider("GDP Growth (%)",        -5.0, 10.0,  2.1,  0.1)
        m2     = c1.slider("Money Supply (M2, $T)", 5.0,  30.0,  21.5, 0.1)
        ppi_v  = c2.slider("Producer Price Index",  100.0,350.0, 230.0,1.0)

        if st.button("Get Forecast →", type="primary", use_container_width=True):
            try:
                resp = requests.post(f"{API_BASE}/predict", json={
                    "m2": m2 * 100, "unemployment": unemp, "fed_funds": ffrate,
                    "oil_wti": oil, "ppi": ppi_v, "gdp_growth": gdp,
                    "month": pd.Timestamp.now().month, "model": model_sel,
                }, timeout=5)
                if resp.status_code == 200:
                    rate = resp.json()["inflation_rate"]
                    if   rate >= 5:   note = "⚠️ High inflation — prices are rising quickly."
                    elif rate >= 3:   note = "🔸 Above target — above the Fed's 2% goal."
                    elif rate >= 1.5: note = "✅ Healthy — close to the Fed's 2% target."
                    else:             note = "❄️ Low inflation — the economy may be cooling."
                    st.markdown(f"""
<div class="result-wrap">
  <div class="result-label">Forecast Result</div>
  <div class="result-value">{rate:.2f}%</div>
  <div class="result-unit">estimated annual inflation rate</div>
  <div class="result-note">{note}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.error(f"API returned {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.warning("API offline — run `uvicorn api.main:app --reload`")

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="sec-head">What Drives Inflation</p><p class="sec-desc">Top factors by mean absolute SHAP value (XGBoost)</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    shap_summary = load_shap_summary()
    xgb_bundle   = load_bundle("xgboost")
    feat_cols    = load_feat_cols()

    if shap_summary:
        fi = pd.DataFrame(shap_summary).head(10)
        fi["label"] = fi["feature"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()))
        fi = fi.sort_values("shap")

        fig2 = go.Figure(go.Bar(
            x=fi["shap"], y=fi["label"], orientation="h",
            marker=dict(color=fi["shap"],
                        colorscale=[[0,"#BFDBFE"],[1,"#1D4ED8"]],
                        showscale=False),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=0, r=10, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(tickfont=dict(size=11, color="#374151"), showgrid=False),
            bargap=0.38,
            font=dict(family="Inter, system-ui, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    elif xgb_bundle and feat_cols:
        # Fallback to model feature importance if SHAP not saved yet
        imps = xgb_bundle["model"].feature_importances_
        fi   = (pd.DataFrame({"f": feat_cols[:len(imps)], "imp": imps})
                .sort_values("imp", ascending=True).tail(10))
        fi["label"] = fi["f"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()))
        fig2 = go.Figure(go.Bar(
            x=fi["imp"], y=fi["label"], orientation="h",
            marker=dict(color=fi["imp"], colorscale=[[0,"#BFDBFE"],[1,"#1D4ED8"]], showscale=False),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        ))
        fig2.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=0, r=10, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(tickfont=dict(size=11, color="#374151"), showgrid=False),
            bargap=0.38, font=dict(family="Inter, system-ui, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Train models to see feature importance.")


# ── SHAP Time-Series Section ──────────────────────────────────────────────────
st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

shap_arr, shap_dates = load_shap_values()
if shap_arr is not None and feat_cols and shap_summary:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-head">SHAP Feature Impact Over Time</p>'
        '<p class="sec-desc">How the contribution of each factor to the model\'s prediction changed across the test period</p>',
        unsafe_allow_html=True,
    )

    top_feats = [item["feature"] for item in shap_summary[:6]]
    top_idxs  = [feat_cols.index(f) for f in top_feats if f in feat_cols]
    shap_dates_parsed = pd.to_datetime(shap_dates)

    fig_shap = go.Figure()
    palette = ["#1D4ED8","#DC2626","#16A34A","#D97706","#7C3AED","#0891B2"]
    for idx, feat in zip(top_idxs, top_feats):
        vals  = shap_arr[:, idx].tolist()
        label = FEATURE_LABELS.get(feat, feat.replace("_"," ").title())
        color = palette[top_idxs.index(idx) % len(palette)]
        fig_shap.add_trace(go.Scatter(
            x=shap_dates_parsed, y=vals, name=label,
            line=dict(width=1.5, color=color),
            hovertemplate=f"%{{x|%b %Y}} — {label}: <b>%{{y:.3f}}</b><extra></extra>",
        ))

    fig_shap.add_hline(y=0, line_color="#E2E8F0", line_width=1)
    fig_shap.update_layout(
        template="plotly_white", height=300,
        margin=dict(l=0, r=10, t=10, b=0), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0,
                    font=dict(size=11, color="#6B7280"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#9CA3AF")),
        yaxis=dict(tickfont=dict(size=11, color="#9CA3AF"), gridcolor="#F1F5F9", zeroline=False,
                   title=dict(text="SHAP value", font=dict(size=11, color="#9CA3AF"))),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── Model Comparison Scorecard (real metrics) ─────────────────────────────────
st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<p class="sec-head">Model Comparison Scorecard</p>'
    '<p class="sec-desc">How well each model predicted inflation on the unseen test period (2019–2025) — lower error is better</p>',
    unsafe_allow_html=True,
)

saved_metrics = load_metrics()
bm = best_model()

MODEL_ORDER = [
    ("ensemble", "Ensemble (Stacking)"),
    ("xgboost",  "XGBoost"),
    ("linear",   "Linear Regression"),
    ("lstm",     "LSTM Neural Net"),
    ("arima",    "ARIMA"),
]

if saved_metrics:
    rows_html = ""
    for key, name in MODEL_ORDER:
        if key not in saved_metrics:
            continue
        m     = saved_metrics[key]
        hl    = "model-row-hl" if key == bm else "model-row-plain"
        badge = '<span class="model-row-badge">Best</span>' if key == bm else ""
        mae_v = m.get("MAE",  "—")
        rmse_v= m.get("RMSE", "—")
        da_v  = m.get("Dir_Acc", "—")
        r2_v  = m.get("R2", "—")
        rows_html += f"""
<div class="model-row {hl}">
  <span class="model-row-name">{name}{badge}</span>
  <span class="model-row-stat">
    MAE <b>{mae_v}</b> &nbsp;·&nbsp; RMSE <b>{rmse_v}</b>
    &nbsp;·&nbsp; R² <b>{r2_v}</b> &nbsp;·&nbsp; Dir. Acc <b>{da_v}%</b>
  </span>
</div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    # Bar chart comparing RMSE
    rmse_vals  = [(k, n, saved_metrics[k]["RMSE"]) for k, n in MODEL_ORDER if k in saved_metrics]
    fig_cmp    = go.Figure(go.Bar(
        x=[n for _, n, _ in rmse_vals],
        y=[r for _, _, r in rmse_vals],
        marker_color=["#1D4ED8" if k == bm else "#93C5FD" for k, _, _ in rmse_vals],
        text=[f"{r:.3f}" for _, _, r in rmse_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>",
    ))
    fig_cmp.update_layout(
        template="plotly_white", height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="RMSE by Model (lower is better)", font=dict(size=12, color="#6B7280")),
        yaxis=dict(gridcolor="#F1F5F9", zeroline=False, tickfont=dict(size=11, color="#9CA3AF")),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#374151")),
        font=dict(family="Inter, system-ui, sans-serif"),
        bargap=0.4,
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
else:
    # Fallback: compute on the fly from the current model selection
    test_df = df_all.loc["2019-01-01":]
    rows_html = ""
    for key, name in MODEL_ORDER:
        preds_m = get_preds(test_df, key)
        if preds_m is None:
            continue
        y      = test_df["inflation_rate"].values[:len(preds_m)]
        err    = round(float(np.mean(np.abs(y - preds_m))), 2)
        da     = round(float(np.mean(np.sign(np.diff(y)) == np.sign(np.diff(preds_m)))) * 100, 1)
        hl     = "model-row-hl" if key == bm else "model-row-plain"
        badge  = '<span class="model-row-badge">Best</span>' if key == bm else ""
        rows_html += f"""
<div class="model-row {hl}">
  <span class="model-row-name">{name}{badge}</span>
  <span class="model-row-stat">Avg error <b>±{err:.2f}pp</b> &nbsp;·&nbsp; Direction accuracy <b>{da:.0f}%</b></span>
</div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

st.markdown("""
<p style="font-size:.74rem;color:#9CA3AF;margin-top:.75rem;line-height:1.6">
  <b>MAE</b> — mean absolute error in percentage points. &nbsp;
  <b>RMSE</b> — root mean squared error (penalises big misses more). &nbsp;
  <b>R²</b> — variance explained (1.0 = perfect). &nbsp;
  <b>Dir. Acc</b> — how often the model correctly calls the direction of change.
</p>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
