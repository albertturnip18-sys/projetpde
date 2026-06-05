# ============================================================
#  STREAMLIT APP — PEMODELAN ODE PERTUMBUHAN PENDUDUK
#  Kota Medan (2020–2030) + Perbandingan dengan Kota Tual
#  Model: Eksponensial & Logistik | Solusi Analitik + Numerik
#  Referensi: Armin & Remetwa, JIMAT Vol.6 No.1, 2025
#  Data Medan: BPS Kota Medan 2020–2025
#  Jalankan: streamlit run app_medan_tual.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ODE Penduduk · Medan & Tual",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    color: #D6E4F0;
    background-color: #87CEEB;
    background-image:
        radial-gradient(ellipse 40% 16% at 10% 7%, rgba(255,255,255,0.88) 0%, transparent 70%),
        radial-gradient(ellipse 30% 12% at 75% 5%, rgba(255,255,255,0.80) 0%, transparent 70%),
        radial-gradient(ellipse 18% 9% at 48% 3%, rgba(255,255,255,0.65) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 100% 0%, rgba(255,230,80,0.18) 0%, transparent 60%),
        linear-gradient(180deg,
            #5BB8E8 0%, #87CEEB 22%, #B0DCEF 42%, #D4EDF8 58%, #E8F5FB 68%,
            #C0D4DE 69%, #AABFCC 70%, #BDCED8 71%, #C8D9E2 72%,
            #AABFCC 73%, #C2D4DE 74%, #D5E6EE 75%, #E8F3F8 82%, #EFF8FC 100%
        );
}
.stApp::before {
    content: '';
    position: fixed; bottom: 0; left: 0; right: 0; height: 200px;
    pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 200' preserveAspectRatio='xMidYMax slice'%3E%3Crect x='0' y='130' width='70' height='70' fill='%23A8BBC6' rx='1'/%3E%3Crect x='10' y='105' width='35' height='95' fill='%2398AEBB' rx='1'/%3E%3Crect x='24' y='95' width='8' height='12' fill='%2398AEBB'/%3E%3Crect x='60' y='115' width='50' height='85' fill='%23AABFCA' rx='1'/%3E%3Crect x='110' y='92' width='28' height='108' fill='%2390A5B2' rx='1'/%3E%3Crect x='113' y='80' width='22' height='14' fill='%2390A5B2'/%3E%3Crect x='122' y='70' width='4' height='12' fill='%2378909C'/%3E%3Crect x='138' y='108' width='48' height='92' fill='%23A3B8C5' rx='1'/%3E%3Crect x='186' y='122' width='36' height='78' fill='%23B5CAD5' rx='1'/%3E%3Crect x='222' y='88' width='58' height='112' fill='%2386A2B0' rx='1'/%3E%3Crect x='234' y='72' width='34' height='17' fill='%2386A2B0'/%3E%3Crect x='249' y='58' width='5' height='16' fill='%2366828E'/%3E%3Crect x='280' y='102' width='42' height='98' fill='%239EB5C2' rx='1'/%3E%3Crect x='322' y='118' width='32' height='82' fill='%23A8BDCA' rx='1'/%3E%3Crect x='354' y='80' width='65' height='120' fill='%2388A8B6' rx='1'/%3E%3Crect x='419' y='110' width='38' height='90' fill='%23B0C5D2' rx='1'/%3E%3Crect x='457' y='92' width='52' height='108' fill='%2393AEBE' rx='1'/%3E%3Crect x='509' y='126' width='36' height='74' fill='%23B8CADA' rx='1'/%3E%3Crect x='545' y='72' width='62' height='128' fill='%2380A0AF' rx='1'/%3E%3Crect x='607' y='102' width='43' height='98' fill='%23A5BCCA' rx='1'/%3E%3Crect x='650' y='90' width='50' height='110' fill='%238EAABC' rx='1'/%3E%3Crect x='700' y='120' width='36' height='80' fill='%23B3C6D4' rx='1'/%3E%3Crect x='736' y='82' width='58' height='118' fill='%2383A3B3' rx='1'/%3E%3Crect x='794' y='106' width='40' height='94' fill='%23A8BAC8' rx='1'/%3E%3Crect x='834' y='94' width='48' height='106' fill='%2396B2C0' rx='1'/%3E%3Crect x='882' y='116' width='33' height='84' fill='%23AEBFCE' rx='1'/%3E%3Crect x='915' y='77' width='65' height='123' fill='%237C9CAC' rx='1'/%3E%3Crect x='980' y='110' width='38' height='90' fill='%23A8BAC8' rx='1'/%3E%3Crect x='1018' y='90' width='53' height='110' fill='%2390AEBE' rx='1'/%3E%3Crect x='1071' y='122' width='36' height='78' fill='%23B6C8D8' rx='1'/%3E%3Crect x='1107' y='84' width='60' height='116' fill='%237EA0B0' rx='1'/%3E%3Crect x='1167' y='98' width='42' height='102' fill='%23A6BCCA' rx='1'/%3E%3Crect x='1209' y='112' width='36' height='88' fill='%23AEBFCC' rx='1'/%3E%3Crect x='1245' y='86' width='53' height='114' fill='%238BAFC0' rx='1'/%3E%3Crect x='1298' y='120' width='38' height='80' fill='%23B8C8D6' rx='1'/%3E%3Crect x='1336' y='88' width='52' height='112' fill='%2396B0BF' rx='1'/%3E%3Crect x='1388' y='108' width='52' height='92' fill='%23A0B5C2' rx='1'/%3E%3Crect x='0' y='196' width='1440' height='4' fill='%23BDD0DA'/%3E%3C/svg%3E");
    background-size: 100% 100%; background-repeat: no-repeat; background-position: bottom; opacity: 0.5;
}
.main .block-container { position: relative; z-index: 1; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6BB8D8 0%, #50A8CC 100%);
    border-right: 1px solid rgba(0,140,255,0.20);
}
[data-testid="stSidebar"] * { color: #B8CDD8 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #060F20 0%, #071525 100%);
    border: 1px solid rgba(0,140,255,0.18);
    border-top: 2px solid rgba(0,180,216,0.35);
    border-radius: 10px;
    padding: 16px 18px 12px;
}
[data-testid="metric-container"] label {
    color: #4E7A96 !important; font-family: 'Space Mono', monospace !important;
    font-size: 9px !important; letter-spacing: 2px; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #48CAE4 !important; font-family: 'Syne', sans-serif !important;
    font-size: 20px !important; font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #52B788 !important; font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(6,15,32,0.8) !important;
    border-bottom: 1px solid rgba(0,140,255,0.12) !important;
    gap: 2px !important; padding: 0 4px !important;
}
.stTabs [data-testid="stTab"] button {
    font-family: 'Syne', sans-serif !important; font-size: 12px !important;
    font-weight: 600 !important; color: #4E7A96 !important;
    border-radius: 6px 6px 0 0 !important; padding: 10px 14px !important;
}
.stTabs [data-testid="stTab"] button[aria-selected="true"] {
    color: #48CAE4 !important; border-bottom: 2px solid #48CAE4 !important;
    background: rgba(0,180,216,0.07) !important;
}
div.stButton > button {
    background: transparent !important; color: #48CAE4 !important;
    border: 1px solid rgba(0,180,216,0.4) !important; border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.8px; padding: 7px 16px !important;
}
.hero {
    background: linear-gradient(135deg, #060F1F 0%, #071828 60%, #060F1F 100%);
    border: 1px solid rgba(0,140,255,0.2); border-radius: 14px;
    padding: 32px 40px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, #0096C7 30%, #48CAE4 50%, #52B788 70%, transparent 100%);
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
    color: #E8F4F8; margin: 0 0 8px; letter-spacing: -1px; line-height: 1.2;
}
.hero-title span.cyan { color: #48CAE4; }
.hero-title span.green { color: #52B788; }
.hero-sub { font-size: 13px; color: #5A8099; line-height: 1.7; max-width: 700px; }
.badge {
    display: inline-flex; align-items: center;
    background: rgba(0,150,199,0.1); border: 1px solid rgba(0,150,199,0.22);
    border-radius: 4px; padding: 3px 10px;
    font-family: 'Space Mono', monospace; font-size: 9px; color: #48CAE4;
    margin-right: 6px; margin-top: 4px;
}
.badge-green { background: rgba(82,183,136,0.1); border-color: rgba(82,183,136,0.22); color: #52B788; }
.badge-amber { background: rgba(244,162,97,0.1); border-color: rgba(244,162,97,0.22); color: #F4A261; }
.section-label {
    font-family: 'Space Mono', monospace; font-size: 9px; color: #0096C7;
    letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 14px;
    padding-bottom: 10px; border-bottom: 1px solid rgba(0,140,255,0.12);
}
.info-card {
    background: rgba(0,150,199,0.04); border: 1px solid rgba(0,140,255,0.12);
    border-left: 2px solid #0096C7; border-radius: 0 8px 8px 0;
    padding: 14px 18px; font-size: 13px; color: #7A9BAD; line-height: 1.85; margin: 8px 0;
}
.info-card-green {
    border-left-color: #52B788;
}
.info-card-amber {
    border-left-color: #F4A261;
}
.formula-box {
    background: #040C18; border: 1px solid rgba(0,140,255,0.18);
    border-radius: 8px; padding: 18px 22px;
    font-family: 'Space Mono', monospace; font-size: 12px;
    color: #48CAE4; text-align: center; line-height: 2.2; margin: 12px 0;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,140,255,0.12) !important;
    border-radius: 8px !important; overflow: hidden !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #03070F; }
::-webkit-scrollbar-thumb { background: rgba(0,140,255,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── COLOR PALETTE ────────────────────────────────────────────
CYAN   = "#48CAE4"
TEAL   = "#00B4D8"
GREEN  = "#52B788"
AMBER  = "#F4A261"
PURPLE = "#A78BFA"
CORAL  = "#F77F6E"
WHITE  = "#E8F4F8"
MUTED  = "#3A6A88"
ORANGE = "#FB8500"

# ── PLOTLY TEMPLATE ──────────────────────────────────────────
# ── PLOTLY LAYOUT HELPER ────────────────────────────────────
# update_layout() bisa menerima **kwargs, tapi nested dict (legend, margin, dll)
# harus diberikan sebagai keyword args biasa — tidak boleh nested dict conflict.
# Solusi: gunakan fungsi apply_layout() yang memanggil update_layout item per item.

def apply_layout(fig, **kwargs):
    """
    Terapkan PTPL base styling + kwargs tambahan ke fig.update_layout().
    Menghindari TypeError saat **PTPL di-unpack bersamaan dengan kwargs
    yang mengandung key yang sama (misal: legend, margin).
    """
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font=dict(family="DM Sans, sans-serif", color="#2A5570", size=12),
        legend=dict(
            bgcolor="rgba(255,255,255,0.90)", bordercolor="rgba(0,140,200,0.22)",
            borderwidth=1, font=dict(size=11, family="DM Sans, sans-serif", color="#2A5570"),
        ),
        margin=dict(l=60, r=30, t=55, b=55),
        hoverlabel=dict(
            bgcolor="rgba(240,252,255,0.97)", bordercolor="rgba(0,150,200,0.4)",
            font=dict(family="Space Mono, monospace", size=11, color="#0096C7"),
        ),
    )
    # kwargs override base untuk key yang sama
    base.update(kwargs)
    fig.update_layout(**base)
    return fig

# Alias agar kompatibel dengan pemanggilan lama
PTPL = {}   # tidak dipakai langsung lagi

def axis_style(title="", fmt=""):
    d = dict(
        gridcolor="rgba(0,120,180,0.12)", linecolor="rgba(0,120,180,0.25)",
        zerolinecolor="rgba(0,120,180,0.15)",
        tickfont=dict(family="Space Mono, monospace", size=10, color="#3A6A88"),
    )
    if title: d["title"] = dict(text=title, font=dict(size=12, color="#2A5570"))
    if fmt:   d["tickformat"] = fmt
    return d

# ── DATA KOTA TUAL (sumber: jurnal Armin & Remetwa, 2025) ────
TUAL_TAHUN_HIST = np.array([2020, 2021, 2022, 2023, 2024], dtype=float)
TUAL_POP_HIST   = np.array([88280, 90322, 93145, 91572, 92744], dtype=float)
TUAL_K          = 0.0122   # laju pertumbuhan dari jurnal
TUAL_P0_PRED    = 92744.0  # populasi awal prediksi (2024)
TUAL_TAHUN_PRED = np.array([2026, 2027, 2028, 2029, 2030])
TUAL_POP_JURNAL = np.array([95035, 96176, 97381, 98587, 99793])

# ── DATA KOTA MEDAN (sumber: BPS Kota Medan 2020–2025) ───────
# Dalam ribuan jiwa → dikonversi ke jiwa penuh
MEDAN_TAHUN_HIST = np.array([2020, 2021, 2022, 2023, 2024, 2025], dtype=float)
MEDAN_POP_HIST   = np.array([
    2_435_252,   # 2020
    2_460_858,   # 2021
    2_494_512,   # 2022
    2_474_166,   # 2023
    2_486_283,   # 2024
    2_498_293,   # 2025
], dtype=float)

# ── ODE FUNGSI ───────────────────────────────────────────────
def ode_exp(P, t, k):           return k * P
def ode_log(P, t, k, K):        return k * P * (1 - P / K)
def sol_exp(t, P0, k):          return P0 * np.exp(k * t)
def sol_log(t, P0, k, K):       return K / (1 + ((K - P0) / P0) * np.exp(-k * t))

def mape(a, p): return np.mean(np.abs((a - p) / a)) * 100
def rmse(a, p): return np.sqrt(np.mean((a - p)**2))

# ── METODE NUMERIK ───────────────────────────────────────────
def euler(f, P0, t_span, dt, args=()):
    ts = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
    Ps = np.zeros(len(ts)); Ps[0] = P0
    for i in range(1, len(ts)):
        Ps[i] = Ps[i-1] + dt * f(Ps[i-1], ts[i-1], *args)
    return ts, Ps

def rk4(f, P0, t_span, dt, args=()):
    ts = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
    Ps = np.zeros(len(ts)); Ps[0] = P0
    for i in range(1, len(ts)):
        h  = dt; ti = ts[i-1]
        k1 = f(Ps[i-1],          ti,       *args)
        k2 = f(Ps[i-1]+h*k1/2,  ti+h/2,   *args)
        k3 = f(Ps[i-1]+h*k2/2,  ti+h/2,   *args)
        k4 = f(Ps[i-1]+h*k3,    ti+h,      *args)
        Ps[i] = Ps[i-1] + (h/6)*(k1+2*k2+2*k3+k4)
    return ts, Ps

# ── FIT MODEL MEDAN ──────────────────────────────────────────
@st.cache_data
def fit_medan():
    t_rel = MEDAN_TAHUN_HIST - MEDAN_TAHUN_HIST[0]
    P0 = MEDAN_POP_HIST[0]
    # Eksponensial: fit k
    popt_e, _ = curve_fit(
        lambda t, k: sol_exp(t, P0, k),
        t_rel, MEDAN_POP_HIST, p0=[0.008], bounds=(0, 0.1)
    )
    # Logistik: fit k dan K
    popt_l, _ = curve_fit(
        lambda t, k, K: sol_log(t, P0, k, K),
        t_rel, MEDAN_POP_HIST, p0=[0.01, 3_500_000],
        bounds=([0, 2_500_000], [0.5, 10_000_000]), maxfev=20000
    )
    return float(popt_e[0]), float(popt_l[0]), float(popt_l[1])

k_medan_exp, k_medan_log, K_medan_log = fit_medan()

# Hitung k Medan menggunakan metode jurnal (sama seperti Tual):
# k = (1/t) * ln(P(t)/P0) menggunakan data 2020–2025 (t=5)
_k_medan_jurnal = (1/5) * np.log(MEDAN_POP_HIST[-1] / MEDAN_POP_HIST[0])

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 8px;">
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#48CAE4;">
        ODE Penduduk
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2A5A78;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">
        Medan · Tual · 2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">⚙ Parameter Medan</div>', unsafe_allow_html=True)

    k_medan_ui = st.slider(
        "k Medan — laju pertumbuhan",
        min_value=0.001, max_value=0.030,
        value=round(_k_medan_jurnal, 4), step=0.001, format="%.4f",
        key="k_medan_slider",
        help=f"Nilai fit dari data BPS: {_k_medan_jurnal:.4f} ({_k_medan_jurnal*100:.2f}%/thn)"
    )
    st.markdown(
        f"<div style='font-family:Space Mono,monospace;font-size:10px;color:#52B788;margin-top:-6px;margin-bottom:10px;'>"
        f"→ {k_medan_ui:.4f} = <b>{k_medan_ui*100:.2f}%</b> / tahun</div>",
        unsafe_allow_html=True
    )

    K_medan_ui = st.slider(
        "K Medan — kapasitas dukung (juta)",
        min_value=2_500_000, max_value=5_000_000,
        value=int(K_medan_log), step=50_000,
        format="%d",
        key="K_medan_slider",
        help="Daya dukung lingkungan untuk model logistik Medan"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">⚙ Parameter Tual</div>', unsafe_allow_html=True)

    k_tual_ui = st.slider(
        "k Tual — laju pertumbuhan",
        min_value=0.003, max_value=0.050,
        value=TUAL_K, step=0.001, format="%.4f",
        key="k_tual_slider",
        help="Nilai jurnal: 0.0122 (1.22%/thn)"
    )
    K_tual_ui = st.slider(
        "K Tual — kapasitas dukung",
        min_value=100_000, max_value=300_000,
        value=150_000, step=5_000, format="%d",
        key="K_tual_slider",
    )

    st.markdown("---")
    dt_ui = st.selectbox("Δt langkah numerik (tahun)", [1.0, 0.5, 0.25, 0.1], index=1, key="dt_selectbox")

    st.markdown("""
    <div style='font-size:10px;color:#1A4060;line-height:2;font-family:Space Mono,monospace;margin-top:8px;'>
    <span style='color:#0096C7;letter-spacing:1.5px;'>REFERENSI</span><br>
    Armin & Remetwa, M.G.K.<br>
    JIMAT Vol.6 No.1, 2025<br>
    DOI: 10.63976/jimat.v6i1.804<br><br>
    <span style='color:#0096C7;letter-spacing:1.5px;'>DATA MEDAN</span><br>
    BPS Kota Medan 2020–2025
    </div>
    """, unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:#0096C7;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;">
    📐 Pemodelan Persamaan Diferensial · BPS 2020–2025
  </div>
  <div class="hero-title">
    Pertumbuhan Penduduk <span class="green">Kota Medan</span><br>
    &amp; Perbandingan dengan <span class="cyan">Kota Tual</span>
  </div>
  <div class="hero-sub" style="margin-top:10px;">
    Aplikasi ODE model eksponensial &amp; logistik · Data BPS Kota Medan 2020–2025 ·
    Metode analitik, Euler, &amp; RK4 · Referensi: Armin &amp; Remetwa, JIMAT Vol.6 No.1, 2025
  </div>
  <div style="margin-top:14px;">
    <span class="badge">dP/dt = k·P</span>
    <span class="badge">P(t) = P₀·e^(kt)</span>
    <span class="badge">dP/dt = k·P·(1−P/K)</span>
    <span class="badge badge-green">Medan: ~{:.2f}%/thn</span>
    <span class="badge badge-amber">Tual: 1.22%/thn</span>
    <span class="badge">Euler &amp; RK4</span>
  </div>
</div>
""".format(_k_medan_jurnal * 100), unsafe_allow_html=True)

# ── KPI METRICS ──────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("P₀ Medan (2020)", f"{int(MEDAN_POP_HIST[0]):,}", "jiwa awal")
c2.metric("P Medan (2025)",  f"{int(MEDAN_POP_HIST[-1]):,}", f"+{int(MEDAN_POP_HIST[-1]-MEDAN_POP_HIST[0]):,}")
c3.metric("k Medan (fit)", f"{_k_medan_jurnal*100:.3f}%/thn", f"k = {_k_medan_jurnal:.5f}")
c4.metric("P₀ Tual (2020)",  f"{int(TUAL_POP_HIST[0]):,}", "jiwa awal")
c5.metric("P Tual (2024)",   f"{int(TUAL_POP_HIST[-1]):,}", f"+{int(TUAL_POP_HIST[-1]-TUAL_POP_HIST[0]):,}")
c6.metric("k Tual (jurnal)", f"{TUAL_K*100:.2f}%/thn", "Armin & Remetwa 2025")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Pemodelan Medan",
    "🔵  Pemodelan Tual",
    "⚖️  Perbandingan",
    "⚙️  Metode Numerik",
    "📐  Teori ODE",
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PEMODELAN KOTA MEDAN                              ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-label">📊 Model ODE Pertumbuhan Penduduk Kota Medan</div>',
                unsafe_allow_html=True)

    col_info, col_k = st.columns([3, 2])
    with col_info:
        st.markdown(f"""
        <div class="info-card info-card-green">
        <b>Kota Medan</b> — ibu kota Provinsi Sumatera Utara, kota terbesar ketiga di Indonesia.
        Data BPS 2020–2025 menunjukkan populasi sekitar <b>2,4–2,5 juta jiwa</b>.
        Laju pertumbuhan k dihitung dengan metode yang sama seperti jurnal referensi:
        k = (1/t)·ln(P(t)/P₀) = <b>{_k_medan_jurnal:.5f}</b> ({_k_medan_jurnal*100:.3f}%/tahun).
        </div>
        """, unsafe_allow_html=True)
    with col_k:
        t5 = MEDAN_TAHUN_HIST[-1] - MEDAN_TAHUN_HIST[0]
        st.markdown(f"""
        <div class="formula-box">
        k = (1/t)·ln(P(t)/P₀)<br>
        k = (1/{int(t5)})·ln({int(MEDAN_POP_HIST[-1]):,} / {int(MEDAN_POP_HIST[0]):,})<br>
        k = (1/{int(t5)})·ln({MEDAN_POP_HIST[-1]/MEDAN_POP_HIST[0]:.5f})<br>
        k = <span style="color:#52B788;font-weight:700;">{_k_medan_jurnal:.5f}</span>
        &nbsp;≈ <span style="color:#52B788;">{_k_medan_jurnal*100:.3f}%/tahun</span>
        </div>
        """, unsafe_allow_html=True)

    # Hitung prediksi Medan
    MEDAN_P0_PRED = MEDAN_POP_HIST[-1]          # 2025 sebagai titik prediksi
    MEDAN_PRED_YR = np.array([2026, 2027, 2028, 2029, 2030])
    t_pred_m      = MEDAN_PRED_YR - 2025        # t relatif dari 2025

    P_exp_pred_m  = sol_exp(t_pred_m, MEDAN_P0_PRED, k_medan_ui)
    P_log_pred_m  = sol_log(t_pred_m, MEDAN_P0_PRED, k_medan_ui, K_medan_ui)

    # Kurva halus
    t_hist_rel   = MEDAN_TAHUN_HIST - MEDAN_TAHUN_HIST[0]
    t_full_m     = np.linspace(0, 10, 400)   # 2020–2030
    tahun_full_m = MEDAN_TAHUN_HIST[0] + t_full_m
    P0_fit       = MEDAN_POP_HIST[0]
    P_exp_full_m = sol_exp(t_full_m, P0_fit, k_medan_ui)
    P_log_full_m = sol_log(t_full_m, P0_fit, k_medan_ui, K_medan_ui)

    # Fit dalam periode historis
    P_exp_hist_m = sol_exp(t_hist_rel, P0_fit, k_medan_ui)
    P_log_hist_m = sol_log(t_hist_rel, P0_fit, k_medan_ui, K_medan_ui)

    # ── Grafik utama Medan ──
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(
        x=tahun_full_m, y=P_exp_full_m, mode="lines", name="Eksponensial",
        line=dict(color=GREEN, width=2.5),
    ))
    fig_m.add_trace(go.Scatter(
        x=tahun_full_m, y=P_log_full_m, mode="lines", name="Logistik",
        line=dict(color=PURPLE, width=2.5, dash="dash"),
    ))
    fig_m.add_trace(go.Scatter(
        x=MEDAN_TAHUN_HIST, y=MEDAN_POP_HIST,
        mode="markers+lines", name="Data BPS Aktual",
        marker=dict(color=AMBER, size=11, symbol="circle",
                    line=dict(color=WHITE, width=1.5)),
        line=dict(color=AMBER, width=1, dash="dot"),
    ))
    fig_m.add_trace(go.Scatter(
        x=tahun_full_m[tahun_full_m >= 2025],
        y=[K_medan_ui] * int((tahun_full_m >= 2025).sum()),
        mode="lines", name=f"K = {K_medan_ui:,}",
        line=dict(color=CORAL, width=1.5, dash="dot"),
    ))

    # Titik prediksi
    fig_m.add_trace(go.Scatter(
        x=MEDAN_PRED_YR, y=P_exp_pred_m,
        mode="markers", name="Pred. Eksponensial",
        marker=dict(color=GREEN, size=9, symbol="diamond",
                    line=dict(color=WHITE, width=1)),
    ))
    fig_m.add_trace(go.Scatter(
        x=MEDAN_PRED_YR, y=P_log_pred_m,
        mode="markers", name="Pred. Logistik",
        marker=dict(color=PURPLE, size=9, symbol="square",
                    line=dict(color=WHITE, width=1)),
    ))

    for yr, pop in zip(MEDAN_TAHUN_HIST, MEDAN_POP_HIST):
        fig_m.add_annotation(
            x=yr, y=pop, text=f"{int(pop/1e6):.3f}M",
            showarrow=False, yshift=18,
            font=dict(color=AMBER, size=9, family="Space Mono"),
            bgcolor="rgba(13,27,46,0.8)", borderpad=3,
        )

    fig_m.add_vline(x=2025, line_dash="dash", line_color=GREEN, line_width=1,
                    opacity=0.5, annotation_text="Batas Prediksi (2025)",
                    annotation_font_color=GREEN, annotation_font_size=10)

    apply_layout(fig_m, height=480,
        title=dict(text="Model ODE Pertumbuhan Penduduk Kota Medan (2020–2030)",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun"), "range": [2019, 2031]},
        yaxis={**axis_style("Jumlah Penduduk (jiwa)", ",d"),
               "range": [min(MEDAN_POP_HIST)*0.97, max(K_medan_ui*1.05, P_exp_full_m.max()*1.03)]},
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # ── Tabel prediksi & validasi ──
    col_tbl, col_val = st.columns(2)
    with col_tbl:
        st.markdown('<div class="section-label">Tabel Prediksi Penduduk Medan 2026–2030</div>',
                    unsafe_allow_html=True)
        df_pred_m = pd.DataFrame({
            "Tahun":          MEDAN_PRED_YR,
            "Eksponensial":   P_exp_pred_m.astype(int),
            "Logistik":       P_log_pred_m.astype(int),
            "Δ Eksponen":     (P_exp_pred_m - MEDAN_P0_PRED).astype(int),
        })
        st.dataframe(
            df_pred_m.style.format({
                "Eksponensial": "{:,}", "Logistik": "{:,}", "Δ Eksponen": "+{:,}"
            }),
            hide_index=True, use_container_width=True
        )

    with col_val:
        st.markdown('<div class="section-label">Validasi Model terhadap Data Historis</div>',
                    unsafe_allow_html=True)
        mape_e_m = mape(MEDAN_POP_HIST, P_exp_hist_m)
        mape_l_m = mape(MEDAN_POP_HIST, P_log_hist_m)
        rmse_e_m = rmse(MEDAN_POP_HIST, P_exp_hist_m)
        rmse_l_m = rmse(MEDAN_POP_HIST, P_log_hist_m)

        df_val_m = pd.DataFrame({
            "Metrik":        ["MAPE (%)", "RMSE (jiwa)"],
            "Eksponensial":  [f"{mape_e_m:.4f}", f"{rmse_e_m:,.1f}"],
            "Logistik":      [f"{mape_l_m:.4f}", f"{rmse_l_m:,.1f}"],
        })
        st.dataframe(df_val_m, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div class="info-card" style="font-size:12px;margin-top:8px;">
        k Medan (fit curve_fit): <b style="color:{GREEN};">{k_medan_exp*100:.4f}%/thn</b><br>
        k Medan (metode jurnal): <b style="color:{GREEN};">{_k_medan_jurnal*100:.4f}%/thn</b><br>
        K Logistik (fit): <b style="color:{PURPLE};">{K_medan_log:,.0f} jiwa</b><br>
        Prediksi 2030 (exp): <b style="color:{GREEN};">{int(P_exp_pred_m[-1]):,} jiwa</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Data historis ──
    st.markdown('<div class="section-label">Data Historis BPS Kota Medan 2020–2025</div>',
                unsafe_allow_html=True)
    laki   = [1_214_331, 1_226_435, 1_242_313, 1_231_644, 1_237_602, 1_243_422]
    permp  = [1_220_921, 1_234_423, 1_252_199, 1_242_522, 1_248_681, 1_254_871]
    df_hist_m = pd.DataFrame({
        "Tahun":             MEDAN_TAHUN_HIST.astype(int),
        "Laki-Laki (Jiwa)": laki,
        "Perempuan (Jiwa)": permp,
        "Total Penduduk":    MEDAN_POP_HIST.astype(int),
        "Pertumbuhan":       [0] + [int(MEDAN_POP_HIST[i]-MEDAN_POP_HIST[i-1])
                                    for i in range(1, len(MEDAN_POP_HIST))],
    })
    st.dataframe(
        df_hist_m.style.format({
            "Laki-Laki (Jiwa)": "{:,}", "Perempuan (Jiwa)": "{:,}",
            "Total Penduduk": "{:,}", "Pertumbuhan": "{:+,}"
        }),
        hide_index=True, use_container_width=True
    )

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — PEMODELAN KOTA TUAL                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-label">🔵 Model ODE Pertumbuhan Penduduk Kota Tual (Replikasi Jurnal)</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
    Replikasi model dari jurnal <b>Armin & Remetwa (JIMAT Vol.6 No.1, 2025)</b>.
    Menggunakan data BPS Provinsi Maluku 2020–2024.
    Laju pertumbuhan k = <b>0,0122 (1,22%/tahun)</b>, prediksi 2026–2030.
    P₀ = 92.744 jiwa (2024).
    </div>
    """, unsafe_allow_html=True)

    # Kurva Tual
    t_full_tual  = np.linspace(0, 10, 400)
    tahun_full_t = TUAL_TAHUN_HIST[0] + t_full_tual
    P0_tual_hist = TUAL_POP_HIST[0]
    P_exp_tual   = sol_exp(t_full_tual, P0_tual_hist, k_tual_ui)
    P_log_tual   = sol_log(t_full_tual, P0_tual_hist, k_tual_ui, K_tual_ui)

    t_hist_tual  = TUAL_TAHUN_HIST - TUAL_TAHUN_HIST[0]
    P_exp_h_t    = sol_exp(t_hist_tual, P0_tual_hist, k_tual_ui)
    P_log_h_t    = sol_log(t_hist_tual, P0_tual_hist, k_tual_ui, K_tual_ui)

    t_pred_tual  = TUAL_TAHUN_PRED - 2024
    P_exp_pred_t = sol_exp(t_pred_tual, TUAL_P0_PRED, k_tual_ui)
    P_log_pred_t = sol_log(t_pred_tual, TUAL_P0_PRED, k_tual_ui, K_tual_ui)

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=tahun_full_t, y=P_exp_tual, mode="lines", name="Eksponensial",
        line=dict(color=CYAN, width=2.5),
    ))
    fig_t.add_trace(go.Scatter(
        x=tahun_full_t, y=P_log_tual, mode="lines", name="Logistik",
        line=dict(color=GREEN, width=2.5, dash="dash"),
    ))
    fig_t.add_trace(go.Scatter(
        x=TUAL_TAHUN_HIST, y=TUAL_POP_HIST,
        mode="markers+lines", name="Data BPS Aktual",
        marker=dict(color=AMBER, size=11, line=dict(color=WHITE, width=1.5)),
        line=dict(color=AMBER, width=1, dash="dot"),
    ))
    fig_t.add_trace(go.Scatter(
        x=TUAL_TAHUN_PRED, y=TUAL_POP_JURNAL,
        mode="markers", name="Pred. Jurnal",
        marker=dict(color=AMBER, size=10, symbol="star",
                    line=dict(color=WHITE, width=1)),
    ))
    for yr, pop in zip(TUAL_TAHUN_HIST, TUAL_POP_HIST):
        fig_t.add_annotation(
            x=yr, y=pop, text=f"{int(pop):,}",
            showarrow=False, yshift=18,
            font=dict(color=AMBER, size=9, family="Space Mono"),
            bgcolor="rgba(13,27,46,0.8)", borderpad=3,
        )
    fig_t.add_vline(x=2024, line_dash="dash", line_color=AMBER,
                    line_width=1, opacity=0.5,
                    annotation_text="Batas Prediksi (2024)",
                    annotation_font_color=AMBER, annotation_font_size=10)
    apply_layout(fig_t, height=460,
        title=dict(text="Model ODE Pertumbuhan Penduduk Kota Tual (2020–2030) · Replikasi Jurnal",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun"), "range": [2019, 2031]},
        yaxis={**axis_style("Jumlah Penduduk (jiwa)", ",d"),
               "range": [85000, max(K_tual_ui * 1.05, P_exp_tual.max() * 1.03)]},
    )
    st.plotly_chart(fig_t, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown('<div class="section-label">Tabel Prediksi Jurnal vs Model Saat Ini</div>',
                    unsafe_allow_html=True)
        df_pred_t = pd.DataFrame({
            "Tahun":          TUAL_TAHUN_PRED,
            "Jurnal (jiwa)":  TUAL_POP_JURNAL,
            "Model Exp.":     P_exp_pred_t.astype(int),
            "Model Log.":     P_log_pred_t.astype(int),
            "Δ (Exp−Jurnal)": (P_exp_pred_t - TUAL_POP_JURNAL).astype(int),
        })
        st.dataframe(
            df_pred_t.style.format({
                "Jurnal (jiwa)": "{:,}", "Model Exp.": "{:,}",
                "Model Log.": "{:,}", "Δ (Exp−Jurnal)": "{:+,}"
            }),
            hide_index=True, use_container_width=True
        )
    with col_t2:
        st.markdown('<div class="section-label">Validasi terhadap Data BPS 2020–2024</div>',
                    unsafe_allow_html=True)
        mape_e_t = mape(TUAL_POP_HIST, P_exp_h_t)
        mape_l_t = mape(TUAL_POP_HIST, P_log_h_t)
        df_val_t = pd.DataFrame({
            "Metrik": ["MAPE (%)", "RMSE (jiwa)"],
            "Eksponensial": [f"{mape_e_t:.4f}", f"{rmse(TUAL_POP_HIST, P_exp_h_t):,.1f}"],
            "Logistik":     [f"{mape_l_t:.4f}", f"{rmse(TUAL_POP_HIST, P_log_h_t):,.1f}"],
        })
        st.dataframe(df_val_t, hide_index=True, use_container_width=True)
        st.markdown(f"""
        <div class="info-card" style="font-size:12px;margin-top:8px;">
        k Jurnal: <b style="color:{CYAN};">0.0122 (1.22%/thn)</b><br>
        Prediksi 2030 (jurnal): <b style="color:{AMBER};">99.793 jiwa</b><br>
        Prediksi 2030 (model): <b style="color:{CYAN};">{int(P_exp_pred_t[-1]):,} jiwa</b>
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — PERBANDINGAN MEDAN vs TUAL                        ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-label">⚖️ Perbandingan Kota Medan vs Kota Tual</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Perbandingan karakteristik pertumbuhan antara Kota Medan (metropolitan, ~2,5 juta jiwa)
    dan Kota Tual (kota kecil, ~93 ribu jiwa). Keduanya dimodelkan dengan ODE eksponensial
    menggunakan metode yang sama dari jurnal referensi. Perbedaan skala populasi divisualisasikan
    dengan sumbu Y ganda (dual axis).
    </div>
    """, unsafe_allow_html=True)

    # ── Grafik 1: Pertumbuhan Relatif (%) ──
    st.markdown('<div class="section-label">Pertumbuhan relatif (%) dari tahun dasar (2020 = 100%)</div>',
                unsafe_allow_html=True)

    t_common     = np.linspace(0, 10, 400)
    tahun_common = 2020 + t_common

    # Relatif dari P0 2020
    P_rel_medan = sol_exp(t_common, 100.0, k_medan_ui)   # skala 100
    P_rel_tual  = sol_exp(t_common, 100.0, k_tual_ui)

    # Data aktual relatif
    tual_rel_act  = TUAL_POP_HIST  / TUAL_POP_HIST[0]  * 100
    medan_rel_act = MEDAN_POP_HIST / MEDAN_POP_HIST[0] * 100

    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=tahun_common, y=P_rel_medan, mode="lines", name="Medan — Model Exp.",
        line=dict(color=GREEN, width=2.5),
    ))
    fig_rel.add_trace(go.Scatter(
        x=tahun_common, y=P_rel_tual, mode="lines", name="Tual — Model Exp.",
        line=dict(color=CYAN, width=2.5, dash="dash"),
    ))
    fig_rel.add_trace(go.Scatter(
        x=MEDAN_TAHUN_HIST, y=medan_rel_act,
        mode="markers", name="Medan — Data BPS",
        marker=dict(color=GREEN, size=10, symbol="circle",
                    line=dict(color=WHITE, width=1.5)),
    ))
    fig_rel.add_trace(go.Scatter(
        x=TUAL_TAHUN_HIST, y=tual_rel_act,
        mode="markers", name="Tual — Data BPS",
        marker=dict(color=CYAN, size=10, symbol="diamond",
                    line=dict(color=WHITE, width=1.5)),
    ))
    fig_rel.add_hline(y=100, line_dash="dot", line_color=MUTED, opacity=0.4,
                      annotation_text="Basis = 100%", annotation_font_color=MUTED)
    fig_rel.add_vline(x=2025, line_dash="dash", line_color=GREEN, line_width=1, opacity=0.4,
                      annotation_text="2025", annotation_font_color=GREEN)
    fig_rel.add_vline(x=2024, line_dash="dash", line_color=CYAN, line_width=1, opacity=0.4,
                      annotation_text="2024", annotation_font_color=CYAN)
    apply_layout(fig_rel, height=420,
        title=dict(text="Pertumbuhan Relatif (2020 = 100%) · Medan vs Tual",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun"), "range": [2019, 2031]},
        yaxis=axis_style("Indeks Pertumbuhan (%)", ".1f"),
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    # ── Grafik 2: Dual axis absolut ──
    st.markdown('<div class="section-label">Populasi absolut — sumbu Y ganda (skala berbeda)</div>',
                unsafe_allow_html=True)

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    # Medan di sumbu kiri
    t_m_full   = np.linspace(0, 10, 300)
    yr_m_full  = MEDAN_TAHUN_HIST[0] + t_m_full
    P_m_curve  = sol_exp(t_m_full, MEDAN_POP_HIST[0], k_medan_ui)

    fig_dual.add_trace(go.Scatter(
        x=yr_m_full, y=P_m_curve, mode="lines", name="Medan — Model Exp.",
        line=dict(color=GREEN, width=2.5),
    ), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=MEDAN_TAHUN_HIST, y=MEDAN_POP_HIST,
        mode="markers", name="Medan — Data BPS",
        marker=dict(color=GREEN, size=11, symbol="circle",
                    line=dict(color=WHITE, width=1.5)),
    ), secondary_y=False)

    # Tual di sumbu kanan
    t_t_full  = np.linspace(0, 10, 300)
    yr_t_full = TUAL_TAHUN_HIST[0] + t_t_full
    P_t_curve = sol_exp(t_t_full, TUAL_POP_HIST[0], k_tual_ui)

    fig_dual.add_trace(go.Scatter(
        x=yr_t_full, y=P_t_curve, mode="lines", name="Tual — Model Exp.",
        line=dict(color=CYAN, width=2.5, dash="dash"),
    ), secondary_y=True)
    fig_dual.add_trace(go.Scatter(
        x=TUAL_TAHUN_HIST, y=TUAL_POP_HIST,
        mode="markers", name="Tual — Data BPS",
        marker=dict(color=CYAN, size=11, symbol="diamond",
                    line=dict(color=WHITE, width=1.5)),
    ), secondary_y=True)

    apply_layout(fig_dual, height=440,
        title=dict(text="Populasi Absolut Medan (kiri) vs Tual (kanan) · 2020–2030",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun"), "range": [2019, 2031]},
    )
    fig_dual.update_yaxes(
        title_text="Penduduk Medan (jiwa)", tickformat=",d",
        gridcolor="rgba(0,140,255,0.06)", secondary_y=False,
        title_font=dict(color=GREEN), tickfont=dict(color=GREEN, family="Space Mono", size=10),
    )
    fig_dual.update_yaxes(
        title_text="Penduduk Tual (jiwa)", tickformat=",d",
        gridcolor="rgba(0,140,255,0.03)", secondary_y=True,
        title_font=dict(color=CYAN), tickfont=dict(color=CYAN, family="Space Mono", size=10),
    )
    st.plotly_chart(fig_dual, use_container_width=True)

    # ── Tabel Ringkasan Perbandingan ──
    st.markdown('<div class="section-label">Ringkasan statistik perbandingan</div>',
                unsafe_allow_html=True)

    P_m_2030 = sol_exp(5, MEDAN_POP_HIST[-1], k_medan_ui)   # 5 thn dari 2025
    P_t_2030 = sol_exp(6, TUAL_P0_PRED, k_tual_ui)           # 6 thn dari 2024

    df_cmp = pd.DataFrame({
        "Parameter":         ["Kota", "Populasi 2020", "Populasi terkini",
                              "k (laju pertumbuhan)", "k (%/tahun)",
                              "Prediksi 2030 (Exp.)", "Pertambahan 2020–2030",
                              "Skala populasi"],
        "Kota Medan":        ["Medan",
                              f"{int(MEDAN_POP_HIST[0]):,}",
                              f"{int(MEDAN_POP_HIST[-1]):,} (2025)",
                              f"{_k_medan_jurnal:.5f}",
                              f"{_k_medan_jurnal*100:.3f}%",
                              f"{int(P_m_2030):,}",
                              f"+{int(P_m_2030 - MEDAN_POP_HIST[0]):,}",
                              "Metropolitan (>2 juta)"],
        "Kota Tual":         ["Tual",
                              f"{int(TUAL_POP_HIST[0]):,}",
                              f"{int(TUAL_P0_PRED):,} (2024)",
                              f"{TUAL_K:.4f}",
                              f"{TUAL_K*100:.2f}%",
                              f"{int(P_t_2030):,}",
                              f"+{int(P_t_2030 - TUAL_POP_HIST[0]):,}",
                              "Kota kecil (<100 ribu)"],
    })
    st.dataframe(df_cmp, hide_index=True, use_container_width=True)

    # ── Bar laju pertumbuhan ──
    st.markdown('<div class="section-label">Perbandingan laju pertumbuhan tahunan (k)</div>',
                unsafe_allow_html=True)

    years_common = np.arange(2021, 2026)
    # Laju per tahun = (P(t)/P(t-1) - 1)*100 untuk data aktual
    def annual_rates(pops):
        return [(pops[i]/pops[i-1] - 1)*100 for i in range(1, len(pops))]

    r_medan = annual_rates(MEDAN_POP_HIST)   # 5 tahun
    r_tual  = annual_rates(TUAL_POP_HIST)    # 4 tahun [2021–2024]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[str(y) for y in range(2021, 2026)],
        y=r_medan,
        name="Medan",
        marker=dict(color=GREEN, opacity=0.85, line=dict(color=WHITE, width=0.5)),
        text=[f"{v:.2f}%" for v in r_medan],
        textposition="outside",
        textfont=dict(color=GREEN, size=10, family="Space Mono"),
    ))
    fig_bar.add_trace(go.Bar(
        x=[str(y) for y in range(2021, 2025)],
        y=r_tual,
        name="Tual",
        marker=dict(color=CYAN, opacity=0.85, line=dict(color=WHITE, width=0.5)),
        text=[f"{v:.2f}%" for v in r_tual],
        textposition="outside",
        textfont=dict(color=CYAN, size=10, family="Space Mono"),
    ))
    fig_bar.add_hline(y=_k_medan_jurnal*100, line_dash="dash", line_color=GREEN,
                      opacity=0.6, annotation_text=f"k̄ Medan={_k_medan_jurnal*100:.3f}%",
                      annotation_font_color=GREEN, annotation_font_size=10)
    fig_bar.add_hline(y=TUAL_K*100, line_dash="dash", line_color=CYAN,
                      opacity=0.6, annotation_text=f"k Tual={TUAL_K*100:.2f}%",
                      annotation_font_color=CYAN, annotation_font_size=10)
    apply_layout(fig_bar, height=380, barmode="group",
        title=dict(text="Laju Pertumbuhan Tahunan Aktual (%)",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun")},
        yaxis={**axis_style("Laju Pertumbuhan (%)", ".2f")},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — METODE NUMERIK                                    ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="section-label">⚙️ Solusi Numerik: Euler & RK4 — Kota Medan</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Persamaan diferensial <b>dP/dt = k·P</b> diselesaikan secara numerik menggunakan
    metode <b>Euler eksplisit (orde 1)</b> dan <b>Runge-Kutta orde 4 (RK4)</b>,
    kemudian dibandingkan dengan solusi analitik eksak P(t) = P₀·e^(kt).
    </div>
    """, unsafe_allow_html=True)

    # Pilih kota untuk analisis numerik
    kota_num = st.radio("Pilih kota untuk analisis numerik:",
                        ["Kota Medan", "Kota Tual"], horizontal=True)

    if kota_num == "Kota Medan":
        k_num  = k_medan_ui
        P0_num = MEDAN_POP_HIST[0]
        t_end  = 10  # 10 tahun
        t0_yr  = 2020
        lbl    = "Medan"
        clr    = GREEN
    else:
        k_num  = k_tual_ui
        P0_num = TUAL_POP_HIST[0]
        t_end  = 10
        t0_yr  = 2020
        lbl    = "Tual"
        clr    = CYAN

    # Hitung solusi numerik
    t_e, P_e = euler(ode_exp, P0_num, (0, t_end), float(dt_ui), args=(k_num,))
    t_r, P_r = rk4(ode_exp,   P0_num, (0, t_end), float(dt_ui), args=(k_num,))

    t_ex = np.linspace(0, t_end, 500)
    P_ex = sol_exp(t_ex, P0_num, k_num)

    fig_num = go.Figure()
    fig_num.add_trace(go.Scatter(
        x=t0_yr + t_ex, y=P_ex, mode="lines", name="Solusi Eksak (analitik)",
        line=dict(color=WHITE, width=2.5),
    ))
    fig_num.add_trace(go.Scatter(
        x=t0_yr + t_e, y=P_e, mode="lines+markers", name=f"Euler (Δt={dt_ui})",
        line=dict(color=AMBER, width=2, dash="dot"),
        marker=dict(color=AMBER, size=7),
    ))
    fig_num.add_trace(go.Scatter(
        x=t0_yr + t_r, y=P_r, mode="lines+markers", name=f"RK4 (Δt={dt_ui})",
        line=dict(color=clr, width=2, dash="dash"),
        marker=dict(color=clr, size=7, symbol="square"),
    ))
    fig_num.add_vline(x=t0_yr + (2025 - t0_yr), line_dash="dot",
                      line_color=MUTED, opacity=0.4)
    apply_layout(fig_num, height=440,
        title=dict(text=f"Solusi Numerik vs Analitik · {lbl} · k={k_num:.4f} · Δt={dt_ui}",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis={**axis_style("Tahun"), "range": [t0_yr - 0.5, t0_yr + t_end + 0.5]},
        yaxis={**axis_style("Jumlah Penduduk (jiwa)", ",d")},
    )
    st.plotly_chart(fig_num, use_container_width=True)

    # ── Error per langkah ──
    P_ex_at_pts = sol_exp(t_e, P0_num, k_num)
    err_euler   = np.abs(P_e - P_ex_at_pts) / P_ex_at_pts * 100
    err_rk4_pts = np.abs(P_r - sol_exp(t_r, P0_num, k_num)) / sol_exp(t_r, P0_num, k_num) * 100

    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.markdown('<div class="section-label">Error Euler per langkah (%)</div>',
                    unsafe_allow_html=True)
        fig_e1 = go.Figure(go.Scatter(
            x=t0_yr + t_e, y=err_euler, mode="lines+markers",
            line=dict(color=AMBER, width=2),
            marker=dict(color=AMBER, size=6),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.07)",
        ))
        apply_layout(fig_e1, height=280,
            title=dict(text="Error Euler (%)", font=dict(size=12, color=WHITE), x=0.01),
            xaxis=axis_style("Tahun"), yaxis=axis_style("Error (%)", ".6f"),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_e1, use_container_width=True)

    with col_err2:
        st.markdown('<div class="section-label">Error RK4 per langkah (%)</div>',
                    unsafe_allow_html=True)
        fig_e2 = go.Figure(go.Scatter(
            x=t0_yr + t_r, y=err_rk4_pts, mode="lines+markers",
            line=dict(color=clr, width=2),
            marker=dict(color=clr, size=6),
            fill="tozeroy", fillcolor=f"rgba(72,202,228,0.07)",
        ))
        apply_layout(fig_e2, height=280,
            title=dict(text="Error RK4 (%)", font=dict(size=12, color=WHITE), x=0.01),
            xaxis=axis_style("Tahun"), yaxis=axis_style("Error (%)", ".2e"),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_e2, use_container_width=True)

    # ── Tabel error pada t = t_end ──
    st.markdown('<div class="section-label">Perbandingan akurasi Euler vs RK4 pada t akhir</div>',
                unsafe_allow_html=True)

    dt_list = [2.0, 1.0, 0.5, 0.25, 0.1]
    rows = []
    for dt in dt_list:
        _, Pe = euler(ode_exp, P0_num, (0, t_end), dt, args=(k_num,))
        _, Pr = rk4(ode_exp,   P0_num, (0, t_end), dt, args=(k_num,))
        P_exact = sol_exp(t_end, P0_num, k_num)
        err_e = abs(Pe[-1] - P_exact) / P_exact * 100
        err_r = abs(Pr[-1] - P_exact) / P_exact * 100
        rows.append({
            "Δt (tahun)": dt,
            "P Eksak (jiwa)": f"{P_exact:,.2f}",
            "P Euler":        f"{Pe[-1]:,.2f}",
            "P RK4":          f"{Pr[-1]:,.2f}",
            "Error Euler (%)": f"{err_e:.8f}",
            "Error RK4 (%)":   f"{err_r:.2e}",
        })
    df_err = pd.DataFrame(rows)
    st.dataframe(df_err, hide_index=True, use_container_width=True)

    st.markdown(f"""
    <div style="background:#040C18;border:1px solid rgba(72,202,228,0.2);border-radius:10px;
    padding:18px 24px;margin-top:8px;font-family:'Space Mono',monospace;font-size:11px;
    color:#5A8099;line-height:2;">
      <b style="color:#C0D8E8;font-family:Syne,sans-serif;font-size:13px;">
        📋 Kesimpulan Numerik
      </b><br><br>
      <b style="color:{AMBER};">Euler (orde 1):</b> error global O(Δt) — halving Δt → error turun ~2×<br>
      <b style="color:{clr};">RK4 (orde 4):</b> error global O(Δt⁴) — halving Δt → error turun ~16×<br>
      → RK4 jauh lebih akurat dengan computational cost yang masih ringan.<br>
      → Untuk prediksi demografi jangka pendek (≤10 tahun), Δt = 1 tahun sudah memadai.
    </div>
    """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 — TEORI ODE                                         ║
# ╚══════════════════════════════════════════════════════════════╝
with tab5:
    st.markdown('<div class="section-label">📐 Landasan Teori Persamaan Diferensial Pertumbuhan Populasi</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Ringkasan teori dan derivasi model ODE yang digunakan dalam pemodelan pertumbuhan
    penduduk Kota Medan dan Kota Tual, mengacu pada kerangka jurnal Armin & Remetwa (2025).
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:#060F20;border:1px solid rgba(72,202,228,0.2);border-radius:10px;
        padding:20px 24px;margin-bottom:12px;">
        <div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#48CAE4;margin-bottom:12px;">
            📈 Model Eksponensial
        </div>
        <div class="formula-box" style="margin:0 0 12px;">
            dP/dt = k · P<br>
            ↓ separasi variabel<br>
            dP/P = k · dt<br>
            ↓ integrasi kedua sisi<br>
            ln P = k·t + C<br>
            ↓ kondisi awal P(0) = P₀<br>
            <span style="color:#52B788;font-weight:700;">P(t) = P₀ · e^(k·t)</span>
        </div>
        <div style="font-size:12px;color:#5A8099;line-height:1.8;">
        <b style="color:#8AABB8;">Asumsi:</b> laju pertumbuhan sebanding dengan populasi saat ini.<br>
        <b style="color:#8AABB8;">Berlaku:</b> jangka pendek, sumber daya tidak terbatas.<br>
        <b style="color:#8AABB8;">Parameter k:</b> dihitung dari data empiris:<br>
        k = (1/t) · ln(P(t)/P₀)
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#060F20;border:1px solid rgba(82,183,136,0.2);border-radius:10px;
        padding:20px 24px;margin-bottom:12px;">
        <div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#52B788;margin-bottom:12px;">
            📉 Model Logistik (Verhulst)
        </div>
        <div class="formula-box" style="margin:0 0 12px;border-color:rgba(82,183,136,0.3);">
            dP/dt = k · P · (1 − P/K)<br>
            ↓ separasi & integrasi parsial<br>
            <span style="color:#52B788;font-weight:700;">P(t) = K / (1 + ((K−P₀)/P₀) · e^(−kt))</span>
        </div>
        <div style="font-size:12px;color:#5A8099;line-height:1.8;">
        <b style="color:#8AABB8;">K:</b> daya dukung lingkungan (carrying capacity).<br>
        <b style="color:#8AABB8;">Jika P ≪ K:</b> tumbuh seperti eksponensial.<br>
        <b style="color:#8AABB8;">Jika P → K:</b> pertumbuhan melambat → P stabil di K.<br>
        <b style="color:#8AABB8;">Berlaku:</b> jangka panjang, sumber daya terbatas.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#060F20;border:1px solid rgba(167,139,250,0.2);border-radius:10px;
    padding:20px 24px;margin-bottom:12px;">
    <div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#A78BFA;margin-bottom:12px;">
        🔢 Metode Numerik: Euler & RK4
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div>
        <div style="font-family:Space Mono,monospace;font-size:10px;color:#A78BFA;letter-spacing:1px;
        text-transform:uppercase;margin-bottom:8px;">Euler Eksplisit (Orde 1)</div>
        <div class="formula-box" style="border-color:rgba(167,139,250,0.25);font-size:11px;padding:12px 16px;text-align:left;">
            P_{n+1} = P_n + Δt · f(P_n, t_n)<br>
            dimana f(P, t) = k · P<br><br>
            Error global: O(Δt)<br>
            Halving Δt → error ÷ 2
        </div>
        </div>
        <div>
        <div style="font-family:Space Mono,monospace;font-size:10px;color:#48CAE4;letter-spacing:1px;
        text-transform:uppercase;margin-bottom:8px;">Runge-Kutta Orde 4 (RK4)</div>
        <div class="formula-box" style="font-size:11px;padding:12px 16px;text-align:left;">
            k1 = f(P_n, t_n)<br>
            k2 = f(P_n + h·k1/2, t_n + h/2)<br>
            k3 = f(P_n + h·k2/2, t_n + h/2)<br>
            k4 = f(P_n + h·k3, t_n + h)<br>
            P_{n+1} = P_n + (h/6)(k1+2k2+2k3+k4)<br><br>
            Error global: O(Δt⁴) · Halving Δt → error ÷ 16
        </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Referensi
    st.markdown("""
    <div style="background:#040C18;border:1px solid rgba(0,140,255,0.12);border-radius:10px;
    padding:18px 24px;margin-top:8px;font-size:12px;color:#5A8099;line-height:2;">
    <b style="color:#C0D8E8;font-family:Syne,sans-serif;font-size:13px;">📚 Referensi</b><br><br>
    <b style="color:#48CAE4;">Jurnal Utama:</b><br>
    Armin & Remetwa, M.G.K. (2025). Aplikasi Persamaan Differensial Dengan Pendekatan Model
    Pertumbuhan Eksponensial Untuk Memprediksi Jumlah Penduduk Kota Tual Tahun 2026–2030.
    <i>JIMAT — Jurnal Ilmiah Matematika</i>, Vol.6, No.1, Hal.327–338.
    DOI: 10.63976/jimat.v6i1.804<br><br>
    <b style="color:#52B788;">Data:</b><br>
    BPS Kota Medan — Jumlah Penduduk Kota Medan 2020–2025<br>
    BPS Provinsi Maluku — Jumlah Penduduk Kota Tual 2020–2024
    </div>
    """, unsafe_allow_html=True)
