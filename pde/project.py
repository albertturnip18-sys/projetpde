# ============================================================
#  STREAMLIT APP — ANIMASI ODE PERTUMBUHAN PENDUDUK KOTA TUAL
#  Model: Eksponensial & Logistik | Animasi Plotly
#  Referensi: Armin & Michael G.K. Remetwa (JIMAT, Vol.6 No.1, 2025)
#  Jalankan: streamlit run app_animasi_tual.py
#  FIX: PLOTLY_TEMPLATE diubah ke dict biasa agar bisa di-unpack
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="ODE Kota Tual · Kelompok 5",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS — PREMIUM EDITION ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #1B6CA8;
    color: #E8F6FF;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(255,255,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(45,212,191,0.10) 0%, transparent 60%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071F20 0%, #051A1B 100%);
    border-right: 1px solid rgba(45,212,191,0.18);
}
[data-testid="stSidebar"] * { color: #A7F3D0 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #14B8A6, #22D3EE) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 4px 0 !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0A2828 0%, #0D3030 100%);
    border: 1px solid rgba(45,212,191,0.18);
    border-top: 2px solid rgba(45,212,191,0.55);
    border-radius: 10px;
    padding: 18px 20px 14px 20px;
    transition: border-color 0.3s ease;
    box-shadow: 0 2px 12px rgba(20,184,166,0.12);
}
[data-testid="metric-container"]:hover {
    border-color: rgba(45,212,191,0.50) !important;
    border-top-color: rgba(34,211,238,0.7) !important;
}
[data-testid="metric-container"] label {
    color: #5EEAD4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2DD4BF !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #34D399 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(5,31,32,0.95) !important;
    border-bottom: 1px solid rgba(45,212,191,0.18) !important;
    gap: 2px !important;
    padding: 0 4px !important;
}
.stTabs [data-testid="stTab"] button {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #3D7A75 !important;
    letter-spacing: 0.3px;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 16px !important;
    transition: color 0.2s ease !important;
}
.stTabs [data-testid="stTab"] button[aria-selected="true"] {
    color: #2DD4BF !important;
    border-bottom: 2px solid #2DD4BF !important;
    background: rgba(45,212,191,0.08) !important;
}
.stTabs [data-testid="stTab"] button:hover {
    color: #5EEAD4 !important;
    background: rgba(45,212,191,0.05) !important;
}

/* ── BUTTONS ── */
div.stButton > button {
    background: transparent !important;
    color: #2DD4BF !important;
    border: 1px solid rgba(45,212,191,0.45) !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.8px;
    padding: 7px 16px !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: rgba(45,212,191,0.10) !important;
    border-color: #2DD4BF !important;
    box-shadow: 0 0 18px rgba(45,212,191,0.25) !important;
}

/* ── HERO PREMIUM ── */
.hero {
    background: linear-gradient(135deg, #071F22 0%, #0A2B2C 60%, #082820 100%);
    border: 1px solid rgba(45,212,191,0.22);
    border-radius: 14px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(45,212,191,0.08);
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #14B8A6 0%, #22D3EE 35%, #34D399 65%, #FBBF24 100%);
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(45,212,191,0.08) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #2DD4BF;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0 0 12px 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #E2F8F5;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero-title span { color: #2DD4BF; }
.hero-sub {
    font-size: 13.5px;
    color: #5EADA8;
    margin: 0 0 20px 0;
    line-height: 1.7;
    max-width: 680px;
}
.badge {
    display: inline-flex;
    align-items: center;
    background: rgba(45,212,191,0.08);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #2DD4BF;
    margin-right: 6px;
    margin-top: 4px;
    letter-spacing: 0.5px;
}
.badge-green {
    background: rgba(52,211,153,0.10);
    border-color: rgba(52,211,153,0.28);
    color: #34D399;
}
.hero-team {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid rgba(45,212,191,0.15);
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}
.hero-team-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #3D7A75;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-member {
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-member-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2DD4BF;
    flex-shrink: 0;
}
.hero-member-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #C8F7F0;
}
.hero-member-nim {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #3D7A75;
    margin-top: 1px;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: #2DD4BF;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(45,212,191,0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 20px; height: 2px;
    background: linear-gradient(90deg, #14B8A6, #22D3EE);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ── INFO CARD ── */
.info-card {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(45,212,191,0.18);
    border-left: 2px solid #14B8A6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: #7ECEC8;
    line-height: 1.85;
    margin: 8px 0;
}

/* ── FORMULA BOX ── */
.formula-box {
    background: #0A2828;
    border: 1px solid rgba(45,212,191,0.22);
    border-radius: 8px;
    padding: 20px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #2DD4BF;
    text-align: center;
    line-height: 2.2;
    margin: 14px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(20,184,166,0.10);
}
.formula-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.45), transparent);
}

/* ── JURNAL CARD ── */
.jurnal-highlight {
    background: linear-gradient(135deg, #071E20 0%, #0A2B2C 100%);
    border: 1px solid rgba(45,212,191,0.18);
    border-radius: 10px;
    padding: 20px 24px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease;
    box-shadow: 0 2px 10px rgba(20,184,166,0.10);
}
.jurnal-highlight:hover { border-color: rgba(45,212,191,0.45); }
.jurnal-highlight::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #2DD4BF, #34D399);
    border-radius: 3px 0 0 3px;
}
.jurnal-num {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: rgba(45,212,191,0.15);
    position: absolute;
    top: 12px; right: 20px;
    line-height: 1;
    user-select: none;
}
.jurnal-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #E2F8F5;
    margin-bottom: 6px;
}
.jurnal-body {
    font-size: 12.5px;
    color: #5EADA8;
    line-height: 1.75;
}
.jurnal-stat {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(45,212,191,0.08);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #2DD4BF;
    margin-top: 10px;
    margin-right: 6px;
}

/* ── DIVIDER ── */
.premium-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.25), transparent);
    margin: 20px 0;
    border: none;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(45,212,191,0.18) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── SELECTBOX & RADIO ── */
[data-baseweb="select"] {
    background: #071E20 !important;
    border-color: rgba(45,212,191,0.25) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #1B6CA8; }
::-webkit-scrollbar-thumb {
    background: rgba(45,212,191,0.35);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY PREMIUM TEMPLATE ─────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(5,31,32,0)",
        plot_bgcolor="#071E20",
        font=dict(family="DM Sans, sans-serif", color="#5EADA8", size=12),
        xaxis=dict(
            gridcolor="rgba(45,212,191,0.08)",
            linecolor="rgba(45,212,191,0.18)",
            zerolinecolor="rgba(45,212,191,0.12)",
            tickfont=dict(family="Space Mono, monospace", size=10, color="#3D7A75")
        ),
        yaxis=dict(
            gridcolor="rgba(45,212,191,0.08)",
            linecolor="rgba(45,212,191,0.18)",
            zerolinecolor="rgba(45,212,191,0.12)",
            tickfont=dict(family="Space Mono, monospace", size=10, color="#3D7A75")
        ),
        legend=dict(
            bgcolor="rgba(7,30,32,0.95)",
            bordercolor="rgba(45,212,191,0.20)",
            borderwidth=1,
            font=dict(size=11, family="DM Sans, sans-serif", color="#A7F0EB"),
        ),
        margin=dict(l=60, r=30, t=55, b=55),
        hoverlabel=dict(
            bgcolor="#071E20",
            bordercolor="rgba(45,212,191,0.45)",
            font=dict(family="Space Mono, monospace", size=11, color="#2DD4BF"),
        ),
    )
)

def make_layout(**kwargs):
    """Merge kwargs into PLOTLY_TEMPLATE['layout'], deep-merging any dict keys that clash."""
    base = dict(PLOTLY_TEMPLATE["layout"])
    # Deep-merge any key that exists in both base and kwargs and both are dicts
    for key in list(kwargs.keys()):
        if key in base and isinstance(base[key], dict) and isinstance(kwargs[key], dict):
            merged = dict(base.pop(key))
            merged.update(kwargs.pop(key))
            kwargs[key] = merged
    base.update(kwargs)
    return go.Layout(**base)

CYAN   = "#2DD4BF"   # Teal-400
TEAL   = "#22D3EE"   # Cyan-400
GREEN  = "#34D399"   # Emerald-400
AMBER  = "#FBBF24"   # Amber-400
PURPLE = "#818CF8"   # Indigo-400
CORAL  = "#FB7185"   # Rose-400
WHITE  = "#E2F8F5"   # Light teal-white
MUTED  = "#3D7A75"   # Ocean muted

# ── DATA & PARAMETER JURNAL ──────────────────────────────────
TAHUN_HIST  = np.array([2020, 2021, 2022, 2023, 2024], dtype=float)
POP_AKTUAL  = np.array([88280, 90322, 93145, 91572, 92744], dtype=float)
TAHUN_PRED  = np.array([2026, 2027, 2028, 2029, 2030])
POP_JURNAL  = np.array([95035, 96176, 97381, 98587, 99793])

P0_HIST = POP_AKTUAL[0]   # 88.280 (2020)
P0_PRED = POP_AKTUAL[-1]  # 92.744 (2024)
T_FIT   = TAHUN_HIST[-1] - TAHUN_HIST[0]  # 4 tahun
K_JURNAL = (1 / T_FIT) * np.log(P0_PRED / P0_HIST)  # 0.0122

K_DEFAULT = 150_000.0

# ── FUNGSI ODE & SOLUSI ──────────────────────────────────────
def ode_exp(P, t, k):    return k * P
def ode_log(P, t, k, K): return k * P * (1 - P / K)

def sol_exp(t, P0, k):    return P0 * np.exp(k * t)
def sol_log(t, P0, k, K): return K / (1 + ((K - P0) / P0) * np.exp(-k * t))

def mape(a, p): return np.mean(np.abs((a - p) / a)) * 100
def rmse(a, p): return np.sqrt(np.mean((a - p)**2))
def r2(a, p):
    ss_res = np.sum((a - p)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return 1 - ss_res / ss_tot

def euler(f, P0, t_span, dt, args=()):
    ts = np.arange(t_span[0], t_span[1] + dt, dt)
    Ps = np.zeros(len(ts)); Ps[0] = P0
    for i in range(1, len(ts)):
        Ps[i] = Ps[i-1] + dt * f(Ps[i-1], ts[i-1], *args)
    return ts, Ps

def rk4(f, P0, t_span, dt, args=()):
    ts = np.arange(t_span[0], t_span[1] + dt, dt)
    Ps = np.zeros(len(ts)); Ps[0] = P0
    for i in range(1, len(ts)):
        h  = dt; ti = ts[i-1]
        k1 = f(Ps[i-1], ti, *args)
        k2 = f(Ps[i-1] + h*k1/2, ti + h/2, *args)
        k3 = f(Ps[i-1] + h*k2/2, ti + h/2, *args)
        k4 = f(Ps[i-1] + h*k3,   ti + h,   *args)
        Ps[i] = Ps[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return ts, Ps

# ── CURVE FIT ────────────────────────────────────────────────
@st.cache_data
def fit_models():
    t_rel = TAHUN_HIST - TAHUN_HIST[0]
    popt_e, _ = curve_fit(
        lambda t, k: sol_exp(t, P0_HIST, k),
        t_rel, POP_AKTUAL, p0=[0.012], bounds=(0, 0.2)
    )
    popt_l, _ = curve_fit(
        lambda t, k, K: sol_log(t, P0_HIST, k, K),
        t_rel, POP_AKTUAL, p0=[0.05, 150000],
        bounds=([0, 93000], [1, 500000]), maxfev=10000
    )
    return popt_e[0], popt_l[0], popt_l[1]

k_fit_e, k_fit_l, K_fit_l = fit_models()
t_rel_h  = TAHUN_HIST - TAHUN_HIST[0]
pred_e_h = sol_exp(t_rel_h, P0_HIST, k_fit_e)
pred_l_h = sol_log(t_rel_h, P0_HIST, k_fit_l, K_fit_l)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
      <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:800;color:#2DD4BF;letter-spacing:-0.5px;line-height:1.2;">
        ODE Kota Tual
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:#3D7A75;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">
        Kelompok 5 · 2025
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">⚙ Parameter Model</div>', unsafe_allow_html=True)
    k_val = st.slider("k — laju pertumbuhan", 0.003, 0.050,
                      float(round(K_JURNAL, 4)), 0.001, format="%.4f",
                      help=f"Nilai jurnal: 0.0122 → setara {0.0122*100:.2f}%/tahun. "
                           f"Nilai saat ini: {float(round(K_JURNAL,4))*100:.2f}%/tahun")
    st.markdown(
        f"<div style='font-family:Space Mono,monospace;font-size:10px;"
        f"color:#34D399;margin-top:-6px;margin-bottom:10px;'>"
        f"→ {k_val:.4f} = <b>{k_val*100:.2f}%</b> / tahun</div>",
        unsafe_allow_html=True
    )
    K_val = st.slider("K — kapasitas dukung (jiwa)",
                      100_000, 300_000, int(K_DEFAULT), 5_000,
                      help="Daya dukung lingkungan untuk model logistik")
    st.markdown("---")
    st.markdown('<div class="section-label">🎬 Animasi</div>', unsafe_allow_html=True)
    anim_speed = st.select_slider("Kecepatan animasi",
                                  options=["Sangat Lambat", "Lambat", "Normal", "Cepat"],
                                  value="Normal")
    speed_map = {"Sangat Lambat": 200, "Lambat": 120, "Normal": 70, "Cepat": 30}
    frame_dur = speed_map[anim_speed]

    st.markdown("---")
    st.markdown('<div class="section-label">🔢 Numerik</div>', unsafe_allow_html=True)
    dt_val = st.selectbox("Langkah Δt (tahun)", [1.0, 0.5, 0.25, 0.1], index=1)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#1A4060;line-height:2;font-family:Space Mono,monospace;'>
    <span style='color:#2DD4BF;letter-spacing:1.5px;'>REFERENSI</span><br>
    Armin & Remetwa, M.G.K.<br>
    JIMAT Vol.6 No.1, 2025<br>
    DOI: 10.63976/jimat.v6i1.804<br><br>
    <span style='color:#2DD4BF;letter-spacing:1.5px;'>DATA SUMBER</span><br>
    BPS Provinsi Maluku<br>
    2020 – 2024<br><br>
    <div style='border-top:1px solid rgba(99,102,241,0.22);padding-top:12px;margin-top:4px;'>
    <span style='color:#2DD4BF;letter-spacing:1.5px;'>KELOMPOK 5</span><br>
    <span style='color:#2DD4BF;font-size:11px;'>Tugas Project</span><br>
    <span style='color:#3D7A75;'>Pemodelan Persamaan<br>Diferensial</span><br><br>
    <span style='color:#8AABB8;'>● Albert Rafael Turnip</span><br>
    <span style='color:#1A4060;'>  4243540002</span><br><br>
    <span style='color:#8AABB8;'>● Apriyani Simbolon</span><br>
    <span style='color:#1A4060;'>  4242240005</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── HERO BANNER ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">📐 Tugas Project · Persamaan Diferensial · Kelompok 5</div>
  <div class="hero-title">Pemodelan <span>Pertumbuhan Penduduk</span><br>Kota Tual 2026–2030</div>
  <div class="hero-sub">
    Aplikasi persamaan diferensial orde pertama dengan model eksponensial &amp; logistik ·
    Data BPS Maluku 2020–2024 · Referensi: Armin &amp; Remetwa, JIMAT Vol.6 No.1, 2025
  </div>
  <div>
    <span class="badge">dP/dt = k·P</span>
    <span class="badge">dP/dt = k·P·(1−P/K)</span>
    <span class="badge">P(t) = P₀·e^(kt)</span>
    <span class="badge">Euler &amp; RK4</span>
    <span class="badge badge-green">k = 1.22%/thn</span>
    <span class="badge badge-green">JIMAT 2025</span>
  </div>
  <div class="hero-team">
    <div class="hero-team-label">Kelompok 5</div>
    <div class="hero-member">
      <div class="hero-member-dot"></div>
      <div>
        <div class="hero-member-name">Albert Rafael Turnip</div>
        <div class="hero-member-nim">NIM · 4243540002</div>
      </div>
    </div>
    <div class="hero-member">
      <div class="hero-member-dot" style="background:#10B981;"></div>
      <div>
        <div class="hero-member-name">Apriyani Simbolon</div>
        <div class="hero-member-nim">NIM · 4242240005</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI METRICS ──────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("P₀ (2020)",     f"{int(P0_HIST):,}",     "jiwa awal")
c2.metric("P (2024)",      f"{int(P0_PRED):,}",     f"+{int(P0_PRED-P0_HIST):,}")
c3.metric("k Jurnal",      f"{K_JURNAL*100:.2f}%/thn",  f"k = {K_JURNAL:.4f}")
# FIX 4a: Hitung selisih dinamis dari data, hindari hardcode "+7.049" yang ambigu
delta_2030 = POP_JURNAL[-1] - P0_PRED
c4.metric("Pred. 2030",    f"{POP_JURNAL[-1]:,} jiwa",  f"+{delta_2030:,} jiwa vs 2024")
c5.metric("MAPE Eksponen", f"{mape(POP_AKTUAL, pred_e_h):.3f}%")
c6.metric("MAPE Logistik", f"{mape(POP_AKTUAL, pred_l_h):.3f}%")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎬  Animasi Utama",
    "🌀  Phase Portrait",
    "⚙️  Metode Numerik",
    "🔬  Sensitivitas",
    "📊  Tabel & Validasi",
    "📐  Derivasi ODE",
    "📖  Ringkasan Jurnal",
])


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — ANIMASI KURVA UTAMA                               ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-label">🎬 Animasi pertumbuhan penduduk 2020–2035</div>',
                unsafe_allow_html=True)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown("""
        <div class="info-card">
        Animasi menunjukkan bagaimana kurva solusi ODE terbentuk secara progresif dari tahun ke tahun.
        Titik kuning = data aktual BPS. Garis biru = model eksponensial. Garis hijau = model logistik.
        Garis putus-putus merah = kapasitas dukung K.
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.button("▶ Jalankan Animasi", key="btn1", use_container_width=True)

    # Data kurva penuh
    t_full  = np.linspace(0, 15, 300)
    tahun_f = 2020 + t_full
    P_exp_f = sol_exp(t_full, P0_HIST, k_val)
    P_log_f = sol_log(t_full, P0_HIST, k_val, K_val)

    # Buat frames untuk animasi
    n_frames = 60
    t_steps  = np.linspace(0, 15, n_frames)

    frames = []
    for i, t_end in enumerate(t_steps):
        mask    = t_full <= t_end
        t_slice = t_full[mask]
        yr_s    = tahun_f[mask]
        Pe_s    = P_exp_f[mask]
        Pl_s    = P_log_f[mask]

        frame_data = [
            go.Scatter(x=yr_s, y=Pe_s, mode="lines",
                       line=dict(color=CYAN, width=3)),
            go.Scatter(x=yr_s, y=Pl_s, mode="lines",
                       line=dict(color=GREEN, width=3)),
            go.Scatter(
                x=TAHUN_HIST[TAHUN_HIST <= 2020 + t_end],
                y=POP_AKTUAL[TAHUN_HIST <= 2020 + t_end],
                mode="markers",
                marker=dict(color=AMBER, size=12, symbol="circle",
                            line=dict(color=WHITE, width=1.5))
            ),
            go.Scatter(
                x=[2020, 2020 + t_end],
                y=[K_val, K_val],
                mode="lines",
                line=dict(color=CORAL, width=1.5, dash="dot"),
            ),
            go.Scatter(
                x=[2020 + t_end],
                y=[sol_exp(t_end, P0_HIST, k_val)],
                mode="markers+text",
                marker=dict(color=CYAN, size=14, symbol="diamond",
                            line=dict(color=WHITE, width=1.5)),
                text=[f"  {int(sol_exp(t_end, P0_HIST, k_val)):,}"],
                textposition="middle right",
                textfont=dict(color=CYAN, size=11, family="JetBrains Mono"),
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig1 = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", name="Eksponensial",
                       line=dict(color=CYAN, width=3)),
            go.Scatter(x=[], y=[], mode="lines", name="Logistik",
                       line=dict(color=GREEN, width=3)),
            go.Scatter(x=[], y=[], mode="markers", name="Data BPS Aktual",
                       marker=dict(color=AMBER, size=12, symbol="circle",
                                   line=dict(color=WHITE, width=1.5))),
            go.Scatter(x=[], y=[], mode="lines", name=f"K = {K_val:,}",
                       line=dict(color=CORAL, width=1.5, dash="dot")),
            go.Scatter(x=[], y=[], mode="markers+text", name="Posisi saat ini",
                       marker=dict(color=CYAN, size=14, symbol="diamond",
                                   line=dict(color=WHITE, width=1.5)),
                       showlegend=False),
        ],
        frames=frames,
        layout=make_layout(
            title=dict(text="Kurva Pertumbuhan Penduduk Kota Tual (2020–2035)",
                       font=dict(size=15, color=WHITE), x=0.01),
            xaxis=dict(title="Tahun", range=[2019, 2036],
                       gridcolor="rgba(255,255,255,0.05)",
                       linecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Jumlah Penduduk (jiwa)",
                       tickformat=",d",
                       # FIX 3: Range dinamis — batas bawah adaptif, fixedrange=False agar bisa zoom manual
                       range=[min(POP_AKTUAL.min() * 0.97, 82000),
                              max(K_val * 1.1, P_exp_f.max() * 1.05)],
                       fixedrange=False,
                       gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(13,27,46,0.85)",
                        bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
            height=520,
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.0, y=-0.12, xanchor="left",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=frame_dur, redraw=True),
                                         fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                    dict(label="↩ Reset",
                         method="animate",
                         args=[["0"], dict(frame=dict(duration=0, redraw=True),
                                           mode="immediate")]),
                ],
                bgcolor="#0D1B2E", bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=CYAN, size=12),
            )],
            sliders=[dict(
                currentvalue=dict(prefix="Tahun: ", font=dict(color=TEAL, size=12)),
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f"{2020 + t_steps[i]:.1f}",
                            method="animate")
                       for i, f in enumerate(frames)],
                x=0.0, y=-0.05, len=1.0,
                bgcolor="#0D1523",
                activebgcolor=CYAN,
                bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=MUTED, size=10),
            )],
        )
    )

    for yr, pop in zip(TAHUN_HIST, POP_AKTUAL):
        fig1.add_annotation(x=yr, y=pop, text=f"{int(pop):,}",
                            showarrow=False, yshift=18,
                            font=dict(color=AMBER, size=9, family="JetBrains Mono"),
                            bgcolor="rgba(13,27,46,0.7)", borderpad=3)
    fig1.add_vline(x=2024, line_dash="dash", line_color=AMBER,
                   line_width=1, opacity=0.5,
                   annotation_text="Batas Prediksi", annotation_font_color=AMBER)

    st.plotly_chart(fig1, use_container_width=True)

    col_e, col_j = st.columns(2)
    with col_e:
        st.markdown('<div class="section-label">Prediksi model (dari sidebar k, K)</div>',
                    unsafe_allow_html=True)
        t_pr = TAHUN_PRED - 2024
        df_pred = pd.DataFrame({
            "Tahun": TAHUN_PRED,
            "Eksponensial": sol_exp(t_pr, P0_PRED, k_val).astype(int),
            "Logistik":     sol_log(t_pr, P0_PRED, k_val, K_val).astype(int),
        })
        st.dataframe(df_pred.style.format({"Eksponensial": "{:,}", "Logistik": "{:,}"}),
                     hide_index=True, use_container_width=True)
    with col_j:
        st.markdown('<div class="section-label">Nilai referensi jurnal (k=0.0122)</div>',
                    unsafe_allow_html=True)
        df_jur = pd.DataFrame({
            "Tahun": TAHUN_PRED,
            "Jurnal (jiwa)": POP_JURNAL,
            "Selisih (app−jurnal)": (sol_exp(t_pr, P0_PRED, k_val).astype(int) - POP_JURNAL),
        })
        st.dataframe(df_jur.style.format({"Jurnal (jiwa)": "{:,}", "Selisih (app−jurnal)": "{:+,}"}),
                     hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — ANIMASI PHASE PORTRAIT                            ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-label">🌀 Animasi ruang fase — lintasan ODE (t, P, dP/dt)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    Phase portrait menampilkan laju perubahan dP/dt terhadap populasi P.
    Lingkaran bergerak menunjukkan posisi sistem saat ini.
    Eksponensial → laju terus naik. Logistik → laju memuncak di P=K/2 lalu turun ke 0 saat P→K.
    </div>
    """, unsafe_allow_html=True)

    P_range  = np.linspace(50000, min(K_val * 1.3, 220000), 500)
    dP_exp   = k_val * P_range
    dP_log   = k_val * P_range * (1 - P_range / K_val)

    t_anim   = np.linspace(0, 12, 80)
    P_traj_e = sol_exp(t_anim, P0_HIST, k_val)
    P_traj_l = sol_log(t_anim, P0_HIST, k_val, K_val)
    dP_traj_e = k_val * P_traj_e
    dP_traj_l = k_val * P_traj_l * (1 - P_traj_l / K_val)

    frames_ph = []
    for i in range(len(t_anim)):
        frames_ph.append(go.Frame(
            data=[
                go.Scatter(x=P_range, y=dP_exp, mode="lines",
                           line=dict(color=CYAN, width=2.5), name="dP/dt Eksponensial"),
                go.Scatter(x=P_range, y=dP_log, mode="lines",
                           line=dict(color=GREEN, width=2.5), name="dP/dt Logistik"),
                go.Scatter(x=[P_traj_e[i]], y=[dP_traj_e[i]], mode="markers",
                           marker=dict(color=CYAN, size=16, symbol="circle",
                                       line=dict(color=WHITE, width=2)),
                           name="Posisi Eksp.",
                           text=[f"P={P_traj_e[i]:,.0f}<br>dP/dt={dP_traj_e[i]:,.0f}"],
                           hoverinfo="text"),
                go.Scatter(x=[P_traj_l[i]], y=[dP_traj_l[i]], mode="markers",
                           marker=dict(color=GREEN, size=16, symbol="diamond",
                                       line=dict(color=WHITE, width=2)),
                           name="Posisi Log.",
                           text=[f"P={P_traj_l[i]:,.0f}<br>dP/dt={dP_traj_l[i]:,.0f}"],
                           hoverinfo="text"),
            ],
            name=str(i)
        ))

    fig2 = go.Figure(
        data=[
            go.Scatter(x=P_range, y=dP_exp, mode="lines", name="dP/dt Eksponensial",
                       line=dict(color=CYAN, width=2.5)),
            go.Scatter(x=P_range, y=dP_log, mode="lines", name="dP/dt Logistik",
                       line=dict(color=GREEN, width=2.5)),
            go.Scatter(x=[P0_HIST], y=[k_val * P0_HIST], mode="markers",
                       marker=dict(color=CYAN, size=16, symbol="circle",
                                   line=dict(color=WHITE, width=2)), name="Posisi Eksp."),
            go.Scatter(x=[P0_HIST], y=[k_val * P0_HIST * (1 - P0_HIST/K_val)],
                       mode="markers",
                       marker=dict(color=GREEN, size=16, symbol="diamond",
                                   line=dict(color=WHITE, width=2)), name="Posisi Log."),
        ],
        frames=frames_ph,
        layout=make_layout(
            title=dict(text="Phase Portrait: dP/dt vs P — Gerak Sistem ODE",
                       font=dict(size=15, color=WHITE), x=0.01),
            xaxis=dict(title="Populasi P (jiwa)", tickformat=",d",
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Laju dP/dt (jiwa/tahun)", tickformat=",d",
                       gridcolor="rgba(255,255,255,0.05)"),
            height=480,
            shapes=[
                dict(type="line", x0=K_val/2, x1=K_val/2,
                     y0=0, y1=k_val * K_val/4,
                     line=dict(color=PURPLE, width=1.5, dash="dot")),
                dict(type="line", x0=K_val, x1=K_val,
                     y0=dP_log.min() * 0.5, y1=k_val * K_val * 0.3,
                     line=dict(color=CORAL, width=1.5, dash="dot")),
            ],
            annotations=[
                dict(x=K_val/2, y=k_val * K_val/4,
                     text=f"Infleksi<br>P=K/2={K_val//2:,}",
                     showarrow=True, arrowhead=2, arrowcolor=PURPLE,
                     font=dict(color=PURPLE, size=10), ax=40, ay=-30),
                dict(x=K_val, y=0,
                     text=f"K={K_val:,}<br>(Ekuilibrium)",
                     showarrow=True, arrowhead=2, arrowcolor=CORAL,
                     font=dict(color=CORAL, size=10), ax=40, ay=-20),
            ],
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.0, y=-0.15, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_dur, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                    dict(label="↩ Reset", method="animate",
                         args=[["0"], dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
                ],
                bgcolor="#0D1B2E", bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=CYAN, size=12),
            )],
            sliders=[dict(
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f"t={t_anim[i]:.1f}", method="animate")
                       for i, f in enumerate(frames_ph)],
                x=0.0, y=-0.05, len=1.0,
                currentvalue=dict(prefix="t = ", suffix=" thn",
                                  font=dict(color=TEAL, size=12)),
                bgcolor="#0D1523", activebgcolor=CYAN,
                bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=MUTED, size=10),
            )],
        )
    )
    fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — ANIMASI METODE NUMERIK                            ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-label">⚙️ Animasi konvergensi metode numerik ODE</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    Animasi menunjukkan bagaimana metode Euler dan RK4 mengaproksimasi solusi analitik secara bertahap.
    RK4 (ungu) jauh lebih akurat dibanding Euler (oranye) terutama saat Δt besar.
    Error Euler ditampilkan sebagai shaded area merah.
    </div>
    """, unsafe_allow_html=True)

    t_an   = np.linspace(0, 10, 500)
    P_anal = sol_exp(t_an, P0_HIST, k_val)

    t_e, P_e = euler(ode_exp, P0_HIST, (0, 10), dt_val, args=(k_val,))
    t_r, P_r = rk4(ode_exp,   P0_HIST, (0, 10), dt_val, args=(k_val,))

    n_num_frames = min(len(t_e), 60)
    step_num = max(1, len(t_e) // n_num_frames)
    indices  = list(range(0, len(t_e), step_num))

    frames_num = []
    for idx in indices:
        t_e_s  = t_e[:idx+1]; P_e_s = P_e[:idx+1]
        t_r_s  = t_r[:idx+1]; P_r_s = P_r[:idx+1]
        mask_a = t_an <= t_e[idx]
        err_e  = np.abs(P_e_s - sol_exp(t_e_s, P0_HIST, k_val))
        frames_num.append(go.Frame(data=[
            go.Scatter(x=t_an[mask_a], y=P_anal[mask_a], mode="lines",
                       line=dict(color=TEAL, width=3, dash="solid"), name="Analitik"),
            go.Scatter(x=t_e_s, y=P_e_s, mode="lines+markers",
                       line=dict(color=AMBER, width=2, dash="dash"),
                       marker=dict(size=5, color=AMBER), name=f"Euler (Δt={dt_val})"),
            go.Scatter(x=t_r_s, y=P_r_s, mode="lines+markers",
                       line=dict(color=PURPLE, width=2, dash="dot"),
                       marker=dict(size=5, color=PURPLE), name=f"RK4 (Δt={dt_val})"),
            go.Scatter(
                x=np.concatenate([t_e_s, t_e_s[::-1]]),
                y=np.concatenate([P_e_s + err_e, (P_e_s - err_e)[::-1]]),
                fill="toself", fillcolor="rgba(252,129,129,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="Error Euler", showlegend=True
            ),
        ], name=str(idx)))

    fig3 = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="lines", name="Analitik",
                       line=dict(color=TEAL, width=3)),
            go.Scatter(x=[], y=[], mode="lines+markers", name=f"Euler Δt={dt_val}",
                       line=dict(color=AMBER, width=2, dash="dash"),
                       marker=dict(size=5, color=AMBER)),
            go.Scatter(x=[], y=[], mode="lines+markers", name=f"RK4 Δt={dt_val}",
                       line=dict(color=PURPLE, width=2, dash="dot"),
                       marker=dict(size=5, color=PURPLE)),
            go.Scatter(x=[], y=[], fill="toself",
                       fillcolor="rgba(252,129,129,0.08)",
                       line=dict(color="rgba(0,0,0,0)"), name="Error Euler"),
        ],
        frames=frames_num,
        layout=make_layout(
            title=dict(text=f"Konvergensi Numerik ODE — Analitik vs Euler vs RK4 (Δt={dt_val})",
                       font=dict(size=15, color=WHITE), x=0.01),
            xaxis=dict(title="Waktu t (tahun dari 2020)", range=[-0.2, 10.5],
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Populasi (jiwa)", tickformat=",d",
                       gridcolor="rgba(255,255,255,0.05)"),
            height=480,
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.0, y=-0.15, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_dur, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                    dict(label="↩ Reset", method="animate",
                         args=[["0"], dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
                ],
                bgcolor="#0D1B2E", bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=CYAN, size=12),
            )],
            sliders=[dict(
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f"t={t_e[int(f.name)]:.2f}",
                            method="animate")
                       for f in frames_num],
                x=0.0, y=-0.05, len=1.0,
                currentvalue=dict(prefix="t = ", suffix=" thn",
                                  font=dict(color=TEAL, size=12)),
                bgcolor="#0D1523", activebgcolor=CYAN,
                bordercolor="rgba(99,179,237,0.3)",
                font=dict(color=MUTED, size=10),
            )],
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-label">Tabel error numerik di titik data historis</div>',
                unsafe_allow_html=True)
    t_chk = np.array([0., 1., 2., 3., 4.])
    P_a_c = sol_exp(t_chk, P0_HIST, k_val)
    # FIX 1: Gunakan argmin agar aman dari floating-point error (misal 1.0/0.1 = 9.9999)
    idx_e_c = [np.abs(t_e - t).argmin() for t in t_chk]
    idx_r_c = [np.abs(t_r - t).argmin() for t in t_chk]
    Pe_c = P_e[idx_e_c]
    Pr_c = P_r[idx_r_c]
    df_err = pd.DataFrame({
        "Tahun":     TAHUN_HIST.astype(int),
        "Analitik":  P_a_c.astype(int),
        "Euler":     Pe_c.astype(int),
        "RK4":       Pr_c.astype(int),
        "Error Euler": np.abs(Pe_c - P_a_c).astype(int),
        "Error RK4":   np.abs(Pr_c - P_a_c).astype(int),
    })
    st.dataframe(df_err.style.format({
        c: "{:,}" for c in df_err.columns[1:]
    }),
    hide_index=True, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — SENSITIVITAS DINAMIS                              ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="section-label">🔬 Analisis sensitivitas dinamis — pengaruh k dan K</div>',
                unsafe_allow_html=True)

    col_sa, col_sb = st.columns(2)

    with col_sa:
        st.markdown("**Variasi laju pertumbuhan k (model eksponensial)**")
        k_vals   = np.arange(0.005, 0.031, 0.005)
        colors_k = px.colors.sequential.Plasma[1::2][:len(k_vals)]
        t_s      = np.linspace(0, 10, 300)
        yr_s     = 2024 + t_s

        fig4a = go.Figure(layout=make_layout(
            title=dict(text="Sensitivitas k — Eksponensial",
                       font=dict(size=14, color=WHITE), x=0.01),
            xaxis=dict(title="Tahun", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Populasi (jiwa)", tickformat=",d",
                       gridcolor="rgba(255,255,255,0.05)"),
            height=360,
        ))
        for k_v, col in zip(k_vals, colors_k):
            P = sol_exp(t_s, P0_PRED, k_v)
            fig4a.add_trace(go.Scatter(
                x=yr_s, y=P, mode="lines", name=f"k={k_v:.3f} ({k_v*100:.1f}%)",
                line=dict(color=col, width=2),
            ))
        fig4a.add_vline(x=2030, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        # FIX 2: Garis batas awal prediksi — x=2024 langsung (bukan ekspresi * 0 yg selalu 2024 tapi menyesatkan)
        fig4a.add_vline(x=2024, line_dash="dash", line_color=AMBER,
                        line_width=1, opacity=0.6,
                        annotation_text="Mulai prediksi",
                        annotation_font_color=AMBER,
                        annotation_position="top right")
        fig4a.add_trace(go.Scatter(
            x=yr_s, y=sol_exp(t_s, P0_PRED, K_JURNAL),
            mode="lines", name="k=0.0122 (Jurnal)",
            line=dict(color=WHITE, width=3, dash="dash"),
        ))
        st.plotly_chart(fig4a, use_container_width=True)

    with col_sb:
        st.markdown("**Variasi kapasitas dukung K (model logistik)**")
        K_vals   = [110000, 130000, 150000, 175000, 200000, 250000]
        colors_K = px.colors.sequential.Viridis[::2][:len(K_vals)]

        fig4b = go.Figure(layout=make_layout(
            title=dict(text="Sensitivitas K — Logistik",
                       font=dict(size=14, color=WHITE), x=0.01),
            xaxis=dict(title="Tahun", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Populasi (jiwa)", tickformat=",d",
                       gridcolor="rgba(255,255,255,0.05)"),
            height=360,
        ))
        for K_v, col in zip(K_vals, colors_K):
            P = sol_log(t_s, P0_PRED, K_JURNAL, K_v)
            fig4b.add_trace(go.Scatter(
                x=yr_s, y=P, mode="lines", name=f"K={K_v:,}",
                line=dict(color=col, width=2),
            ))
            fig4b.add_hline(y=K_v, line_dash="dot", line_color=col,
                            opacity=0.3, line_width=1)
        fig4b.add_vline(x=2030, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig4b, use_container_width=True)

    st.markdown('<div class="section-label">Heatmap MAPE vs k — seberapa sensitif akurasi terhadap k?</div>',
                unsafe_allow_html=True)
    k_grid    = np.linspace(0.005, 0.030, 30)
    mape_grid = []
    for kk in k_grid:
        pred = sol_exp(t_rel_h, P0_HIST, kk)
        mape_grid.append(mape(POP_AKTUAL, pred))

    fig4c = go.Figure(layout=make_layout(
        title=dict(text="MAPE Model Eksponensial sebagai Fungsi k",
                   font=dict(size=14, color=WHITE), x=0.01),
        xaxis=dict(title="Laju Pertumbuhan k", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="MAPE (%)", gridcolor="rgba(255,255,255,0.05)"),
        height=300,
    ))
    fig4c.add_trace(go.Scatter(
        x=k_grid, y=mape_grid, mode="lines+markers",
        line=dict(color=CYAN, width=2.5),
        marker=dict(size=5, color=CYAN),
        fill="tozeroy", fillcolor="rgba(99,179,237,0.07)",
        name="MAPE(%)"
    ))
    fig4c.add_vline(x=K_JURNAL, line_dash="dash", line_color=AMBER,
                    annotation_text=f"k jurnal = {K_JURNAL:.4f}",
                    annotation_font_color=AMBER)
    fig4c.add_vline(x=k_fit_e, line_dash="dash", line_color=GREEN,
                    annotation_text=f"k fit = {k_fit_e:.4f}",
                    annotation_font_color=GREEN,
                    annotation_position="bottom right")
    st.plotly_chart(fig4c, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 5 — TABEL & VALIDASI                                  ║
# ╚══════════════════════════════════════════════════════════════╝
with tab5:
    st.markdown('<div class="section-label">📊 Tabel lengkap & validasi terhadap jurnal</div>',
                unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Metrik akurasi — fit ke data historis 2020–2024**")
        df_met = pd.DataFrame({
            "Metrik":        ["k laju pertumbuhan", "MAPE (%)", "RMSE (jiwa)", "R²"],
            "Eksponensial":  [f"{k_fit_e*100:.4f}%/thn",
                              f"{mape(POP_AKTUAL, pred_e_h):.4f}%",
                              f"{rmse(POP_AKTUAL, pred_e_h):,.2f}",
                              f"{r2(POP_AKTUAL, pred_e_h):.6f}"],
            "Logistik":      [f"{k_fit_l*100:.4f}%/thn",
                              f"{mape(POP_AKTUAL, pred_l_h):.4f}%",
                              f"{rmse(POP_AKTUAL, pred_l_h):,.2f}",
                              f"{r2(POP_AKTUAL, pred_l_h):.6f}"],
        })
        st.dataframe(df_met, hide_index=True, use_container_width=True)

    with col_m2:
        st.markdown("**Validasi prediksi vs nilai jurnal (k=0.0122)**")
        t_pr = TAHUN_PRED - 2024
        app_pred = sol_exp(t_pr, P0_PRED, K_JURNAL).astype(int)
        df_val = pd.DataFrame({
            "Tahun":          TAHUN_PRED,
            "App (jiwa)":     app_pred,
            "Jurnal (jiwa)":  POP_JURNAL,
            "Deviasi (jiwa)": app_pred - POP_JURNAL,
            "Deviasi (%)":    ((app_pred - POP_JURNAL) / POP_JURNAL * 100).round(4),
        })
        st.dataframe(df_val.style.format({
            "App (jiwa)":     "{:,}",
            "Jurnal (jiwa)":  "{:,}",
            "Deviasi (jiwa)": "{:+,}",
            "Deviasi (%)":    "{:+.4f}%",
        }),
        hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Data historis BPS vs prediksi model</div>',
                unsafe_allow_html=True)
    df_hist = pd.DataFrame({
        "Tahun":              TAHUN_HIST.astype(int),
        "Aktual BPS (jiwa)":  POP_AKTUAL.astype(int),
        "Pred. Eksp. (jiwa)": pred_e_h.astype(int),
        "Pred. Log. (jiwa)":  pred_l_h.astype(int),
        "Error Eksp.":        (POP_AKTUAL - pred_e_h).astype(int),
        "Error Log.":         (POP_AKTUAL - pred_l_h).astype(int),
        "APE Eksp. (%)":      (np.abs(POP_AKTUAL - pred_e_h) / POP_AKTUAL * 100).round(4),
        "APE Log. (%)":       (np.abs(POP_AKTUAL - pred_l_h) / POP_AKTUAL * 100).round(4),
    })
    st.dataframe(df_hist.style.format({
        c: "{:,}" for c in ["Aktual BPS (jiwa)", "Pred. Eksp. (jiwa)", "Pred. Log. (jiwa)",
                             "Error Eksp.", "Error Log."]
    } | {"APE Eksp. (%)": "{:.4f}%", "APE Log. (%)": "{:.4f}%"}),
    hide_index=True, use_container_width=True)

    st.markdown('<div class="section-label">Visualisasi perbandingan prediksi 2026–2030</div>',
                unsafe_allow_html=True)
    t_pr2  = TAHUN_PRED - 2024
    p_e_pr = sol_exp(t_pr2, P0_PRED, k_val).astype(int)
    p_l_pr = sol_log(t_pr2, P0_PRED, k_val, K_val).astype(int)

    fig5 = go.Figure(layout=make_layout(
        title=dict(text="Prediksi Jumlah Penduduk Kota Tual 2026–2030",
                   font=dict(size=15, color=WHITE), x=0.01),
        xaxis=dict(title="Tahun", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Populasi (jiwa)", tickformat=",d",
                   range=[88000, max(p_e_pr.max(), p_l_pr.max()) * 1.07],
                   gridcolor="rgba(255,255,255,0.05)"),
        barmode="group", height=380,
    ))
    fig5.add_trace(go.Bar(x=TAHUN_PRED, y=p_e_pr, name="Eksponensial",
                          marker_color=CYAN, marker_line_color="rgba(99,179,237,0.5)",
                          marker_line_width=1,
                          text=[f"{v:,}" for v in p_e_pr], textposition="outside",
                          textfont=dict(color=CYAN, size=10)))
    fig5.add_trace(go.Bar(x=TAHUN_PRED, y=p_l_pr, name="Logistik",
                          marker_color=GREEN, marker_line_color="rgba(104,211,145,0.5)",
                          marker_line_width=1,
                          text=[f"{v:,}" for v in p_l_pr], textposition="outside",
                          textfont=dict(color=GREEN, size=10)))
    fig5.add_trace(go.Scatter(x=TAHUN_PRED, y=POP_JURNAL, mode="markers+lines",
                              name="Referensi Jurnal",
                              marker=dict(color=AMBER, size=10, symbol="star"),
                              line=dict(color=AMBER, width=2, dash="dot")))
    st.plotly_chart(fig5, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 6 — DERIVASI ODE & PENJELASAN                         ║
# ╚══════════════════════════════════════════════════════════════╝
with tab6:
    st.markdown('<div class="section-label">📐 Derivasi matematis ODE sesuai jurnal</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Model Eksponensial")
        st.markdown("""
        <div class="formula-box">
        Persamaan ODE:<br>
        dP/dt = k · P<br><br>
        Separasi variabel:<br>
        dP / P = k · dt<br><br>
        Integrasi kedua sisi:<br>
        ln P = k·t + C<br><br>
        Solusi Analitik:<br>
        P(t) = P₀ · e^(kt)<br><br>
        Estimasi k:<br>
        k = (1/t) · ln(P(t) / P₀)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
        Dengan data jurnal:<br>
        P₀ = 88.280 (2020) · P(t) = 92.744 (2024) · t = 4 tahun<br><br>
        k = (1/4) · ln(92.744 / 88.280)<br>
        k = (1/4) · ln(1.0505)<br>
        k = (1/4) · 0.0492 = <strong>0.0122 (1.22%/tahun)</strong>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("#### Model Logistik (Pembanding)")
        st.markdown("""
        <div class="formula-box">
        Persamaan ODE:<br>
        dP/dt = k · P · (1 − P/K)<br><br>
        K = kapasitas dukung lingkungan<br><br>
        Saat P &lt;&lt; K: dP/dt ≈ k·P (eksponensial)<br>
        Saat P → K: dP/dt → 0 (stagnasi)<br>
        Saat P = K/2: dP/dt maksimum<br><br>
        Solusi Analitik:<br>
        P(t) = K / (1 + ((K−P₀)/P₀)·e^(−kt))
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
        Catatan jurnal: Model eksponensial cocok untuk prediksi
        <strong>jangka pendek</strong> (2026–2030) dengan sumber daya besar.<br><br>
        Untuk prediksi jangka panjang, disarankan menggunakan
        <strong>model logistik</strong> yang mempertimbangkan daya dukung lingkungan (K),
        kebijakan pemerintah, migrasi, dan keterbatasan sumber daya.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # FIX 3: Gunakan k_val dari sidebar agar tabel dinamis mengikuti input pengguna.
    # Label seksi menyertakan k aktif vs k jurnal agar pembaca tahu angka dari mana.
    st.markdown(
        f'<div class="section-label">Tabel prediksi — k aktif = ' +
        f'{k_val:.4f} ({k_val*100:.2f}%/thn) | k jurnal = {K_JURNAL:.4f} ({K_JURNAL*100:.2f}%/thn)</div>',
        unsafe_allow_html=True
    )

    t_pr3    = TAHUN_PRED - 2024
    col_ekt  = f"e^(k·t)  k={k_val:.4f}"
    col_pt   = "P(t) = P₀·e^(kt)"
    p_model  = (P0_PRED * np.exp(k_val * t_pr3)).astype(int)
    df_full  = pd.DataFrame({
        "Tahun":           TAHUN_PRED,
        "t (dari 2024)":   t_pr3,
        col_ekt:           np.exp(k_val * t_pr3).round(4),
        col_pt:            p_model,
        "Nilai Jurnal":    POP_JURNAL,
        "Selisih vs Jrn":  p_model - POP_JURNAL,
        "Pertambahan":     np.concatenate([[0], np.diff(p_model)]).astype(int),
    })
    st.dataframe(df_full.style.format({
        col_pt:            "{:,}",
        "Nilai Jurnal":    "{:,}",
        "Selisih vs Jrn":  "{:+,}",
        "Pertambahan":     "{:+,}",
    }),
    hide_index=True, use_container_width=True)

    st.markdown('<div class="section-label">Waterfall — pertambahan penduduk per tahun</div>',
                unsafe_allow_html=True)
    diffs      = np.diff(np.concatenate([[P0_PRED], POP_JURNAL]))
    all_years  = np.concatenate([[2024], TAHUN_PRED])
    all_pop    = np.concatenate([[P0_PRED], POP_JURNAL])

    fig6 = go.Figure(layout=make_layout(
        title=dict(text="Pertambahan Penduduk Kota Tual per Tahun (Jurnal)",
                   font=dict(size=15, color=WHITE), x=0.01),
        height=360,
    ))
    fig6.add_trace(go.Waterfall(
        x=[str(y) for y in all_years],
        y=[P0_PRED] + list(diffs),
        measure=["absolute"] + ["relative"] * len(diffs),
        text=[f"{int(v):+,}" if i > 0 else f"{int(v):,}" for i, v in enumerate([P0_PRED] + list(diffs))],
        textposition="outside",
        textfont=dict(color=WHITE, size=10),
        increasing_marker_color=GREEN,
        decreasing_marker_color=CORAL,
        totals_marker_color=CYAN,
        connector_line_color="rgba(255,255,255,0.2)",
    ))
    st.plotly_chart(fig6, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 7 — RINGKASAN JURNAL INTERAKTIF                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tab7:
    st.markdown('<div class="section-label">📖 Ringkasan jurnal referensi — Armin & Remetwa, JIMAT 2025</div>',
                unsafe_allow_html=True)

    # ── HEADER JURNAL ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#071E22,#0A2B2C);border:1px solid rgba(45,212,191,0.20);
    border-radius:10px;padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,#14B8A6,#22D3EE,transparent);"></div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2DD4BF;letter-spacing:2px;margin-bottom:8px;">
        JIMAT · Vol.6 No.1 · Juni 2025 · e-ISSN: 2774-1729
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#C8F7F0;
      line-height:1.4;margin-bottom:10px;">
        Aplikasi Persamaan Differensial Dengan Pendekatan Model Pertumbuhan Eksponensial<br>
        Untuk Memprediksi Jumlah Penduduk Kota Tual Tahun 2026–2030
      </div>
      <div style="font-family:'DM Sans',sans-serif;font-size:12.5px;color:#5EADA8;">
        <b style="color:#2DD4BF;">Armin</b> · Politeknik Perikanan Negeri Tual (Teknologi Kelautan)
        &nbsp;|&nbsp;
        <b style="color:#34D399;">Michael Gerits Kriswanto Remetwa</b> · Politeknik Perikanan Negeri Tual (Agrowisata Bahari)
      </div>
      <div style="margin-top:12px;">
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:#3D7A75;background:rgba(45,212,191,0.08);
        border:1px solid rgba(45,212,191,0.18);border-radius:3px;padding:3px 8px;margin-right:6px;">
          DOI: 10.63976/jimat.v6i1.804
        </span>
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:#3D7A75;background:rgba(45,212,191,0.06);
        border:1px solid rgba(52,211,153,0.18);border-radius:3px;padding:3px 8px;">
          Halaman 327–338
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── POIN-POIN KUNCI ──
    col_j1, col_j2 = st.columns(2)

    with col_j1:
        st.markdown("""
        <div class="jurnal-highlight">
          <div class="jurnal-num">01</div>
          <div class="jurnal-title">Latar Belakang & Tujuan</div>
          <div class="jurnal-body">
            Kota Tual sebagai kota persinggahan wilayah Maluku Tenggara belum memiliki proyeksi
            penduduk resmi. Penelitian ini bertujuan memprediksi jumlah penduduk 2026–2030
            menggunakan persamaan diferensial dengan asumsi pertumbuhan eksponensial —
            model yang cocok untuk prediksi <b style="color:#2DD4BF">jangka pendek dengan sumber daya besar</b>.
          </div>
          <span class="jurnal-stat">🎯 Prediksi jangka pendek</span>
          <span class="jurnal-stat">📍 Kota Tual, Maluku</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="jurnal-highlight" style="margin-top:12px;">
          <div class="jurnal-num">03</div>
          <div class="jurnal-title">Perhitungan Laju Pertumbuhan k</div>
          <div class="jurnal-body">
            Menggunakan data P₀ = 88.280 jiwa (2020) dan P(t) = 92.744 jiwa (2024), t = 4 tahun:<br><br>
            <code style="font-family:'Space Mono',monospace;color:#2DD4BF;font-size:11px;">
              k = (1/4) · ln(92.744 / 88.280)<br>
              k = (1/4) · ln(1,0505)<br>
              k = (1/4) · 0,0488 = <b>0,0122</b>
            </code>
          </div>
          <span class="jurnal-stat">📈 k = 0.0122</span>
          <span class="jurnal-stat" style="color:#34D399;border-color:rgba(16,185,129,0.2);">1.22% / tahun</span>
        </div>
        """, unsafe_allow_html=True)

    with col_j2:
        st.markdown("""
        <div class="jurnal-highlight">
          <div class="jurnal-num">02</div>
          <div class="jurnal-title">Data & Metode</div>
          <div class="jurnal-body">
            Data sekunder dari <b style="color:#2DD4BF">BPS Provinsi Maluku</b> tahun 2020–2024.
            Jenis penelitian: kuantitatif deskriptif. Model matematika persamaan diferensial:<br><br>
            <code style="font-family:'Space Mono',monospace;color:#2DD4BF;font-size:11px;">
              dP/P = k dt &nbsp;→&nbsp; P(t) = P₀·e^(kt)
            </code>
          </div>
          <span class="jurnal-stat">🏛 BPS Maluku 2020–2024</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="jurnal-highlight" style="margin-top:12px;">
          <div class="jurnal-num">04</div>
          <div class="jurnal-title">Hasil Prediksi Utama</div>
          <div class="jurnal-body">
            Dengan P₀ = 92.744 jiwa (2024) dan k = 0,0122, prediksi jumlah penduduk:
          </div>
        """, unsafe_allow_html=True)

        # Tabel mini hasil jurnal
        df_j = pd.DataFrame({
            "Tahun": [2026, 2027, 2028, 2029, 2030],
            "Jiwa":  [95035, 96176, 97381, 98587, 99793],
            "Tambah": ["+2.291", "+1.141", "+1.205", "+1.206", "+1.206"],
        })
        st.dataframe(
            df_j.style
                .format({"Jiwa": "{:,}"})
                .set_properties(**{"font-family": "Space Mono, monospace", "font-size": "11px"}),
            hide_index=True,
            use_container_width=True,
            height=215,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='premium-divider'>", unsafe_allow_html=True)

    # ── INSIGHT & REKOMENDASI ──
    col_j3, col_j4, col_j5 = st.columns(3)

    with col_j3:
        st.markdown("""
        <div class="jurnal-highlight">
          <div class="jurnal-num">05</div>
          <div class="jurnal-title">Temuan Kunci</div>
          <div class="jurnal-body">
            Selama 6 tahun (2024–2030) penduduk Kota Tual diproyeksikan bertambah
            <b style="color:#2DD4BF;">7.049 jiwa</b> — dari 92.744 menjadi 99.793 jiwa,
            dengan tren positif yang konsisten dan laju 1,22% per tahun.
          </div>
          <span class="jurnal-stat">+7.049 jiwa total</span>
        </div>
        """, unsafe_allow_html=True)

    with col_j4:
        st.markdown("""
        <div class="jurnal-highlight">
          <div class="jurnal-num">06</div>
          <div class="jurnal-title">Batasan Model</div>
          <div class="jurnal-body">
            Model eksponensial <b style="color:#F59E0B;">hanya disarankan jangka pendek</b>.
            Untuk jangka panjang disarankan model logistik yang mempertimbangkan:
            daya dukung lingkungan (K), migrasi, dan kebijakan pemerintah.
          </div>
          <span class="jurnal-stat" style="color:#F59E0B;border-color:rgba(245,158,11,0.2);">⚠ Jangka pendek saja</span>
        </div>
        """, unsafe_allow_html=True)

    with col_j5:
        st.markdown("""
        <div class="jurnal-highlight">
          <div class="jurnal-num">07</div>
          <div class="jurnal-title">Rekomendasi Kebijakan</div>
          <div class="jurnal-body">
            Hasil penelitian ini menjadi acuan untuk:
            pengendalian laju pertumbuhan berbasis data,
            optimalisasi tata ruang, perencanaan infrastruktur,
            dan strategi sosio-ekonomi jangka pendek pemerintah Kota Tual.
          </div>
          <span class="jurnal-stat" style="color:#34D399;border-color:rgba(16,185,129,0.2);">🏛 Kebijakan publik</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='premium-divider'>", unsafe_allow_html=True)

    # ── VISUALISASI RINGKASAN JURNAL ──
    st.markdown('<div class="section-label">Visualisasi ringkasan: perbandingan data aktual vs prediksi jurnal</div>',
                unsafe_allow_html=True)

    all_y = np.concatenate([TAHUN_HIST, TAHUN_PRED])
    all_p_jurnal = np.concatenate([POP_AKTUAL, POP_JURNAL])
    all_p_model  = np.concatenate([
        sol_exp(TAHUN_HIST - TAHUN_HIST[0], P0_HIST, K_JURNAL),
        sol_exp(TAHUN_PRED - 2024, P0_PRED, K_JURNAL)
    ])

    fig_j = go.Figure(layout=make_layout(
        title=dict(text="Data BPS 2020–2024 + Prediksi Jurnal 2026–2030 · Kota Tual",
                   font=dict(size=14, color=WHITE, family="Syne, sans-serif"), x=0.01),
        xaxis=dict(title="Tahun", dtick=1, gridcolor="rgba(45,212,191,0.08)"),
        yaxis=dict(title="Jumlah Penduduk (jiwa)", tickformat=",d",
                   gridcolor="rgba(45,212,191,0.08)"),
        height=400,
        shapes=[dict(
            type="rect", x0=2024.5, x1=2030.5,
            y0=0, y1=1, yref="paper",
            fillcolor="rgba(99,102,241,0.04)",
            line=dict(color="rgba(0,0,0,0)")
        )],
        annotations=[dict(
            x=2027, y=1, yref="paper",
            text="ZONA PREDIKSI",
            showarrow=False,
            font=dict(family="Space Mono, monospace", size=9,
                      color="rgba(99,102,241,0.3)"),
            yshift=-12
        )]
    ))

    # Shaded area antara aktual dan model
    fig_j.add_trace(go.Scatter(
        x=np.concatenate([TAHUN_HIST, TAHUN_HIST[::-1]]),
        y=np.concatenate([POP_AKTUAL, sol_exp(TAHUN_HIST[::-1]-TAHUN_HIST[0], P0_HIST, K_JURNAL)]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name=""
    ))

    fig_j.add_trace(go.Scatter(
        x=all_y, y=all_p_model, mode="lines",
        name="Model P(t)=P₀·e^(kt)",
        line=dict(color=CYAN, width=2.5),
    ))
    fig_j.add_trace(go.Scatter(
        x=TAHUN_HIST, y=POP_AKTUAL, mode="markers+lines",
        name="Data Aktual BPS",
        marker=dict(color=AMBER, size=11, symbol="circle",
                    line=dict(color=WHITE, width=1.5)),
        line=dict(color=AMBER, width=1.5, dash="dot"),
    ))
    fig_j.add_trace(go.Scatter(
        x=TAHUN_PRED, y=POP_JURNAL, mode="markers+lines",
        name="Prediksi Jurnal (k=0.0122)",
        marker=dict(color=GREEN, size=11, symbol="diamond",
                    line=dict(color=WHITE, width=1.5)),
        line=dict(color=GREEN, width=2),
    ))

    for yr, pop in zip(TAHUN_PRED, POP_JURNAL):
        fig_j.add_annotation(
            x=yr, y=pop, text=f"{pop:,}",
            showarrow=False, yshift=18,
            font=dict(color=GREEN, size=9, family="Space Mono, monospace"),
            bgcolor="rgba(7,30,32,0.92)", borderpad=3
        )
    for yr, pop in zip(TAHUN_HIST, POP_AKTUAL):
        fig_j.add_annotation(
            x=yr, y=pop, text=f"{int(pop):,}",
            showarrow=False, yshift=-22,
            font=dict(color=AMBER, size=9, family="Space Mono, monospace"),
            bgcolor="rgba(7,30,32,0.92)", borderpad=3
        )

    fig_j.add_vline(x=2024, line_dash="dash", line_color=AMBER,
                    line_width=1, opacity=0.4,
                    annotation_text="2024 · Batas Prediksi",
                    annotation_font_color=AMBER,
                    annotation_font_size=10)

    st.plotly_chart(fig_j, use_container_width=True)

    # ── ABSTRACT BOX ──
    st.markdown('<div class="section-label">Abstrak jurnal</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#071E20;border:1px solid rgba(45,212,191,0.20);border-radius:8px;
    padding:20px 26px;font-size:12.5px;color:#5EADA8;line-height:1.9;font-family:'DM Sans',sans-serif;">
      <span style="font-family:'Syne',sans-serif;font-weight:700;color:#2DD4BF;font-size:13px;">Abstrak · </span>
      Pertumbuhan penduduk merupakan faktor penting dalam perencanaan pembangunan suatu wilayah.
      Penelitian ini bertujuan memprediksi jumlah penduduk Kota Tual pada tahun 2026–2030
      menggunakan persamaan diferensial dengan asumsi model tumbuh secara eksponensial.
      Data bersumber dari BPS Provinsi Maluku tahun 2020–2024.
      Diperoleh laju pertumbuhan sebesar <b style="color:#2DD4BF;">1,22% per tahun</b>.
      Diperkirakan penduduk akan bertambah <b style="color:#34D399;">7.049 jiwa</b> selama 6 tahun —
      dari <b style="color:#C8F7F0;">92.744 jiwa (2024)</b> menjadi <b style="color:#C8F7F0;">99.793 jiwa (2030)</b>.
      Rekomendasi: perlunya pengendalian laju pertumbuhan melalui kebijakan kependudukan berbasis data
      serta optimalisasi tata ruang dan infrastruktur.
    </div>
    """, unsafe_allow_html=True)
