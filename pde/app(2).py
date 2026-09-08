import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie


# -----------------------------------------------------------------------------
# CONFIG HALAMAN & CSS ANIMASI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deutschsprung - Belajar Bahasa Jerman",
    page_icon="🇩🇪",
    layout="wide",
)


# Fungsi untuk memuat animasi Lottie via URL JSON
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# Menginjeksi CSS untuk animasi antarmuka
st.markdown(
    """
    <style>
    /* Animasi Fade In Smooth untuk Container */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-container {
        animation: fadeIn 0.6s ease-out;
    }

    /* Efek Hover dan Kaca pada Kartu Kosakata */
    .vocab-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .vocab-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        border-color: #FFCC00;
    }

    .badge-xp {
        background-color: #FFCC00;
        color: #000;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INITIALIZATION SESSION STATE
# -----------------------------------------------------------------------------
if "xp" not in st.session_state:
    st.session_state.xp = 50
if "streak" not in st.session_state:
    st.session_state.streak = 3
if "mastered_words" not in st.session_state:
    st.session_state.mastered_words = set()

# -----------------------------------------------------------------------------
# DATASET DUMMY KOSAKATA & KUIS
# -----------------------------------------------------------------------------
vocab_data = [
    {
        "id": 1,
        "word": "das Haus",
        "meaning": "Rumah",
        "level": "A1",
        "example": "Das Haus ist groß.",
    },
    {
        "id": 2,
        "word": "lernen",
        "meaning": "Belajar",
        "level": "A1",
        "example": "Ich lerne Deutsch.",
    },
    {
        "id": 3,
        "word": "die Entscheidung",
        "meaning": "Keputusan",
        "level": "A2",
        "example": "Das ist eine gute Entscheidung.",
    },
    {
        "id": 4,
        "word": "schön",
        "meaning": "Indah / Cantik",
        "level": "A1",
        "example": "Der Tag ist schön.",
    },
]

# Asset Animasi Lottie (JSON Public)
ANIMATION_WELCOME = "https://assets5.lottiefiles.com/packages/lf20_d1b82ipv.json"
ANIMATION_TROPHY = "https://assets10.lottiefiles.com/packages/lf20_tououxu0.json"

lottie_welcome = load_lottie_url(ANIMATION_WELCOME)
lottie_trophy = load_lottie_url(ANIMATION_TROPHY)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🇩🇪 Deutschsprung")

    # Display Profil & Status
    st.subheader("Profil Pembelajar")
    st.markdown(
        f"<span class='badge-xp'>⚡ XP: {st.session_state.xp}</span> &nbsp; 🔥 Streak: {st.session_state.streak} Hari",
        unsafe_allow_html=True,
    )
    st.progress(min(st.session_state.xp / 200, 1.0))

    st.write("---")
    menu = st.radio(
        "Pilih Modul:",
        ["Wortschatz (Kosakata)", "Grammatik (Kuis)", "Dashboard Progres"],
    )

# -----------------------------------------------------------------------------
# MODUL 1: KOSAKATA INTERAKTIF
# -----------------------------------------------------------------------------
if menu == "Wortschatz (Kosakata)":
    st.markdown("<div class='animated-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("📚 Wortschatz - Kartu Kosakata")
        st.caption(
            "Pelajari kata-kata baru dan klik tombol untuk menandai sebagai dikuasai."
        )
    with col2:
        if lottie_welcome:
            st_lottie(lottie_welcome, height=120, key="welcome")

    level_filter = st.selectbox("Filter Level:", ["Semua", "A1", "A2"])

    filtered_vocab = [
        item
        for item in vocab_data
        if level_filter == "Semua" or item["level"] == level_filter
    ]

    for item in filtered_vocab:
        is_mastered = item["id"] in st.session_state.mastered_words
        status_icon = "✅ Dikuasai" if is_mastered else "📖 Belum Dikuasai"

        st.markdown(
            f"""
            <div class="vocab-card">
                <h3>{item['word']} <small style="font-size: 14px; color: #888;">({item['level']})</small></h3>
                <p><b>Arti:</b> {item['meaning']}</p>
                <p><i>Contoh: "{item['example']}"</i></p>
                <p><small>Status: {status_icon}</small></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 4])
        with c1:
            if not is_mastered:
                if st.button(f"Tandai Kuasai", key=f"btn_{item['id']}"):
                    st.session_state.mastered_words.add(item["id"])
                    st.session_state.xp += 10
                    st.balloons()  # Animasi selebrasi saat menguasai kata
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODUL 2: KUIS & GRAMMAR
# -----------------------------------------------------------------------------
elif menu == "Grammatik (Kuis)":
    st.markdown("<div class='animated-container'>", unsafe_allow_html=True)
    st.title("✍️ Kuis Konjugasi Kata Kerja")

    st.write(
        "Lengkapi konjugasi dari kata kerja **'lernen'** (Belajar) untuk subjek berikut:"
    )

    with st.form("quiz_form"):
        ans = st.text_input("Ich ... (Saya belajar):").strip().lower()
        submitted = st.form_submit_button("Periksa Jawaban")

        if submitted:
            if ans == "lerne":
                st.success("Richtig! (Benar!) +20 XP")
                st.session_state.xp += 20
                st.balloons()  # Animasi selebrasi baloon
            else:
                st.error("Falsch! Jawaban yang benar adalah **lerne**.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODUL 3: DASHBOARD PROGRES
# -----------------------------------------------------------------------------
elif menu == "Dashboard Progres":
    st.markdown("<div class='animated-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("📊 Statistik Pembelajaran")
        st.metric(
            label="Total Kosakata Dikuasai",
            value=f"{len(st.session_state.mastered_words)} / {len(vocab_data)}",
        )
        st.metric(label="Total XP Terkumpul", value=st.session_state.xp)
        st.metric(
            label="Streak Belajar", value=f"{st.session_state.streak} Hari"
        )

    with col2:
        if lottie_trophy:
            st_lottie(lottie_trophy, height=200, key="trophy")

    # Grafik Ringkas Progres
    df = pd.DataFrame(
        {
            "Status": ["Dikuasai", "Belum Dikuasai"],
            "Jumlah": [
                len(st.session_state.mastered_words),
                len(vocab_data) - len(st.session_state.mastered_words),
            ],
        }
    )
    st.bar_chart(df.set_index("Status"))

    st.markdown("</div>", unsafe_allow_html=True)
