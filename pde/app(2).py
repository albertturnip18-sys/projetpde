"""
Deutschsprung – Dein spielerischer Weg von A1 zu A2
Streamlit app (revisi v2)

Perubahan besar pada revisi ini:
  - Fix bug: reset kartu vocab (flipped) tiap ganti kategori
  - Fix bug: filter level A1/A2 tidak lagi "bocor" menampilkan level lain
  - Fix bug: logout memakai st.session_state.clear() (reset total)
  - Fitur baru: Audio pengucapan (gTTS) di setiap kartu kosakata
  - Fitur baru: Modul latihan konjugasi verba (sein, haben, lernen, fahren)
  - Fitur baru: Dashboard progres (kosakata dilihat & skor quiz tertinggi)
  - UX: navigasi halaman berbasis tombol (pill nav), grid kartu lebih rapi,
        badge/alert umpan balik saat quiz dijawab

Struktur file:
  1. Konfigurasi halaman & data (vocab, grammar, verbs, questions)
  2. Session state
  3. Util: TTS (gTTS) helper
  4. CSS kustom
  5. Layar login (simulasi)
  6. Nav atas + dashboard progres
  7. Bagian: hero, vocab, grammar, verben, quiz, footer
  8. Router berbasis st.session_state.page
"""

import io

import streamlit as st

# ---------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deutschsprung – Dein spielerischer Weg von A1 zu A2",
    page_icon="🇩🇪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
VOCAB = {
    "Begrüßung": [
        {"de": "Hallo", "id": "Halo", "lvl": "A1"},
        {"de": "Guten Morgen", "id": "Selamat pagi", "lvl": "A1"},
        {"de": "Tschüss", "id": "Sampai jumpa", "lvl": "A1"},
        {"de": "Danke", "id": "Terima kasih", "lvl": "A1"},
        {"de": "Entschuldigung", "id": "Maaf / Permisi", "lvl": "A1"},
        {"de": "Wie geht's?", "id": "Apa kabar?", "lvl": "A1"},
    ],
    "Zahlen & Farben": [
        {"de": "eins, zwei, drei", "id": "satu, dua, tiga", "lvl": "A1"},
        {"de": "zehn", "id": "sepuluh", "lvl": "A1"},
        {"de": "rot", "id": "merah", "lvl": "A1"},
        {"de": "blau", "id": "biru", "lvl": "A1"},
        {"de": "gelb", "id": "kuning", "lvl": "A1"},
    ],
    "Familie": [
        {"de": "die Mutter", "id": "ibu", "lvl": "A1"},
        {"de": "der Vater", "id": "ayah", "lvl": "A1"},
        {"de": "die Schwester", "id": "saudari perempuan", "lvl": "A1"},
        {"de": "der Bruder", "id": "saudara laki-laki", "lvl": "A1"},
        {"de": "das Kind", "id": "anak", "lvl": "A1"},
    ],
    "Im Hotel": [
        {"de": "die Rezeption", "id": "resepsionis", "lvl": "A2"},
        {"de": "das Zimmer", "id": "kamar", "lvl": "A2"},
        {"de": "der Gast", "id": "tamu", "lvl": "A2"},
        {"de": "der Schlüssel", "id": "kunci", "lvl": "A2"},
        {"de": "die Buchung", "id": "pemesanan", "lvl": "A2"},
    ],
    "Alltag & Zeit": [
        {"de": "heute", "id": "hari ini", "lvl": "A2"},
        {"de": "morgen", "id": "besok", "lvl": "A2"},
        {"de": "pünktlich", "id": "tepat waktu", "lvl": "A2"},
        {"de": "die Verabredung", "id": "janji temu", "lvl": "A2"},
    ],
}

GRAMMAR = [
    {"lvl": "A1", "title": "Personalpronomen + sein",
     "text": "Die Basis für jeden Satz: wer bin ich, wer bist du?",
     "ex": "ich bin, du bist, er/sie/es ist"},
    {"lvl": "A1", "title": "Artikel: der, die, das",
     "text": "Jedes Nomen hat ein Geschlecht — das muss man einfach mitlernen.",
     "ex": "der Tisch, die Lampe, das Buch"},
    {"lvl": "A1", "title": "Verben im Präsens",
     "text": "Regelmäßige Verben bekommen im Präsens feste Endungen.",
     "ex": "ich lerne, du lernst, wir lernen"},
    {"lvl": "A2", "title": "Perfekt (Vergangenheit)",
     "text": "Mit haben oder sein + Partizip II erzählt man, was passiert ist.",
     "ex": "Ich habe Deutsch gelernt."},
    {"lvl": "A2", "title": "Modalverben",
     "text": "können, müssen, wollen verändern die Bedeutung des Hauptverbs.",
     "ex": "Ich muss heute arbeiten."},
    {"lvl": "A2", "title": "Akkusativ",
     "text": "Das direkte Objekt eines Satzes steht im Akkusativ.",
     "ex": "Ich sehe den Gast."},
]

# Data konjugasi verba dasar A1-A2 (untuk modul latihan baru)
VERBS = {
    "sein": {
        "meaning": "menjadi / berada (to be)",
        "conj": {"ich": "bin", "du": "bist", "er/sie/es": "ist",
                  "wir": "sind", "ihr": "seid", "sie/Sie": "sind"},
    },
    "haben": {
        "meaning": "memiliki (to have)",
        "conj": {"ich": "habe", "du": "hast", "er/sie/es": "hat",
                  "wir": "haben", "ihr": "habt", "sie/Sie": "haben"},
    },
    "lernen": {
        "meaning": "belajar (to learn)",
        "conj": {"ich": "lerne", "du": "lernst", "er/sie/es": "lernt",
                  "wir": "lernen", "ihr": "lernt", "sie/Sie": "lernen"},
    },
    "fahren": {
        "meaning": "berkendara / pergi (to drive / go)",
        "conj": {"ich": "fahre", "du": "fährst", "er/sie/es": "fährt",
                  "wir": "fahren", "ihr": "fahrt", "sie/Sie": "fahren"},
    },
}

QUESTIONS = [
    {"q": "Wie sagt man 'terima kasih' auf Deutsch?",
     "opts": ["Bitte", "Danke", "Tschüss", "Hallo"], "a": 1},
    {"q": "'Die Mutter' bedeutet auf Indonesisch...",
     "opts": ["ayah", "ibu", "anak", "kakak"], "a": 1},
    {"q": "Welcher Artikel passt zu 'Buch'?",
     "opts": ["der", "die", "das", "den"], "a": 2},
    {"q": "Ich ___ Student. (sein)",
     "opts": ["bin", "bist", "ist", "sind"], "a": 0},
    {"q": "Wähle die richtige Präsensform: du ___ (lernen)",
     "opts": ["lerne", "lernst", "lernt", "lernen"], "a": 1},
    {"q": "Perfekt von 'lernen' mit haben:",
     "opts": ["Ich habe gelernt.", "Ich bin gelernt.", "Ich lernte habe.", "Ich hatte lernen."], "a": 0},
    {"q": "Modalverb für 'harus' (müssen):",
     "opts": ["kann", "will", "muss", "mag"], "a": 2},
    {"q": "'Der Gast' im Hotelkontext bedeutet:",
     "opts": ["kunci", "kamar", "tamu", "resepsionis"], "a": 2},
]

PAGES = ["beranda", "vokabeln", "grammatik", "verben", "quiz"]
PAGE_LABELS = {
    "beranda": "🏠 Beranda",
    "vokabeln": "🗂️ Vokabeln",
    "grammatik": "📘 Grammatik",
    "verben": "🔤 Verben",
    "quiz": "📝 Quiz",
}

TOTAL_VOCAB = sum(len(v) for v in VOCAB.values())

# ---------------------------------------------------------------------------
# 2. SESSION STATE
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "logged_in": False,
    "username": "",
    "level": "A1",                       # "A1" | "A2"
    "page": "beranda",                   # halaman aktif (nav pill)
    "current_cat": list(VOCAB.keys())[0],
    "flipped": {},                       # {card_key: bool} — direset tiap ganti kategori
    "viewed_vocab": set(),               # kata (de) yang pernah dibuka terjemahannya
    "audio_cache": {},                   # {card_key: bytes mp3}
    "q_index": 0,
    "q_score": 0,
    "q_selected": None,
    "q_answered": False,
    "q_done": False,
    "q_high_score": 0,
    "verb_choice": list(VERBS.keys())[0],
    "verb_best": {},                     # {verb: skor_terbaik}
    "verb_result": None,                 # hasil cek terakhir (untuk ditampilkan)
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_quiz():
    st.session_state.q_index = 0
    st.session_state.q_score = 0
    st.session_state.q_selected = None
    st.session_state.q_answered = False
    st.session_state.q_done = False


def go_to(page: str):
    st.session_state.page = page


# ---------------------------------------------------------------------------
# 3. UTIL — Text-to-Speech (gTTS)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_tts_audio(text: str):
    """Menghasilkan audio MP3 (bytes) untuk teks bahasa Jerman.

    Mengembalikan None jika gTTS tidak tersedia / gagal (mis. tidak ada
    koneksi internet), agar UI tetap berjalan tanpa error.
    """
    try:
        from gtts import gTTS  # import lokal agar app tetap jalan tanpa gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang="de").write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. CSS KUSTOM
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

        :root{
            --ink:#1B2430;
            --paper:#F2ECDD;
            --mustard:#E8A93B;
            --brick:#C1440E;
            --teal:#3A7D7B;
            --line:rgba(242,236,221,0.18);
        }

        html, body, [class^="st-"], [class*=" st-"] { font-family:'Inter', sans-serif; }
        h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; }

        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}

        .stApp { background:var(--ink); color:var(--paper); }
        .block-container { max-width:900px; padding-top:1.5rem; padding-bottom:3rem; }

        /* ---------- Tombol umum ---------- */
        .stButton > button {
            font-family:'Inter', sans-serif; font-weight:600; font-size:0.9rem;
            border-radius:999px; border:1.5px solid var(--line);
            background:transparent; color:var(--paper);
            transition:transform .15s ease, background .2s, color .2s, border-color .2s;
            padding:0.5rem 1rem;
        }
        .stButton > button:hover { transform:translateY(-2px); border-color:var(--mustard); color:var(--mustard); }
        .stButton > button:active { transform:translateY(0); }
        button[kind="primary"], .stButton > button[kind="primary"] {
            background:var(--mustard) !important; color:var(--ink) !important; border-color:var(--mustard) !important;
        }
        button[kind="primary"]:hover { color:var(--ink) !important; filter:brightness(1.05); }
        button[kind="secondary"] { background:transparent !important; }

        /* ---------- Kartu umum (st.container(border=True)) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }

        /* ---------- Eyebrow / label kecil ---------- */
        .eyebrow { color:var(--mustard); font-weight:600; font-size:0.9rem; margin-bottom:6px; }

        /* ---------- Login ---------- */
        .login-shapes .shape { position:fixed; z-index:0; pointer-events:none; }
        .login-shapes .circle { width:240px; height:240px; border-radius:50%; background:var(--brick); top:-70px; left:-80px; opacity:0.9; }
        .login-shapes .square { width:56px; height:56px; background:var(--mustard); bottom:70px; right:60px; transform:rotate(12deg); }
        .login-shapes .triangle { width:0; height:0; border-left:64px solid transparent; border-right:64px solid transparent; border-bottom:110px solid var(--teal); opacity:.85; bottom:0; left:80px; }

        .st-key-login_card { position:relative; z-index:2; background:var(--paper) !important; border-color:rgba(27,36,48,0.1) !important; padding:0.5rem 0.25rem; }
        .st-key-login_card, .st-key-login_card p, .st-key-login_card span, .st-key-login_card div { color:var(--ink); }
        .login-brand { display:flex; align-items:center; gap:9px; font-weight:700; font-size:1.1rem; font-family:'Space Grotesk',sans-serif; margin-bottom:6px; }
        .login-sub { color:rgba(27,36,48,0.65); font-size:0.92rem; margin-bottom:6px; }
        .brand-dot { width:20px; height:20px; border-radius:50%; background:var(--mustard); display:inline-block; flex-shrink:0; }
        .divider { text-align:center; color:rgba(27,36,48,0.45); font-size:0.8rem; margin:14px 0 10px; position:relative; }
        .st-key-login_card .stButton > button { border-color:rgba(27,36,48,0.25); color:var(--ink); }
        .st-key-login_card .stButton > button[kind="primary"] { background:var(--ink) !important; color:var(--paper) !important; border-color:var(--ink) !important; }

        /* ---------- Nav / top bar ---------- */
        .topnav { display:flex; align-items:center; justify-content:space-between; padding:10px 0 16px; border-bottom:1px solid var(--line); margin-bottom:14px; }
        .brand { display:flex; align-items:center; gap:9px; font-weight:700; font-size:1.1rem; font-family:'Space Grotesk',sans-serif; }
        .user-chip { display:flex; align-items:center; gap:8px; font-size:0.88rem; font-weight:600; }
        .avatar { width:28px; height:28px; border-radius:50%; background:var(--mustard); color:var(--ink); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.82rem; flex-shrink:0; }

        /* ---------- Dashboard progres ---------- */
        div[data-testid="stMetric"] {
            background:rgba(242,236,221,0.06); border:1px solid var(--line);
            border-radius:12px; padding:10px 14px;
        }
        div[data-testid="stMetricLabel"] { color:rgba(242,236,221,0.75) !important; }
        div[data-testid="stMetricValue"] { color:var(--mustard) !important; }

        /* ---------- Page pill nav ---------- */
        .st-key-page_nav .stButton > button { padding:0.4rem 0.85rem; font-size:0.85rem; }

        /* ---------- Hero ---------- */
        .hero-title { font-size:clamp(2.1rem, 6vw, 3.4rem); font-weight:700; line-height:1.08; margin:6px 0 14px; }
        .hero-title .accent { color:var(--mustard); }
        .hero-desc { color:rgba(242,236,221,0.85); font-size:1.05rem; max-width:34em; margin-bottom:22px; }

        /* ---------- Section heading umum ---------- */
        .section-head h2 { font-size:clamp(1.4rem,3.2vw,1.9rem); font-weight:700; margin-bottom:6px; }
        .section-head p { font-size:0.98rem; opacity:0.85; margin-bottom:6px; }

        /* ---------- Vokabel (section terang, gaya "paper") ---------- */
        .st-key-vocab_wrap { background:var(--paper) !important; border:none !important; border-radius:16px; padding:1.6rem 1.4rem; color:var(--ink); }
        .st-key-vocab_wrap .section-head p { color:rgba(27,36,48,0.7); }
        .st-key-vocab_wrap .stButton > button { border:1.5px solid var(--ink); color:var(--ink); }
        .st-key-vocab_wrap .stButton > button[kind="primary"] { background:var(--ink) !important; color:var(--paper) !important; border-color:var(--ink) !important; }
        div[class*="st-key-vcard_"] { min-height:118px; display:flex; flex-direction:column; justify-content:center; }
        div[class*="st-key-vcard_f_"] { background:var(--paper) !important; border:1.5px solid var(--ink) !important; }
        div[class*="st-key-vcard_b_"] { background:var(--brick) !important; border:1.5px solid var(--brick) !important; }
        .card-front { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.08rem; text-align:center; color:var(--ink); }
        .card-front .card-tag { display:block; font-size:0.7rem; color:var(--brick); margin-top:5px; font-weight:600; }
        .card-back { font-family:'Inter',sans-serif; font-weight:600; font-size:1rem; text-align:center; color:var(--paper); }

        /* ---------- Grammatik (section gelap, seperti asli) ---------- */
        .g-lvl { display:inline-block; font-size:0.7rem; font-weight:700; color:var(--ink); background:var(--mustard); padding:3px 10px; border-radius:999px; margin-bottom:8px; }
        .g-title { font-size:1.05rem; margin:2px 0 6px; font-weight:700; font-family:'Space Grotesk',sans-serif; }
        .g-text { color:rgba(242,236,221,0.8); font-size:0.92rem; margin-bottom:8px; }
        .g-ex { font-family:'Space Grotesk',sans-serif; font-size:0.92rem; border-left:3px solid var(--teal); padding-left:10px; }

        /* ---------- Verben (modul baru) ---------- */
        .st-key-verb_wrap { background:var(--paper) !important; border:none !important; border-radius:16px; padding:1.6rem 1.4rem; color:var(--ink); }
        .st-key-verb_wrap .section-head p { color:rgba(27,36,48,0.7); }
        .verb-meaning { color:rgba(27,36,48,0.65); font-size:0.92rem; margin-bottom:10px; }
        .verb-row { padding:10px 14px; border-radius:10px; margin-bottom:8px; font-size:0.92rem; border:1.5px solid rgba(27,36,48,0.15); }
        .verb-row.correct { background:rgba(58,125,123,0.18); border-color:var(--teal); }
        .verb-row.wrong { background:rgba(193,68,14,0.14); border-color:var(--brick); }
        .verb-best { display:inline-block; background:var(--mustard); color:var(--ink); font-weight:700; font-size:0.78rem; padding:3px 10px; border-radius:999px; margin-left:8px; }

        /* ---------- Quiz (section terang berisi kotak gelap) ---------- */
        .st-key-quiz_wrap { background:var(--paper) !important; border:none !important; border-radius:16px; padding:1.6rem 1.4rem; color:var(--ink); }
        .st-key-quiz_wrap .section-head p { color:rgba(27,36,48,0.7); }
        .st-key-quiz_box { background:var(--ink) !important; border-color:var(--line) !important; color:var(--paper) !important; padding:0.75rem 0.5rem; }
        .q-progress { font-size:0.82rem; color:var(--mustard); font-weight:600; margin-bottom:6px; }
        .q-text { font-size:1.15rem; font-weight:600; font-family:'Space Grotesk',sans-serif; margin-bottom:14px; }
        .q-score { font-weight:600; color:var(--mustard); font-size:0.9rem; }
        .q-opt-static { text-align:left; padding:13px 15px; border-radius:10px; border:1.5px solid var(--line); margin-bottom:9px; font-size:0.94rem; }
        .q-opt-static.correct { border-color:var(--teal); background:rgba(58,125,123,0.25); }
        .q-opt-static.wrong { border-color:var(--brick); background:rgba(193,68,14,0.25); }
        .q-opt-static.neutral { opacity:0.45; }
        .big-score { font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:700; color:var(--mustard); text-align:center; }
        .quiz-done-caption { text-align:center; color:rgba(242,236,221,0.7); font-size:0.9rem; }

        /* ---------- Badge umpan balik (feedback) ---------- */
        .badge-success, .badge-error {
            display:block; text-align:center; font-weight:700; font-size:0.95rem;
            padding:10px 16px; border-radius:12px; margin-bottom:14px;
            animation: pop-in .25s ease;
        }
        .badge-success { background:rgba(58,125,123,0.28); border:1.5px solid var(--teal); color:var(--paper); }
        .badge-error { background:rgba(193,68,14,0.24); border:1.5px solid var(--brick); color:var(--paper); }
        @keyframes pop-in { from { transform:scale(0.92); opacity:0; } to { transform:scale(1); opacity:1; } }

        /* ---------- Footer ---------- */
        .app-footer { text-align:center; color:rgba(242,236,221,0.5); font-size:0.85rem; padding-top:28px; margin-top:20px; border-top:1px solid var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 5. LAYAR LOGIN (simulasi — sama seperti versi HTML asli, tanpa backend)
# ---------------------------------------------------------------------------
def login_screen():
    st.markdown(
        """
        <div class="login-shapes">
          <div class="shape circle"></div>
          <div class="shape square"></div>
          <div class="shape triangle"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_mid, col_r = st.columns([1, 3, 1])
    with col_mid:
        with st.container(border=True, key="login_card"):
            st.markdown(
                """
                <div class="login-brand"><span class="brand-dot"></span>Deutschsprung</div>
                <h2 style="margin-bottom:4px;">Willkommen zurück</h2>
                <p class="login-sub">Melde dich an und mach weiter, wo du aufgehört hast.</p>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔵  Mit Google fortfahren", key="google_demo", use_container_width=True):
                st.info(
                    "Demo: Eine echte Google-Anmeldung braucht ein eigenes Backend (OAuth). "
                    "Nutze bitte das Formular unten, um fortzufahren.",
                    icon="ℹ️",
                )

            st.markdown('<div class="divider">oder mit E-Mail</div>', unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                gmail = st.text_input("Gmail-Adresse", placeholder="deinname@gmail.com")
                username = st.text_input("Benutzername", placeholder="z. B. Immanuel")
                submitted = st.form_submit_button("Loslegen", type="primary", use_container_width=True)

                if submitted:
                    gmail_clean = gmail.strip().lower()
                    if not gmail_clean.endswith("@gmail.com") or len(gmail_clean) <= len("@gmail.com"):
                        st.error("Bitte gib eine gültige Gmail-Adresse ein.")
                    elif not username.strip():
                        st.error("Bitte gib einen Benutzernamen ein.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.username = username.strip()
                        st.rerun()


# ---------------------------------------------------------------------------
# 6. NAV ATAS + DASHBOARD PROGRES
# ---------------------------------------------------------------------------
def top_nav():
    initial = st.session_state.username[:1].upper() if st.session_state.username else "?"
    st.markdown(
        f"""
        <div class="topnav">
          <div class="brand"><span class="brand-dot"></span>Deutschsprung</div>
          <div class="user-chip">
            <span class="avatar">{initial}</span>
            <span>{st.session_state.username}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Dashboard progres ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Kosakata Dilihat", f"{len(st.session_state.viewed_vocab)}/{TOTAL_VOCAB}")
    with m2:
        st.metric("Skor Quiz Tertinggi", f"{st.session_state.q_high_score}/{len(QUESTIONS)}")
    with m3:
        st.metric("Level Aktif", st.session_state.level)

    st.write("")

    # --- Kontrol level + logout ---
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1.2])
    with c1:
        if st.button("A1", key="lvl_a1", type="primary" if st.session_state.level == "A1" else "secondary",
                      use_container_width=True):
            st.session_state.level = "A1"
            st.rerun()
    with c2:
        if st.button("A2", key="lvl_a2", type="primary" if st.session_state.level == "A2" else "secondary",
                      use_container_width=True):
            st.session_state.level = "A2"
            st.rerun()
    with c4:
        if st.button("Abmelden", key="logout_btn", use_container_width=True):
            st.session_state.clear()   # reset total seluruh sesi (fix bug logout)
            st.rerun()

    # --- Pill nav antar halaman ---
    with st.container(key="page_nav"):
        nav_cols = st.columns(len(PAGES))
        for col, pg in zip(nav_cols, PAGES):
            with col:
                if st.button(PAGE_LABELS[pg], key=f"nav_{pg}",
                             type="primary" if st.session_state.page == pg else "secondary",
                             use_container_width=True):
                    go_to(pg)
                    st.rerun()
    st.write("")


# ---------------------------------------------------------------------------
# 7. BAGIAN APLIKASI UTAMA
# ---------------------------------------------------------------------------
def hero_section():
    st.markdown('<div class="eyebrow">Deine Deutschreise beginnt hier</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h1 class="hero-title">Von <span class="accent">Null</span> auf Deutsch –
        <span class="accent">Schritt</span> für Schritt.</h1>
        <p class="hero-desc">Vokabeln, Grammatik, Verbkonjugation und ein kleines Quiz für die Niveaus
        A1 und A2. Kein Auswendiglernen ohne Sinn – nur klare Häppchen, die wirklich hängen bleiben.</p>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗂️ Vokabeln entdecken", key="cta_vocab", type="primary", use_container_width=True):
            go_to("vokabeln")
            st.rerun()
    with c2:
        if st.button("🔤 Verben üben", key="cta_verben", use_container_width=True):
            go_to("verben")
            st.rerun()
    with c3:
        if st.button("📝 Quiz starten", key="cta_quiz", use_container_width=True):
            go_to("quiz")
            st.rerun()
    st.write("")


def vocab_section():
    with st.container(key="vocab_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Vokabelkarten</h2>
              <p>Klick auf eine Karte, um die Übersetzung zu sehen, und höre dir die Aussprache an.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Tab kategori (wrap 3 per baris) ---
        cats = list(VOCAB.keys())
        per_row = 3
        for row_start in range(0, len(cats), per_row):
            row_cats = cats[row_start: row_start + per_row]
            cols = st.columns(len(row_cats))
            for col, cat in zip(cols, row_cats):
                with col:
                    is_active = cat == st.session_state.current_cat
                    if st.button(cat, key=f"cat_{cat}", type="primary" if is_active else "secondary",
                                 use_container_width=True):
                        if cat != st.session_state.current_cat:
                            st.session_state.current_cat = cat
                            # FIX BUG: reset kartu ke posisi depan tiap ganti kategori
                            st.session_state.flipped = {}
                        st.rerun()

        st.write("")

        # --- Grid kartu, difilter murni berdasarkan level aktif ---
        # FIX BUG: A1 hanya menampilkan A1; A2 menampilkan A1+A2 (tanpa fallback
        # yang membocorkan level lain saat suatu kategori kosong di A1).
        all_items = VOCAB[st.session_state.current_cat]
        if st.session_state.level == "A1":
            items = [v for v in all_items if v["lvl"] == "A1"]
        else:
            items = list(all_items)  # A2 = gabungan A1 + A2

        if not items:
            st.info(
                f"Belum ada kosakata level A1 di kategori **{st.session_state.current_cat}**. "
                "Coba beralih ke level A2 di atas. 👆"
            )
            return

        card_cols = st.columns(3)
        for idx, item in enumerate(items):
            safe_cat = st.session_state.current_cat.replace(" ", "_").replace("&", "und")
            card_key = f"{safe_cat}_{item['de']}".replace(" ", "_").replace(",", "").replace("'", "")
            flipped = st.session_state.flipped.get(card_key, False)
            side = "b" if flipped else "f"
            with card_cols[idx % 3]:
                with st.container(border=True, key=f"vcard_{side}_{card_key}"):
                    if flipped:
                        st.markdown(f'<div class="card-back">{item["id"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="card-front">{item["de"]}<span class="card-tag">{item["lvl"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                    fc1, fc2 = st.columns([2, 1])
                    with fc1:
                        label = "↺ Kembali" if flipped else "↺ Terjemahan"
                        if st.button(label, key=f"flip_{card_key}", use_container_width=True):
                            new_state = not flipped
                            st.session_state.flipped[card_key] = new_state
                            if new_state:
                                # Progress tracker: catat kata yang sudah dibuka
                                st.session_state.viewed_vocab.add(item["de"])
                            st.rerun()
                    with fc2:
                        if st.button("🔊", key=f"audio_{card_key}", use_container_width=True,
                                      help="Dengarkan pengucapan"):
                            st.session_state.audio_cache[card_key] = generate_tts_audio(item["de"])

                    audio_bytes = st.session_state.audio_cache.get(card_key)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                    elif card_key in st.session_state.audio_cache:
                        st.caption("🔇 Audio tidak tersedia (periksa koneksi internet)")


def grammar_section():
    st.markdown(
        """
        <div class="section-head">
          <h2>Grammatik-Häppchen</h2>
          <p>Die wichtigsten Bausteine für A1 und A2 – kurz erklärt, mit Beispiel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, g in enumerate(GRAMMAR):
        with cols[i % 2]:
            with st.container(border=True, key=f"gcard_{i}"):
                st.markdown(
                    f"""
                    <span class="g-lvl">{g['lvl']}</span>
                    <div class="g-title">{g['title']}</div>
                    <div class="g-text">{g['text']}</div>
                    <div class="g-ex">{g['ex']}</div>
                    """,
                    unsafe_allow_html=True,
                )
    st.write("")


def verb_section():
    with st.container(key="verb_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Verbkonjugation üben</h2>
              <p>Wähle ein Verb und fülle die richtige Präsensform für jedes Pronomen aus.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        verb_names = list(VERBS.keys())
        choice = st.selectbox(
            "Verb auswählen", verb_names,
            index=verb_names.index(st.session_state.verb_choice),
            key="verb_choice_select",
        )
        if choice != st.session_state.verb_choice:
            st.session_state.verb_choice = choice
            st.session_state.verb_result = None
            st.rerun()

        verb_data = VERBS[choice]
        best = st.session_state.verb_best.get(choice)
        best_html = f'<span class="verb-best">Rekor: {best}/6</span>' if best is not None else ""
        st.markdown(
            f'<div class="verb-meaning"><b>{choice}</b> — {verb_data["meaning"]} {best_html}</div>',
            unsafe_allow_html=True,
        )

        pronouns = list(verb_data["conj"].keys())
        with st.form(f"verb_form_{choice}"):
            cols = st.columns(2)
            inputs = {}
            for i, pron in enumerate(pronouns):
                with cols[i % 2]:
                    inputs[pron] = st.text_input(f"{pron} ___", key=f"verb_in_{choice}_{pron}",
                                                  placeholder="Konjugation eingeben")
            submitted = st.form_submit_button("✅ Cek Jawaban", type="primary", use_container_width=True)

        if submitted:
            correct_count = 0
            rows_html = ""
            for pron in pronouns:
                user_ans = (inputs[pron] or "").strip().lower()
                correct_ans = verb_data["conj"][pron].lower()
                if user_ans == correct_ans:
                    correct_count += 1
                    rows_html += (
                        f'<div class="verb-row correct">✅ <b>{pron}</b> {verb_data["conj"][pron]} — Benar!</div>'
                    )
                else:
                    shown = inputs[pron] or "(kosong)"
                    rows_html += (
                        f'<div class="verb-row wrong">❌ <b>{pron}</b> — jawabanmu: "{shown}", '
                        f'yang benar: <b>{verb_data["conj"][pron]}</b></div>'
                    )
            st.markdown(rows_html, unsafe_allow_html=True)

            total = len(pronouns)
            prev_best = st.session_state.verb_best.get(choice, 0)
            st.session_state.verb_best[choice] = max(prev_best, correct_count)

            if correct_count == total:
                st.markdown(
                    f'<div class="badge-success">🎉 Sempurna! {correct_count}/{total} benar untuk "{choice}".</div>',
                    unsafe_allow_html=True,
                )
                st.balloons()
            elif correct_count >= total * 0.5:
                st.markdown(
                    f'<div class="badge-success">👍 Bagus! {correct_count}/{total} benar.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="badge-error">💪 Terus berlatih! {correct_count}/{total} benar.</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📋 Lihat tabel konjugasi lengkap"):
            for pron, form in verb_data["conj"].items():
                st.markdown(f"- **{pron}** → {form}")


def quiz_section():
    with st.container(key="quiz_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Teste dich selbst</h2>
              <p>Acht Fragen, gemischt aus A1 und A2. Viel Erfolg!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True, key="quiz_box"):
            if not st.session_state.q_done:
                item = QUESTIONS[st.session_state.q_index]
                st.markdown(
                    f'<div class="q-progress">Frage {st.session_state.q_index + 1} / {len(QUESTIONS)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="q-text">{item["q"]}</div>', unsafe_allow_html=True)

                if not st.session_state.q_answered:
                    for i, opt in enumerate(item["opts"]):
                        if st.button(opt, key=f"opt_{st.session_state.q_index}_{i}", use_container_width=True):
                            st.session_state.q_selected = i
                            st.session_state.q_answered = True
                            if i == item["a"]:
                                st.session_state.q_score += 1
                            st.rerun()
                else:
                    # --- Umpan balik visual (badge) ---
                    is_correct = st.session_state.q_selected == item["a"]
                    if is_correct:
                        st.markdown(
                            '<div class="badge-success">✅ Richtig! Sehr gut gemacht.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="badge-error">❌ Nicht ganz. Richtige Antwort: '
                            f'{item["opts"][item["a"]]}</div>',
                            unsafe_allow_html=True,
                        )

                    for i, opt in enumerate(item["opts"]):
                        if i == item["a"]:
                            cls = "correct"
                        elif i == st.session_state.q_selected:
                            cls = "wrong"
                        else:
                            cls = "neutral"
                        st.markdown(f'<div class="q-opt-static {cls}">{opt}</div>', unsafe_allow_html=True)

                    is_last = st.session_state.q_index == len(QUESTIONS) - 1
                    next_label = "Ergebnis anzeigen →" if is_last else "Nächste Frage →"
                    if st.button(next_label, key="next_q", type="primary", use_container_width=True):
                        if is_last:
                            st.session_state.q_done = True
                            if st.session_state.q_score > st.session_state.q_high_score:
                                st.session_state.q_high_score = st.session_state.q_score
                        else:
                            st.session_state.q_index += 1
                            st.session_state.q_answered = False
                            st.session_state.q_selected = None
                        st.rerun()

                st.markdown(f'<div class="q-score">Punkte: {st.session_state.q_score}</div>',
                             unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="big-score">{st.session_state.q_score} / {len(QUESTIONS)}</div>',
                             unsafe_allow_html=True)
                st.markdown('<p class="quiz-done-caption">Richtige Antworten von 8</p>', unsafe_allow_html=True)

                score_ratio = st.session_state.q_score / len(QUESTIONS)
                if score_ratio == 1.0:
                    st.success("🏆 Luar biasa! Kamu meraih skor sempurna!")
                    st.balloons()
                elif score_ratio >= 0.75:
                    st.info("👍 Hasil yang sangat baik, terus pertahankan!")
                elif score_ratio >= 0.5:
                    st.warning("🙂 Lumayan! Coba ulangi untuk skor lebih tinggi.")
                else:
                    st.warning("💪 Tetap semangat, latihan lagi yuk!")

                st.write("")
                if st.button("Nochmal versuchen", key="restart_quiz", type="primary", use_container_width=True):
                    reset_quiz()
                    st.rerun()


def footer():
    st.markdown('<div class="app-footer">Deutschsprung · Übung macht den Meister 🇩🇪</div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 8. ROUTER
# ---------------------------------------------------------------------------
inject_css()

if not st.session_state.logged_in:
    login_screen()
else:
    top_nav()

    page = st.session_state.page
    if page == "beranda":
        hero_section()
    elif page == "vokabeln":
        vocab_section()
    elif page == "grammatik":
        grammar_section()
    elif page == "verben":
        verb_section()
    elif page == "quiz":
        quiz_section()

    st.write("")
    footer()
