"""
Deutschsprung – Dein spielerischer Weg von A1 zu A2
Streamlit port dari deutschsprung-1.html

Struktur aplikasi:
  1. Konfigurasi halaman & data (vocab, grammar, quiz)
  2. Session state
  3. CSS kustom (meniru palet warna & font versi HTML)
  4. Layar login (simulasi, sama seperti versi asli)
  5. Bagian aplikasi utama: nav, hero, vocab, grammar, quiz, footer
  6. Router sederhana berbasis st.session_state
"""

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
# DATA — dipindahkan 1:1 dari objek `vocab`, `grammar`, `questions` di JS asli
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

# ---------------------------------------------------------------------------
# 2. SESSION STATE
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "logged_in": False,
    "username": "",
    "level": "A1",                       # "A1" | "A2"
    "current_cat": list(VOCAB.keys())[0],
    "flipped": {},                       # {card_key: bool}
    "q_index": 0,
    "q_score": 0,
    "q_selected": None,
    "q_answered": False,
    "q_done": False,
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


# ---------------------------------------------------------------------------
# 3. CSS KUSTOM — meniru palet & font dari deutschsprung-1.html
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
        .topnav { display:flex; align-items:center; justify-content:space-between; padding:10px 0 16px; border-bottom:1px solid var(--line); margin-bottom:18px; }
        .brand { display:flex; align-items:center; gap:9px; font-weight:700; font-size:1.1rem; font-family:'Space Grotesk',sans-serif; }
        .user-chip { display:flex; align-items:center; gap:8px; font-size:0.88rem; font-weight:600; }
        .avatar { width:28px; height:28px; border-radius:50%; background:var(--mustard); color:var(--ink); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.82rem; flex-shrink:0; }

        /* ---------- Hero ---------- */
        .hero-title { font-size:clamp(2.1rem, 6vw, 3.4rem); font-weight:700; line-height:1.08; margin:6px 0 14px; }
        .hero-title .accent { color:var(--mustard); }
        .hero-desc { color:rgba(242,236,221,0.85); font-size:1.05rem; max-width:34em; margin-bottom:22px; }
        .cta-link { display:block; text-align:center; padding:13px 22px; border-radius:8px; font-weight:600; text-decoration:none; font-family:'Inter',sans-serif; font-size:0.95rem; transition:transform .15s ease; }
        .cta-link:hover { transform:translateY(-2px); }
        .cta-primary { background:var(--mustard); color:var(--ink); }
        .cta-ghost { background:transparent; color:var(--paper); border:1px solid var(--line); }

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

        /* ---------- Footer ---------- */
        .app-footer { text-align:center; color:rgba(242,236,221,0.5); font-size:0.85rem; padding-top:28px; margin-top:20px; border-top:1px solid var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 4. LAYAR LOGIN (simulasi — sama seperti versi HTML asli, tanpa backend)
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
# 5. BAGIAN APLIKASI UTAMA
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
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()


def hero_section():
    st.markdown('<div class="eyebrow">Deine Deutschreise beginnt hier</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h1 class="hero-title">Von <span class="accent">Null</span> auf Deutsch –
        <span class="accent">Schritt</span> für Schritt.</h1>
        <p class="hero-desc">Vokabeln, Grammatik und ein kleines Quiz für die Niveaus A1 und A2.
        Kein Auswendiglernen ohne Sinn – nur klare Häppchen, die wirklich hängen bleiben.</p>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<a href="#vokabeln" class="cta-link cta-primary">Vokabeln entdecken</a>',
                     unsafe_allow_html=True)
    with c2:
        st.markdown('<a href="#quiz" class="cta-link cta-ghost">Quiz starten</a>', unsafe_allow_html=True)
    st.write("")


def vocab_section():
    st.markdown('<div id="vokabeln"></div>', unsafe_allow_html=True)
    with st.container(key="vocab_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Vokabelkarten</h2>
              <p>Klick auf eine Karte, um die Übersetzung zu sehen. Wechsle die Kategorie nach Interesse.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Tab kategori (wrap 3 per baris, mirip flex-wrap di versi asli) ---
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
                        st.session_state.current_cat = cat
                        st.rerun()

        st.write("")

        # --- Grid kartu, difilter berdasarkan level aktif (logika sama seperti JS asli) ---
        items = [v for v in VOCAB[st.session_state.current_cat]
                 if st.session_state.level == "A2" or v["lvl"] == "A1"]
        if not items:
            items = VOCAB[st.session_state.current_cat]

        card_cols = st.columns(3)
        for idx, item in enumerate(items):
            card_key = f"{st.session_state.current_cat}_{idx}".replace(" ", "_").replace("&", "und")
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
                    label = "↺ Kembali" if flipped else "↺ Terjemahan"
                    if st.button(label, key=f"flip_{card_key}", use_container_width=True):
                        st.session_state.flipped[card_key] = not flipped
                        st.rerun()


def grammar_section():
    st.markdown('<div id="grammatik"></div>', unsafe_allow_html=True)
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


def quiz_section():
    st.markdown('<div id="quiz"></div>', unsafe_allow_html=True)
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
                st.write("")
                if st.button("Nochmal versuchen", key="restart_quiz", type="primary", use_container_width=True):
                    reset_quiz()
                    st.rerun()


def footer():
    st.markdown('<div class="app-footer">Deutschsprung · Übung macht den Meister 🇩🇪</div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. ROUTER
# ---------------------------------------------------------------------------
inject_css()

if not st.session_state.logged_in:
    login_screen()
else:
    top_nav()
    hero_section()
    vocab_section()
    st.write("")
    grammar_section()
    st.write("")
    quiz_section()
    footer()
