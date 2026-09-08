"""
Deutschsprung – Dein spielerischer Weg von A1 zu A2
Streamlit app (revisi v3 — gabungan v2 + fitur pilihan dari draft "Gemini")

Perubahan besar pada revisi v2:
  - Fix bug: reset kartu vocab (flipped) tiap ganti kategori
  - Fix bug: filter level A1/A2 tidak lagi "bocor" menampilkan level lain
  - Fix bug: logout memakai st.session_state.clear() (reset total)
  - Fitur: Kartu kosakata dilengkapi contoh kalimat + tips mnemonic,
        dan tombol "Tandai Hafal" (mastery tracker)
  - Fitur: Modul latihan konjugasi verba (sein, haben, lernen, fahren)
  - Fitur: Dashboard progres (kosakata dikuasai/dilihat & skor quiz)
  - UX: landing page (highlight fitur, tip harian, kartu navigasi,
        ringkasan progres), navigasi berbasis tombol (pill nav), grid kartu
        rapi, badge/alert umpan balik saat quiz & latihan verba dijawab
  - Fix bug tampilan mobile: header topnav tidak lagi terpotong, dan ikon
        expander tidak lagi menimpa teks ("arrow_right" overlap)

Tambahan revisi v3 (diadaptasi dari draft "Gemini", tanpa mengubah
struktur/estetika yang sudah stabil di atas):
  - Audio pengucapan native (Web Speech API browser, tanpa API key eksternal)
    pada setiap kartu kosakata
  - Badge warna gender pedagogis: der = biru, die = merah muda, das = hijau,
    otomatis dideteksi dari kata (mis. "der Vater")
  - Filter tambahan "Semua / Belum Hafal / Sudah Hafal" pada halaman Vokabeln
  - Pembahasan (penjelasan tata bahasa singkat) pada setiap soal quiz setelah
    dijawab, bukan cuma benar/salah

Struktur file:
  1. Konfigurasi halaman & data (vocab, grammar, verbs, questions, tips)
  2. Session state
  3. Data tambahan (tip harian)
  4. Audio engine (TTS native browser)
  5. CSS kustom
  6. Layar login (simulasi)
  7. Nav atas + dashboard progres
  8. Bagian: hero (landing), vocab, grammar, verben, quiz, footer
  9. Router berbasis st.session_state.page
"""

import random
import datetime

import streamlit as st
import streamlit.components.v1 as components

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
        {"de": "Hallo", "id": "Halo", "lvl": "A1",
         "ex": "Hallo, wie heißt du?", "tip": "Sapaan paling umum, cocok untuk siapa saja."},
        {"de": "Guten Morgen", "id": "Selamat pagi", "lvl": "A1",
         "ex": "Guten Morgen, Frau Schmidt!", "tip": "Dipakai sampai sekitar jam 10-11 pagi."},
        {"de": "Tschüss", "id": "Sampai jumpa", "lvl": "A1",
         "ex": "Tschüss, bis morgen!", "tip": "Versi santai dari 'Auf Wiedersehen'."},
        {"de": "Danke", "id": "Terima kasih", "lvl": "A1",
         "ex": "Danke für deine Hilfe.", "tip": "Tambahkan 'sehr' → 'Danke sehr' untuk lebih sopan."},
        {"de": "Entschuldigung", "id": "Maaf / Permisi", "lvl": "A1",
         "ex": "Entschuldigung, wo ist der Bahnhof?", "tip": "Dua fungsi: minta maaf & menyapa orang asing."},
        {"de": "Wie geht's?", "id": "Apa kabar?", "lvl": "A1",
         "ex": "Wie geht's? – Mir geht's gut, danke!", "tip": "Singkatan santai dari 'Wie geht es dir?'."},
    ],
    "Zahlen & Farben": [
        {"de": "eins, zwei, drei", "id": "satu, dua, tiga", "lvl": "A1",
         "ex": "Ich zähle: eins, zwei, drei.", "tip": "Fondasi semua angka bahasa Jerman."},
        {"de": "zehn", "id": "sepuluh", "lvl": "A1",
         "ex": "Ich habe zehn Finger.", "tip": "Angka 11-19 biasanya berakhiran '-zehn'."},
        {"de": "rot", "id": "merah", "lvl": "A1",
         "ex": "Die Ampel ist rot.", "tip": "Ingat lewat warna lampu lalu lintas."},
        {"de": "blau", "id": "biru", "lvl": "A1",
         "ex": "Der Himmel ist blau.", "tip": "Mirip kata Inggris 'blue', mudah diingat."},
        {"de": "gelb", "id": "kuning", "lvl": "A1",
         "ex": "Die Banane ist gelb.", "tip": "Bayangkan pisang kuning agar cepat melekat."},
    ],
    "Familie": [
        {"de": "die Mutter", "id": "ibu", "lvl": "A1",
         "ex": "Meine Mutter kocht gern.", "tip": "Anggota keluarga perempuan sering pakai artikel 'die'."},
        {"de": "der Vater", "id": "ayah", "lvl": "A1",
         "ex": "Mein Vater arbeitet viel.", "tip": "Anggota keluarga laki-laki sering pakai artikel 'der'."},
        {"de": "die Schwester", "id": "saudari perempuan", "lvl": "A1",
         "ex": "Ich habe eine Schwester.", "tip": "'Schwester' mirip 'sister' dalam bahasa Inggris."},
        {"de": "der Bruder", "id": "saudara laki-laki", "lvl": "A1",
         "ex": "Mein Bruder ist älter als ich.", "tip": "'Bruder' mirip 'brother' dalam bahasa Inggris."},
        {"de": "das Kind", "id": "anak", "lvl": "A1",
         "ex": "Das Kind spielt im Park.", "tip": "'Kind' selalu netral (das), meski merujuk siapa saja."},
    ],
    "Im Hotel": [
        {"de": "die Rezeption", "id": "resepsionis", "lvl": "A2",
         "ex": "Die Rezeption ist im Erdgeschoss.", "tip": "Mirip kata 'resepsi/reception' dalam bahasa Indonesia/Inggris."},
        {"de": "das Zimmer", "id": "kamar", "lvl": "A2",
         "ex": "Mein Zimmer hat die Nummer 12.", "tip": "Kata dasar untuk semua jenis ruangan/kamar."},
        {"de": "der Gast", "id": "tamu", "lvl": "A2",
         "ex": "Der Gast wartet an der Rezeption.", "tip": "Mirip kata 'guest' dalam bahasa Inggris."},
        {"de": "der Schlüssel", "id": "kunci", "lvl": "A2",
         "ex": "Wo ist mein Schlüssel?", "tip": "Bayangkan bentuk kunci saat mengucapkan 'Schlüssel'."},
        {"de": "die Buchung", "id": "pemesanan", "lvl": "A2",
         "ex": "Ich habe eine Buchung für zwei Nächte.", "tip": "Berasal dari kata 'buchen' (memesan)."},
    ],
    "Alltag & Zeit": [
        {"de": "heute", "id": "hari ini", "lvl": "A2",
         "ex": "Heute ist ein schöner Tag.", "tip": "Kata waktu paling sering dipakai sehari-hari."},
        {"de": "morgen", "id": "besok", "lvl": "A2",
         "ex": "Morgen fahre ich nach Berlin.", "tip": "Huruf kecil = 'besok'; huruf besar 'Morgen' = kata benda 'pagi'."},
        {"de": "pünktlich", "id": "tepat waktu", "lvl": "A2",
         "ex": "Der Zug kommt pünktlich.", "tip": "Ciri khas budaya Jerman: sangat menghargai ketepatan waktu."},
        {"de": "die Verabredung", "id": "janji temu", "lvl": "A2",
         "ex": "Ich habe eine Verabredung um 15 Uhr.", "tip": "Berasal dari kata 'sich verabreden' (membuat janji)."},
    ],
    "Essen & Trinken": [
        {"de": "das Brot", "id": "roti", "lvl": "A1",
         "ex": "Ich esse jeden Morgen Brot.", "tip": "Makanan pokok orang Jerman sehari-hari."},
        {"de": "das Wasser", "id": "air", "lvl": "A1",
         "ex": "Ein Glas Wasser, bitte.", "tip": "Kata paling penting saat memesan minuman."},
        {"de": "der Apfel", "id": "apel", "lvl": "A1",
         "ex": "Der Apfel ist süß.", "tip": "Mirip kata Inggris 'apple', mudah dihafal."},
        {"de": "das Restaurant", "id": "restoran", "lvl": "A2",
         "ex": "Wir gehen heute ins Restaurant.", "tip": "Kata serapan internasional, pengucapan mirip bahasa Prancis."},
        {"de": "die Rechnung", "id": "tagihan / bon", "lvl": "A2",
         "ex": "Können wir bitte die Rechnung haben?", "tip": "Kalimat wajib sebelum meninggalkan restoran."},
    ],
    "Reisen & Verkehr": [
        {"de": "der Zug", "id": "kereta", "lvl": "A1",
         "ex": "Der Zug fährt um acht Uhr.", "tip": "Transportasi favorit di Jerman, selalu diusahakan pünktlich."},
        {"de": "das Flugzeug", "id": "pesawat terbang", "lvl": "A1",
         "ex": "Das Flugzeug landet in Berlin.", "tip": "Gabungan 'fliegen' (terbang) + 'Zeug' (alat)."},
        {"de": "der Bahnhof", "id": "stasiun kereta", "lvl": "A2",
         "ex": "Der Bahnhof ist gleich um die Ecke.", "tip": "Gabungan 'Bahn' (rel) + 'Hof' (halaman/tempat)."},
        {"de": "die Fahrkarte", "id": "tiket perjalanan", "lvl": "A2",
         "ex": "Ich brauche eine Fahrkarte nach München.", "tip": "Berasal dari 'fahren' (berkendara) + 'Karte' (kartu)."},
        {"de": "der Flughafen", "id": "bandara", "lvl": "A2",
         "ex": "Wir treffen uns am Flughafen.", "tip": "Gabungan 'Flug' (penerbangan) + 'Hafen' (pelabuhan)."},
    ],
    "Beruf & Schule": [
        {"de": "die Arbeit", "id": "pekerjaan", "lvl": "A1",
         "ex": "Meine Arbeit macht mir Spaß.", "tip": "Kata dasar untuk semua topik seputar pekerjaan."},
        {"de": "die Schule", "id": "sekolah", "lvl": "A1",
         "ex": "Die Kinder gehen zur Schule.", "tip": "Mirip 'school' dalam bahasa Inggris."},
        {"de": "der Lehrer", "id": "guru (laki-laki)", "lvl": "A2",
         "ex": "Der Lehrer erklärt die Grammatik.", "tip": "Versi perempuan: 'die Lehrerin'."},
        {"de": "der Kollege", "id": "rekan kerja (laki-laki)", "lvl": "A2",
         "ex": "Mein Kollege hilft mir gern.", "tip": "Versi perempuan: 'die Kollegin'."},
        {"de": "das Büro", "id": "kantor", "lvl": "A2",
         "ex": "Ich arbeite im Büro bis 17 Uhr.", "tip": "Kata serapan dari bahasa Prancis 'bureau'."},
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
     "opts": ["Bitte", "Danke", "Tschüss", "Hallo"], "a": 1,
     "exp": "'Danke' = terima kasih. 'Bitte' dipakai untuk membalas ucapan terima kasih ('sama-sama')."},
    {"q": "'Die Mutter' bedeutet auf Indonesisch...",
     "opts": ["ayah", "ibu", "anak", "kakak"], "a": 1,
     "exp": "'Die Mutter' = ibu. Anggota keluarga perempuan hampir selalu berartikel 'die'."},
    {"q": "Welcher Artikel passt zu 'Buch'?",
     "opts": ["der", "die", "das", "den"], "a": 2,
     "exp": "'Buch' bergender netral, jadi artikelnya 'das Buch'."},
    {"q": "Ich ___ Student. (sein)",
     "opts": ["bin", "bist", "ist", "sind"], "a": 0,
     "exp": "Konjugasi 'sein' untuk subjek 'ich' adalah 'bin' → Ich bin Student."},
    {"q": "Wähle die richtige Präsensform: du ___ (lernen)",
     "opts": ["lerne", "lernst", "lernt", "lernen"], "a": 1,
     "exp": "Verba beraturan untuk subjek 'du' mendapat akhiran '-st' → du lernst."},
    {"q": "Perfekt von 'lernen' mit haben:",
     "opts": ["Ich habe gelernt.", "Ich bin gelernt.", "Ich lernte habe.", "Ich hatte lernen."], "a": 0,
     "exp": "'Lernen' adalah verba transitif biasa, jadi Perfekt-nya pakai 'haben' + Partizip II (gelernt)."},
    {"q": "Modalverb für 'harus' (müssen):",
     "opts": ["kann", "will", "muss", "mag"], "a": 2,
     "exp": "'Müssen' (harus) → konjugasi 'ich muss'. 'Kann' = bisa, 'will' = mau, 'mag' = suka."},
    {"q": "'Der Gast' im Hotelkontext bedeutet:",
     "opts": ["kunci", "kamar", "tamu", "resepsionis"], "a": 2,
     "exp": "'Der Gast' = tamu, mirip kata 'guest' dalam bahasa Inggris."},
    {"q": "Wie sagt man 'roti' auf Deutsch?",
     "opts": ["das Wasser", "das Brot", "der Apfel", "die Rechnung"], "a": 1,
     "exp": "'Das Brot' = roti, makanan pokok sehari-hari di Jerman."},
    {"q": "'Der Bahnhof' bedeutet auf Indonesisch...",
     "opts": ["bandara", "stasiun kereta", "tiket", "pesawat"], "a": 1,
     "exp": "'Der Bahnhof' = gabungan 'Bahn' (rel) + 'Hof' (tempat) → stasiun kereta."},
    {"q": "Welches Wort passt: 'Ich brauche eine ___ nach München.'",
     "opts": ["Fahrkarte", "Rechnung", "Schule", "Büro"], "a": 0,
     "exp": "'Die Fahrkarte' = tiket perjalanan, dibutuhkan sebelum naik kereta/bus."},
    {"q": "'Die Schule' bedeutet:",
     "opts": ["kantor", "sekolah", "pekerjaan", "restoran"], "a": 1,
     "exp": "'Die Schule' = sekolah, mirip kata 'school' dalam bahasa Inggris."},
    {"q": "Perfekt yang benar untuk 'fahren' (dengan sein):",
     "opts": ["Ich habe gefahren.", "Ich bin gefahren.", "Ich fahre gewesen.", "Ich war fahren."], "a": 1,
     "exp": "'Fahren' menyatakan perpindahan tempat, sehingga Perfekt-nya wajib pakai 'sein', bukan 'haben'."},
    {"q": "Wähle die richtige Präsensform: er ___ (fahren)",
     "opts": ["fahre", "fährst", "fährt", "fahren"], "a": 2,
     "exp": "'Fahren' mengalami Umlautwechsel (a→ä) untuk 'du' dan 'er/sie/es' → er fährt."},
]

# Data untuk mini-game "Satz-Bauer" (susun kalimat acak jadi kalimat benar)
SENTENCE_BUILDER_DATA = [
    {"target": "Ich lerne heute Deutsch", "meaning": "Saya belajar bahasa Jerman hari ini",
     "words": ["Ich", "lerne", "heute", "Deutsch"]},
    {"target": "Der Zug kommt pünktlich", "meaning": "Kereta datang tepat waktu",
     "words": ["Der", "Zug", "kommt", "pünktlich"]},
    {"target": "Wir gehen ins Restaurant", "meaning": "Kita pergi ke restoran",
     "words": ["Wir", "gehen", "ins", "Restaurant"]},
    {"target": "Das Kind spielt im Park", "meaning": "Anak itu bermain di taman",
     "words": ["Das", "Kind", "spielt", "im", "Park"]},
    {"target": "Mein Vater arbeitet im Büro", "meaning": "Ayah saya bekerja di kantor",
     "words": ["Mein", "Vater", "arbeitet", "im", "Büro"]},
]

PAGES = ["beranda", "vokabeln", "grammatik", "verben", "games", "quiz"]
PAGE_LABELS = {
    "beranda": "🏠 Beranda",
    "vokabeln": "🗂️ Vokabeln",
    "grammatik": "📘 Grammatik",
    "verben": "🔤 Verben",
    "games": "🎮 Mini-Games",
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
    "mastered_vocab": set(),             # kata (de) yang ditandai "sudah hafal"
    "filter_mastery": "Semua",           # "Semua" | "Belum Hafal" | "Sudah Hafal"
    "daily_tip_idx": None,               # index tip harian (diacak sekali per sesi)
    "q_index": 0,
    "q_score": 0,
    "q_selected": None,
    "q_answered": False,
    "q_done": False,
    "q_high_score": 0,
    "verb_choice": list(VERBS.keys())[0],
    "verb_best": {},                     # {verb: skor_terbaik}
    "verb_result": None,                 # hasil cek terakhir (untuk ditampilkan)
    # --- Gamifikasi & Mini-Games ---
    "xp": 0,
    "streak": 1,
    "last_active_date": None,            # diisi & dicek sekali per sesi login
    "badges": set(),
    "match_pairs": None,                 # {"de":[...], "id":[...], "raw":[...]}
    "match_selected_de": None,
    "match_selected_id": None,
    "match_solved": set(),
    "sb_index": 0,
    "sb_user_words": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def add_xp(points: int, reason: str = ""):
    """Tambah XP dan cek apakah ada lencana baru yang terbuka."""
    st.session_state.xp += points
    check_badges()
    if reason:
        st.toast(f"⚡ +{points} XP — {reason}")


def check_badges():
    """Buka lencana baru berdasarkan pencapaian saat ini (idempotent, aman dipanggil berkali-kali)."""
    milestones = [
        (len(st.session_state.mastered_vocab) >= 5, "🎯 Wortschatz-Anfänger (5 kata hafal)"),
        (len(st.session_state.mastered_vocab) >= 15, "🔥 Wortschatz-Meister (15 kata hafal)"),
        (st.session_state.q_high_score >= 10, "🧠 Quiz-Profi (skor quiz ≥ 10)"),
        (st.session_state.xp >= 100, "⭐ Deutsch-Lernender (100+ XP)"),
        (st.session_state.streak >= 3, "📅 Konsisten (3 hari berturut-turut)"),
    ]
    for achieved, label in milestones:
        if achieved:
            st.session_state.badges.add(label)


def update_daily_streak():
    """Naikkan streak jika login di hari berikutnya, reset jika ada hari yang terlewat.
    Dipanggil sekali per sesi (setelah login), tidak setiap render halaman."""
    today = datetime.date.today()
    last_str = st.session_state.last_active_date
    if last_str is None:
        st.session_state.streak = 1
    else:
        last_date = datetime.date.fromisoformat(last_str)
        delta_days = (today - last_date).days
        if delta_days == 1:
            st.session_state.streak += 1
        elif delta_days > 1:
            st.session_state.streak = 1
        # delta_days == 0 → sudah login hari ini, streak tidak berubah
    st.session_state.last_active_date = today.isoformat()
    check_badges()


def reset_quiz():
    st.session_state.q_index = 0
    st.session_state.q_score = 0
    st.session_state.q_selected = None
    st.session_state.q_answered = False
    st.session_state.q_done = False


def go_to(page: str):
    st.session_state.page = page


# ---------------------------------------------------------------------------
# 3. DATA TAMBAHAN — Tip harian untuk landing page
# ---------------------------------------------------------------------------
DAILY_TIPS = [
    "Buat 3 kalimat sendiri dengan kosakata yang baru kamu pelajari hari ini.",
    "Ulangi kosakata yang sudah 'Dikuasai' seminggu sekali agar tidak lupa.",
    "Fokus satu kategori dulu sebelum pindah ke kategori berikutnya.",
    "Latihan konjugasi verba 5 menit tiap hari lebih efektif daripada sekali seminggu.",
    "Ucapkan kosakata dengan lantang — otot mulutmu juga perlu latihan!",
    "Coba tulis buku harian singkat pakai kosakata yang sudah kamu kuasai.",
    "Salah itu wajar. Fokus pada progres, bukan kesempurnaan.",
]


# ---------------------------------------------------------------------------
# 4. AUDIO ENGINE — TTS native browser (Web Speech API, tanpa API key)
# ---------------------------------------------------------------------------
def play_pronunciation(text: str):
    """Memutar pengucapan Jerman lewat speechSynthesis bawaan browser.

    Tidak butuh koneksi ke layanan pihak ketiga (mis. Google Translate TTS),
    jadi tetap berfungsi offline selama browser mendukung Web Speech API.
    """
    safe_text = text.replace("\\", "").replace("'", "\\'").replace('"', '\\"')
    components.html(
        f"""
        <script>
        (function() {{
            if (!('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance('{safe_text}');
            msg.lang = 'de-DE';
            msg.rate = 0.92;
            window.speechSynthesis.speak(msg);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def render_speech_rec(target_text: str, key: str):
    """Komponen mikrofon untuk latihan pengucapan (Web Speech API — SpeechRecognition).
    Sepenuhnya opsional/best-effort: browser yang tidak mendukung akan
    menampilkan pesan singkat, tanpa mengganggu bagian lain dari kartu."""
    safe_target = target_text.replace("\\", "").replace("'", "\\'").replace('"', '\\"')
    components.html(
        f"""
        <div style="font-family:'Inter',sans-serif; text-align:center;">
            <button id="recBtn_{key}" style="background:transparent; border:1.5px solid #E8A93B;
                color:#E8A93B; padding:6px 14px; font-weight:600; font-size:0.82rem;
                border-radius:999px; cursor:pointer; width:100%;">
                🎙️ Latihan ucapkan
            </button>
            <div id="out_{key}" style="margin-top:6px; font-size:0.8rem; color:#F2ECDD; min-height:1.2em;"></div>
        </div>
        <script>
        (function() {{
            const btn = document.getElementById('recBtn_{key}');
            const out = document.getElementById('out_{key}');
            const target = "{safe_target}".toLowerCase();
            btn.addEventListener('click', () => {{
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) {{
                    out.innerText = 'Browser ini belum mendukung pengenalan suara.';
                    return;
                }}
                const rec = new SpeechRecognition();
                rec.lang = 'de-DE';
                out.innerText = 'Mendengarkan... ucapkan dalam bahasa Jerman.';
                rec.start();
                rec.onresult = (e) => {{
                    const transcript = e.results[0][0].transcript.toLowerCase();
                    if (transcript.includes(target) || target.includes(transcript)) {{
                        out.innerHTML = '✅ Tepat! Terdengar: "' + transcript + '"';
                        out.style.color = '#5FBF8E';
                    }} else {{
                        out.innerHTML = '❌ Kurang tepat. Terdengar: "' + transcript + '"';
                        out.style.color = '#C1440E';
                    }}
                }};
                rec.onerror = () => {{ out.innerText = 'Gagal mendeteksi suara, coba lagi.'; }};
            }});
        }})();
        </script>
        """,
        height=68,
    )


def gendered_word_html(de_text: str) -> str:
    """Mewarnai artikel der/die/das sesuai konvensi pedagogis DaF
    (der = biru, die = merah muda, das = hijau), tanpa mengubah struktur
    data VOCAB (artikel tetap menyatu dalam field 'de', mis. 'der Vater')."""
    parts = de_text.split(" ", 1)
    if len(parts) == 2 and parts[0] in ("der", "die", "das"):
        article, rest = parts
        cls = {"der": "gender-der", "die": "gender-die", "das": "gender-das"}[article]
        return f'<span class="{cls}">{article}</span> {rest}'
    return de_text


# ---------------------------------------------------------------------------
# 5. CSS KUSTOM
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

            /* Warna gender pedagogis DaF: der/die/das */
            --gender-der:#4E8FDE;
            --gender-die:#E1548C;
            --gender-das:#5FBF8E;
        }

        html, body, [class^="st-"], [class*=" st-"] { font-family:'Inter', sans-serif; }
        h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; }

        /* FIX BUG #2 — "arrow_right menimpa teks Mengingat":
           Aturan font-family umum di atas ikut menimpa font ikon Streamlit
           (Material Symbols), yang dipakai a.l. oleh panah expander.
           Font ikon itu pakai teknik ligature: kalau font-nya diganti ke
           'Inter', ligature gagal dan yang tampil malah TEKS MENTAH
           "arrow_right" bertumpuk di atas label "Tips mengingat".
           Kembalikan font asli khusus untuk elemen ikon. */
        [data-testid="stIconMaterial"],
        span[data-testid="stExpanderToggleIcon"],
        .material-symbols-rounded,
        .material-symbols-outlined {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
            font-size: 1.15rem !important;
            line-height: 1 !important;
        }

        /* Rapikan header expander agar ikon & teks tidak bertumpuk lagi */
        div[data-testid="stExpander"] summary {
            display:flex !important;
            align-items:center !important;
            gap:8px;
            line-height:1.5 !important;
            padding:10px 14px !important;
            min-height:2.4em;
        }
        div[data-testid="stExpander"] summary p {
            margin:0 !important;
            font-size:0.92rem;
        }

        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}

        /* Header bawaan Streamlit (tombol Fork/GitHub) dibuat transparan tapi
           tetap memakan ruang, supaya topnav kustom tidak ketiban / terpotong. */
        header[data-testid="stHeader"] {
            background:transparent;
            height:3rem;
        }

        .stApp { background:var(--ink); color:var(--paper); }
        .block-container {
            max-width:900px;
            padding-top:2rem;      /* FIX BUG #1: jarak aman dari header Streamlit */
            padding-bottom:3rem;
        }

        /* ---------- Responsif khusus layar HP ---------- */
        @media (max-width: 640px) {
            .block-container { padding-top:3.25rem !important; padding-left:1rem; padding-right:1rem; }
            .topnav { padding-top:0.75rem; }
        }

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
        /* FIX BUG #1: flex-wrap + align-items memastikan brand & user-chip
           tetap rapi bila lebar layar sempit (HP), bukan terdorong keluar. */
        .topnav {
            display:flex;
            flex-wrap:wrap;
            align-items:center;
            justify-content:space-between;
            row-gap:10px;
            column-gap:12px;
            padding:10px 0 16px;
            border-bottom:1px solid var(--line);
            margin-bottom:14px;
        }
        .brand { display:flex; align-items:center; gap:9px; font-weight:700; font-size:1.1rem; font-family:'Space Grotesk',sans-serif; flex-shrink:0; }
        .user-chip { display:flex; align-items:center; gap:8px; font-size:0.88rem; font-weight:600; max-width:100%; overflow:hidden; }
        .user-chip span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:38vw; }
        .user-chip .username-text { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:32vw; }
        .avatar { width:28px; height:28px; border-radius:50%; background:var(--mustard); color:var(--ink); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.82rem; flex-shrink:0; }

        /* ---------- Dashboard progres ---------- */
        div[data-testid="stMetric"] {
            background:rgba(242,236,221,0.06); border:1px solid var(--line);
            border-radius:12px; padding:10px 14px;
        }
        div[data-testid="stMetricLabel"] { color:rgba(242,236,221,0.75) !important; }
        div[data-testid="stMetricValue"] { color:var(--mustard) !important; }

        /* ---------- Gamifikasi: badge XP / streak / lencana ---------- */
        .badge-pill {
            display:inline-block; background:rgba(232,169,59,0.16); border:1px solid var(--mustard);
            color:var(--mustard); padding:3px 10px; border-radius:999px;
            font-size:0.75rem; font-weight:700; margin:2px 4px 2px 0;
        }

        /* ---------- Mini-Games ---------- */
        .st-key-games_wrap { background:var(--paper) !important; border:none !important; border-radius:16px; padding:1.6rem 1.4rem; color:var(--ink); }
        .st-key-games_wrap .section-head p { color:rgba(27,36,48,0.7); }
        .game-box {
            background:rgba(58,125,123,0.08); border:1.5px solid var(--teal);
            border-radius:12px; padding:14px 16px; margin-bottom:12px; color:var(--ink);
        }
        .sb-sentence {
            font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600;
            background:rgba(27,36,48,0.06); border-radius:10px; padding:10px 14px;
            min-height:2.2em; margin-bottom:10px; color:var(--ink);
        }

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
        div[class*="st-key-vcard_"] {
            min-height:118px; display:flex; flex-direction:column; justify-content:center;
            transition:transform .18s ease, box-shadow .18s ease;
        }
        div[class*="st-key-vcard_"]:hover { transform:translateY(-3px); box-shadow:0 8px 18px rgba(0,0,0,0.18); }
        div[class*="st-key-vcard_f_"] { background:var(--paper) !important; border:1.5px solid var(--ink) !important; }
        div[class*="st-key-vcard_b_"] { background:var(--brick) !important; border:1.5px solid var(--brick) !important; }
        .card-front { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.08rem; text-align:center; color:var(--ink); line-height:1.4; }
        .card-front .card-tag { display:block; font-size:0.7rem; color:var(--brick); margin-top:6px; font-weight:600; }
        .card-back { font-family:'Inter',sans-serif; font-weight:600; font-size:1rem; text-align:center; color:var(--paper); line-height:1.4; }
        .mastered-badge { margin-left:6px; }

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

        /* ---------- Landing page (Beranda) ---------- */
        .why-card { text-align:center; padding:4px 2px; }
        .why-icon { font-size:1.7rem; margin-bottom:6px; }
        .why-title { font-weight:700; font-family:'Space Grotesk',sans-serif; font-size:0.95rem; margin-bottom:4px; }
        .why-desc { font-size:0.82rem; opacity:0.8; }
        .tip-of-day { background:rgba(232,169,59,0.12); border:1.5px solid var(--mustard); border-radius:12px; padding:14px 16px; margin:18px 0; }
        .tip-of-day .tip-label { color:var(--mustard); font-weight:700; font-size:0.82rem; margin-bottom:4px; }
        .feature-icon { font-size:1.5rem; margin-bottom:4px; }
        .feature-title { font-weight:700; font-family:'Space Grotesk',sans-serif; font-size:1.02rem; margin-bottom:4px; }
        .feature-desc { font-size:0.85rem; opacity:0.82; margin-bottom:10px; min-height:2.6em; }
        .progress-snapshot { background:rgba(58,125,123,0.14); border:1.5px solid var(--teal); border-radius:12px; padding:14px 16px; margin:18px 0; }
        .progress-snapshot b { color:var(--mustard); }

        /* ---------- Kartu vocab: tandai hafal ---------- */
        div[class*="st-key-vcard_f_m1_"], div[class*="st-key-vcard_b_m1_"] {
            border-color:var(--mustard) !important; box-shadow:0 0 0 2px rgba(232,169,59,0.35) inset;
        }
        .mastered-badge { display:inline-block; background:var(--mustard); color:var(--ink); font-size:0.68rem; font-weight:700; padding:2px 9px; border-radius:999px; margin-left:6px; vertical-align:middle; }
        .card-example { font-size:0.82rem; opacity:0.85; margin-top:8px; line-height:1.45; font-style:italic; }

        /* ---------- Badge warna gender (der/die/das) ---------- */
        .gender-der { color:var(--gender-der); font-weight:800; }
        .gender-die { color:var(--gender-die); font-weight:800; }
        .gender-das { color:var(--gender-das); font-weight:800; }

        /* ---------- Pembahasan soal quiz ---------- */
        .quiz-exp-box {
            background:rgba(232,169,59,0.12); border:1.5px solid var(--mustard);
            border-radius:10px; padding:12px 14px; margin:10px 0 14px;
            font-size:0.88rem; line-height:1.5; color:var(--paper);
        }
        .quiz-exp-box b { color:var(--mustard); }

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
                        update_daily_streak()
                        add_xp(10, "Bonus login harian")
                        st.rerun()


# ---------------------------------------------------------------------------
# 6. NAV ATAS + DASHBOARD PROGRES
# ---------------------------------------------------------------------------
def top_nav():
    check_badges()  # sinkronkan lencana dengan progres terbaru sebelum ditampilkan
    initial = st.session_state.username[:1].upper() if st.session_state.username else "?"
    st.markdown(
        f"""
        <div class="topnav">
          <div class="brand"><span class="brand-dot"></span>Deutschsprung</div>
          <div class="user-chip">
            <span class="avatar">{initial}</span>
            <span class="username-text">{st.session_state.username}</span>
            <span class="badge-pill">⚡ {st.session_state.xp} XP</span>
            <span class="badge-pill">🔥 {st.session_state.streak} hari</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Dashboard progres ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Kosakata Dikuasai", f"{len(st.session_state.mastered_vocab)}/{TOTAL_VOCAB}")
    with m2:
        st.metric("Kosakata Dilihat", f"{len(st.session_state.viewed_vocab)}/{TOTAL_VOCAB}")
    with m3:
        st.metric("Skor Quiz Tertinggi", f"{st.session_state.q_high_score}/{len(QUESTIONS)}")
    with m4:
        st.metric("Lencana Didapat", f"{len(st.session_state.badges)}")

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
    # --- Hero utama ---
    st.markdown('<div class="eyebrow">Deine Deutschreise beginnt hier</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h1 class="hero-title">Von <span class="accent">Null</span> auf Deutsch –
        <span class="accent">Schritt</span> für Schritt.</h1>
        <p class="hero-desc">Halo, {st.session_state.username or 'Freund'}! Vokabeln, Grammatik,
        Verbkonjugation, dan quiz seru untuk level A1 & A2. Kein Auswendiglernen ohne Sinn –
        nur klare Häppchen, die wirklich hängen bleiben.</p>
        """,
        unsafe_allow_html=True,
    )

    # --- Kenapa belajar di sini (3 highlight) ---
    why_cols = st.columns(3)
    highlights = [
        ("🎯", "Interaktif", "Kartu, latihan verba, dan quiz — bukan sekadar teks untuk dihafal."),
        ("🧠", "Mudah Diingat", "Setiap kata punya contoh kalimat dan tips mnemonic sendiri."),
        ("🏆", "Progres Terukur", "Lacak kosakata yang sudah kamu kuasai dan skor terbaikmu."),
    ]
    for col, (icon, title, desc) in zip(why_cols, highlights):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="why-card">
                      <div class="why-icon">{icon}</div>
                      <div class="why-title">{title}</div>
                      <div class="why-desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --- Tip harian (diacak sekali per sesi) ---
    if st.session_state.daily_tip_idx is None:
        st.session_state.daily_tip_idx = random.randrange(len(DAILY_TIPS))
    st.markdown(
        f"""
        <div class="tip-of-day">
          <div class="tip-label">💡 TIPP DES TAGES</div>
          <div>{DAILY_TIPS[st.session_state.daily_tip_idx]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Ringkasan progres (hanya jika sudah ada aktivitas) ---
    has_progress = (
        st.session_state.viewed_vocab
        or st.session_state.mastered_vocab
        or st.session_state.q_high_score > 0
    )
    if has_progress:
        st.markdown(
            f"""
            <div class="progress-snapshot">
              📊 Progres kamu: <b>{len(st.session_state.mastered_vocab)}/{TOTAL_VOCAB}</b> kosakata dikuasai ·
              <b>{len(st.session_state.viewed_vocab)}/{TOTAL_VOCAB}</b> sudah dilihat ·
              skor quiz tertinggi <b>{st.session_state.q_high_score}/{len(QUESTIONS)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Lencana yang sudah didapat (jika ada) ---
    if st.session_state.badges:
        st.markdown(
            '<div class="section-head" style="margin-top:10px;"><h2>🏆 Lencanamu</h2></div>',
            unsafe_allow_html=True,
        )
        badge_html = "".join(f'<span class="badge-pill">{b}</span>' for b in sorted(st.session_state.badges))
        st.markdown(badge_html, unsafe_allow_html=True)
        st.write("")

    # --- Kartu navigasi ke tiap mode belajar ---
    st.markdown('<div class="section-head" style="margin-top:6px;"><h2>Pilih mode belajar</h2></div>',
                 unsafe_allow_html=True)

    feature_cards = [
        ("vokabeln", "🗂️", "Vokabeln", "Kartu kosakata bertema, lengkap dengan audio, contoh kalimat & tips mengingat."),
        ("grammatik", "📘", "Grammatik", "Ringkasan grammar A1-A2 yang paling sering dipakai sehari-hari."),
        ("verben", "🔤", "Verben", "Latihan konjugasi verba dasar seperti sein, haben, lernen, fahren."),
        ("games", "🎮", "Mini-Games", "Wort-Match & Satz-Bauer — asah kosakata & tata kalimat sambil main."),
        ("quiz", "📝", "Quiz", f"Uji pemahamanmu lewat {len(QUESTIONS)} soal campuran A1 & A2."),
    ]
    fcols = st.columns(2)
    for i, (page_key, icon, title, desc) in enumerate(feature_cards):
        with fcols[i % 2]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Buka {title} →", key=f"cta_{page_key}", type="primary", use_container_width=True):
                    go_to(page_key)
                    st.rerun()
    st.write("")


def vocab_section():
    with st.container(key="vocab_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Vokabelkarten</h2>
              <p>Klick auf eine Karte, um die Übersetzung samt Beispielsatz zu sehen, und markiere
              Wörter, die du schon auswendig kannst.</p>
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

        # --- Filter tambahan: status hafalan (Semua / Belum / Sudah Hafal) ---
        st.session_state.filter_mastery = st.radio(
            "Filter status",
            ["Semua", "Belum Hafal", "Sudah Hafal"],
            horizontal=True,
            key="filter_mastery_radio",
            index=["Semua", "Belum Hafal", "Sudah Hafal"].index(st.session_state.filter_mastery),
        )

        # --- Grid kartu, difilter murni berdasarkan level aktif ---
        # FIX BUG: A1 hanya menampilkan A1; A2 menampilkan A1+A2 (tanpa fallback
        # yang membocorkan level lain saat suatu kategori kosong di A1).
        all_items = VOCAB[st.session_state.current_cat]
        if st.session_state.level == "A1":
            items = [v for v in all_items if v["lvl"] == "A1"]
        else:
            items = list(all_items)  # A2 = gabungan A1 + A2

        # Terapkan filter status hafalan di atas hasil filter level
        if st.session_state.filter_mastery == "Belum Hafal":
            items = [v for v in items if v["de"] not in st.session_state.mastered_vocab]
        elif st.session_state.filter_mastery == "Sudah Hafal":
            items = [v for v in items if v["de"] in st.session_state.mastered_vocab]

        if not items:
            st.info(
                f"Tidak ada kosakata yang cocok di kategori **{st.session_state.current_cat}** "
                "dengan filter saat ini. Coba ganti level atau filter status di atas. 👆"
            )
            return

        card_cols = st.columns(3)
        for idx, item in enumerate(items):
            safe_cat = st.session_state.current_cat.replace(" ", "_").replace("&", "und")
            card_key = f"{safe_cat}_{item['de']}".replace(" ", "_").replace(",", "").replace("'", "")
            flipped = st.session_state.flipped.get(card_key, False)
            mastered = item["de"] in st.session_state.mastered_vocab
            side = "b" if flipped else "f"
            mstate = "m1" if mastered else "m0"
            with card_cols[idx % 3]:
                with st.container(border=True, key=f"vcard_{side}_{mstate}_{card_key}"):
                    badge = '<span class="mastered-badge">✓ Hafal</span>' if mastered else ""
                    de_html = gendered_word_html(item["de"])  # pewarnaan der/die/das
                    if flipped:
                        st.markdown(
                            f'<div class="card-back">{item["id"]}{badge}'
                            f'<div class="card-example">„{item["ex"]}"</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="card-front">{de_html}{badge}'
                            f'<span class="card-tag">{item["lvl"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                    fc0, fc1, fc2 = st.columns([1, 2.4, 2.4])
                    with fc0:
                        if st.button("🔊", key=f"tts_{card_key}", use_container_width=True,
                                      help="Dengarkan pengucapan (butuh browser yang mendukung Web Speech API)"):
                            play_pronunciation(item["de"])
                            st.session_state.viewed_vocab.add(item["de"])
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
                        star_label = "★ Hafal" if mastered else "☆ Hafal"
                        if st.button(star_label, key=f"master_{card_key}", use_container_width=True,
                                      type="primary" if mastered else "secondary",
                                      help="Tandai kosakata ini sudah kamu kuasai"):
                            if mastered:
                                st.session_state.mastered_vocab.discard(item["de"])
                            else:
                                st.session_state.mastered_vocab.add(item["de"])
                                add_xp(5, "Kosakata baru dikuasai")
                            st.rerun()

                    with st.expander("💡 Tips mengingat"):
                        st.caption(item["tip"])
                    render_speech_rec(item["de"], key=card_key)


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
            add_xp(correct_count * 3, "Latihan konjugasi verba")

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


def _init_match_round():
    """Ambil 4 pasangan kata acak dari SELURUH kategori (bukan hanya 1-2 kategori
    seperti draft awal), supaya tiap ronde terasa berbeda dan tetap valid
    walau kategori tertentu jumlah katanya sedikit."""
    pool = [item for cat_items in VOCAB.values() for item in cat_items]
    sample_size = min(4, len(pool))
    chosen = random.sample(pool, sample_size)
    de_list = [w["de"] for w in chosen]
    id_list = [w["id"] for w in chosen]
    random.shuffle(de_list)
    random.shuffle(id_list)
    st.session_state.match_pairs = {"de": de_list, "id": id_list, "raw": chosen}
    st.session_state.match_solved = set()
    st.session_state.match_selected_de = None
    st.session_state.match_selected_id = None


def games_section():
    with st.container(key="games_wrap"):
        st.markdown(
            """
            <div class="section-head">
              <h2>Mini-Games</h2>
              <p>Belajar sambil main: jodohkan kata di Wort-Match, atau susun kalimat di Satz-Bauer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["🧩 Wort-Match", "🧱 Satz-Bauer"])

        # ---------------- GAME 1: WORT-MATCH ----------------
        with tab1:
            st.markdown('<div class="game-box">Klik satu kata Jerman, lalu klik artinya dalam Bahasa Indonesia. '
                        'Pasangan yang benar akan otomatis tertandai selesai.</div>', unsafe_allow_html=True)

            if not st.session_state.match_pairs:
                _init_match_round()

            col_de, col_id = st.columns(2)
            with col_de:
                st.markdown("**🇩🇪 Deutsch**")
                for word in st.session_state.match_pairs["de"]:
                    is_solved = word in st.session_state.match_solved
                    if st.button(
                        word, key=f"m_de_{word}", disabled=is_solved, use_container_width=True,
                        type="primary" if st.session_state.match_selected_de == word else "secondary",
                    ):
                        st.session_state.match_selected_de = word
                        st.rerun()
            with col_id:
                st.markdown("**🇮🇩 Indonesisch**")
                for word in st.session_state.match_pairs["id"]:
                    is_solved = word in st.session_state.match_solved
                    if st.button(
                        word, key=f"m_id_{word}", disabled=is_solved, use_container_width=True,
                        type="primary" if st.session_state.match_selected_id == word else "secondary",
                    ):
                        st.session_state.match_selected_id = word
                        st.rerun()

            sel_de = st.session_state.match_selected_de
            sel_id = st.session_state.match_selected_id
            if sel_de and sel_id:
                is_correct_pair = any(
                    w["de"] == sel_de and w["id"] == sel_id for w in st.session_state.match_pairs["raw"]
                )
                if is_correct_pair:
                    st.success(f"✅ Benar! {sel_de} = {sel_id}")
                    st.session_state.match_solved.add(sel_de)
                    st.session_state.match_solved.add(sel_id)
                    st.session_state.match_selected_de = None
                    st.session_state.match_selected_id = None
                    add_xp(10, "Mencocokkan kata dengan benar")
                    st.rerun()
                else:
                    st.error("❌ Pasangan kurang tepat, coba lagi.")
                    st.session_state.match_selected_de = None
                    st.session_state.match_selected_id = None

            total_pairs = len(st.session_state.match_pairs["de"])
            if total_pairs and len(st.session_state.match_solved) >= total_pairs * 2:
                st.balloons()
                st.success("🎉 Selamat! Kamu menyelesaikan ronde ini!")
                if st.button("🔄 Mainkan Ronde Baru", key="reset_match", type="primary"):
                    st.session_state.match_pairs = None
                    st.rerun()

        # ---------------- GAME 2: SATZ-BAUER ----------------
        with tab2:
            st.markdown('<div class="game-box">Susun kembali kata-kata acak menjadi kalimat bahasa Jerman '
                        'yang benar sesuai arti yang diberikan.</div>', unsafe_allow_html=True)

            sb_item = SENTENCE_BUILDER_DATA[st.session_state.sb_index]
            st.info(f"Arti kalimat: **{sb_item['meaning']}**")

            current_sentence = " ".join(st.session_state.sb_user_words)
            st.markdown(f'<div class="sb-sentence">{current_sentence or "&nbsp;"}</div>', unsafe_allow_html=True)

            # Blok kata yang tersedia — hitung sisa kemunculan tiap kata agar
            # kata yang berulang (mis. dua "im") tetap bisa dipakai penuh.
            remaining_words = list(sb_item["words"])
            for used in st.session_state.sb_user_words:
                if used in remaining_words:
                    remaining_words.remove(used)

            if remaining_words:
                word_cols = st.columns(len(remaining_words))
                for i, w in enumerate(remaining_words):
                    with word_cols[i]:
                        if st.button(w, key=f"sb_w_{st.session_state.sb_index}_{i}_{w}", use_container_width=True):
                            st.session_state.sb_user_words.append(w)
                            st.rerun()

            st.write("")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅ Cek Kalimat", type="primary", use_container_width=True):
                    if current_sentence.strip() == sb_item["target"]:
                        st.success("🎉 Perfekt! Struktur kalimatmu benar!")
                        add_xp(15, "Menyusun kalimat dengan benar")
                        st.balloons()
                    else:
                        st.error(f"❌ Belum tepat. Kalimat yang benar: {sb_item['target']}")
            with b2:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state.sb_user_words = []
                    st.rerun()
            with b3:
                if st.button("➡️ Kalimat Berikutnya", use_container_width=True):
                    st.session_state.sb_index = (st.session_state.sb_index + 1) % len(SENTENCE_BUILDER_DATA)
                    st.session_state.sb_user_words = []
                    st.rerun()


def quiz_section():
    with st.container(key="quiz_wrap"):
        st.markdown(
            f"""
            <div class="section-head">
              <h2>Teste dich selbst</h2>
              <p>{len(QUESTIONS)} Fragen, gemischt aus A1 und A2. Viel Erfolg!</p>
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
                                add_xp(8, "Jawaban quiz benar")
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

                    if item.get("exp"):
                        st.markdown(
                            f'<div class="quiz-exp-box">💡 <b>Pembahasan:</b> {item["exp"]}</div>',
                            unsafe_allow_html=True,
                        )

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
                st.markdown('<p class="quiz-done-caption">Richtige Antworten von {}</p>'.format(len(QUESTIONS)),
                             unsafe_allow_html=True)

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
    elif page == "games":
        games_section()
    elif page == "quiz":
        quiz_section()

    st.write("")
    footer()
