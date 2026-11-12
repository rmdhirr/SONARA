from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from inference import (
    DETECTOR_FILES,
    SPECIES_MODEL_FILE,
    TARGET_SR,
    WINDOW_SECONDS,
    HOP_SECONDS,
    analyze_recording,
    load_model_bundle,
    make_species_prediction,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("CETUSID_MODEL_DIR", APP_DIR / "models"))

st.set_page_config(
    page_title="SONARA | Dolphin Vocalisation Analysis",
    page_icon="🐬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The pixel font is decorative; system monospace remains the fallback if Google Fonts
# cannot be reached by the browser. Body copy stays larger than the pixel labels.
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
:root {
  --bg: #07131f;
  --bg-2: #0a1d2b;
  --panel: #10283a;
  --panel-2: #15364a;
  --panel-3: #1b4356;
  --line: #2d6070;
  --line-soft: rgba(104, 200, 195, .22);
  --ink: #edf8f2;
  --muted: #a8c2c6;
  --teal: #6be3cf;
  --aqua: #83cfff;
  --yellow: #f4d477;
  --coral: #ff8c85;
  --pixel-shadow: 4px 4px 0 #030a12;
}
html, body, [class*="css"] { font-family: 'VT323', 'Courier New', monospace; }
.stApp {
  color: var(--ink);
  background-color: var(--bg);
  background-image:
    linear-gradient(rgba(76, 146, 160, .045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(76, 146, 160, .045) 1px, transparent 1px),
    linear-gradient(180deg, #07131f 0%, #0a1b29 58%, #07131f 100%);
  background-size: 24px 24px, 24px 24px, auto;
}
[data-testid="stHeader"] { background: rgba(7, 19, 31, .94); }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stSidebar"] {
  background: #091a27;
  border-right: 2px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.15rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
.block-container { max-width: 1280px; padding: 1.45rem clamp(1rem, 3vw, 2.4rem) 2.5rem; }

/* Type system: readable retro terminal copy, pixel font reserved for micro-labels. */
h1, h2, h3, h4 { color: var(--ink) !important; font-family: 'VT323', monospace !important; letter-spacing: .015em; }
h1 { font-size: clamp(2.45rem, 5vw, 4.1rem) !important; line-height: .98 !important; }
h2 { font-size: clamp(2rem, 3vw, 2.7rem) !important; }
h3 { font-size: 1.85rem !important; }
p, li, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
p { font-size: 1.22rem; line-height: 1.35; }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-testid="stCaptionContainer"] p { font-size: 1.05rem; }
hr { border-color: var(--line) !important; }

.pixel-label, .topline, .brand-sub, .pixel-chip, .pixel-kicker, .status-pill, .step-index {
  font-family: 'Press Start 2P', 'Courier New', monospace !important;
  text-transform: uppercase;
  letter-spacing: .035em;
}
.topline { color: var(--aqua); font-size: .52rem; line-height: 1.8; margin: .2rem 0 .55rem; }
.brand-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.brand { color: var(--ink); font: 400 clamp(2.5rem, 5vw, 4rem)/.9 'VT323', monospace; letter-spacing: .06em; }
.brand-mark { color: var(--teal); }
.brand-sub { color: var(--yellow); font-size: .48rem; line-height: 1.8; margin-top: .45rem; }
.header-readout {
  border: 2px solid var(--line); background: #0b2231; box-shadow: var(--pixel-shadow);
  padding: .7rem .9rem; text-align: right; min-width: 180px;
}
.header-readout .big { color: var(--teal); font: 400 2rem/1 'VT323', monospace; }
.header-readout .small { color: var(--muted); font: .5rem/1.8 'Press Start 2P', monospace; }

.hero-panel {
  display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(220px, .8fr);
  gap: 1.25rem; align-items: center; margin: 1.3rem 0 1.1rem;
  padding: clamp(1.1rem, 3vw, 2.1rem); background: #0d2638;
  border: 3px solid var(--teal); box-shadow: 6px 6px 0 #030a12;
}
.hero-copy { min-width: 0; }
.pixel-kicker { color: var(--yellow); font-size: .52rem; line-height: 1.9; margin-bottom: .85rem; }
.hero-panel h1 { margin: 0 0 .7rem !important; }
.hero-panel p { color: #c5d9d7; max-width: 650px; margin: 0; }
.hero-visual {
  min-width: 0; border: 2px solid var(--line); background: #071a28;
  padding: .5rem; box-shadow: inset 0 0 0 3px rgba(107,227,207,.06);
}
.hero-visual svg { display: block; width: 100%; height: auto; image-rendering: pixelated; shape-rendering: crispEdges; }
.pixel-chip {
  display: inline-block; padding: .42rem .55rem; margin-top: .9rem;
  color: var(--teal); background: #0a1c29; border: 1px solid var(--line);
  font-size: .45rem; line-height: 1.8;
}

.pixel-card {
  border: 2px solid var(--line); background: rgba(13, 38, 55, .95);
  padding: .9rem 1rem; box-shadow: 4px 4px 0 #030a12; height: 100%;
}
.pixel-card-title { color: var(--yellow); font-size: .52rem; line-height: 1.9; margin-bottom: .45rem; }
.section-kicker { color: var(--aqua); font: .5rem/1.8 'Press Start 2P', monospace; margin: 1.2rem 0 .45rem; }
.steps-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .85rem; margin: .6rem 0 1.2rem; }
.step-card { background: #0d2638; border: 2px solid var(--line); box-shadow: 4px 4px 0 #030a12; padding: .85rem; min-width: 0; }
.step-index { color: var(--teal); font-size: .55rem; margin-bottom: .6rem; }
.step-card b { color: var(--yellow); font-size: 1.2rem; letter-spacing: .04em; }
.step-card p { color: var(--muted); font-size: 1.08rem; margin: .3rem 0 0; }

.status-row { display: flex; align-items: flex-start; justify-content: space-between; gap: .55rem; padding: .38rem 0; border-bottom: 1px dashed rgba(132,190,202,.18); color: var(--muted); }
.status-row span:first-child { min-width: 0; overflow-wrap: anywhere; }
.status-pill { flex: 0 0 auto; font-size: .38rem; padding: .3rem .35rem; border: 1px solid; line-height: 1.5; }
.status-pill.ok { color: var(--teal); border-color: var(--teal); }
.status-pill.off { color: var(--coral); border-color: var(--coral); }

/* Streamlit controls and output blocks: square pixel borders, consistent spacing. */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  min-height: 2.7rem; height: auto; white-space: normal; overflow-wrap: anywhere;
  border-radius: 0 !important; border: 2px solid var(--teal) !important;
  background: #17495b !important; color: var(--ink) !important;
  font: 1.15rem 'VT323', monospace !important; letter-spacing: .04em;
  box-shadow: 3px 3px 0 #030a12; padding: .45rem .8rem !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #1d6170 !important; border-color: var(--yellow) !important;
}
.stButton > button:focus, .stDownloadButton > button:focus { outline: 2px solid var(--yellow) !important; }
[data-testid="stMetric"] {
  min-width: 0; border-radius: 0; border: 2px solid var(--line);
  background: #0d2638; box-shadow: 4px 4px 0 #030a12; padding: .8rem .9rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 1.05rem !important; }
[data-testid="stMetricValue"] { color: var(--teal) !important; font-family: 'VT323', monospace !important; font-size: 2.15rem !important; overflow-wrap: anywhere; }
[data-testid="stFileUploader"] {
  border: 2px dashed var(--teal); border-radius: 0; background: rgba(13, 38, 55, .72); padding: .45rem;
}
[data-testid="stFileUploader"] section { border-radius: 0; }
[data-testid="stAlert"] { border-radius: 0 !important; border-width: 2px !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 2px solid var(--line); }
.stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; gap: .35rem; }
.stTabs [data-baseweb="tab"] {
  min-height: 2.35rem; background: #102b3d; border: 1px solid var(--line);
  border-radius: 0; color: var(--ink); padding: .4rem .7rem;
}
.stTabs [aria-selected="true"] { border: 2px solid var(--teal) !important; color: var(--teal) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
  border-radius: 0 !important; border-color: var(--line) !important; background: #0c2231 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .35rem; }
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--ink) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 0; background: rgba(12,34,49,.65); }

.info-strip {
  border: 2px solid var(--yellow); background: #292b29; color: #fff2c8;
  padding: .75rem .9rem; box-shadow: 4px 4px 0 #030a12; margin: .5rem 0 1rem;
}
.warning-strip { border-color: var(--coral); background: #2c2028; color: #ffe0dc; }
.footer-line { color: var(--muted); text-align: center; font-size: 1.05rem; border-top: 1px dashed var(--line); padding-top: .8rem; margin-top: 1.5rem; }

@media (max-width: 850px) {
  .hero-panel { grid-template-columns: minmax(0, 1fr); }
  .hero-visual { max-width: 440px; width: 100%; margin: 0 auto; }
  .header-readout { min-width: 0; text-align: left; }
}
@media (max-width: 620px) {
  .block-container { padding-left: .85rem; padding-right: .85rem; }
  .steps-grid { grid-template-columns: minmax(0, 1fr); }
  .topline, .brand-sub, .pixel-kicker { overflow-wrap: anywhere; }
  .header-readout { margin-top: .5rem; }
  [data-testid="stHorizontalBlock"] { gap: .65rem; }
  [data-testid="stMetric"] { padding: .65rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

LANGUAGES = ["English", "Bahasa Indonesia"]
PAGE_LABELS = {
    "English": {
        "overview": "Mission briefing",
        "analysis": "Dolphin detection",
        "history": "Detection log",
        "research": "Research bench",
        "guide": "Field notes",
    },
    "Bahasa Indonesia": {
        "overview": "Ringkasan misi",
        "analysis": "Deteksi lumba-lumba",
        "history": "Log deteksi",
        "research": "Meja riset",
        "guide": "Catatan lapangan",
    },
}

COPY = {
    "English": {
        "language": "LANGUAGE / BAHASA",
        "mission": "DOLPHIN PASSIVE ACOUSTIC MONITORING",
        "subtitle": "DOLPHIN VOCALISATION ANALYSIS CONSOLE",
        "header_note": "TARGET CLASS: DOLPHIN VOCALISATIONS",
        "hero_kicker": "FIELD CONSOLE / MISSION 01",
        "hero_title": "Find the dolphin calls in the noise.",
        "hero_body": "SONARA scans underwater WAV recordings for candidate dolphin vocalisations. It is a focused dolphin-acoustics tool — not a general ocean-sound classifier.",
        "hero_chip": "PRIMARY TASK: VOCALISATION / BACKGROUND DETECTION",
        "notice_title": "READ THIS BEFORE INTERPRETING A DETECTION",
        "notice": "A positive result is a candidate dolphin-vocalisation window, not proof of a dolphin encounter. Review the audio before drawing conclusions.",
        "how": "THREE-STEP WORKFLOW",
        "step1": "UPLOAD",
        "step1p": "Add an underwater WAV recording.",
        "step2": "DETECT",
        "step2p": "Three detector variants scan overlapping two-second windows for dolphin-vocalisation patterns.",
        "step3": "REVIEW",
        "step3p": "Listen to candidate events, inspect timestamps, and export results.",
        "analysis_title": "Dolphin-vocalisation detection",
        "analysis_body": "The detector answers a focused question: does this two-second acoustic window resemble the dolphin-vocalisation class learned during training, rather than the negative/background class?",
        "upload": "Underwater WAV recording",
        "upload_help": "WAV only. Very long recordings can take longer to scan on free hosting.",
        "threshold": "Detection threshold",
        "models": "Detector variants",
        "run": "RUN DOLPHIN DETECTION",
        "need_audio": "Upload a WAV recording to begin.",
        "short_audio": "The recording must be at least 2 seconds long.",
        "invalid_audio": "The uploaded file could not be decoded as audio. Please check that it is a valid WAV file.",
        "analyzing": "Building spectrogram windows and running the selected detectors…",
        "analysis_complete": "Detection scan complete.",
        "no_detectors": "No detector model files are available yet. Add the three detector .keras files to the repository's models/ folder.",
        "results": "DETECTION REPORT",
        "duration": "Recording duration",
        "windows": "Windows scanned",
        "detectors_run": "Detectors run",
        "event_counts": "Candidate events by detector",
        "events": "candidate events",
        "no_events": "No windows crossed the selected threshold. This does not establish that dolphin vocalisations are absent.",
        "some_events": "Candidate dolphin-vocalisation windows were detected. Review the clips before drawing conclusions.",
        "timeline": "Window-level detection timeline",
        "details": "Window-level results and exports",
        "start": "Start (s)",
        "end": "End (s)",
        "download_csv": "Download window results (CSV)",
        "download_json": "Download analysis report (JSON)",
        "inspect_event": "Choose a candidate event to listen to",
        "download_clip": "Download candidate event WAV",
        "species_title": "EXPERIMENTAL SPECIES CLASSIFICATION",
        "species_body": "This optional second stage classifies a selected candidate event into one of three trained labels: Delphinus delphis, Sousa plumbea, and Tursiops aduncus.",
        "species_event": "Candidate event to classify",
        "species_run": "RUN EXPERIMENTAL SPECIES MODEL",
        "species_no_events": "Run detection first and select a candidate event. Species classification is only available for detected event clips.",
        "species_result": "Model prediction",
        "species_missing": "The experimental species model file is missing.",
        "status": "MODEL FILE STATUS",
        "detector_status": "Detectors",
        "species_status": "Experimental species model",
        "available": "READY",
        "missing": "MISSING",
        "history_title": "Detection log",
        "history_empty": "No scans in this session yet.",
        "clear_history": "Clear session log",
        "research_title": "Research bench: controlled SNR experiment",
        "research_body": "Compare the three detector variants on the demonstration-subset experiment. These values are preliminary and are not a guarantee of performance on uploaded field recordings.",
        "metric": "Benchmark metric",
        "snr_note": "The SNR results are from a demonstration subset and are not directly comparable with the paper's full independent test benchmarks. Repeated identical values across multiple SNR levels should be audited before strong claims are made.",
        "guide_title": "Field notes",
        "what_detector": "What does the main detector do?",
        "what_detector_p": "It performs binary detection: dolphin-vocalisation-like audio versus the negative/background class represented during training. It does not identify arbitrary ocean sounds, and it does not tell you which species produced every positive window.",
        "what_species": "What does the optional species stage do?",
        "what_species_p": "It is a separate model with three output labels: Delphinus delphis (common dolphin), Sousa plumbea (Indian Ocean humpback dolphin), and Tursiops aduncus (Indo-Pacific bottlenose dolphin).",
        "what_snr": "What is SNR?",
        "what_snr_p": "Signal-to-noise ratio compares signal power with background-noise power. Lower SNR means a signal is more heavily masked by noise. The SNR curriculum is a training strategy for the detector, not a separate sound category.",
        "limitations": "Scientific limitations",
        "limitations_p": "The app uses regular sliding windows at inference, while the training notebook used annotation-associated windows. Uploads have no ground-truth labels or measured SNR by default. Real ocean recordings may differ in hydrophone response, propagation, ambient noise, species distribution, and recording conditions. Treat detections as screening cues requiring expert review.",
        "footer": "SONARA · CetusID-derived dolphin-vocalisation research prototype · Not a validated ecological monitoring instrument",
        "about": "ABOUT SONARA",
        "about_body": "Focused dolphin-vocalisation detection, with a separate experimental three-class species stage.",
        "sample_rate": "Target sample rate",
        "window": "Window length",
        "hop": "Hop length",
        "source": "Source recording",
        "original_rate": "Original sample rate",
        "all_models": "Select all available detectors",
        "model_name": "Detector",
        "created": "Scanned at",
        "clear_notice": "New recording selected. Previous results have been cleared.",
    },
    "Bahasa Indonesia": {
        "language": "BAHASA / LANGUAGE",
        "mission": "PEMANTAUAN AKUSTIK PASIF LUMBA-LUMBA",
        "subtitle": "KONSOL ANALISIS VOKALISASI LUMBA-LUMBA",
        "header_note": "KELAS TARGET: VOKALISASI LUMBA-LUMBA",
        "hero_kicker": "KONSOL LAPANGAN / MISI 01",
        "hero_title": "Temukan suara lumba-lumba di tengah derau.",
        "hero_body": "SONARA memindai rekaman WAV bawah air untuk mencari kandidat vokalisasi lumba-lumba. Ini adalah alat akustik khusus lumba-lumba — bukan pengklasifikasi semua suara laut.",
        "hero_chip": "TUGAS UTAMA: DETEKSI VOKALISASI / LATAR",
        "notice_title": "BACA SEBELUM MENAFSIRKAN HASIL DETEKSI",
        "notice": "Hasil positif adalah kandidat jendela vokalisasi lumba-lumba, bukan bukti pasti adanya perjumpaan dengan lumba-lumba. Dengarkan audio sebelum menarik kesimpulan.",
        "how": "ALUR KERJA TIGA LANGKAH",
        "step1": "UNGGAH",
        "step1p": "Masukkan rekaman WAV bawah air.",
        "step2": "DETEKSI",
        "step2p": "Tiga varian detektor memindai jendela dua detik yang saling bertumpang tindih untuk mencari pola vokalisasi lumba-lumba.",
        "step3": "TINJAU",
        "step3p": "Dengarkan kandidat kejadian, periksa waktunya, dan ekspor hasil.",
        "analysis_title": "Deteksi vokalisasi lumba-lumba",
        "analysis_body": "Detektor menjawab pertanyaan khusus: apakah jendela akustik dua detik ini menyerupai kelas vokalisasi lumba-lumba yang dipelajari saat pelatihan, dibandingkan kelas negatif/latar?",
        "upload": "Rekaman WAV bawah air",
        "upload_help": "Hanya WAV. Rekaman yang sangat panjang membutuhkan waktu pemindaian lebih lama pada hosting gratis.",
        "threshold": "Ambang deteksi",
        "models": "Varian detektor",
        "run": "JALANKAN DETEKSI LUMBA-LUMBA",
        "need_audio": "Unggah rekaman WAV untuk memulai.",
        "short_audio": "Durasi rekaman minimal 2 detik.",
        "invalid_audio": "Berkas tidak dapat dibaca sebagai audio. Pastikan berkas merupakan WAV yang valid.",
        "analyzing": "Membuat jendela spektrogram dan menjalankan detektor yang dipilih…",
        "analysis_complete": "Pemindaian deteksi selesai.",
        "no_detectors": "Berkas model detektor belum tersedia. Tambahkan tiga berkas .keras detektor ke folder models/ pada repositori.",
        "results": "LAPORAN DETEKSI",
        "duration": "Durasi rekaman",
        "windows": "Jendela dipindai",
        "detectors_run": "Detektor dijalankan",
        "event_counts": "Kandidat kejadian per detektor",
        "events": "kandidat kejadian",
        "no_events": "Tidak ada jendela yang melewati ambang terpilih. Ini tidak membuktikan bahwa vokalisasi lumba-lumba tidak ada.",
        "some_events": "Kandidat jendela vokalisasi lumba-lumba terdeteksi. Tinjau klip sebelum menarik kesimpulan.",
        "timeline": "Linimasa deteksi per jendela",
        "details": "Hasil per jendela dan ekspor",
        "start": "Mulai (dtk)",
        "end": "Selesai (dtk)",
        "download_csv": "Unduh hasil jendela (CSV)",
        "download_json": "Unduh laporan analisis (JSON)",
        "inspect_event": "Pilih kandidat kejadian untuk didengarkan",
        "download_clip": "Unduh WAV kandidat kejadian",
        "species_title": "KLASIFIKASI SPESIES EKSPERIMENTAL",
        "species_body": "Tahap kedua opsional ini mengklasifikasikan kandidat kejadian terpilih ke dalam salah satu dari tiga label pelatihan: Delphinus delphis, Sousa plumbea, dan Tursiops aduncus.",
        "species_event": "Kandidat kejadian untuk diklasifikasikan",
        "species_run": "JALANKAN MODEL SPESIES EKSPERIMENTAL",
        "species_no_events": "Jalankan deteksi terlebih dahulu dan pilih kandidat kejadian. Klasifikasi spesies hanya tersedia untuk klip kejadian yang terdeteksi.",
        "species_result": "Prediksi model",
        "species_missing": "Berkas model spesies eksperimental tidak ditemukan.",
        "status": "STATUS BERKAS MODEL",
        "detector_status": "Detektor",
        "species_status": "Model spesies eksperimental",
        "available": "SIAP",
        "missing": "HILANG",
        "history_title": "Log deteksi",
        "history_empty": "Belum ada pemindaian pada sesi ini.",
        "clear_history": "Hapus log sesi",
        "research_title": "Meja riset: eksperimen SNR terkontrol",
        "research_body": "Bandingkan tiga varian detektor pada eksperimen subset demonstrasi. Nilai bersifat awal dan bukan jaminan kinerja pada rekaman lapangan yang diunggah.",
        "metric": "Metrik benchmark",
        "snr_note": "Hasil SNR berasal dari subset demonstrasi dan tidak dapat dibandingkan langsung dengan benchmark uji independen penuh pada paper. Nilai identik berulang pada beberapa tingkat SNR perlu diaudit sebelum membuat klaim kuat.",
        "guide_title": "Catatan lapangan",
        "what_detector": "Apa fungsi detektor utama?",
        "what_detector_p": "Detektor melakukan deteksi biner: audio yang menyerupai vokalisasi lumba-lumba dibandingkan kelas negatif/latar yang direpresentasikan selama pelatihan. Model ini tidak mengidentifikasi semua jenis suara laut dan tidak menentukan spesies dari setiap jendela positif.",
        "what_species": "Apa fungsi tahap spesies opsional?",
        "what_species_p": "Ini adalah model terpisah dengan tiga label keluaran: Delphinus delphis (lumba-lumba biasa), Sousa plumbea (lumba-lumba bungkuk Samudra Hindia), dan Tursiops aduncus (lumba-lumba hidung botol Indo-Pasifik).",
        "what_snr": "Apa itu SNR?",
        "what_snr_p": "Signal-to-noise ratio membandingkan daya sinyal dengan daya derau latar. SNR yang lebih rendah berarti sinyal semakin tertutup derau. Kurikulum SNR merupakan strategi pelatihan detektor, bukan kategori suara tersendiri.",
        "limitations": "Keterbatasan ilmiah",
        "limitations_p": "Aplikasi menggunakan sliding window reguler saat inferensi, sedangkan notebook pelatihan menggunakan jendela yang terkait anotasi. Rekaman unggahan tidak otomatis memiliki label ground truth atau SNR terukur. Rekaman laut nyata dapat berbeda dalam respons hidrofon, propagasi, derau lingkungan, distribusi spesies, dan kondisi perekaman. Perlakukan deteksi sebagai petunjuk penyaringan yang memerlukan tinjauan ahli.",
        "footer": "SONARA · Prototipe riset vokalisasi lumba-lumba turunan CetusID · Bukan instrumen pemantauan ekologi tervalidasi",
        "about": "TENTANG SONARA",
        "about_body": "Deteksi khusus vokalisasi lumba-lumba, dengan tahap spesies tiga kelas yang terpisah dan masih eksperimental.",
        "sample_rate": "Frekuensi sampling target",
        "window": "Panjang jendela",
        "hop": "Langkah jendela",
        "source": "Rekaman sumber",
        "original_rate": "Frekuensi sampling awal",
        "all_models": "Pilih semua detektor yang tersedia",
        "model_name": "Detektor",
        "created": "Waktu pemindaian",
        "clear_notice": "Rekaman baru dipilih. Hasil sebelumnya telah dihapus.",
    },
}

MODEL_NAMES = {
    "fixed": {"English": "Fixed-mix reference", "Bahasa Indonesia": "Referensi fixed-mix"},
    "random": {"English": "Random-SNR control", "Bahasa Indonesia": "Kontrol random-SNR"},
    "curriculum": {"English": "SNR curriculum", "Bahasa Indonesia": "Kurikulum SNR"},
}


def tr(key: str) -> str:
    lang = st.session_state.get("language", "English")
    return COPY[lang][key]


def detector_status() -> dict[str, bool]:
    return {key: (MODEL_DIR / filename).is_file() for key, filename in DETECTOR_FILES.items()}


def render_status(status: dict[str, bool], species_exists: bool) -> None:
    ready = sum(status.values())
    st.markdown(f"**{tr('status')}**")
    st.progress(ready / max(len(status), 1), text=f"{ready}/{len(status)} {tr('detector_status').lower()}")
    for key, exists in status.items():
        label = MODEL_NAMES[key][st.session_state.language]
        st.markdown(
            f"<div class='status-row'><span>{label}</span><span class='status-pill {'ok' if exists else 'off'}'>{tr('available') if exists else tr('missing')}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div class='status-row'><span>{tr('species_status')}</span><span class='status-pill {'ok' if species_exists else 'off'}'>{tr('available') if species_exists else tr('missing')}</span></div>",
        unsafe_allow_html=True,
    )


def load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV, convert to mono, and resample to the model's 96 kHz input rate."""
    try:
        raw, original_sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
        if raw.size == 0 or original_sr <= 0:
            raise ValueError("Empty or invalid WAV audio")
        mono = np.mean(raw, axis=1).astype(np.float32)
        if original_sr != TARGET_SR:
            mono = librosa.resample(mono, orig_sr=original_sr, target_sr=TARGET_SR)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        return mono, int(original_sr)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


@st.cache_resource(show_spinner=False)
def cached_model_bundle(model_dir: str, requested: tuple[str, ...]):
    """Load selected Keras models once per Streamlit worker instead of on every rerun."""
    return load_model_bundle(Path(model_dir), list(requested))


def plot_timeline(window_df: pd.DataFrame):
    colors = {"fixed": "#6be3cf", "random": "#f4d477", "curriculum": "#83cfff"}
    fig = go.Figure()
    for model_key, group in window_df.groupby("model_key", sort=False):
        fig.add_trace(go.Scatter(
            x=group["start_s"],
            y=[MODEL_NAMES[model_key][st.session_state.language]] * len(group),
            mode="markers",
            marker={"size": 9, "color": colors.get(model_key, "#edf8f2"), "opacity": .88,
                     "line": {"width": 1, "color": "#07131f"}},
            customdata=group["end_s"],
            hovertemplate="Start: %{x:.2f}s<br>End: %{customdata:.2f}s<extra></extra>",
            name=MODEL_NAMES[model_key][st.session_state.language],
        ))
    fig.update_layout(
        height=max(210, 78 * max(1, window_df["model_key"].nunique())),
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,19,31,.4)",
        font={"color": "#edf8f2", "family": "monospace", "size": 12},
        xaxis_title="Time (seconds)",
        yaxis_title="",
        showlegend=False,
        xaxis={"gridcolor": "rgba(160,190,205,.16)", "zerolinecolor": "rgba(160,190,205,.2)"},
        yaxis={"gridcolor": "rgba(160,190,205,.1)", "autorange": "reversed"},
    )
    return fig


def flatten_window_results(result: dict) -> pd.DataFrame:
    rows = []
    for model_key, windows in result["windows"].items():
        for row in windows:
            rows.append({
                "model_key": model_key,
                "detector": MODEL_NAMES[model_key][st.session_state.language],
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "candidate": row["detected"],
            })
    return pd.DataFrame(rows)


def show_results(result: dict, audio: np.ndarray, threshold: float) -> None:
    st.markdown(f"### {tr('results')}")
    st.caption(f"{tr('threshold')}: {threshold:.2f}")
    m1, m2, m3 = st.columns(3)
    m1.metric(tr("duration"), f"{result['duration_s']:.1f} s")
    m2.metric(tr("windows"), f"{result['window_count']:,}")
    m3.metric(tr("detectors_run"), len(result["events"]))

    rows_by_detector = []
    for key, events in result["events"].items():
        rows_by_detector.append({
            tr("model_name"): MODEL_NAMES[key][st.session_state.language],
            tr("events"): len(events),
        })
    st.markdown(f"#### {tr('event_counts')}")
    st.dataframe(pd.DataFrame(rows_by_detector), use_container_width=True, hide_index=True)

    total_events = sum(len(events) for events in result["events"].values())
    if total_events:
        st.success(tr("some_events"))
    else:
        st.info(tr("no_events"))

    window_df = flatten_window_results(result)
    if not window_df.empty:
        st.markdown(f"#### {tr('timeline')}")
        st.plotly_chart(plot_timeline(window_df), use_container_width=True, config={"displayModeBar": False})

    model_keys = list(result["events"].keys())
    tabs = st.tabs([MODEL_NAMES[key][st.session_state.language] for key in model_keys])
    for tab, key in zip(tabs, model_keys):
        with tab:
            events = result["events"][key]
            if not events:
                st.caption(tr("no_events"))
                continue
            event_df = pd.DataFrame(events).rename(columns={
                "start_s": tr("start"),
                "end_s": tr("end"),
                "peak_probability": tr("peak"),
                "window_count": "Windows",
            })
            st.dataframe(event_df, use_container_width=True, hide_index=True)
            selected_event = st.selectbox(
                tr("inspect_event"),
                options=list(range(len(events))),
                format_func=lambda idx, ev=events: (
                    f"EVENT {idx + 1:02d}  |  {ev[idx]['start_s']:.2f}–{ev[idx]['end_s']:.2f} s"
                ),
                key=f"inspect_{key}",
            )
            event = events[selected_event]
            clip_start = max(0, int(event["start_s"] * TARGET_SR))
            clip_end = min(len(audio), int(event["end_s"] * TARGET_SR))
            if clip_end > clip_start:
                clip = audio[clip_start:clip_end]
                st.audio(clip, sample_rate=TARGET_SR)
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, clip, TARGET_SR, format="WAV")
                st.download_button(
                    tr("download_clip"),
                    data=wav_buffer.getvalue(),
                    file_name=f"sonara_{key}_candidate_{selected_event + 1}.wav",
                    mime="audio/wav",
                    key=f"download_{key}_{selected_event}",
                )

    with st.expander(tr("details")):
        st.dataframe(
            window_df.drop(columns=["model_key", "score"], errors="ignore").rename(columns={
                "detector": tr("model_name"),
                "start_s": tr("start"),
                "end_s": tr("end"),
                "candidate": "Candidate",
            }),
            use_container_width=True,
            hide_index=True,
        )
        csv_data = window_df.drop(columns=["model_key", "score"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(tr("download_csv"), csv_data, "sonara_detection_windows.csv", "text/csv")
        payload = {
            "project": "SONARA",
            "task": "candidate_dolphin_vocalisation_detection",
            "model_lineage": "CetusID research lineage; see README for paper and dataset attribution.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "duration_s": result["duration_s"],
            "window_count": result["window_count"],
            "events": result["events"],
            "windows": result["windows"],
            "scientific_note": "A positive prediction is a candidate dolphin-vocalisation detection, not confirmation of dolphin presence or species.",
        }
        st.download_button(
            tr("download_json"),
            json.dumps(payload, indent=2).encode("utf-8"),
            "sonara_detection_report.json",
            "application/json",
        )


def render_pixel_dolphin() -> None:
    # Small, self-contained pixel-art dolphin/sonar illustration; no external image dependency.
    st.markdown(
        """
        <div class="hero-visual" aria-label="Pixel-art dolphin in a sonar display">
          <svg viewBox="0 0 320 210" role="img" aria-hidden="true">
            <rect x="1" y="1" width="318" height="208" fill="#071a28" stroke="#2d6070" stroke-width="2"/>
            <g stroke="#1c5060" stroke-width="1" fill="none" shape-rendering="crispEdges">
              <path d="M20 36H300 M20 72H300 M20 108H300 M20 144H300 M20 180H300" stroke-dasharray="4 7"/>
              <path d="M56 16V194 M104 16V194 M152 16V194 M200 16V194 M248 16V194" stroke-dasharray="4 7"/>
              <rect x="76" y="27" width="168" height="150"/>
              <rect x="96" y="45" width="128" height="114"/>
              <rect x="116" y="63" width="88" height="78"/>
            </g>
            <g fill="#6be3cf" shape-rendering="crispEdges">
              <rect x="91" y="90" width="22" height="18"/>
              <rect x="113" y="72" width="22" height="36"/>
              <rect x="135" y="63" width="66" height="45"/>
              <rect x="201" y="72" width="25" height="27"/>
              <rect x="226" y="81" width="18" height="18"/>
              <rect x="244" y="87" width="16" height="9"/>
              <rect x="153" y="54" width="17" height="9"/>
              <rect x="161" y="45" width="9" height="9"/>
              <rect x="148" y="108" width="34" height="9" fill="#3faeaa"/>
              <rect x="169" y="117" width="27" height="9" fill="#3faeaa"/>
              <rect x="108" y="108" width="27" height="9"/>
              <rect x="89" y="117" width="18" height="9"/>
              <rect x="73" y="108" width="16" height="9"/>
              <rect x="66" y="99" width="9" height="9"/>
            </g>
            <g fill="#071a28" shape-rendering="crispEdges">
              <rect x="214" y="81" width="7" height="7"/>
              <rect x="244" y="87" width="5" height="5"/>
            </g>
            <g fill="#f4d477" shape-rendering="crispEdges">
              <rect x="231" y="81" width="7" height="7"/>
              <rect x="48" y="47" width="5" height="5"/>
              <rect x="269" y="148" width="5" height="5"/>
              <rect x="42" y="158" width="5" height="5"/>
            </g>
            <g fill="#83cfff" font-family="monospace" font-size="9">
              <text x="18" y="18">SONAR / 96 kHz</text>
              <text x="222" y="198">DOLPHIN</text>
            </g>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left, right = st.columns([1.6, 1], vertical_alignment="center")
    with left:
        st.markdown(
            f"<div class='topline'>PASSIVE ACOUSTIC MONITORING / SONARA RESEARCH CONSOLE</div>"
            f"<div class='brand'>SONARA<span class='brand-mark'> ◈</span></div>"
            f"<div class='brand-sub'>{tr('subtitle')}</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div class='header-readout'><div class='small'>{tr('header_note')}</div>"
            f"<div class='big'>96,000 Hz</div><div class='small'>2.0 s WINDOW / 1.0 s HOP</div></div>",
            unsafe_allow_html=True,
        )


def _go_to_analysis() -> None:
    # Streamlit callbacks run before widgets are re-created on the next rerun.
    st.session_state.page = "analysis"


def render_overview() -> None:
    left, right = st.columns([1.35, .85], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            f"<div class='pixel-kicker'>{tr('hero_kicker')}</div>"
            f"<h1>{tr('hero_title')}</h1>"
            f"<p>{tr('hero_body')}</p>"
            f"<div class='pixel-chip'>{tr('hero_chip')}</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "▶  " + PAGE_LABELS[st.session_state.language]["analysis"],
            type="primary",
            use_container_width=True,
            on_click=_go_to_analysis,
        )
    with right:
        render_pixel_dolphin()

    st.markdown(f"<div class='section-kicker'>{tr('notice_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-strip'>{tr('notice')}</div>", unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric(tr("sample_rate"), "96 kHz")
    b.metric(tr("window"), "2.0 s")
    c.metric(tr("hop"), "1.0 s")

    st.markdown(f"<div class='section-kicker'>{tr('how')}</div>", unsafe_allow_html=True)
    step_items = [("01", tr("step1"), tr("step1p")), ("02", tr("step2"), tr("step2p")), ("03", tr("step3"), tr("step3p"))]
    step_html = "<div class='steps-grid'>" + "".join(
        f"<div class='step-card'><div class='step-index'>{number}</div><b>{title}</b><p>{body}</p></div>"
        for number, title, body in step_items
    ) + "</div>"
    st.markdown(step_html, unsafe_allow_html=True)


def render_species_stage(result: dict | None, audio: np.ndarray | None, species_exists: bool) -> None:
    st.divider()
    st.markdown(f"<div class='section-kicker'>{tr('species_title')}</div>", unsafe_allow_html=True)
    st.write(tr("species_body"))
    if not species_exists:
        st.warning(tr("species_missing"))
        return
    if result is None or audio is None:
        st.info(tr("species_no_events"))
        return

    candidates = []
    for model_key, events in result["events"].items():
        for event_index, event in enumerate(events):
            candidates.append((model_key, event_index, event))
    if not candidates:
        st.info(tr("species_no_events"))
        return

    choice = st.selectbox(
        tr("species_event"),
        options=list(range(len(candidates))),
        format_func=lambda i: (
            f"{MODEL_NAMES[candidates[i][0]][st.session_state.language]} · "
            f"{candidates[i][2]['start_s']:.2f}–{candidates[i][2]['end_s']:.2f} s"
        ),
        key="species_candidate_choice",
    )
    model_key, event_index, event = candidates[choice]
    if st.button(tr("species_run"), key="run_species_model"):
        start = max(0, int(event["start_s"] * TARGET_SR))
        end = min(len(audio), int(event["end_s"] * TARGET_SR))
        event_audio = audio[start:end]
        if len(event_audio) < int(WINDOW_SECONDS * TARGET_SR):
            st.error("Selected candidate is shorter than the model's 2-second input window.")
            return
        try:
            with st.spinner("Classifying selected candidate event…"):
                species_model = cached_model_bundle(str(MODEL_DIR), ("species",))["species"]
                prediction = make_species_prediction(
                    event_audio,
                    species_model,
                    metadata_path=MODEL_DIR / "experiment_metadata.json",
                )
            st.metric(tr("species_result"), prediction["label"].replace("_", " "))
        except Exception as exc:
            st.error(f"Species inference failed: {exc}")


def render_analysis(status: dict[str, bool], species_exists: bool) -> None:
    st.markdown(f"## {tr('analysis_title')}")
    st.write(tr("analysis_body"))
    available = [key for key, exists in status.items() if exists]
    uploaded = st.file_uploader(tr("upload"), type=["wav"], help=tr("upload_help"), key="dolphin_wav_upload")
    threshold = st.slider(tr("threshold"), 0.05, 0.95, 0.50, 0.05, key="detection_threshold")
    selected = st.multiselect(
        tr("models"),
        options=available,
        default=available,
        format_func=lambda key: MODEL_NAMES[key][st.session_state.language],
        key="selected_detectors",
        help=tr("all_models"),
    )

    current_result = None
    current_audio = None
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        upload_token = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("active_upload_token") != upload_token:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_audio", None)
            st.session_state.pop("species_prediction", None)
            st.session_state.active_upload_token = upload_token
        try:
            audio, original_sr = load_audio(file_bytes)
            if len(audio) / TARGET_SR < WINDOW_SECONDS:
                st.error(tr("short_audio"))
            else:
                st.caption(
                    f"{tr('source')}: {uploaded.name} · {tr('duration')}: {len(audio) / TARGET_SR:.2f} s · "
                    f"{tr('original_rate')}: {original_sr:,} Hz → {TARGET_SR:,} Hz"
                )
                st.audio(file_bytes, format="audio/wav")
                if st.button(tr("run"), type="primary", use_container_width=True, disabled=(not selected)):
                    try:
                        with st.spinner(tr("analyzing")):
                            models = cached_model_bundle(str(MODEL_DIR), tuple(selected))
                            scan_result = analyze_recording(audio, models, threshold=threshold)
                        st.session_state.last_result = scan_result
                        st.session_state.last_audio = audio
                        st.session_state.last_threshold = float(threshold)
                        st.session_state.history.insert(0, {
                            "file": uploaded.name,
                            tr("created"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                            tr("duration"): round(scan_result["duration_s"], 2),
                            tr("detectors_run"): len(scan_result["events"]),
                            tr("events"): sum(len(v) for v in scan_result["events"].values()),
                        })
                        st.success(tr("analysis_complete"))
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

                if st.session_state.get("active_upload_token") == upload_token:
                    current_result = st.session_state.get("last_result")
                    current_audio = st.session_state.get("last_audio")
                    if current_result is not None and current_audio is not None:
                        show_results(
                            current_result,
                            current_audio,
                            st.session_state.get("last_threshold", threshold),
                        )
        except Exception:
            st.error(tr("invalid_audio"))
    else:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_audio", None)
        st.session_state.pop("active_upload_token", None)
        st.info(tr("need_audio"))

    if not available:
        st.warning(tr("no_detectors"))
    elif not selected:
        st.info("Select at least one available detector to run the analysis.")

    if current_result is None and st.session_state.get("active_upload_token"):
        current_result = st.session_state.get("last_result")
        current_audio = st.session_state.get("last_audio")
    render_species_stage(current_result, current_audio, species_exists)


def render_research() -> None:
    st.markdown(f"## {tr('research_title')}")
    st.write(tr("research_body"))
    benchmark_path = MODEL_DIR / "snr_robustness_results.csv"
    if benchmark_path.is_file():
        benchmark = pd.read_csv(benchmark_path)
        metric = st.selectbox(
            tr("metric"),
            ["accuracy", "precision", "recall", "macro_f1", "roc_auc"],
            format_func=lambda value: {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "macro_f1": "Macro-F1",
                "roc_auc": "ROC-AUC",
            }[value],
        )
        pivot = benchmark.pivot(index="condition", columns="model", values=metric)
        order = ["20 dB", "15 dB", "10 dB", "5 dB", "0 dB", "-5 dB"]
        pivot = pivot.reindex(order)
        pivot.columns = [
            {
                "Fixed-mix reference": MODEL_NAMES["fixed"][st.session_state.language],
                "Random-SNR control": MODEL_NAMES["random"][st.session_state.language],
                "SNR curriculum": MODEL_NAMES["curriculum"][st.session_state.language],
            }.get(column, column)
            for column in pivot.columns
        ]
        st.dataframe((pivot * 100).round(2).astype(str) + "%", use_container_width=True)
    else:
        st.warning("Benchmark CSV not found / Berkas benchmark CSV tidak ditemukan.")
    st.markdown(f"<div class='info-strip warning-strip'>{tr('snr_note')}</div>", unsafe_allow_html=True)


def render_guide() -> None:
    st.markdown(f"## {tr('guide_title')}")
    guide_items = [
        ("what_detector", "what_detector_p"),
        ("what_species", "what_species_p"),
        ("what_snr", "what_snr_p"),
        ("limitations", "limitations_p"),
    ]
    for heading_key, body_key in guide_items:
        with st.expander(tr(heading_key)):
            st.write(tr(body_key))


def main() -> None:
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "page" not in st.session_state:
        st.session_state.page = "overview"
    if "history" not in st.session_state:
        st.session_state.history = []

    status = detector_status()
    species_exists = (MODEL_DIR / SPECIES_MODEL_FILE).is_file()

    with st.sidebar:
        st.markdown("# ◈ SONARA")
        st.caption(tr("mission"))
        st.radio(tr("language"), LANGUAGES, key="language", label_visibility="visible")
        st.divider()
        st.markdown("**MISSION MENU**")
        page = st.radio(
            "Navigation",
            options=list(PAGE_LABELS[st.session_state.language].keys()),
            format_func=lambda key: PAGE_LABELS[st.session_state.language][key],
            key="page",
            label_visibility="collapsed",
        )
        st.divider()
        render_status(status, species_exists)
        st.divider()
        st.markdown(f"**{tr('about')}**")
        st.caption(tr("about_body"))

    render_header()
    if page == "overview":
        render_overview()
    elif page == "analysis":
        render_analysis(status, species_exists)
    elif page == "history":
        st.markdown(f"## {tr('history_title')}")
        if not st.session_state.history:
            st.info(tr("history_empty"))
        else:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
            if st.button(tr("clear_history")):
                st.session_state.history = []
                st.rerun()
    elif page == "research":
        render_research()
    elif page == "guide":
        render_guide()

    st.markdown(f"<div class='footer-line'>{tr('footer')}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
