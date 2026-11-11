from __future__ import annotations

import io
import json
import os
import zipfile
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
    merge_candidate_events,
    make_species_prediction,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("CETUSID_MODEL_DIR", APP_DIR / "models"))

st.set_page_config(
    page_title="SONARA | Ocean Sound Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --abyss: #071522;
  --deep: #0b2030;
  --panel: #102b3c;
  --panel-hi: #15384a;
  --line: rgba(132, 190, 202, .22);
  --text: #e4f0f2;
  --muted: #91aeb9;
  --teal: #69d2ca;
  --amber: #e9c77d;
  --coral: #f39a83;
}
.stApp {
  background:
    radial-gradient(ellipse at 78% 0%, rgba(30, 102, 123, .18), transparent 42%),
    linear-gradient(180deg, #071522 0%, #0a1d2c 48%, #081725 100%);
  color: var(--text);
}
[data-testid="stHeader"] { background: rgba(7, 21, 34, .86); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1c2a 0%, #0d2637 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
.block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 2.5rem; }
h1, h2, h3 { color: var(--text) !important; letter-spacing: -.025em; }
p, li, label, .stMarkdown { color: var(--text); }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
hr { border-color: var(--line); }
.topline {
  color: #80a7b5; font: 600 .68rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .15em; margin-bottom: .7rem;
}
.topline-dot {
  display: inline-block; width: 7px; height: 7px; margin-right: 8px;
  border-radius: 50%; background: var(--teal); box-shadow: 0 0 10px var(--teal);
}
.topline-sep { color: #466c7d; padding: 0 .45rem; }
.brand {
  font: 800 clamp(2rem, 4vw, 3.25rem)/.95 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .06em; color: #eaf7f6;
}
.brand-mark { color: var(--teal); }
.brand-sub {
  color: #8db6c0; font: 600 .72rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .26em; margin-top: .55rem;
}
.depth-readout {
  text-align: right; border: 1px solid var(--line); border-radius: 3px;
  background: rgba(12, 39, 55, .55); padding: .65rem .8rem; margin-top: .2rem;
}
.depth-readout p { margin: 0; }
.depth-value { color: var(--teal); font: 700 1.05rem ui-monospace, monospace; letter-spacing: .18em; }
.hero-panel {
  position: relative; overflow: hidden; min-height: 370px; margin: 1.4rem 0 1rem;
  padding: clamp(1.5rem, 4vw, 3.4rem); border: 1px solid rgba(105,210,202,.28);
  border-radius: 5px;
  background:
    linear-gradient(105deg, rgba(8,27,42,.97) 0%, rgba(11,42,58,.94) 52%, rgba(13,55,69,.78) 100%),
    repeating-linear-gradient(0deg, transparent 0 31px, rgba(116,185,194,.045) 32px);
}
.hero-panel:after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(90deg, transparent 0 79px, rgba(116,185,194,.035) 80px);
}
.hero-kicker {
  color: var(--amber); font: 600 .7rem ui-monospace, monospace;
  letter-spacing: .18em; position: relative; z-index: 2;
}
.hero-kicker span { color: var(--teal); padding: 0 .35rem; }
.hero-panel h1 {
  max-width: 620px; margin: 1.1rem 0 .8rem; position: relative; z-index: 2;
  font-size: clamp(2rem, 4vw, 3.5rem) !important; line-height: 1.06;
}
.hero-panel p {
  max-width: 580px; color: #afc7ce; line-height: 1.7;
  position: relative; z-index: 2;
}
.sonar-wrap {
  position: absolute; width: 320px; height: 320px; right: 6%; top: 25px;
  border: 1px solid rgba(105,210,202,.22); border-radius: 50%;
  background: radial-gradient(circle, rgba(54,144,155,.15), rgba(8,30,45,.04) 68%);
}
.sonar-ring {
  position: absolute; border: 1px solid rgba(105,210,202,.27); border-radius: 50%;
  inset: 15%; 
}
.ring-b { inset: 30%; border-color: rgba(105,210,202,.22); }
.ring-c { inset: 45%; border-color: rgba(105,210,202,.2); }
.sonar-core {
  position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
  color: var(--teal); font-size: 2.1rem; text-shadow: 0 0 18px rgba(105,210,202,.8);
}
.sonar-sweep {
  position: absolute; width: 50%; height: 50%; left: 50%; top: 50%;
  transform-origin: 0 0; border-radius: 0 100% 0 0;
  background: linear-gradient(45deg, rgba(105,210,202,.17), transparent 68%);
  animation: sweep 8s linear infinite;
}
@keyframes sweep { to { transform: rotate(360deg); } }
.sonar-label {
  position: absolute; color: #82b7bf; font: 600 .58rem ui-monospace, monospace;
  letter-spacing: .12em; white-space: nowrap;
}
.label-a { right: -10px; top: 48%; }
.label-b { left: -24px; bottom: 22%; }
.steps-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .8rem; margin: 1rem 0 2rem; }
.step-card {
  padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 4px;
  background: rgba(14,42,58,.72);
}
.step-card span {
  color: var(--teal); font: 700 .72rem ui-monospace, monospace; display: block; margin-bottom: .6rem;
}
.step-card b { color: #e6f1f2; font: 700 .7rem ui-monospace, monospace; letter-spacing: .08em; }
.step-card p { color: var(--muted); font-size: .88rem; margin: .5rem 0 0; }
.status-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: .5rem; padding: .42rem 0; border-bottom: 1px solid rgba(132,190,202,.1);
  color: #b9cdd2; font-size: .82rem;
}
.status-pill {
  font: 600 .6rem ui-monospace, monospace; letter-spacing: .06em;
  padding: .18rem .42rem; border: 1px solid; border-radius: 2px;
}
.status-pill.ok { color: var(--teal); border-color: rgba(105,210,202,.4); }
.status-pill.off { color: var(--coral); border-color: rgba(243,154,131,.4); }
.stButton > button, .stDownloadButton > button {
  border-radius: 3px; border: 1px solid rgba(105,210,202,.55);
  background: #17485a; color: #edfbfa; font: 700 .72rem ui-monospace, monospace;
  letter-spacing: .06em; min-height: 2.65rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--teal); background: #1d5a68; color: white;
}
.stButton > button[kind="primary"] {
  background: #277d83; border-color: #6dd4ca; color: #fff;
}
[data-testid="stMetric"] {
  background: rgba(14,42,58,.72); border: 1px solid var(--line);
  border-radius: 4px; padding: .85rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--teal) !important; }
[data-testid="stFileUploader"] {
  border: 1px dashed rgba(105,210,202,.42); border-radius: 4px;
  background: rgba(13,43,57,.35); padding: .35rem;
}
[data-testid="stAlert"] { border-radius: 3px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] {
  background: rgba(14,42,58,.72); border: 1px solid var(--line);
  border-radius: 3px 3px 0 0; color: #bdd2d7; padding: .5rem .85rem;
}
.stTabs [aria-selected="true"] { border-color: var(--teal) !important; color: var(--teal) !important; }
@media (max-width: 760px) {
  .sonar-wrap { opacity: .22; right: -110px; top: 70px; }
  .hero-panel { min-height: 390px; }
  .steps-grid { grid-template-columns: 1fr; }
  .depth-readout { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .sonar-sweep { animation: none; }
}
</style>
""", unsafe_allow_html=True)

COPY = {
    "English": {
        "nav": "NAVIGATION",
        "home": "Mission overview",
        "explore": "Acoustic analysis",
        "discoveries": "Discovery log",
        "lab": "Research laboratory",
        "guide": "Field guide",
        "language": "LANGUAGE / BAHASA",
        "title": "SONARA",
        "subtitle": "OCEAN SOUND EXPLORER",
        "mission": "Passive acoustic exploration",
        "home_h": "Listen beneath the surface.",
        "home_p": "Explore underwater recordings with an AI-assisted acoustic detector. Upload a WAV file, inspect candidate vocalisations, and review how the models respond.",
        "start": "START ACOUSTIC ANALYSIS",
        "how": "HOW IT WORKS",
        "status": "SYSTEM STATUS",
        "ready": "Console ready",
        "models": "Detector models",
        "models_missing": "Model files not found",
        "sample_rate": "Target sample rate",
        "window": "Analysis window",
        "hop": "Window step",
        "notice_title": "Scientific note",
        "notice": "A detection is a model-generated candidate, not confirmation of dolphin presence or species. Performance on uploaded field recordings has not been established.",
        "explore_h": "Acoustic analysis",
        "explore_p": "Upload an underwater WAV recording. The application resamples it to the training sample rate and scans overlapping two-second windows.",
        "upload": "Choose a WAV recording",
        "upload_help": "WAV audio only. Longer recordings take more time to analyze.",
        "threshold": "Detection threshold",
        "run": "RUN ANALYSIS",
        "need_models": "Model artifacts are missing. Add the three detector files to the `models/` directory. See README.md.",
        "need_audio": "Upload a WAV recording to begin.",
        "analyzing": "Preparing acoustic windows…",
        "results": "ANALYSIS REPORT",
        "duration": "Recording duration",
        "windows": "Windows analyzed",
        "candidate_events": "Candidate events",
        "no_events": "No candidate events crossed the selected threshold. This does not prove that dolphin vocalisations are absent.",
        "events_found": "Candidate acoustic events detected",
        "model": "Model",
        "time_start": "Start (s)",
        "time_end": "End (s)",
        "confidence": "Peak probability",
        "event_count": "Events",
        "timeline": "Detection timeline",
        "details": "Window-level results",
        "download_csv": "Download CSV",
        "download_json": "Download JSON",
        "species_h": "Experimental species stage",
        "species_warning": "Experimental only — the demonstration-subset validation was limited (33.3% reported in this run). Do not use this output as a biological identification.",
        "species_run": "RUN EXPERIMENTAL SPECIES CLASSIFIER",
        "species_missing": "Species classifier file not found.",
        "species_result": "Highest model score",
        "history_h": "Discovery log",
        "history_empty": "No analyses in this session yet. Your results will appear here after an analysis.",
        "lab_h": "Research laboratory",
        "lab_p": "Compare the three detector variants under controlled SNR conditions. These benchmark values come from the notebook's demonstration-subset run and are not a field-performance guarantee.",
        "fixed": "Fixed-mix reference",
        "random": "Random-SNR control",
        "curriculum": "SNR curriculum",
        "snr_note": "Controlled SNR results are not directly comparable with the paper's full independent test benchmarks.",
        "field_h": "Field guide",
        "field_p": "A short guide to interpreting passive acoustic monitoring outputs.",
        "what_detection": "What is detection?",
        "what_detection_p": "The detector examines short acoustic windows and estimates whether each window resembles the positive dolphin-vocalisation class learned during training.",
        "what_snr": "What is SNR?",
        "what_snr_p": "Signal-to-noise ratio compares signal power with background-noise power. Lower SNR means the signal is more heavily masked by noise.",
        "what_spectrogram": "What is a spectrogram?",
        "what_spectrogram_p": "A spectrogram shows how acoustic energy is distributed across frequency over time. It can help reveal whistles, clicks, and other structured sounds.",
        "limitations": "Limitations",
        "limitations_p": "This release uses a demonstration subset. Recording conditions, hydrophone response, species distribution, and real ocean noise can differ substantially from training data. Treat all predictions as screening cues that require expert review.",
        "footer": "SONARA · Research prototype · Not a validated biodiversity-monitoring instrument",
        "about": "ABOUT THIS CONSOLE",
        "about_p": "An acoustic research prototype using three SNR-training strategies for candidate dolphin-vocalisation detection.",
        "model_file": "Model artifact",
        "available": "Available",
        "missing": "Missing",
        "listen": "Listen to recording",
        "select_model": "Models to run",
        "all_models": "Run all available detectors",
        "analysis_complete": "Analysis complete",
        "invalid_audio": "The uploaded file could not be decoded as audio.",
        "short_audio": "Recording must be at least 2 seconds long.",
        "source": "Source recording",
        "peak": "Peak amplitude",
        "no_model": "No detector models are available.",
    },
    "Bahasa Indonesia": {
        "nav": "NAVIGASI",
        "home": "Ringkasan misi",
        "explore": "Analisis akustik",
        "discoveries": "Log temuan",
        "lab": "Laboratorium riset",
        "guide": "Panduan lapangan",
        "language": "BAHASA / LANGUAGE",
        "title": "SONARA",
        "subtitle": "PENJELAJAH SUARA LAUT",
        "mission": "Eksplorasi akustik pasif",
        "home_h": "Dengarkan suara di bawah permukaan.",
        "home_p": "Jelajahi rekaman bawah laut dengan detektor akustik berbantuan AI. Unggah berkas WAV, periksa kandidat vokalisasi, dan tinjau respons model.",
        "start": "MULAI ANALISIS AKUSTIK",
        "how": "CARA KERJA",
        "status": "STATUS SISTEM",
        "ready": "Konsol siap",
        "models": "Model detektor",
        "models_missing": "Berkas model tidak ditemukan",
        "sample_rate": "Frekuensi sampling target",
        "window": "Jendela analisis",
        "hop": "Langkah jendela",
        "notice_title": "Catatan ilmiah",
        "notice": "Deteksi adalah kandidat yang dihasilkan model, bukan konfirmasi keberadaan lumba-lumba atau spesies tertentu. Kinerja pada rekaman lapangan yang diunggah belum ditetapkan.",
        "explore_h": "Analisis akustik",
        "explore_p": "Unggah rekaman bawah laut berformat WAV. Aplikasi akan melakukan resampling ke frekuensi sampling pelatihan dan memindai jendela dua detik yang saling bertumpang tindih.",
        "upload": "Pilih rekaman WAV",
        "upload_help": "Hanya audio WAV. Rekaman yang lebih panjang membutuhkan waktu analisis lebih lama.",
        "threshold": "Ambang deteksi",
        "run": "JALANKAN ANALISIS",
        "need_models": "Artefak model belum tersedia. Tambahkan tiga berkas detektor ke direktori `models/`. Lihat README.md.",
        "need_audio": "Unggah rekaman WAV untuk memulai.",
        "analyzing": "Menyiapkan jendela akustik…",
        "results": "LAPORAN ANALISIS",
        "duration": "Durasi rekaman",
        "windows": "Jendela dianalisis",
        "candidate_events": "Kandidat kejadian",
        "no_events": "Tidak ada kandidat kejadian yang melewati ambang terpilih. Ini tidak membuktikan bahwa vokalisasi lumba-lumba tidak ada.",
        "events_found": "Kandidat kejadian akustik terdeteksi",
        "model": "Model",
        "time_start": "Mulai (dtk)",
        "time_end": "Selesai (dtk)",
        "confidence": "Probabilitas puncak",
        "event_count": "Kejadian",
        "timeline": "Linimasa deteksi",
        "details": "Hasil per jendela",
        "download_csv": "Unduh CSV",
        "download_json": "Unduh JSON",
        "species_h": "Tahap klasifikasi spesies eksperimental",
        "species_warning": "Hanya eksperimental — validasi pada subset demonstrasi terbatas (akurasi 33,3% dilaporkan pada eksperimen ini). Jangan gunakan keluaran ini sebagai identifikasi biologis.",
        "species_run": "JALANKAN KLASIFIKATOR SPESIES EKSPERIMENTAL",
        "species_missing": "Berkas klasifikator spesies tidak ditemukan.",
        "species_result": "Skor model tertinggi",
        "history_h": "Log temuan",
        "history_empty": "Belum ada analisis pada sesi ini. Hasil akan muncul di sini setelah analisis dilakukan.",
        "lab_h": "Laboratorium riset",
        "lab_p": "Bandingkan tiga varian detektor pada kondisi SNR terkontrol. Nilai benchmark ini berasal dari eksperimen subset demonstrasi di notebook dan bukan jaminan kinerja lapangan.",
        "fixed": "Referensi fixed-mix",
        "random": "Kontrol random-SNR",
        "curriculum": "Kurikulum SNR",
        "snr_note": "Hasil SNR terkontrol tidak dapat dibandingkan langsung dengan benchmark uji independen penuh pada paper.",
        "field_h": "Panduan lapangan",
        "field_p": "Panduan singkat untuk memahami keluaran pemantauan akustik pasif.",
        "what_detection": "Apa itu deteksi?",
        "what_detection_p": "Detektor memeriksa jendela akustik pendek dan memperkirakan apakah setiap jendela menyerupai kelas positif vokalisasi lumba-lumba yang dipelajari selama pelatihan.",
        "what_snr": "Apa itu SNR?",
        "what_snr_p": "Signal-to-noise ratio membandingkan daya sinyal dengan daya derau latar. SNR yang lebih rendah berarti sinyal semakin tertutup oleh derau.",
        "what_spectrogram": "Apa itu spektrogram?",
        "what_spectrogram_p": "Spektrogram menunjukkan distribusi energi akustik pada frekuensi terhadap waktu. Visualisasi ini dapat membantu memperlihatkan siulan, klik, dan suara terstruktur lainnya.",
        "limitations": "Keterbatasan",
        "limitations_p": "Rilis ini menggunakan subset demonstrasi. Kondisi perekaman, respons hidrofon, distribusi spesies, dan derau laut nyata dapat berbeda dari data pelatihan. Perlakukan semua prediksi sebagai petunjuk penyaringan yang memerlukan tinjauan ahli.",
        "footer": "SONARA · Prototipe riset · Bukan instrumen pemantauan biodiversitas tervalidasi",
        "about": "TENTANG KONSOL INI",
        "about_p": "Prototipe riset akustik yang menggunakan tiga strategi pelatihan SNR untuk mendeteksi kandidat vokalisasi lumba-lumba.",
        "model_file": "Artefak model",
        "available": "Tersedia",
        "missing": "Tidak tersedia",
        "listen": "Dengarkan rekaman",
        "select_model": "Model yang dijalankan",
        "all_models": "Jalankan semua detektor yang tersedia",
        "analysis_complete": "Analisis selesai",
        "invalid_audio": "Berkas yang diunggah tidak dapat dibaca sebagai audio.",
        "short_audio": "Durasi rekaman minimal 2 detik.",
        "source": "Rekaman sumber",
        "peak": "Amplitudo puncak",
        "no_model": "Tidak ada model detektor yang tersedia.",
    },
}


def tr(key: str) -> str:
    return COPY[st.session_state.get("language", "English")][key]


def model_status() -> dict[str, bool]:
    return {name: (MODEL_DIR / filename).exists() for name, filename in DETECTOR_FILES.items()}


def render_header():
    st.markdown(
        """
        <div class="topline">
          <span class="topline-dot"></span>
          PASSIVE ACOUSTIC MONITORING <span class="topline-sep">/</span>
          SONARA RESEARCH CONSOLE
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b = st.columns([3, 1])
    with a:
        st.markdown(f"<div class='brand'>{tr('title')}<span class='brand-mark'> ◈</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='brand-sub'>{tr('subtitle')}</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='depth-readout'>", unsafe_allow_html=True)
        st.caption("ACOUSTIC ARRAY  ·  96 kHz")
        st.markdown("<span class='depth-value'>—  —  —</span>", unsafe_allow_html=True)
        st.caption("SYSTEM / STANDBY")
        st.markdown("</div>", unsafe_allow_html=True)


def render_model_status(status: dict[str, bool]):
    ready = sum(status.values())
    st.markdown(f"**{tr('status')}**")
    st.progress(ready / len(status), text=f"{ready}/{len(status)} {tr('models').lower()}")
    for key, ok in status.items():
        label = {
            "fixed": tr("fixed"),
            "random": tr("random"),
            "curriculum": tr("curriculum"),
        }[key]
        st.markdown(
            f"<div class='status-row'><span>{label}</span><span class='status-pill {'ok' if ok else 'off'}'>{tr('available') if ok else tr('missing')}</span></div>",
            unsafe_allow_html=True,
        )


def load_audio(uploaded):
    try:
        raw, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32", always_2d=True)
        audio = np.mean(raw, axis=1).astype(np.float32)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return audio, sr
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def plot_timeline(df: pd.DataFrame):
    fig = go.Figure()
    colors = {
        "Fixed-mix reference": "#67c6c1",
        "Random-SNR control": "#e6c77a",
        "SNR curriculum": "#ff9b78",
    }
    for model_name, group in df.groupby("model"):
        y = [model_name] * len(group)
        fig.add_trace(go.Scatter(
            x=group["start_s"], y=y, mode="markers",
            marker=dict(size=10, color=colors.get(model_name, "#9eb4c5"),
                        opacity=0.85,
                        line=dict(width=1, color="#e9f1f5")),
            customdata=np.stack([group["end_s"], group["probability"]], axis=-1),
            hovertemplate="Start: %{x:.2f}s<br>End: %{customdata[0]:.2f}s<br>Probability: %{customdata[1]:.3f}<extra></extra>",
            name=model_name,
        ))
    fig.update_layout(
        height=250, margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe7ed"), xaxis_title="Time (seconds)",
        yaxis_title="", showlegend=False,
        xaxis=dict(gridcolor="rgba(160,190,205,.12)"),
        yaxis=dict(gridcolor="rgba(160,190,205,.08)"),
    )
    return fig


def show_results(result: dict, audio: np.ndarray, threshold: float):
    st.markdown(f"### {tr('results')}")
    m1, m2, m3 = st.columns(3)
    m1.metric(tr("duration"), f"{result['duration_s']:.1f} s")
    m2.metric(tr("windows"), f"{result['window_count']:,}")
    total = sum(len(v) for v in result["events"].values())
    m3.metric(tr("candidate_events"), total)
    if total:
        st.success(tr("events_found"))
    else:
        st.info(tr("no_events"))

    all_rows = []
    display_names = {
        "fixed": tr("fixed"),
        "random": tr("random"),
        "curriculum": tr("curriculum"),
    }
    for key, rows in result["windows"].items():
        for row in rows:
            all_rows.append({
                "model": display_names[key],
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "probability": row["probability"],
                "detected": row["detected"],
            })
    window_df = pd.DataFrame(all_rows)
    if not window_df.empty:
        st.markdown(f"#### {tr('timeline')}")
        st.plotly_chart(plot_timeline(window_df), use_container_width=True)

    tabs = st.tabs([display_names[k] for k in result["events"]])
    for tab, key in zip(tabs, result["events"]):
        with tab:
            events = result["events"][key]
            if not events:
                st.caption(tr("no_events"))
            else:
                event_df = pd.DataFrame(events).rename(columns={
                    "start_s": tr("time_start"),
                    "end_s": tr("time_end"),
                    "peak_probability": tr("confidence"),
                })
                st.dataframe(event_df, use_container_width=True, hide_index=True)
                chosen = st.selectbox(
                    "Inspect event / Periksa kejadian",
                    options=list(range(len(events))),
                    format_func=lambda i: f"Event {i+1} · {events[i]['start_s']:.2f}–{events[i]['end_s']:.2f} s",
                    key=f"event_{key}",
                )
                ev = events[chosen]
                start = max(0, int(ev["start_s"] * TARGET_SR))
                end = min(len(audio), int(ev["end_s"] * TARGET_SR))
                if end > start:
                    clip = audio[start:end]
                    st.audio(clip, sample_rate=TARGET_SR)
                    wav_buf = io.BytesIO()
                    sf.write(wav_buf, clip, TARGET_SR, format="WAV")
                    st.download_button(
                        "Download event WAV / Unduh WAV kejadian",
                        data=wav_buf.getvalue(),
                        file_name=f"sonara_{key}_event_{chosen+1}.wav",
                        mime="audio/wav",
                        key=f"clip_{key}_{chosen}",
                    )

    with st.expander(tr("details")):
        st.dataframe(window_df, use_container_width=True, hide_index=True)
        st.download_button(
            tr("download_csv"),
            data=window_df.to_csv(index=False).encode("utf-8"),
            file_name="sonara_window_predictions.csv",
            mime="text/csv",
        )
        payload = {
            "project": "SONARA",
            "model_lineage": "Trained detector artifacts exported from the Sonara research experiment; see README for paper and dataset attribution.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "duration_s": result["duration_s"],
            "window_count": result["window_count"],
            "events": result["events"],
            "windows": result["windows"],
            "scientific_note": "Model predictions are candidate detections, not confirmation of dolphin presence.",
        }
        st.download_button(
            tr("download_json"),
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name="sonara_analysis.json",
            mime="application/json",
        )


def main():
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown("## ◈ SONARA")
        st.caption(tr("mission"))
        st.session_state.language = st.radio(
            tr("language"), ["English", "Bahasa Indonesia"],
            index=0 if st.session_state.language == "English" else 1,
            horizontal=False,
        )
        st.divider()
        st.markdown(f"**{tr('nav')}**")
        nav_items = [tr("home"), tr("explore"), tr("discoveries"), tr("lab"), tr("guide")]
        requested_page = st.session_state.pop("nav_jump", None)
        default_index = nav_items.index(requested_page) if requested_page in nav_items else 0
        page = st.radio(
            "Navigation",
            nav_items,
            index=default_index,
            label_visibility="collapsed",
        )
        st.divider()
        status = model_status()
        render_model_status(status)
        st.divider()
        st.markdown(f"**{tr('about')}**")
        st.caption(tr("about_p"))

    render_header()

    if page == tr("home"):
        st.markdown(
            f"""
            <div class="hero-panel">
              <div class="hero-kicker">MISSION 01 <span>·</span> HYDROACOUSTIC SURVEY</div>
              <h1>{tr('home_h')}</h1>
              <p>{tr('home_p')}</p>
              <div class="sonar-wrap">
                <div class="sonar-ring ring-a"></div><div class="sonar-ring ring-b"></div>
                <div class="sonar-ring ring-c"></div><div class="sonar-core">◈</div>
                <div class="sonar-sweep"></div>
                <div class="sonar-label label-a">96 kHz / ARRAY</div>
                <div class="sonar-label label-b">PASSIVE LISTENING</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(tr("start"), type="primary", use_container_width=True):
            st.session_state["nav_jump"] = tr("explore")
            st.rerun()
        st.markdown(f"### {tr('notice_title')}")
        st.info(tr("notice"))
        c1, c2, c3 = st.columns(3)
        c1.metric(tr("sample_rate"), "96,000 Hz")
        c2.metric(tr("window"), "2.0 s")
        c3.metric(tr("hop"), "1.0 s")
        st.markdown(f"### {tr('how')}")
        if st.session_state.language == "English":
            step_html = """
            <div class="steps-grid">
              <div class="step-card"><span>01</span><b>UPLOAD</b><p>Choose an underwater WAV recording.</p></div>
              <div class="step-card"><span>02</span><b>SCAN</b><p>Three detector variants examine overlapping windows.</p></div>
              <div class="step-card"><span>03</span><b>REVIEW</b><p>Inspect candidate events, scores, and audio clips.</p></div>
            </div>"""
        else:
            step_html = """
            <div class="steps-grid">
              <div class="step-card"><span>01</span><b>UNGGAH</b><p>Pilih rekaman bawah laut berformat WAV.</p></div>
              <div class="step-card"><span>02</span><b>PINDAI</b><p>Tiga varian detektor memeriksa jendela audio yang saling bertumpang tindih.</p></div>
              <div class="step-card"><span>03</span><b>TINJAU</b><p>Periksa kandidat kejadian, skor, dan klip audio.</p></div>
            </div>"""
        st.markdown(step_html, unsafe_allow_html=True)

    elif page == tr("explore"):
        st.markdown(f"### {tr('explore_h')}")
        st.write(tr("explore_p"))
        uploaded = st.file_uploader(tr("upload"), type=["wav"], help=tr("upload_help"))
        threshold = st.slider(tr("threshold"), 0.05, 0.95, 0.50, 0.05)
        available = [k for k, v in status.items() if v]
        selected = st.multiselect(
            tr("select_model"),
            options=available,
            default=available,
            format_func=lambda k: {"fixed": tr("fixed"), "random": tr("random"), "curriculum": tr("curriculum")}[k],
        )
        if uploaded is not None:
            if st.session_state.get("active_upload_name") != uploaded.name:
                st.session_state.pop("last_result", None)
                st.session_state.pop("last_audio", None)
                st.session_state.active_upload_name = uploaded.name
            try:
                preview_audio, original_sr = load_audio(uploaded)
                if len(preview_audio) / TARGET_SR < WINDOW_SECONDS:
                    st.error(tr("short_audio"))
                else:
                    st.caption(
                        f"{tr('source')}: {uploaded.name} · {tr('duration')}: {len(preview_audio)/TARGET_SR:.2f} s · "
                        f"{tr('sample_rate')}: {original_sr:,} → {TARGET_SR:,} Hz · "
                        f"{tr('peak')}: {np.max(np.abs(preview_audio)):.3f}"
                    )
                    st.audio(uploaded.getvalue(), format="audio/wav")
                    if st.button(tr("run"), type="primary", disabled=not selected):
                        try:
                            with st.spinner(tr("analyzing")):
                                models = load_model_bundle(MODEL_DIR, selected)
                                result = analyze_recording(preview_audio, models, threshold=threshold)
                            st.session_state.last_result = result
                            st.session_state.last_audio = preview_audio
                            st.session_state.history.insert(0, {
                                "file": uploaded.name,
                                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "duration_s": result["duration_s"],
                                "events": sum(len(v) for v in result["events"].values()),
                            })
                            st.success(tr("analysis_complete"))
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")
                    if "last_result" in st.session_state:
                        show_results(st.session_state.last_result, st.session_state.last_audio, threshold)
            except Exception:
                st.error(tr("invalid_audio"))
        else:
            st.info(tr("need_audio"))
        if not available:
            st.warning(tr("need_models"))

        st.divider()
        st.markdown(f"### {tr('species_h')}")
        st.warning(tr("species_warning"))
        if (MODEL_DIR / SPECIES_MODEL_FILE).exists() and "last_audio" in st.session_state:
            if st.button(tr("species_run")):
                try:
                    species_model = load_model_bundle(MODEL_DIR, ["species"])["species"]
                    best = make_species_prediction(
                        st.session_state.last_audio, species_model,
                        metadata_path=MODEL_DIR / "experiment_metadata.json"
                    )
                    st.metric(tr("species_result"), f"{best['label']} · {best['probability']:.3f}")
                    st.caption("Scores are model outputs, not calibrated probabilities or confirmed taxonomy.")
                except Exception as exc:
                    st.error(f"Species inference failed: {exc}")
        else:
            st.caption(tr("species_missing"))

    elif page == tr("discoveries"):
        st.markdown(f"### {tr('history_h')}")
        if not st.session_state.history:
            st.info(tr("history_empty"))
        else:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
            if st.button("Clear session log / Hapus log sesi"):
                st.session_state.history = []
                st.rerun()

    elif page == tr("lab"):
        st.markdown(f"### {tr('lab_h')}")
        st.write(tr("lab_p"))
        benchmark_path = MODEL_DIR / "snr_robustness_results.csv"
        if benchmark_path.exists():
            benchmark = pd.read_csv(benchmark_path)
            metric = st.selectbox(
                "Benchmark metric / Metrik benchmark",
                ["accuracy", "precision", "recall", "macro_f1", "roc_auc"],
                format_func=lambda x: {
                    "accuracy": "Accuracy / Akurasi",
                    "precision": "Precision",
                    "recall": "Recall",
                    "macro_f1": "Macro-F1",
                    "roc_auc": "ROC-AUC",
                }[x],
            )
            pivot = benchmark.pivot(index="condition", columns="model", values=metric)
            order = ["20 dB", "15 dB", "10 dB", "5 dB", "0 dB", "-5 dB"]
            pivot = pivot.reindex(order)
            pivot.columns = [
                {
                    "Fixed-mix reference": tr("fixed"),
                    "Random-SNR control": tr("random"),
                    "SNR curriculum": tr("curriculum"),
                }.get(col, col) for col in pivot.columns
            ]
            st.dataframe((pivot * 100).round(2).astype(str) + "%", use_container_width=True)
        else:
            st.warning("Benchmark CSV not found / Berkas benchmark CSV tidak ditemukan.")
        st.caption(tr("snr_note"))
        st.info("Preliminary demo-subset results only. Confirm the evaluation split and measured waveform SNR before making comparative claims.")

    elif page == tr("guide"):
        st.markdown(f"### {tr('field_h')}")
        st.write(tr("field_p"))
        for h, p in [
            (tr("what_detection"), tr("what_detection_p")),
            (tr("what_snr"), tr("what_snr_p")),
            (tr("what_spectrogram"), tr("what_spectrogram_p")),
            (tr("limitations"), tr("limitations_p")),
        ]:
            with st.expander(h):
                st.write(p)

    st.divider()
    st.caption(tr("footer"))


if __name__ == "__main__":
    # Honour a navigation request made by the home CTA.
    main()
