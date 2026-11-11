from __future__ import annotations

import json
from pathlib import Path

import librosa
import librosa.display
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

TARGET_SR = 96_000
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
DETECTION_DPI = 70
SPECIES_DPI = 90

DETECTOR_FILES = {
    "fixed": "sonara_fixed_mix_reference.keras",
    "random": "sonara_random_snr_control.keras",
    "curriculum": "sonara_snr_curriculum_detector.keras",
}
SPECIES_MODEL_FILE = "sonara_species_classifier.keras"


def waveform_to_cetus_image(audio: np.ndarray, dpi: int) -> np.ndarray:
    """Reproduce the notebook's 1024-point STFT + square Matplotlib rendering."""
    audio = np.asarray(audio, dtype=np.float32)
    spectrum = librosa.stft(audio, n_fft=1024)
    spectrum_db = librosa.amplitude_to_db(np.abs(spectrum))

    fig = Figure(figsize=(5, 5), dpi=int(dpi))
    canvas = FigureCanvasAgg(fig)
    try:
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        librosa.display.specshow(spectrum_db, ax=ax)
        canvas.draw()
        # The training notebook used the first RGBA channel, not a grayscale transform.
        image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :1].copy()
        return image
    finally:
        fig.clear()
        del canvas, fig, spectrum, spectrum_db, audio


def sliding_windows(audio: np.ndarray):
    win = int(round(WINDOW_SECONDS * TARGET_SR))
    hop = int(round(HOP_SECONDS * TARGET_SR))
    if len(audio) < win:
        raise ValueError("Audio must be at least two seconds long.")
    starts = list(range(0, len(audio) - win + 1, hop))
    windows = [(start, audio[start:start + win]) for start in starts]
    return windows


def load_model_bundle(model_dir: Path, requested: list[str]):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model inference. Install dependencies from requirements.txt."
        ) from exc
    model_dir = Path(model_dir)
    loaded = {}
    for key in requested:
        if key == "species":
            path = model_dir / SPECIES_MODEL_FILE
        else:
            if key not in DETECTOR_FILES:
                raise ValueError(f"Unknown model key: {key}")
            path = model_dir / DETECTOR_FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        loaded[key] = keras.models.load_model(path, compile=False)
    return loaded


def _predict_images(model, images: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start:start + batch_size]).astype(np.uint8, copy=False)
        pred = np.asarray(model.predict(batch, verbose=0))
        outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def merge_candidate_events(rows: list[dict], gap_tolerance_s: float = 0.0) -> list[dict]:
    """Merge overlapping positive windows into candidate events."""
    positives = sorted((r for r in rows if r["detected"]), key=lambda r: r["start_s"])
    if not positives:
        return []
    merged = []
    current = {
        "start_s": float(positives[0]["start_s"]),
        "end_s": float(positives[0]["end_s"]),
        "peak_probability": float(positives[0]["probability"]),
        "window_count": 1,
    }
    for row in positives[1:]:
        if float(row["start_s"]) < current["end_s"] + gap_tolerance_s - 1e-9:
            current["end_s"] = max(current["end_s"], float(row["end_s"]))
            current["peak_probability"] = max(current["peak_probability"], float(row["probability"]))
            current["window_count"] += 1
        else:
            merged.append(current)
            current = {
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "peak_probability": float(row["probability"]),
                "window_count": 1,
            }
    merged.append(current)
    return merged


def analyze_recording(audio: np.ndarray, models: dict, threshold: float = 0.5,
                     batch_size: int = 8) -> dict:
    """Analyze in small batches so long recordings do not materialize all spectrograms in RAM."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Expected mono audio.")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)
    windows = sliding_windows(audio)
    model_keys = [key for key in models if key != "species"]
    if not model_keys:
        raise ValueError("No detector models are available.")

    expected_shape = (5 * DETECTION_DPI, 5 * DETECTION_DPI, 1)
    for key in model_keys:
        expected = tuple(models[key].input_shape[1:])
        if expected != expected_shape:
            raise ValueError(
                f"{key} model expects input {expected}; notebook-matched preprocessing "
                f"produces {expected_shape}."
            )

    probabilities = {key: [] for key in model_keys}
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]
        images = [waveform_to_cetus_image(segment, DETECTION_DPI)
                  for _, segment in batch_windows]
        batch = np.stack(images).astype(np.uint8, copy=False)
        for key in model_keys:
            pred = np.asarray(models[key].predict(batch, verbose=0))
            # Training notebook mapping: positive vocalisation class = 1; negative/noise = 0.
            probabilities[key].extend(pred[:, 1].astype(float).tolist())
        del images, batch

    starts = [start_sample / TARGET_SR for start_sample, _ in windows]
    window_results, event_results = {}, {}
    for key in model_keys:
        rows = []
        for start_s, probability in zip(starts, probabilities[key]):
            rows.append({
                "start_s": float(start_s),
                "end_s": float(start_s + WINDOW_SECONDS),
                "probability": float(probability),
                "detected": bool(probability >= threshold),
            })
        window_results[key] = rows
        event_results[key] = merge_candidate_events(rows)

    return {
        "duration_s": len(audio) / TARGET_SR,
        "window_count": len(windows),
        "windows": window_results,
        "events": event_results,
    }


def make_species_prediction(audio: np.ndarray, model, metadata_path: Path | None = None) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < int(WINDOW_SECONDS * TARGET_SR):
        raise ValueError("Audio must be at least two seconds long.")
    # Species CNN was trained on 2-second source-model windows rendered at 90 dpi.
    segment = audio[:int(WINDOW_SECONDS * TARGET_SR)]
    image = waveform_to_cetus_image(segment, SPECIES_DPI)
    expected = tuple(model.input_shape[1:])
    if tuple(image.shape) != expected:
        raise ValueError(f"Species model expects {expected}, preprocessing produced {image.shape}.")
    scores = np.asarray(model.predict(np.expand_dims(image, axis=0), verbose=0))[0]
    classes = []
    if metadata_path and Path(metadata_path).exists():
        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            classes = metadata.get("species_classes", [])
        except Exception:
            classes = []
    if len(classes) != len(scores):
        classes = [f"Class {i}" for i in range(len(scores))]
    index = int(np.argmax(scores))
    return {"label": classes[index], "probability": float(scores[index]), "scores": scores.tolist()}
