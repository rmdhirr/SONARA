# SONARA — Ocean Sound Explorer

**SONARA** is a bilingual (English / Bahasa Indonesia) Streamlit research prototype for exploring underwater WAV recordings with three SNR-trained acoustic detection CNNs.

The application and packaged model artifacts are branded Sonara. The supplied trained weights are copied unchanged; only the artifact filenames are normalized to the Sonara naming scheme. The original research lineage is credited below.

## Current implementation

- Scientific submarine-console interface with responsive layout and reduced-motion support.
- English and Bahasa Indonesia interface toggle.
- WAV upload and playback.
- Mono conversion and resampling to **96 kHz**.
- Overlapping **2-second windows**, **1-second hop**.
- Notebook-matched spectrogram rendering: 1024-point STFT, librosa amplitude-to-dB conversion, square Matplotlib canvas, 70 dpi for detection and 90 dpi for species classification.
- Three detector variants: fixed-mix reference, random-SNR control, SNR curriculum.
- Candidate-event grouping for overlapping positive windows.
- Window-level results, event playback, WAV/CSV/JSON downloads.
- Experimental species-classifier panel with a prominent reliability warning.
- Research benchmark and bilingual field guide.

## Model artifacts

The supplied model bundle has been extracted into `models/`:

```text
models/
├── sonara_fixed_mix_reference.keras
├── sonara_random_snr_control.keras
├── sonara_snr_curriculum_detector.keras
├── sonara_species_classifier.keras
├── experiment_metadata.json
└── snr_robustness_results.csv
```

The three detector models and optional species classifier were confirmed to be Keras v3 `.keras` archives. Their weight contents are unchanged from the uploaded research export. Their serialized input shapes are:

| Artifact | Input shape | Output classes |
|---|---:|---:|
| Fixed-mix detector | 350 × 350 × 1 | 2 |
| Random-SNR detector | 350 × 350 × 1 | 2 |
| SNR-curriculum detector | 350 × 350 × 1 | 2 |
| Species classifier | 450 × 450 × 1 | 3 |

The app checks detector input dimensions at runtime before prediction.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_models.py   # load every model and run a zero-input smoke test
streamlit run app.py
```

The trained model files are included in this project package. If you need to reinstall them from the original export bundle:

```bash
python scripts/install_models.py /path/to/CetusID_R_outputs.zip
```

Alternatively set `CETUSID_MODEL_DIR` to the directory containing the model files.

## Public deployment

For Hugging Face Spaces:
1. Create a new **Streamlit** Space.
2. Upload this project.
3. Ensure the `models/` directory and all four `.keras` files are included.
4. Choose CPU hardware initially; the CNNs are small, though inference latency scales with recording length.
5. Test with representative WAV recordings before sharing the Space.

For Streamlit Community Cloud, connect the repository and set the app entry point to `app.py`. Ensure model files are committed and available at runtime. The combined model bundle is approximately 3.9 MB.

## Inference pipeline

Derived from the supplied checkpointed training notebook and model bundle:

- Target sample rate: 96,000 Hz.
- Window duration: 2 seconds.
- Hop: 1 second.
- Spectrogram: `librosa.stft(..., n_fft=1024)`, followed by `librosa.amplitude_to_db`.
- Detection image: square 5 × 5 inch Matplotlib render at 70 dpi (350 × 350).
- Species image: square 5 × 5 inch Matplotlib render at 90 dpi (450 × 450).
- Pixel extraction uses the first RGBA channel (`[:, :, :1]`) and retains the `uint8` pixel values. Keras performs its input dtype handling when the batch is passed to the model.

The app scans a recording using regular, overlapping windows. The training notebook's windows were associated with annotation timestamps. This regular sliding-window inference is an inference-time adaptation; it is not identical to the notebook's annotated training-window extraction procedure.

## Scientific interpretation and limitations

- A positive window is a **candidate detection**, not proof of dolphin presence.
- Overlapping positive windows are merged into candidate events. Event counts are not counts of individual vocalisations.
- Uploaded recordings have no ground-truth labels or measured SNR by default. The app therefore does not report accuracy for an arbitrary upload.
- The species classifier is experimental. The notebook reports 33.3% validation accuracy on its demonstration subset. Its score must not be treated as reliable species identification or a calibrated probability.
- The benchmark CSV comes from the notebook's demonstration-subset experiment. The metadata explicitly warns that these results are not directly comparable with the paper's full independent test benchmarks; any window-level fallback split is not independent-recording validation.
- Repeatedly identical metric values across several SNR settings should be investigated by auditing the noise-generation/evaluation path before drawing strong conclusions.
- Real ocean recordings may differ in hydrophone response, propagation, ambient noise, species distribution, and recording conditions. Treat outputs as screening cues requiring expert review.

## Research attribution

The model lineage and dataset context are recorded in `models/experiment_metadata.json`. The original research implementation is cited by its authors and repository; this attribution is not the name of the Sonara application. The experiment references Frainer et al. (2023), *Automatic Detection and Taxonomic Identification of Dolphin Vocalisations Using Convolutional Neural Networks for Passive Acoustic Monitoring*, DOI: [10.1016/j.ecoinf.2023.102291](https://doi.org/10.1016/j.ecoinf.2023.102291), and Zenodo record 8074949.

Sonara is the application/project name used for this implementation. The source paper and original model lineage are credited here; the original model artifact filenames are preserved for traceability.

## Project structure

```text
.
├── app.py
├── inference.py
├── requirements.txt
├── Dockerfile
├── README.md
├── models/
│   ├── *.keras
│   ├── experiment_metadata.json
│   └── snr_robustness_results.csv
└── scripts/
    └── install_models.py
```
