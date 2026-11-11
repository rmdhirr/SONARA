# SONARA — Dolphin Vocalisation Analysis Console

SONARA is a bilingual (English / Bahasa Indonesia) research prototype for **detecting candidate dolphin vocalisations** in underwater WAV recordings. It is not a general-purpose ocean-sound classifier.

The project adapts the model lineage and acoustic workflow of CetusID. SONARA is the application name; CetusID is the original research implementation and attribution is retained below.

## What the models do

### Primary stage: dolphin-vocalisation detection

Three binary CNN detector variants scan overlapping audio windows and estimate whether each window resembles the positive **dolphin-vocalisation** class learned during training rather than the negative/background class:

- Fixed-mix reference
- Random-SNR control
- SNR curriculum

A positive window is a **candidate detection**, not confirmation that a dolphin was present. Adjacent overlapping positive windows are merged into candidate events. Candidate-event counts are not counts of individual whistles, clicks, or burst pulses.

### Optional stage: experimental species classification

A separate three-class model scores a selected candidate-event clip against these labels:

- `Delphinus delphis`
- `Sousa plumbea`
- `Tursiops aduncus`

The current SONARA reimplementation reported **33.3% validation accuracy on its demonstration subset** for this species model. It is therefore marked experimental and must not be treated as reliable taxonomic identification. The species stage is deliberately separated from the primary detector.

## Interface

- Pixel/8-bit inspired acoustic-monitoring console, with responsive layout.
- English and Bahasa Indonesia.
- WAV upload, playback, mono conversion, and resampling to 96 kHz.
- Three detector variants, configurable detection threshold, event grouping, and event playback.
- CSV and JSON result exports, plus candidate-event WAV downloads.
- Optional experimental species scoring on a selected detected event.
- Demonstration-subset SNR benchmark table and field notes.

## Model artifacts

Place these files under `models/`:

```text
models/
├── sonara_fixed_mix_reference.keras
├── sonara_random_snr_control.keras
├── sonara_snr_curriculum_detector.keras
├── sonara_species_classifier.keras
├── experiment_metadata.json
└── snr_robustness_results.csv
```

The three detector files are required for the primary analysis. The species classifier is optional. Benchmark and metadata files support the research page and species-label lookup.

## Inference configuration

The preprocessing configuration is based on the supplied checkpointed training notebook and model bundle:

- Target sample rate: **96,000 Hz**
- Window duration: **2 seconds**
- Hop: **1 second**
- STFT: 1024-point
- Detection spectrogram: 350 × 350 × 1 (5 × 5 inches at 70 dpi)
- Species spectrogram: 450 × 450 × 1 (5 × 5 inches at 90 dpi)
- Positive detector class index: **1**

The app uses regular sliding windows at inference because uploaded recordings do not arrive with annotation timestamps. This is an inference-time adaptation; it is not identical to the annotation-associated window extraction used in training.

## Run locally

Use Python 3.12 (recommended for the pinned TensorFlow wheel):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_models.py   # load models and run a zero-input smoke test
streamlit run app.py
```

The actual model-loading smoke test requires TensorFlow and all model artifacts. A successful Python syntax check alone does not verify model compatibility or runtime memory use.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub, including `models/` and the three detector `.keras` files.
2. Create or redeploy the app from the GitHub repository with `app.py` as the entry point.
3. Select **Python 3.12** in the deployment's Advanced settings.
4. Let Community Cloud install `requirements.txt` and inspect the deployment logs.
5. Confirm the sidebar reports `3/3` detector files ready, then run a short WAV test.

The model files total only a few megabytes, but inference time depends on recording length and the free hosting CPU allocation. Start with short test recordings before trying long survey files.

## Scientific interpretation and limitations

- The primary model performs candidate dolphin-vocalisation detection against the negative/background class represented during training; it does not classify arbitrary marine sound sources.
- The species model currently covers only the three labels listed above, and its demo-subset validation is too limited for dependable species identification.
- Uploaded recordings have no ground-truth labels or measured SNR by default. The app does not calculate accuracy for arbitrary uploads.
- The SNR benchmark CSV comes from a demonstration-subset experiment. It is not directly comparable with the paper's full independent test benchmarks. Repeatedly identical metrics across several SNR settings should be investigated before drawing strong conclusions.
- Synthetic-noise robustness is not proof of generalization to real ocean recordings. Hydrophone response, propagation, ambient noise, species distribution, and recording conditions may differ from the training data.
- Treat predictions as screening cues requiring expert review, not as confirmed biological observations.

## Research attribution

The original research lineage is:

Frainer, G. et al. (2023). *Automatic detection and taxonomic identification of dolphin vocalisations using convolutional neural networks for passive acoustic monitoring*. **Ecological Informatics**, 78, 102291. https://doi.org/10.1016/j.ecoinf.2023.102291

- Original implementation: https://github.com/Gui-Frainer/CetusID
- Demonstration data: https://doi.org/10.5281/zenodo.8074949

SONARA is the application/project name used for this adaptation. Supplied model weights are packaged under SONARA filenames; the training lineage and scientific limitations remain documented here.

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
    ├── check_models.py
    └── install_models.py
```
