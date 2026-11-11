from __future__ import annotations

from pathlib import Path
import numpy as np
from tensorflow import keras

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
EXPECTED = {
    "sonara_fixed_mix_reference.keras": (350, 350, 1),
    "sonara_random_snr_control.keras": (350, 350, 1),
    "sonara_snr_curriculum_detector.keras": (350, 350, 1),
    "sonara_species_classifier.keras": (450, 450, 1),
}

def main():
    for filename, shape in EXPECTED.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"MISSING  {filename}")
            continue
        model = keras.models.load_model(path, compile=False)
        actual = tuple(model.input_shape[1:])
        if actual != shape:
            raise SystemExit(f"SHAPE ERROR {filename}: expected {shape}, got {actual}")
        probe = np.zeros((1, *shape), dtype=np.float32)
        output = np.asarray(model.predict(probe, verbose=0))
        if output.ndim != 2 or output.shape[0] != 1:
            raise SystemExit(f"OUTPUT ERROR {filename}: got {output.shape}")
        print(f"OK  {filename}  input={actual}  output={tuple(output.shape)}")
    print("Model archive loading and forward-pass smoke test complete.")

if __name__ == "__main__":
    main()
