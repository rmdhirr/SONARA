from __future__ import annotations
import argparse
import zipfile
from pathlib import Path

RENAMES = {
    "cetusid_fixed_mix_reference.keras": "sonara_fixed_mix_reference.keras",
    "cetusid_random_snr_control.keras": "sonara_random_snr_control.keras",
    "cetusid_snr_curriculum_detector.keras": "sonara_snr_curriculum_detector.keras",
    "cetusid_species_classifier.keras": "sonara_species_classifier.keras",
}
REQUIRED = {
    "sonara_fixed_mix_reference.keras",
    "sonara_random_snr_control.keras",
    "sonara_snr_curriculum_detector.keras",
}

def main():
    parser = argparse.ArgumentParser(description="Install Sonara model artifacts from the exported research bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    if not args.bundle.exists():
        raise SystemExit(f"Bundle not found: {args.bundle}")
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.bundle) as archive:
        names = archive.namelist()
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"Unsafe archive path rejected: {name}")
        for name in names:
            base = Path(name).name
            target_name = RENAMES.get(base, base)
            if not base:
                continue
            target = args.output / target_name
            target.write_bytes(archive.read(name))
    found = {p.name for p in args.output.glob("*.keras")}
    missing = REQUIRED - found
    if missing:
        raise SystemExit("Extraction finished, but required detector files are missing: " + ", ".join(sorted(missing)))
    print(f"Installed Sonara model artifacts under: {args.output.resolve()}")
    for p in sorted(args.output.iterdir()):
        if p.is_file():
            print(f" - {p.name} ({p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
