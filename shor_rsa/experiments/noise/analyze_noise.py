import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "noisy"

# period r=4 for N=15, a=2, t=4
USEFUL_PHASES = {"0100", "1100"}

for path in sorted(RESULTS_DIR.glob("N15_a2_p*.json")):

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    counts = data["counts"]
    shots = data["shots"]

    useful_shots = sum(
        counts.get(phase, 0)
        for phase in USEFUL_PHASES
    )

    useful_fraction = useful_shots / shots

    print(f"\nCX error: {data['cx_error_probability'] * 100:.1f} %")
    print(f"Ideal peak fraction: {data['peak_fraction']:.3f}")
    print(f"Useful phase fraction: {useful_fraction:.3f}")

    print("Distribution:")
    for bitstring, count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:6]:
        print(f"  {bitstring}: {count}")