import json
from pathlib import Path
from statistics import mean, stdev

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from ...shor_circuit import create_shor_circuit


# =====================================
# config
# =====================================

N = 15
A = 2
COUNTING_QUBITS = 4

SHOTS = 512
SEEDS = [42, 10042, 20042]

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "results" / "ibm_noise" / "batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IDEAL_PEAKS = {"0000", "0100", "1000", "1100"}
USEFUL_PHASES = {"0100", "1100"}


# =====================================
# ibm backend
# =====================================

backend = FakeBrisbane()

print("Processor:", backend.name)

properties = backend.properties()
calibration_date = (
    properties.last_update_date.isoformat()
    if properties is not None
    else None
)

print("Calibration date:", calibration_date)


# =====================================
# shor circuit
# =====================================

qc = create_shor_circuit(
    N,
    A,
    counting_qubits=COUNTING_QUBITS
)

compiled = transpile(
    qc,
    backend=backend,
    optimization_level=2,
    seed_transpiler=42
)

print("Depth:", compiled.depth())
print("ECR gates:", compiled.count_ops().get("ecr", 0))


# =====================================
# simulators
# =====================================

ideal_simulator = AerSimulator.from_backend(
    backend,
    noise_model=None,
    method="matrix_product_state"
)

noisy_simulator = AerSimulator.from_backend(
    backend,
    method="matrix_product_state"
)

assert ideal_simulator.options.noise_model is None
assert noisy_simulator.options.noise_model is not None


# =====================================
# run batches
# =====================================

def run_batch(name, simulator):

    print(f"\n{'=' * 45}")
    print(name.upper())
    print(f"{'=' * 45}")

    peak_fractions = []
    useful_fractions = []

    for seed in SEEDS:

        print(f"\nRunning seed {seed} ...", flush=True)

        result = simulator.run(
            compiled,
            shots=SHOTS,
            seed_simulator=seed
        ).result()

        if not result.success:
            raise RuntimeError(str(result.results[0].status))

        counts = result.get_counts()

        peak_fraction = sum(
            counts.get(bits, 0)
            for bits in IDEAL_PEAKS
        ) / SHOTS

        useful_fraction = sum(
            counts.get(bits, 0)
            for bits in USEFUL_PHASES
        ) / SHOTS

        peak_fractions.append(peak_fraction)
        useful_fractions.append(useful_fraction)

        print(f"Ideal peaks: {peak_fraction:.3f}")
        print(f"Useful phases: {useful_fraction:.3f}")

        # save each run
        data = {
            "backend": backend.name,
            "calibration_date": calibration_date,
            "experiment": name,
            "N": N,
            "a": A,
            "counting_qubits": COUNTING_QUBITS,
            "shots": SHOTS,
            "seed": seed,
            "optimization_level": 2,
            "seed_transpiler": 42,
            "simulation_method": "matrix_product_state",
            "circuit_depth": compiled.depth(),
            "ecr_gates": compiled.count_ops().get("ecr", 0),
            "peak_fraction": peak_fraction,
            "useful_fraction": useful_fraction,
            "counts": counts
        }

        filename = OUTPUT_DIR / f"{name}_seed{seed}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # =====================================
    # statistics
    # =====================================

    mean_peak = mean(peak_fractions)
    mean_useful = mean(useful_fractions)
    std_useful = stdev(useful_fractions)

    expected_shots = (
        1 / mean_useful
        if mean_useful > 0
        else None
    )

    print(f"\n{'-' * 45}")
    print("AVERAGE")
    print(f"Ideal peaks: {mean_peak:.3f}")
    print(f"Useful phases: {mean_useful:.3f}")
    print(f"Standard deviation: {std_useful:.4f}")

    if expected_shots is not None:
        print(
            "Expected number of measurements: "
            f"{expected_shots:.2f}"
        )

    summary = {
        "backend": backend.name,
        "calibration_date": calibration_date,
        "experiment": name,
        "N": N,
        "a": A,
        "shots_per_seed": SHOTS,
        "seeds": SEEDS,
        "mean_peak_fraction": mean_peak,
        "mean_useful_fraction": mean_useful,
        "std_useful_fraction": std_useful,
        "expected_shots": expected_shots,
        "peak_fractions": peak_fractions,
        "useful_fractions": useful_fractions
    }

    filename = OUTPUT_DIR / f"{name}_summary.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# =====================================
# run both experiments
# =====================================

ideal = run_batch(
    "ideal",
    ideal_simulator
)

noisy = run_batch(
    "ibm_noise",
    noisy_simulator
)


# =====================================
# final comparison
# =====================================

print(f"\n{'=' * 45}")
print("FINAL COMPARISON")
print(f"{'=' * 45}")

print(
    "Ideal - useful phases:",
    f"{ideal['mean_useful_fraction']:.3f}"
)

print(
    "IBM noise - useful phases:",
    f"{noisy['mean_useful_fraction']:.3f}"
)

difference = (
    ideal["mean_useful_fraction"]
    - noisy["mean_useful_fraction"]
)

print(
    "Decrease in percentage points:",
    f"{difference * 100:.2f}"
)

print("\nAll experiments completed!")