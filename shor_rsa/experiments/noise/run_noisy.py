import json
from pathlib import Path

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from ...shor_circuit import create_shor_circuit


N = 15
a = 2

COUNTING_QUBITS = 4
SHOTS = 4096

SEEDS = [42, 10042, 20042]

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01]

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "results" / "noisy" / "batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IDEAL_PEAKS = {"0000", "0100", "1000", "1100"}
USEFUL_PHASES = {"0100", "1100"}


qc = create_shor_circuit(
    N,
    a,
    counting_qubits=COUNTING_QUBITS
)

# reuse the transpiled circuit for all simulations
compiled = transpile(
    qc,
    basis_gates=["u", "cx"],
    optimization_level=0,
    seed_transpiler=42
)

print(f"Depth: {compiled.depth()}")
print(f"CX gates: {compiled.count_ops().get('cx', 0)}")


for p in NOISE_LEVELS:

    print(f"\n{'=' * 40}")
    print(f"CX error: {p * 100:.1f} %")
    print(f"{'=' * 40}")

    noise_model = NoiseModel()

    if p > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p, 2),
            ["cx"]
        )

    simulator = AerSimulator(
        noise_model=noise_model,
        method="matrix_product_state"
    )

    peak_fractions = []
    useful_fractions = []

    for seed in SEEDS:

        result = simulator.run(
            compiled,
            shots=SHOTS,
            seed_simulator=seed
        ).result()

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

        print(
            f"Seed {seed}: "
            f"ideal={peak_fraction:.3f}, "
            f"useful={useful_fraction:.3f}"
        )

        # save each simulation
        data = {
            "N": N,
            "a": a,
            "counting_qubits": COUNTING_QUBITS,
            "shots": SHOTS,
            "seed": seed,
            "cx_error_probability": p,
            "peak_fraction": peak_fraction,
            "useful_fraction": useful_fraction,
            "counts": counts
        }

        filename = OUTPUT_DIR / f"N{N}_a{a}_p{p}_seed{seed}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # summarize the runs
    mean_peak = np.mean(peak_fractions)
    mean_useful = np.mean(useful_fractions)

    std_useful = np.std(useful_fractions, ddof=1)

    print("\nAVERAGE:")
    print(f"Ideal peaks: {mean_peak:.3f}")
    print(f"Useful phases: {mean_useful:.3f}")
    print(f"Standard deviation: {std_useful:.4f}")

    if mean_useful > 0:
        print(
            f"Expected number of measurements: "
            f"{1 / mean_useful:.2f}"
        )