import json
from pathlib import Path

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from classical.postprocess import analyse_counts, load_keys
from ...shor_circuit import create_shor_circuit


# Shor for N = 15 with depolarizing noise on every two-qubit gate.
# Question: how does the success per shot p_s fall as the gate error rate grows?
# Expected: p_s - baseline ~ (p_ideal - baseline) * exp(-eps * G), G = two-qubit gates.

N = 15
a = 2

# period 4 divides 2^4, so 4 counting qubits already give the exact phases
COUNTING_QUBITS = 4
MULTIPLIER = "binary"

SHOTS = 512
SEEDS = [42, 10042, 20042]

# about 12 500 two-qubit gates, so the interesting range is eps ~ 1 / 12 500
# by 3e-3 the quantum signal is gone (p_s at the pure-noise level); higher rates add
# nothing and are slow to simulate (more errors make the simulation state larger)
NOISE_LEVELS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "results" / "noisy" / "batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


qc = create_shor_circuit(
    N,
    a,
    counting_qubits=COUNTING_QUBITS,
    multiplier=MULTIPLIER
)

# reuse the transpiled circuit for all simulations
compiled = transpile(
    qc,
    basis_gates=["u", "cx"],
    optimization_level=1,
    seed_transpiler=42
)

two_qubit_gates = compiled.count_ops().get("cx", 0)

print(f"Qubits: {compiled.num_qubits}")
print(f"Depth: {compiled.depth()}")
print(f"CX gates: {two_qubit_gates}")

keys = load_keys()


for p in NOISE_LEVELS:

    print(f"\n{'=' * 40}")
    print(f"CX error: {p:g}  (eps * G = {p * two_qubit_gates:.2f})")
    print(f"{'=' * 40}")

    noise_model = NoiseModel()

    if p > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p, 2),
            ["cx"]
        )

    # matrix product state: about 4x faster than statevector for this circuit
    simulator = AerSimulator(
        noise_model=noise_model,
        method="matrix_product_state"
    )

    success = []

    for seed in SEEDS:

        result = simulator.run(
            compiled,
            shots=SHOTS,
            seed_simulator=seed
        ).result()

        counts = result.get_counts()

        data = {
            "N": N,
            "a": a,
            "counting_qubits": COUNTING_QUBITS,
            "multiplier": MULTIPLIER,
            "shots": SHOTS,
            "seed": seed,
            "cx_error_probability": p,
            "circuit": {
                "qubits": compiled.num_qubits,
                "depth": compiled.depth(),
                "two_qubit_gates": two_qubit_gates
            },
            "counts": dict(sorted(counts.items()))
        }

        # success per shot: the full classical pipeline must break the keys
        analysis = analyse_counts(data, keys)
        success.append(analysis["p_s"])

        print(
            f"Seed {seed}: "
            f"p_s = {analysis['p_s']:.3f} "
            f"(pure noise {analysis['p_s_uniform']:.3f})",
            flush=True
        )

        # save each simulation
        filename = OUTPUT_DIR / f"N{N}_a{a}_t{COUNTING_QUBITS}_p{p}_seed{seed}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # summarize the runs
    mean_success = np.mean(success)
    std_success = np.std(success, ddof=1)

    print("\nAVERAGE:")
    print(f"Success per shot: {mean_success:.3f} +- {std_success:.3f}")

    if mean_success > 0:
        print(
            f"Expected number of measurements: "
            f"{1 / mean_success:.2f}"
        )
