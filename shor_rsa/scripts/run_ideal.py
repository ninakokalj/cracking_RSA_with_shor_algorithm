import json

from pathlib import Path
from qiskit import transpile
from qiskit_aer import AerSimulator

from ..shor_circuit import create_shor_circuit


# config

ROOT_DIR = Path(__file__).resolve().parents[1]

EXPERIMENTS_FILE = ROOT_DIR.parent / "data" / "experiments.json"

RESULTS_DIR = ROOT_DIR / "results" / "ideal"

SHOTS = 4096

# the binary multiplier needs 4n + 4 qubits at t = 2n: 28 qubits for 6-bit N,
# about 3 minutes each with the matrix product state simulator
MAX_N = 63

MULTIPLIER = "binary"

SEED = 0


# run one experiment

def run_experiment(N: int, a: int, counting_qubits: int, simulator):

    print(f"\nRunning: N = {N}, a = {a}, t = {counting_qubits}", flush=True)

    # build the circuit
    qc = create_shor_circuit(
        N=N,
        a=a,
        counting_qubits=counting_qubits,
        multiplier=MULTIPLIER
    )

    # transpile to one- and two-qubit gates (also gives the circuit size)
    compiled = transpile(
        qc,
        basis_gates=["u", "cx"],
        optimization_level=1,
        seed_transpiler=SEED
    )

    # run the simulation
    job = simulator.run(
        compiled,
        shots=SHOTS,
        seed_simulator=SEED
    )

    # get measurement counts
    counts = job.result().get_counts()

    # check the shot count
    assert sum(counts.values()) == SHOTS

    result = {
        "N": N,
        "a": a,
        "counting_qubits": counting_qubits,
        "work_qubits": (N - 1).bit_length(),
        "multiplier": MULTIPLIER,
        "shots": SHOTS,
        "simulator": "ideal",
        "simulation_method": "matrix_product_state",
        "seed_simulator": SEED,
        "circuit": {
            "qubits": compiled.num_qubits,
            "depth": compiled.depth(),
            "two_qubit_gates": compiled.count_ops().get("cx", 0)
        },
        "counts": dict(sorted(counts.items()))
    }

    return result


# run all experiments

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # load experiments
    with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["experiments"]

    # keep smaller N
    experiments = [
        exp for exp in experiments
        if exp["N"] <= MAX_N
    ]

    print(f"Number of experiments: {len(experiments)}")

    # ideal simulator; matrix product state handles these circuits far faster than a full
    # statevector (N = 21: 24 qubits, 137k two-qubit gates in about 20 s)
    simulator = AerSimulator(method="matrix_product_state")

    for exp in experiments:

        N = exp["N"]
        a = exp["a"]
        t = exp["counting_qubits"]

        # run the experiment
        result = run_experiment(N, a, t, simulator)

        result["keygen_seed"] = data["keygen_seed"]

        filename = f"N{N}_a{a}_t{t}.json"

        output_file = RESULTS_DIR / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved: {output_file.name}")

        top_counts = sorted(
            result["counts"].items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]

        print("Most frequent measurements:")

        for state, count in top_counts:
            print(f"  {state}: {count / SHOTS:.2%}")

    print("\nAll selected experiments are complete!")


if __name__ == "__main__":
    main()