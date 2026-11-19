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
COUNTING_QUBITS = 4

# limit runs to N <= 21
MAX_N = 21

SEED = 0


# run one experiment

def run_experiment(N: int, a: int, simulator):

    print(f"\nRunning: N = {N}, a = {a}")

    # build the circuit
    qc = create_shor_circuit(
        N=N,
        a=a,
        counting_qubits=COUNTING_QUBITS
    )

    # transpile for the simulator
    compiled = transpile(
        qc,
        simulator,
        optimization_level=0,
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
        "counting_qubits": COUNTING_QUBITS,
        "work_qubits": (N - 1).bit_length(),
        "shots": SHOTS,
        "simulator": "ideal",
        "seed_simulator": SEED,
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

    # ideal simulator
    simulator = AerSimulator()

    for exp in experiments:

        N = exp["N"]
        a = exp["a"]

        # run the experiment
        result = run_experiment(N, a, simulator)

        result["keygen_seed"] = data["keygen_seed"]

        filename = f"N{N}_a{a}_t{COUNTING_QUBITS}.json"

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