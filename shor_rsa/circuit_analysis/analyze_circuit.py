import json
from pathlib import Path
from time import perf_counter

from qiskit import transpile

from ..shor_circuit import create_shor_circuit


ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / "results" / "circuit_analysis"

# small test case
N = 65
a = 12
COUNTING_QUBITS = 4

SEED = 0

# basis gates for transpilation
BASIS_GATES = ["u", "cx"]


def get_circuit_metrics(qc):

    # count gates by number of qubits
    two_qubit_gates = 0
    multi_qubit_gates = 0

    for instruction in qc.data:

        operation = instruction.operation

        if operation.num_qubits == 2:
            two_qubit_gates += 1

        elif operation.num_qubits > 2:
            multi_qubit_gates += 1

    return {
        "qubits": qc.num_qubits,
        "depth": qc.depth(),
        "two_qubit_gates": two_qubit_gates,
        "multi_qubit_gates": multi_qubit_gates,
        "operations": dict(qc.count_ops())
    }


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # build the circuit
    qc = create_shor_circuit(
        N=N,
        a=a,
        counting_qubits=COUNTING_QUBITS
    )

    # get metrics before transpilation
    before = get_circuit_metrics(qc)

    print("\nBEFORE TRANSPILATION")
    print("------------------------")
    print(f"Qubits: {before['qubits']}")
    print(f"Depth: {before['depth']}")
    print(f"Two-qubit gates: {before['two_qubit_gates']}")
    print(f"Multi-qubit gates (>2): {before['multi_qubit_gates']}")

    # transpile to basis gates
    print("\nTranspiling quantum circuit ...")

    start = perf_counter()

    compiled = transpile(
        qc,
        basis_gates=BASIS_GATES,
        optimization_level=0,
        seed_transpiler=SEED
    )

    translation_time = perf_counter() - start

    # get metrics after transpilation
    after = get_circuit_metrics(compiled)

    print("\nAFTER TRANSPILATION")
    print("------------------------")
    print(f"Qubits: {after['qubits']}")
    print(f"Depth: {after['depth']}")
    print(f"Two-qubit gates: {after['two_qubit_gates']}")
    print(f"Multi-qubit gates (>2): {after['multi_qubit_gates']}")
    print(f"Transpilation time: {translation_time:.2f} s")

    result = {
        "N": N,
        "a": a,
        "counting_qubits": COUNTING_QUBITS,
        "basis_gates": BASIS_GATES,
        "optimization_level": 0,
        "before": before,
        "after": after,
        "translation_time_seconds": translation_time
    }

    output_file = RESULTS_DIR / f"N{N}_a{a}_t{COUNTING_QUBITS}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()