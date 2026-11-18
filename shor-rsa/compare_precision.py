from qiskit import transpile
from qiskit_aer import AerSimulator

from shor_circuit import create_shor_circuit


N = 21
a = 2
shots = 4096

# qubits to compare
counting_qubits_options = [4, 6, 8]

simulator = AerSimulator()

for t in counting_qubits_options:

    print(f"\n{'=' * 50}")
    print(f"Number of counting qubits: {t}")
    print(f"{'=' * 50}")

    # build the circuit
    qc = create_shor_circuit(N, a, counting_qubits=t)

    # transpile for the simulator
    compiled = transpile(qc, simulator)

    # run the simulation
    result = simulator.run(compiled, shots=shots).result()

    # get measurement counts
    counts = result.get_counts()

    sorted_counts = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print(f"Total number of qubits: {qc.num_qubits}")
    print(f"Number of distinct results: {len(counts)}")

    print("\n10 most frequent results:")

    for state, count in sorted_counts[:10]:

        y = int(state, 2)
        phase = y / (2**t)
        probability = count / shots

        print(
            f"{state} | "
            f"phase = {phase:.6f} | "
            f"probability = {probability:.2%}"
        )