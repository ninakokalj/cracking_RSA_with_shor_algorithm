from time import perf_counter
from qiskit import transpile

from ..shor_circuit import create_shor_circuit


N = 15
a = 2
counting_qubits = 4

qc = create_shor_circuit(N, a, counting_qubits)

print(f"Original depth: {qc.depth()}")
print(f"Number of qubits: {qc.num_qubits}")

for level in [0, 1, 2, 3]:

    print(f"\nOptimization level: {level}")

    start = perf_counter()

    compiled = transpile(
        qc,
        basis_gates=["u", "cx"],
        optimization_level=level,
        seed_transpiler=0
    )

    elapsed = perf_counter() - start

    print(f"Depth: {compiled.depth()}")
    print(f"CX gates: {compiled.count_ops().get('cx', 0)}")
    print(f"Total operations: {compiled.size()}")
    print(f"Transpilation time: {elapsed:.3f} s")