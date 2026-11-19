from time import perf_counter

from qiskit import QuantumCircuit, transpile

from ..modular_gate import modular_multiplication_gate
from ..experiments.alternative_arithmetic.modular_multiplication_clean import (
    modular_multiplication_clean,
)


N = 15
a = 2

n = (N - 1).bit_length()


# =====================================
# old implementation
# =====================================

old_qc = QuantumCircuit(n)

old_gate = modular_multiplication_gate(a, N)

old_qc.append(old_gate, range(n))


# =====================================
# new implementation
# =====================================

new_qc = QuantumCircuit(2 * n + 4)

x_reg = list(range(n))
y_reg = list(range(n, 2 * n))

aux = list(range(2 * n, 2 * n + 3))
valid = 2 * n + 3

modular_multiplication_clean(
    new_qc,
    x_reg,
    y_reg,
    aux,
    valid,
    N,
    a
)


# =====================================
# analysis
# =====================================

def analyze(name, qc):

    print(f"\n{'=' * 40}")
    print(name)
    print(f"{'=' * 40}")

    print(f"Qubits: {qc.num_qubits}")
    print(f"Depth before transpilation: {qc.depth()}")
    print(f"Operations before transpilation: {qc.size()}")

    print("\nTranspiling circuit ...", flush=True)

    start = perf_counter()

    compiled = transpile(
        qc,
        basis_gates=["u", "cx"],
        optimization_level=0,
        seed_transpiler=0
    )

    elapsed = perf_counter() - start

    print("\nRESULTS:")
    print(f"Depth: {compiled.depth()}")
    print(f"CX gates: {compiled.count_ops().get('cx', 0)}")
    print(f"Total operations: {compiled.size()}")
    print(f"Transpilation time: {elapsed:.3f} s")


analyze("OLD IMPLEMENTATION", old_qc)
analyze("NEW IMPLEMENTATION", new_qc)