from math import pi, gcd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from .arithmetic.modular_gate import modular_multiplication_gate
from .arithmetic.modular_multiplication_clean import (
    modular_multiplication_clean,
)

MULTIPLIERS = ("matrix", "binary")


# inverse qft
def inverse_qft(qc, qubits):

    n = len(qubits)

    # reverse qubit order
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - i - 1])

    # inverse qft
    for j in range(n):
        for m in range(j):
            angle = -pi / (2 ** (j - m))
            qc.cp(angle, qubits[m], qubits[j])

        qc.h(qubits[j])


# Shor circuit
# multiplier="matrix":  general permutation matrix (modular_gate.py), gates grow like 4^n;
#                       a simulation shortcut only, not how Shor works on real hardware
# multiplier="binary":  binary adders with carry chains
#                       (arithmetic/), gates grow like n^3.6,
#                       needs n+4 extra helper qubits that start and end in |0>
def create_shor_circuit(N: int, a: int, counting_qubits: int = 4, multiplier: str = "binary"):

    if N < 3 or gcd(a, N) != 1:
        raise ValueError("N must be >= 3, and a must be coprime to N.")
    if multiplier not in MULTIPLIERS:
        raise ValueError(f"multiplier must be one of {MULTIPLIERS}")

    # number of work qubits
    n = (N - 1).bit_length()

    # create registers
    counting = QuantumRegister(counting_qubits, "count")
    work = QuantumRegister(n, "work")
    classical = ClassicalRegister(counting_qubits, "c")

    if multiplier == "binary":
        helper = QuantumRegister(n + 4, "helper")  # y (n) + overflow, valid, restore + valid
        qc = QuantumCircuit(counting, work, helper, classical)
    else:
        qc = QuantumCircuit(counting, work, classical)

    # prepare the work register in |1>
    qc.x(work[0])

    # put the counting register in superposition
    qc.h(counting)

    # controlled modular multiplication
    for k in range(counting_qubits):

        factor = pow(a, 2**k, N)

        # skip multiplication by 1
        if factor == 1:
            continue

        if multiplier == "binary":
            modular_multiplication_clean(
                qc, list(work), list(helper[:n]), list(helper[n:n + 3]), helper[n + 3],
                N, factor, control=counting[k]
            )
        else:
            gate = modular_multiplication_gate(factor, N)
            qc.append(gate.control(1), [counting[k], *work])

    # inverse qft
    inverse_qft(qc, list(counting))

    # measure the counting register
    qc.measure(counting, classical)

    return qc


# test
if __name__ == "__main__":

    N = 21
    a = 2
    counting_qubits = 4

    qc = create_shor_circuit(N, a, counting_qubits)

    print("Quantum circuit:")
    print(qc.draw(fold=120))

    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    result = simulator.run(compiled, shots=4096).result()

    counts = result.get_counts()

    print("\nMeasurement results:")

    for state, count in sorted(counts.items()):
        y = int(state, 2)
        phase = y / (2**counting_qubits)

        print(f"{state}: {count} measurements, phase = {phase:.4f}")