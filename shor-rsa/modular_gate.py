import numpy as np
from math import gcd

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


def modular_multiplication_gate(a: int, N: int):

    # are a and N coprime?
    if gcd(a, N) != 1:
        raise ValueError("a and N must be coprime.")

    # number of qubits
    n = (N - 1).bit_length()

    # number of basis states
    size = 2**n

    # zero matrix
    matrix = np.zeros((size, size), dtype=complex)

    # map each basis state
    for x in range(size):

        if x < N:
            y = (a * x) % N
        else:
            y = x

        matrix[y, x] = 1

    # gate -> matrix
    return UnitaryGate(matrix, label=f"×{a} mod {N}")


if __name__ == "__main__":
    # test

    N = 15
    a = 2

    n = (N - 1).bit_length()

    # build the circuit
    qc = QuantumCircuit(n, n)

    # prepare |1>
    qc.x(0)

    # add the modular multiplication gate
    gate = modular_multiplication_gate(a, N)
    qc.append(gate, range(n))

    # measure the qubits
    qc.measure(range(n), range(n))

    # draw the circuit
    print(qc.draw())

    # run the simulation
    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    result = simulator.run(compiled, shots=100).result()

    print("Results:", result.get_counts())