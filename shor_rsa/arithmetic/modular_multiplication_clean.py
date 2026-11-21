from math import gcd

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .modular_addition import add_constant, modular_addition


def mark_valid(qc, x_reg, overflow, valid, N, control=None):
    """
    flip valid when x < N (and control is 1, if a control qubit is given)

    leave x and overflow unchanged
    """

    extended = x_reg + [overflow]

    # subtracting N sets the top bit when x < N
    add_constant(qc, extended, -N)

    if control is None:
        qc.cx(overflow, valid)
    else:
        # valid = (x < N) and control, so everything controlled on valid
        # is automatically controlled on the Shor counting qubit too
        qc.ccx(overflow, control, valid)

    # restore x
    add_constant(qc, extended, N)


def controlled_accumulate(
    qc, x_reg, y_reg, aux, valid, N, a,
    inverse=False
):
    """
    if valid=1:
        |x>|y> -> |x>|(y + a*x) mod N>

    undo the operation if inverse=True

    assume y < N and clean ancillas
    """

    n = len(x_reg)

    indices = range(n)

    if inverse:
        indices = reversed(range(n))

    overflow, inner_valid, restore = aux

    for i in indices:

        c = (a * (2 ** i)) % N

        # adding (N - c) mod N undoes adding c (for y < N)
        if inverse:
            c = (-c) % N

        # control on valid and bit i of x
        # pass the controls into the addition instead of using gate.control(2):
        # same result, about 40x fewer gates
        modular_addition(
            qc, y_reg, c, N,
            overflow, inner_valid, restore,
            controls=[valid, x_reg[i]]
        )


def modular_multiplication_clean(
    qc, x_reg, y_reg, aux, valid, N, a, control=None
):
    """
    for x < N:
        |x>|0> -> |a*x mod N>|0>

    for x >= N:
        |x>|0> -> |x>|0>

    if a control qubit is given, multiply only when it is 1 (as Shor needs)

    return ancillas to |0>
    """

    n = (N - 1).bit_length()

    if len(x_reg) != n or len(y_reg) != n:
        raise ValueError("Invalid register size.")

    if len(aux) != 3:
        raise ValueError("Three ancilla qubits are required.")

    if gcd(a, N) != 1:
        raise ValueError("a and N must be coprime.")

    # check whether x < N
    mark_valid(qc, x_reg, aux[0], valid, N, control)

    # compute a*x mod N in y
    controlled_accumulate(
        qc, x_reg, y_reg, aux, valid, N, a
    )

    # swap registers for valid inputs
    for i in range(n):
        qc.cswap(valid, x_reg[i], y_reg[i])

    # get the modular inverse of a
    a_inverse = pow(a, -1, N)

    # clear the original x from y
    controlled_accumulate(
        qc,
        x_reg,
        y_reg,
        aux,
        valid,
        N,
        a_inverse,
        inverse=True
    )

    # clear valid
    # valid outputs are still < N; invalid inputs are unchanged
    mark_valid(qc, x_reg, aux[0], valid, N, control)


def test_clean_multiplication():

    N = 15
    a = 2

    n = (N - 1).bit_length()

    test_values = range(2**n)

    for x in test_values:

        qc = QuantumCircuit(2 * n + 4)

        x_reg = list(range(n))
        y_reg = list(range(n, 2 * n))

        aux = list(range(2 * n, 2 * n + 3))
        valid = 2 * n + 3

        # prepare |x>
        for i in range(n):
            if (x >> i) & 1:
                qc.x(x_reg[i])

        modular_multiplication_clean(
            qc, x_reg, y_reg, aux, valid, N, a
        )

        state = Statevector.from_instruction(qc)

        probabilities = state.probabilities()

        result = int(probabilities.argmax())

        if x < N:
            expected = (a * x) % N
        else:
            expected = x

        # check the result in x, with y and ancillas at 0
        assert abs(probabilities[expected] - 1.0) < 1e-9, (
            f"N={N}, a={a}, x={x}: "
            f"expected {expected}, got {result}"
        )

        print(
            f"x={x}: "
            f"result={result}, "
            f"expected={expected} OK"
        )

    print("\nAll tests passed!")

def test_superposition():
    N = 15
    a = 2
    n = 4

    qc = QuantumCircuit(2 * n + 4)

    x_reg = list(range(n))
    y_reg = list(range(n, 2 * n))
    aux = list(range(2 * n, 2 * n + 3))
    valid = 2 * n + 3

    # prepare the superposition
    # (|2> + |6>) / sqrt(2)
    qc.x(x_reg[1])
    qc.h(x_reg[2])

    modular_multiplication_clean(
        qc, x_reg, y_reg, aux, valid, N, a
    )

    state = Statevector.from_instruction(qc)

    # expect:
    # (|4> + |12>) / sqrt(2)
    # both outputs must have clean auxiliary registers
    expected = np.zeros(2 ** qc.num_qubits, dtype=complex)

    expected[4] = 1 / np.sqrt(2)
    expected[12] = 1 / np.sqrt(2)

    assert np.allclose(state.data, expected, atol=1e-9)

    print("Superposition: OK")

if __name__ == "__main__":
    test_clean_multiplication()
    test_superposition()