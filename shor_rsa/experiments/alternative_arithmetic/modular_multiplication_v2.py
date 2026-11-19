from math import gcd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .modular_addition import modular_addition


def create_modular_add_gate(N: int, c: int):
    """create a gate that adds c modulo N"""

    n = (N - 1).bit_length()

    qc = QuantumCircuit(n + 3)

    data = list(range(n))
    overflow = n
    valid = n + 1
    restore = n + 2

    modular_addition(
        qc, data, c, N,
        overflow, valid, restore
    )

    return qc.to_gate(label=f"+{c} mod {N}")


def multiply_into_accumulator(qc, x_reg, y_reg, aux, N, a):
    """
    |x>|0> -> |x>|a*x mod N>

    for 0 <= x < N
    """

    if gcd(a, N) != 1:
        raise ValueError("a and N must be coprime.")

    n = len(x_reg)

    if len(y_reg) != n or len(aux) != 3:
        raise ValueError("Invalid register size.")

    for i in range(n):

        # constant for bit i of x
        c = (a * (2 ** i)) % N

        # add modulo N when bit i is 1
        add_gate = create_modular_add_gate(N, c)

        controlled_gate = add_gate.control(1)

        qc.append(
            controlled_gate,
            [x_reg[i]] + y_reg + aux
        )


def test_multiplication():

    test_cases = [
        (15, 2),
        (21, 8),
    ]

    for N, a in test_cases:

        n = (N - 1).bit_length()

        for x in [0, 1, 2, N - 1]:

            qc = QuantumCircuit(2 * n + 3)

            x_reg = list(range(n))
            y_reg = list(range(n, 2 * n))
            aux = list(range(2 * n, 2 * n + 3))

            # prepare |x>
            for i in range(n):
                if (x >> i) & 1:
                    qc.x(x_reg[i])

            multiply_into_accumulator(
                qc, x_reg, y_reg, aux, N, a
            )

            state = Statevector.from_instruction(qc)

            result = int(state.probabilities().argmax())

            expected_y = (a * x) % N

            # expect x unchanged, the result in y and ancillas at 0
            expected_state = x | (expected_y << n)

            assert result == expected_state, (
                f"N={N}, a={a}, x={x}: "
                f"expected state {expected_state}, "
                f"got {result}"
            )

            print(
                f"N={N}, a={a}: "
                f"{a} * {x} mod {N} = {expected_y} OK"
            )

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_multiplication()