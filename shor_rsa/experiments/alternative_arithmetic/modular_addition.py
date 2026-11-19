from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def add_constant(qc, qubits, c, controls=None):
    """
    reversibly add the constant c modulo 2^n

    if controls are given, add only when all control qubits are 1
    """

    controls = list(controls or [])

    n = len(qubits)
    c = c % (2**n)

    def controlled_x(target, control_qubits):

        if len(control_qubits) == 0:
            qc.x(target)

        elif len(control_qubits) == 1:
            qc.cx(control_qubits[0], target)

        else:
            qc.mcx(control_qubits, target)

    for bit in range(n):

        if not ((c >> bit) & 1):
            continue

        # carry to higher bits
        for target in range(n - 1, bit, -1):

            all_controls = controls + qubits[bit:target]

            controlled_x(qubits[target], all_controls)

        controlled_x(qubits[bit], controls)


def modular_addition(qc, data, c, N, overflow, valid, restore):
    """
    |x> -> |(x+c) mod N>  for x < N

    leave x >= N unchanged

    return all three ancillas to |0>
    """

    n = len(data)

    if not (1 <= N <= 2**n):
        raise ValueError("N is not valid for the register size.")

    c = c % N

    if c == 0:
        return

    extended = data + [overflow]

    # check whether x < N
    add_constant(qc, extended, -N)

    qc.cx(overflow, valid)

    add_constant(qc, extended, N)

    # add c when x < N
    add_constant(qc, extended, c, controls=[valid])

    # subtract N
    add_constant(qc, extended, -N, controls=[valid])

    # mark whether N must be added back
    qc.ccx(valid, overflow, restore)

    add_constant(qc, extended, N, controls=[restore])

    # clear restore
    # for valid inputs, the result shows whether N was added back
    add_constant(qc, extended, -c)

    qc.x(overflow)
    qc.ccx(valid, overflow, restore)
    qc.x(overflow)

    add_constant(qc, extended, c)

    # clear valid
    add_constant(qc, extended, -N)

    qc.cx(overflow, valid)

    add_constant(qc, extended, N)


def test_modular_addition():

    test_cases = [
        (15, 4),
        (21, 8),
    ]

    for N, c in test_cases:

        n = (N - 1).bit_length()

        for x in range(2**n):

            qc = QuantumCircuit(n + 3)

            data = list(range(n))

            overflow = n
            valid = n + 1
            restore = n + 2

            # prepare |x>
            for i in range(n):
                if (x >> i) & 1:
                    qc.x(i)

            modular_addition(
                qc,
                data,
                c,
                N,
                overflow,
                valid,
                restore
            )

            state = Statevector.from_instruction(qc)

            result = int(state.probabilities().argmax())

            if x < N:
                expected = (x + c) % N
            else:
                expected = x

            assert result == expected, (
                f"N={N}, x={x}, c={c}: "
                f"expected {expected}, got {result}"
            )

        print(f"N={N}, c={c}: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_modular_addition()