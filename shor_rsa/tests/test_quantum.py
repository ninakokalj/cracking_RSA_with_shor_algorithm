import numpy as np
import pytest

from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from ..modular_gate import modular_multiplication_gate
from ..shor_circuit import create_shor_circuit


# =========================================
# TEST 1: Modular multiplication
# =========================================

@pytest.mark.parametrize(
    "N,a",
    [
        (15, 2),
        (15, 4),
        (21, 2),
        (21, 8),
    ]
)
def test_modular_multiplication(N, a):

    gate = modular_multiplication_gate(a, N)

    # get the gate matrix
    matrix = Operator(gate).data

    n = (N - 1).bit_length()
    size = 2**n

    # check each input state
    for x in range(size):

        if x < N:
            expected = (a * x) % N
        else:
            expected = x

        # column x is the output for |x>
        column = matrix[:, x]

        # check the expected output
        assert np.isclose(
            abs(column[expected]),
            1.0
        )

        # check for one nonzero output amplitude
        assert np.count_nonzero(
            abs(column) > 1e-9
        ) == 1


# =========================================
# TEST 2: Invalid base
# =========================================

def test_invalid_base():

    # 3 and 15 are not coprime
    with pytest.raises(ValueError):
        modular_multiplication_gate(3, 15)


# =========================================
# TEST 3: Shor for N = 15
# =========================================

@pytest.mark.parametrize(
    "a,expected_states",
    [
        (2, {"0000", "0100", "1000", "1100"}),
        (4, {"0000", "1000"}),
    ]
)
def test_shor_n15(a, expected_states):

    qc = create_shor_circuit(
        N=15,
        a=a,
        counting_qubits=4
    )

    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    counts = simulator.run(
        compiled,
        shots=1024,
        seed_simulator=0
    ).result().get_counts()

    # check the shot count
    assert sum(counts.values()) == 1024

    # no unexpected states
    assert set(counts).issubset(expected_states)

    # all expected states must occur
    assert set(counts) == expected_states


# =========================================
# TEST 4: Shor with period 6
# =========================================

def test_shor_period_six():

    t = 6

    qc = create_shor_circuit(
        N=21,
        a=2,
        counting_qubits=t
    )

    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    counts = simulator.run(
        compiled,
        shots=2048,
        seed_simulator=0
    ).result().get_counts()

    # expected phases for period 6
    expected_phases = [
        s / 6 for s in range(6)
    ]

    near_expected = 0

    for state, count in counts.items():

        y = int(state, 2)
        phase = y / (2**t)

        # distance to the nearest expected phase
        distance = min(
            abs(phase - expected)
            for expected in expected_phases
        )

        # check whether the phase is close enough
        if distance <= 1 / (2**t) + 1e-9:
            near_expected += count

    probability = near_expected / 2048

    # most measurements should be near expected phases
    assert probability > 0.70