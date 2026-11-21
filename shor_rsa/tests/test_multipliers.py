import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from classical.postprocess import analyse_counts
from keygen import generate
from ..arithmetic.modular_multiplication_clean import (
    modular_multiplication_clean,
)
from ..shor_circuit import create_shor_circuit


SIM = AerSimulator(method="statevector")


def run_basis_states(circuit, inputs):
    """Final statevector of `circuit` for each basis-state input index."""
    compiled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1)
    circuits = []
    for idx in inputs:
        qc = QuantumCircuit(compiled.num_qubits)
        for q in range(compiled.num_qubits):
            if idx >> q & 1:
                qc.x(q)
        qc.compose(compiled, inplace=True)
        qc.save_statevector()
        circuits.append(qc)
    result = SIM.run(circuits).result()
    return [np.asarray(result.get_statevector(i)) for i in range(len(circuits))]


# =========================================
# TEST 1: controlled modular multiplication, every input
# =========================================

@pytest.mark.parametrize("N,a", [(15, 2), (15, 7), (21, 5)])
def test_binary_controlled_multiplication(N, a):
    n = (N - 1).bit_length()
    control = 2 * n + 4
    qc = QuantumCircuit(2 * n + 5)  # x, y, overflow/valid/restore, valid, control
    modular_multiplication_clean(qc, list(range(n)), list(range(n, 2 * n)),
                                 list(range(2 * n, 2 * n + 3)), 2 * n + 3, N, a, control=control)
    cases = [(ctrl, x) for ctrl in (0, 1) for x in range(2**n)]
    states = run_basis_states(qc, [x | ctrl << control for ctrl, x in cases])
    for (ctrl, x), sv in zip(cases, states):
        y = (a * x) % N if ctrl and x < N else x
        # x register holds the result, all helper qubits are back in |0>
        assert abs(sv[y | ctrl << control]) ** 2 == pytest.approx(1.0), (ctrl, x)


# =========================================
# TEST 2: both multipliers give the same Shor result
# =========================================

@pytest.mark.parametrize("multiplier", ["matrix", "binary"])
def test_shor_n15_success(multiplier):
    t = 8
    qc = create_shor_circuit(N=15, a=7, counting_qubits=t, multiplier=multiplier)
    compiled = transpile(qc, SIM)
    counts = SIM.run(compiled, shots=2048, seed_simulator=0).result().get_counts()
    record = {"N": 15, "a": 7, "counting_qubits": t, "shots": 2048, "counts": counts}
    keys = generate(4, 3, seed=0, n_messages=3)
    # ideal success per shot for N = 15, a = 7 is exactly 0.75
    assert analyse_counts(record, keys)["p_s"] == pytest.approx(0.75, abs=0.04)
