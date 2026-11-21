"""Time per Shor attempt vs key size on an IBM processor (compiled, not simulated).

One attempt = one shot of the Shor circuit:
    2n controlled multiplications (t = 2n counting qubits, worst case: none skipped)
    + measurement + the processor's reset delay between shots.
Each multiplication is compiled for the real chip layout of FakeBrisbane (including the
SWAPs needed to route it), and its duration comes from the processor's gate times.
Compiling the whole circuit for large n would take too long; for small n the whole
circuit is compiled too, to check the estimate.

Output: shor_rsa/results/circuit_analysis/time_per_attempt.csv

    python -m shor_rsa.circuit_analysis.time_per_attempt [--max-n 10]
"""

import argparse
import csv
import math
from math import gcd
from pathlib import Path
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from keygen import generate
from ..arithmetic.modular_multiplication_clean import (
    modular_multiplication_clean,
)
from ..shor_circuit import create_shor_circuit

ROOT_DIR = Path(__file__).resolve().parents[1]  # shor_rsa
OUTPUT = ROOT_DIR / "results" / "circuit_analysis" / "time_per_attempt.csv"
OPTIMIZATION_LEVEL = 1
SEED = 0
TWO_QUBIT_GATES = ("ecr", "cz", "cx")


def controlled_multiplication(N, a):
    n = (N - 1).bit_length()
    qc = QuantumCircuit(2 * n + 5)  # x, y, overflow/valid/restore, valid, control
    modular_multiplication_clean(qc, list(range(n)), list(range(n, 2 * n)),
                                 list(range(2 * n, 2 * n + 3)), 2 * n + 3, N, a,
                                 control=2 * n + 4)
    return qc


def measure(isa, backend):
    ops = isa.count_ops()
    return {"two_qubit_gates": sum(ops.get(g, 0) for g in TWO_QUBIT_GATES),
            "depth": isa.depth(),
            "duration_s": isa.estimate_duration(backend.target, unit="s")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--max-n", type=int, default=10)
    ap.add_argument("--full-circuit-max-n", type=int, default=5)
    args = ap.parse_args()

    backend = FakeBrisbane()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=OPTIMIZATION_LEVEL,
                                      seed_transpiler=SEED)
    readout_s = backend.target["measure"][(0,)].duration
    rep_delay_s = backend.configuration().default_rep_delay
    print(f"{backend.name}: readout {readout_s * 1e6:.2f} us, reset delay {rep_delay_s * 1e6:.0f} us")

    rows = []
    for n in range(args.min_n, args.max_n + 1):
        N = generate(n, 1, seed=SEED, n_messages=1, balanced=n >= 5)[0]["N"]
        a = next(a for a in range(2, N) if gcd(a, N) == 1)
        start = perf_counter()
        mult = measure(pm.run(controlled_multiplication(N, a)), backend)
        t = 2 * n
        attempt_s = t * mult["duration_s"] + readout_s + rep_delay_s
        row = {"n": n, "N": N, "a": a, "counting_qubits": t, "qubits": 4 * n + 4,
               "mult_two_qubit_gates": mult["two_qubit_gates"],
               "mult_depth": mult["depth"], "mult_duration_s": mult["duration_s"],
               "attempt_two_qubit_gates_est": t * mult["two_qubit_gates"],
               "attempt_duration_s_est": attempt_s,
               "full_two_qubit_gates": "", "full_duration_s": "",
               "compile_seconds": 0.0}
        if n <= args.full_circuit_max_n:
            # check: compile the whole Shor circuit (some multiplications may be skipped)
            full = measure(pm.run(create_shor_circuit(N, a, t)), backend)
            row["full_two_qubit_gates"] = full["two_qubit_gates"]
            row["full_duration_s"] = full["duration_s"] + readout_s + rep_delay_s
        row["compile_seconds"] = round(perf_counter() - start, 1)
        rows.append(row)
        print(f"n={n:>2} N={N:<5} one multiplication: {mult['two_qubit_gates']:>7} two-qubit gates, "
              f"{mult['duration_s'] * 1e3:8.2f} ms  ->  attempt ~ {attempt_s * 1e3:9.2f} ms"
              + (f"  (full circuit: {row['full_duration_s'] * 1e3:.2f} ms)" if row["full_duration_s"] != "" else "")
              + f"  [{row['compile_seconds']} s]", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUTPUT}")

    xs = [math.log(r["n"]) for r in rows]
    ys = [math.log(r["attempt_duration_s_est"] - rep_delay_s - readout_s) for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    print(f"time per attempt (gates part) grows like n^{k:.2f}")


if __name__ == "__main__":
    main()
