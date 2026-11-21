"""Gate count of one controlled modular multiplication vs key size, for both multipliers.

This is the part of Shor's time per attempt that depends on the multiplier: a full run
does up to t = 2n of these multiplications (fewer when a^(2^k) mod N = 1).
Circuits are only built and compiled, not simulated, so large n is cheap for "binary".
"matrix" is limited to small n: synthesizing a general 2^n x 2^n matrix takes more
than 10x longer per extra bit (32 s at n = 7; n = 8 had not finished after 8 minutes).
Memory is not the limit: peak about 250 MiB at n = 7 and n = 8.

Output: shor_rsa/results/circuit_analysis/gate_scaling.csv, plus fits printed at the end:
    matrix: log2(cx) vs n       -> slope about 2 means 4^n growth
    binary: log(cx) vs log(n)   -> slope k means n^k growth

    python -m shor_rsa.circuit_analysis.gate_scaling [--max-n 16] [--max-n-matrix 7]
"""

import argparse
import csv
import math
from math import gcd
from pathlib import Path
from time import perf_counter

from qiskit import QuantumCircuit, transpile

from keygen import generate
from ..arithmetic.modular_multiplication_clean import (
    modular_multiplication_clean,
)
from ..arithmetic.modular_gate import modular_multiplication_gate

ROOT_DIR = Path(__file__).resolve().parents[1]  # shor_rsa
OUTPUT = ROOT_DIR / "results" / "circuit_analysis" / "gate_scaling.csv"
BASIS_GATES = ["u", "cx"]
OPTIMIZATION_LEVEL = 1
SEED = 0


def controlled_multiplication(kind, a, N):
    n = (N - 1).bit_length()
    if kind == "binary":
        # x (n), y (n), overflow/valid/restore (3), valid (1), control (1)
        qc = QuantumCircuit(2 * n + 5)
        modular_multiplication_clean(qc, list(range(n)), list(range(n, 2 * n)),
                                     list(range(2 * n, 2 * n + 3)), 2 * n + 3, N, a,
                                     control=2 * n + 4)
        return qc
    qc = QuantumCircuit(n + 1)
    qc.append(modular_multiplication_gate(a, N).control(1), range(n + 1))
    return qc


def linfit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-n", type=int, default=4)
    ap.add_argument("--max-n", type=int, default=16)
    ap.add_argument("--max-n-matrix", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for n in range(args.min_n, args.max_n + 1):
        # a key with exactly n bits (same generator as the experiments)
        N = generate(n, 1, seed=SEED, n_messages=1, balanced=n >= 5)[0]["N"]
        a = next(a for a in range(2, N) if gcd(a, N) == 1)
        for kind in ("matrix", "binary"):
            if kind == "matrix" and n > args.max_n_matrix:
                continue
            start = perf_counter()
            compiled = transpile(controlled_multiplication(kind, a, N), basis_gates=BASIS_GATES,
                                 optimization_level=OPTIMIZATION_LEVEL, seed_transpiler=SEED)
            ops = compiled.count_ops()
            rows.append({"n": n, "N": N, "a": a, "multiplier": kind,
                         "qubits": compiled.num_qubits, "cx": ops.get("cx", 0),
                         "u": ops.get("u", 0), "depth": compiled.depth(),
                         "compile_seconds": round(perf_counter() - start, 2)})
            r = rows[-1]
            print(f"n={n:>2}  {kind:<6}  {r['qubits']:>3} qubits  {r['cx']:>8} cx  "
                  f"depth {r['depth']:>8}  ({r['compile_seconds']} s)", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUTPUT}")

    m = [(r["n"], math.log2(r["cx"])) for r in rows if r["multiplier"] == "matrix"]
    if len(m) >= 2:
        slope, _ = linfit(*zip(*m))
        print(f"matrix: cx grows by x{2 ** slope:.2f} per bit (4^n would be x4)")
    for kind in ("binary",):
        pts = [(math.log(r["n"]), math.log(r["cx"])) for r in rows if r["multiplier"] == kind]
        if len(pts) >= 2:
            slope, _ = linfit(*zip(*pts))
            print(f"{kind}: cx grows like n^{slope:.2f}")


if __name__ == "__main__":
    main()
