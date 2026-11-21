"""postprocess.py: classical post-processing of Shor measurements.

Turns the counting-register measurements of one Shor run into the period, the factors
and the private keys, and reports how often a single shot is enough.

RECORD: one JSON file per run, as written by the shor_rsa scripts
    {"N": int, "a": int, "counting_qubits": int, "shots": int, "counts": {"bitstring": int},
     ...optional: "simulator" / "backend", "cx_error_probability", "multiplier", "circuit"}
    One run per (N, a) attacks every key with that N: factoring N breaks all of them.

Bit order: bitstrings as Qiskit returns them for create_shor_circuit, so the measured
value is y = int(bitstring, 2) and the phase is y / 2^t (t = counting qubits).

Per measured value y (one shot):
    1. phase y / 2^t  ->  Fraction.limit_denominator(N)  ->  denominator r
    2. a^r = 1 (mod N)? if not, try 2r and 3r (the fraction may have come out reduced)
    3. r even and a^(r/2) != -1 (mod N)?
    4. p = gcd(a^(r/2) - 1, N) (or +1), q = N / p          Euclid (euclid.py)
    5. for every key with this N: d = e^-1 mod (p-1)(q-1)   extended Euclid (euclid.py)
       must decrypt its ciphertexts
p_s = successful shots / shots.
p_s_uniform = p_s that pure noise (every y equally likely) would reach. It is far from 0
for small N (about 0.19 for N = 15, a = 2), so a run only shows quantum signal if p_s
beats it.

CLI
    python -m classical.postprocess [records ...]   (default: shor_rsa/results/ideal/*.json)
"""

import argparse
import glob
import json
from fractions import Fraction
from math import inf
from pathlib import Path

from .euclid import break_key, gcd

ROOT_DIR = Path(__file__).resolve().parent.parent
KEY_FILES = str(ROOT_DIR / "data" / "keys_*bit.json")


def load_keys(pattern=KEY_FILES):
    return [k for path in sorted(glob.glob(pattern)) for k in json.load(open(path))["keys"]]


def period_from_phase(y, t, a, N):
    """Steps 1 and 2: period candidate r from one measured value, or None."""
    if y == 0:
        return None
    r = Fraction(y, 2**t).limit_denominator(N).denominator
    for mult in (1, 2, 3):
        if pow(a, r * mult, N) == 1:
            return r * mult
    return None


def factors_from_period(a, r, N):
    """Steps 3 and 4: non-trivial (p, q) from period r, or None."""
    if r % 2:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    for cand in (gcd(x - 1, N), gcd(x + 1, N)):
        if 1 < cand < N:
            return min(cand, N // cand), max(cand, N // cand)
    return None


def keys_broken(p, q, keys):
    """Step 5: True if the factors give a working private key for every key in `keys`."""
    for key in keys:
        try:
            _, decrypted = break_key(p, q, key)
        except ValueError:  # e has no inverse: these "factors" cannot be right
            return False
        if decrypted != key["messages"]:
            return False
    return True


def outcome(y, t, a, N, keys):
    """Steps 1 to 5 for one measured value: (period or None, success)."""
    r = period_from_phase(y, t, a, N)
    pq = factors_from_period(a, r, N) if r else None
    return r, bool(pq) and keys_broken(*pq, keys)


def uniform_baseline(t, a, N, keys, max_t=20):
    """p_s if every y were equally likely (pure noise). A real run must beat this, not 0."""
    if t > max_t:
        return None
    return sum(outcome(y, t, a, N, keys)[1] for y in range(2**t)) / 2**t


def ideal_p_s(N, a, t, keys):
    """Exact p_s of a perfect quantum computer (the ideal phase-estimation distribution)."""
    import math
    keys = [k for k in keys if k["N"] == N]
    r = next(r for r in range(1, N) if pow(a, r, N) == 1)
    Q = 2**t
    total = 0.0
    for y in range(Q):
        prob = 0.0
        for s in range(r):
            theta = 2 * math.pi * (s / r - y / Q)
            den = math.sin(theta / 2) ** 2
            prob += 1.0 if den < 1e-15 else math.sin(Q * theta / 2) ** 2 / (Q * Q * den)
        if outcome(y, t, a, N, keys)[1]:
            total += prob / r
    return total


def counting_qubits(record):
    return record.get("counting_qubits", record.get("n_count"))


def analyse_counts(record, keys):
    """Success probability per shot for one run, checked against all keys with the run's N."""
    N, a, t = record["N"], record["a"], counting_qubits(record)
    keys = [k for k in keys if k["N"] == N]
    if not keys:
        raise ValueError(f"no key with N = {N}")

    if gcd(a, N) != 1:  # a already shares a factor: solved classically, no quantum step
        return {"N": N, "a": a, "t": t, "trivial_gcd": True, "p_s": None, "shots": record["shots"]}

    cache, periods = {}, {}
    success = total = 0
    for bits, count in record["counts"].items():
        y = int(bits.replace(" ", ""), 2)
        total += count
        if y not in cache:
            cache[y] = outcome(y, t, a, N, keys)
        r, ok = cache[y]
        if r:
            periods[r] = periods.get(r, 0) + count
        if ok:
            success += count

    return {
        "N": N, "a": a, "t": t, "trivial_gcd": False, "keys": len(keys),
        "shots": total, "successes": success,
        "p_s": success / total if total else 0.0,
        "p_s_uniform": uniform_baseline(t, a, N, keys),
        "true_period": next(r for r in range(1, N) if pow(a, r, N) == 1),
        "periods_found": dict(sorted(periods.items())),
    }


def estimated_qpu_seconds(record, p_s):
    """Expected QPU time to break the key: (1 / p_s) shots x (circuit duration + rep delay)."""
    c = record.get("circuit", {})
    dur, rep = c.get("duration_s"), c.get("rep_delay_s")
    if dur is None or rep is None:
        return None
    if not p_s:
        return inf
    return (dur + rep) / p_s


def main():
    ap = argparse.ArgumentParser(description="Analyse Shor measurement records.")
    ap.add_argument("records", nargs="*",
                    default=[str(ROOT_DIR / "shor_rsa" / "results" / "ideal" / "*.json")])
    ap.add_argument("--keys", default=KEY_FILES, help="glob of key files")
    args = ap.parse_args()

    keys = load_keys(args.keys)
    paths = sorted(p for g in args.records for p in glob.glob(g))
    for path in paths:
        rec = json.load(open(path))
        res = analyse_counts(rec, keys)
        name = Path(path).name
        if res["trivial_gcd"]:
            print(f"{name}: a = {res['a']} shares a factor with N, no quantum step needed")
            continue
        base = res["p_s_uniform"]
        base_txt = f", pure noise {base:.3f}" if base is not None else ""
        print(f"{name}: p_s = {res['p_s']:.3f}{base_txt}  (true r = {res['true_period']}, "
              f"{res['keys']} keys)")


if __name__ == "__main__":
    main()
