"""break_keys.py: break real keys from Shor measurements, step by step.

For each Shor run: take the most frequent measured value that leads to the factors,
then for every key with that N compute the private key with extended Euclid and
decrypt its ciphertexts. The attack only uses N, e and the ciphertexts; the original
messages from keygen are used only at the end, to confirm the decryption is right.

    python -m classical.break_keys [records ...]   (default: shor_rsa/results/ideal/*.json)

Writes classical/results/broken_keys.json.
"""

import argparse
import glob
import json
from pathlib import Path

from .euclid import break_key
from .postprocess import ROOT_DIR, KEY_FILES, load_keys, period_from_phase, factors_from_period

OUTPUT = ROOT_DIR / "classical" / "results" / "broken_keys.json"


def first_breaking_shot(record):
    """Most frequent measured value whose period gives the factors: (y, r, p, q) or None."""
    N, a, t = record["N"], record["a"], record["counting_qubits"]
    for bits, _ in sorted(record["counts"].items(), key=lambda item: -item[1]):
        y = int(bits.replace(" ", ""), 2)
        r = period_from_phase(y, t, a, N)
        pq = factors_from_period(a, r, N) if r else None
        if pq:
            return y, r, *pq
    return None


def main():
    ap = argparse.ArgumentParser(description="Break keys from Shor measurement records.")
    ap.add_argument("records", nargs="*",
                    default=[str(ROOT_DIR / "shor_rsa" / "results" / "ideal" / "*.json")])
    ap.add_argument("--keys", default=KEY_FILES)
    args = ap.parse_args()

    keys = load_keys(args.keys)
    out = []
    for path in sorted(p for g in args.records for p in glob.glob(g)):
        rec = json.load(open(path))
        N, a, t = rec["N"], rec["a"], rec["counting_qubits"]
        hit = first_breaking_shot(rec)
        print(f"\n{Path(path).name}: N = {N}, a = {a}")
        if hit is None:
            print("    no measured value gives the factors (bad base or too much noise)")
            continue
        y, r, p, q = hit
        print(f"    measured y = {y}  ->  phase {y}/{2**t}  ->  period r = {r}")
        print(f"    gcd({a}^{r // 2} - 1, {N}) and gcd({a}^{r // 2} + 1, {N})  ->  p = {p}, q = {q}")
        for key in (k for k in keys if k["N"] == N):
            d, decrypted = break_key(p, q, key)
            ok = decrypted == key["messages"]
            print(f"    key e = {key['e']:>3}:  d = {d:>3}   ciphertexts {key['ciphertexts']}"
                  f"  ->  {decrypted}   {'correct' if ok else 'WRONG'}")
            out.append({"record": Path(path).name, "N": N, "a": a, "y": y, "period": r,
                        "p": p, "q": q, "e": key["e"], "d": d, "keygen_d": key["d"],
                        "ciphertexts": key["ciphertexts"], "decrypted": decrypted,
                        "correct": ok})

    if out:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(out, indent=2))
        n_ok = sum(o["correct"] for o in out)
        print(f"\n{n_ok}/{len(out)} keys broken and decrypted correctly -> {OUTPUT.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
