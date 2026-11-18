"""keygen.py: simplified RSA key generator (Shor vs brute force project, group 29).

Purpose
    Produces small RSA keys that the Shor circuit and the brute-force attack try to break.
    Standard library only (Python 3.8+), no install needed.

Difficulty
    BITS = bit length of the modulus N = p * q (not of each prime). It sets the Shor
    work register size n = BITS. Minimum is 4 (N = 15).

CLI
    python keygen.py BITS [--count 50] [--seed 0] [--messages 3] [--out keys.json]

Output JSON
    {"bits": int, "count": int, "seed": int,
     "keys": [{"p", "q", "N", "phi", "e", "d", "messages": [int], "ciphertexts": [int]}, ...]}
    p < q are distinct odd primes, phi = (p-1)(q-1), e * d = 1 mod phi,
    ciphertexts[i] = messages[i]^e mod N, messages are in [2, N-2].

Python use
    from keygen import generate
    keys = generate(bits=6, count=50, seed=0, n_messages=3)   # list of the key dicts above

Notes for consumers
    - Treat p, q, phi and d as ground truth for checking a break; the attack itself
      must only use N, e and the ciphertexts.
    - Same seed gives identical output; record the seed in the report.
    - Small BITS repeat N: 4 -> {15}, 5 -> {21}, 6 -> 6 distinct N. Keys sharing N
      differ only in e, d and messages, so deduplicate on N for circuit runs.
    - All keys in one file have the same BITS; run once per difficulty for a mix.
"""

import argparse
import json
import random
from math import gcd

MIN_BITS = 4  # smallest odd N = p * q with distinct odd primes is 15 (4 bits)


def primes_below(limit):
    """All primes < limit (sieve of Eratosthenes)."""
    sieve = bytearray([1]) * limit
    sieve[:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(limit) if sieve[i]]


def candidate_moduli(bits):
    """All (p, q) with 3 <= p < q prime and N = p * q exactly `bits` bits long."""
    lo, hi = 1 << (bits - 1), 1 << bits
    primes = [p for p in primes_below(hi // 3 + 1) if p > 2]
    return [(p, q) for i, p in enumerate(primes) for q in primes[i + 1 :] if lo <= p * q < hi]


def make_key(p, q, rng, n_messages):
    n, phi = p * q, (p - 1) * (q - 1)
    e = rng.choice([x for x in range(3, phi) if gcd(x, phi) == 1])
    d = pow(e, -1, phi)
    messages = rng.sample(range(2, n - 1), min(n_messages, n - 3))
    ciphertexts = [pow(m, e, n) for m in messages]
    assert all(pow(c, d, n) == m for m, c in zip(messages, ciphertexts))
    return {"p": p, "q": q, "N": n, "phi": phi, "e": e, "d": d,
            "messages": messages, "ciphertexts": ciphertexts}


def generate(bits, count, seed, n_messages):
    if bits < MIN_BITS:
        raise ValueError(f"difficulty must be at least {MIN_BITS} bits (N = 15)")
    rng = random.Random(seed)
    pairs = candidate_moduli(bits)
    return [make_key(*rng.choice(pairs), rng, n_messages) for _ in range(count)]


def main():
    ap = argparse.ArgumentParser(description="Generate simplified RSA keys (see module docstring).")
    ap.add_argument("bits", type=int, help="difficulty: bit length of N")
    ap.add_argument("--count", type=int, default=50, help="number of keys (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for reproducibility (default 0)")
    ap.add_argument("--messages", type=int, default=3, help="sample messages per key (default 3)")
    ap.add_argument("--out", default="keys.json", help="output JSON file (default keys.json)")
    args = ap.parse_args()

    try:
        keys = generate(args.bits, args.count, args.seed, args.messages)
    except ValueError as err:
        ap.error(str(err))

    with open(args.out, "w") as f:
        json.dump({"bits": args.bits, "count": args.count, "seed": args.seed, "keys": keys}, f, indent=2)

    distinct = sorted({k["N"] for k in keys})
    print(f"Wrote {len(keys)} keys ({args.bits}-bit N) to {args.out}")
    print(f"Distinct N ({len(distinct)}): {distinct}")
    if len(distinct) < len(keys):
        print("Some keys share N; they differ in e, d and messages.")


if __name__ == "__main__":
    main()
