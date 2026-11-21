"""bruteforce.py: classical attacks on the simplified RSA keys from keygen.py.

Two families, each written as a ladder of versions. Every version changes exactly one
thing compared with the previous one; the change is named in the banner above it.
The ladder documents how the attack was optimised; the experiments only use the end
result, `optimised_brute_force` (= F3, the fastest in wall time).
All functions return (result, ops).

Factoring N (result = smallest factor found, ops = trial divisions or rho iterations)
    F0  every candidate 2 .. N-1
    F1  + stop at isqrt(N)
    F2  + skip even candidates
    F3  + 6k +- 1 wheel (skip multiples of 2 and 3)
    F4  + divide by primes only (sieve up to isqrt(N), sieve time is included)
    R   Pollard rho: not brute force, the best practical classical method at this size
    P   F3 split across CPU cores (parallel_trial_division), for the compute-power experiment

Searching the private exponent d directly (result = d, ops = modular multiplications)
    The attacker knows only N and e. They encrypt their own test messages with e and try
    candidates d until decryption returns the test messages.
    D0  every d, full pow() per test message
    D1  + odd d only (e * d = 1 mod an even number forces d odd)
    D2  + incremental powers: c^(d+2) = c^d * c^2, one multiplication per step
    D3  + check one test message per step, confirm the others only on a hit
    The d found is the smallest working one; it can differ from keygen's d (any d that is
    congruent modulo lambda(N) works), so check success by decrypting, not by comparing d.

CLI (self-check on generated keys)
    python -m classical.bruteforce [--bits 4 6 8 10] [--balanced 20 32]
"""

import argparse
from math import gcd, isqrt
from multiprocessing import get_context

from keygen import generate, primes_below


# ---------------------------------------------------------------------------
# Factoring
# ---------------------------------------------------------------------------

# === F0: naive. Try every candidate from 2 to N-1. ===========================
def f0_naive(N):
    ops = 0
    for d in range(2, N):
        ops += 1
        if N % d == 0:
            return d, ops
    return None, ops


# === F1: + stop at isqrt(N). If N = p*q with p <= q then p <= sqrt(N). ========
def f1_sqrt_bound(N):
    ops = 0
    for d in range(2, isqrt(N) + 1):
        ops += 1
        if N % d == 0:
            return d, ops
    return None, ops


# === F2: + skip even candidates. Test 2 once, then only odd numbers. =========
def f2_skip_even(N):
    ops = 1
    if N % 2 == 0:
        return 2, ops
    for d in range(3, isqrt(N) + 1, 2):
        ops += 1
        if N % d == 0:
            return d, ops
    return None, ops


# === F3: + 6k +- 1 wheel. Every prime > 3 is 6k-1 or 6k+1, so skip the rest. ==
def f3_wheel(N):
    ops = 0
    for d in (2, 3):
        ops += 1
        if N % d == 0:
            return d, ops
    limit = isqrt(N)
    d = 5
    while d <= limit:
        ops += 2
        if N % d == 0:
            return d, ops
        if N % (d + 2) == 0:
            return d + 2, ops
        d += 6
    return None, ops


# === F4: + primes only. Sieve all primes up to isqrt(N) and divide by those. ==
def f4_primes_only(N):
    ops = 0
    for d in primes_below(isqrt(N) + 1):
        ops += 1
        if N % d == 0:
            return d, ops
    return None, ops


# === R: Pollard rho (Floyd cycle detection). Not brute force. ==================
# Iterates x -> x^2 + c mod N; a factor p shows up as a cycle mod p after about
# sqrt(p) steps, so the cost is about N^(1/4) instead of N^(1/2).
def pollard_rho(N):
    if N % 2 == 0:
        return 2, 1
    ops = 0
    for c in range(1, 100):
        x = y = 2
        g = 1
        while g == 1:
            ops += 1
            x = (x * x + c) % N
            y = (y * y + c) % N
            y = (y * y + c) % N
            g = gcd(abs(x - y), N)
        if g != N:
            return g, ops
    return None, ops


# === P: F3 split across CPU cores. =============================================
# The candidates are cut into many small blocks that workers take one at a time, so
# fast cores simply take more blocks. (Equal shares per worker would make every run
# wait for the slowest core; this laptop has 2 fast and 8 slow cores.)
# A shared stop flag lets every worker quit as soon as one of them finds the factor,
# just like the serial version stops at the first hit.
_stop = None
CHECK_EVERY = 1024  # wheel steps between looks at the stop flag
BLOCKS_PER_WORKER = 64


def _init_worker(flag):
    global _stop
    _stop = flag


def make_pool(workers):
    """Process pool with the shared stop flag. Create it once, outside any timing."""
    ctx = get_context()  # platform default: spawn on Windows/macOS, forkserver on Linux
    return ctx.Pool(workers, initializer=_init_worker, initargs=(ctx.Value("b", 0, lock=False),))


def _block_worker(args):
    """Test candidates 6k - 1 and 6k + 1 for k in [k_start, k_end)."""
    N, k_start, k_end = args
    ops = 0
    k = k_start
    while k < k_end:
        if _stop.value:
            break
        for k in range(k, min(k + CHECK_EVERY, k_end)):
            ops += 2
            if N % (6 * k - 1) == 0:
                _stop.value = 1
                return 6 * k - 1, ops
            if N % (6 * k + 1) == 0:
                _stop.value = 1
                return 6 * k + 1, ops
        k += 1
    return None, ops


def _reset_stop(_):
    _stop.value = 0


def parallel_trial_division(N, workers, pool=None):
    """F3 with the candidates split into blocks shared by `workers` processes.

    Pass a pool from make_pool() to keep process start-up out of the measured time.
    ops is the total number of divisions over all workers.
    """
    for d in (2, 3):
        if N % d == 0:
            return d, 1
    k_max = (isqrt(N) + 1) // 6 + 1  # last k with 6k - 1 <= isqrt(N), plus one
    size = max(CHECK_EVERY, k_max // (workers * BLOCKS_PER_WORKER) + 1)
    blocks = [(N, k, min(k + size, k_max + 1)) for k in range(1, k_max + 1, size)]
    own = pool is None
    if own:
        pool = make_pool(workers)
    try:
        pool.apply(_reset_stop, (None,))  # the flag object is shared by all workers
        total = 2
        found = None
        # blocks are handed out in order; after a hit the rest return immediately
        for factor, ops in pool.imap_unordered(_block_worker, blocks):
            total += ops
            if factor is not None and (found is None or factor < found):
                found = factor
        return found, total
    finally:
        if own:
            pool.close()
            pool.join()


# === The method used in the experiments. ====================================
# F3 is the fastest version in wall time. F4 does fewer divisions, but building
# the sieve on every call costs more than it saves, so it is slower in practice.
optimised_brute_force = f3_wheel


FACTOR_METHODS = {
    "F0": f0_naive,
    "F1": f1_sqrt_bound,
    "F2": f2_skip_even,
    "F3": f3_wheel,
    "F4": f4_primes_only,
    "R": pollard_rho,
}


# ---------------------------------------------------------------------------
# Searching d directly
# ---------------------------------------------------------------------------

def pow_cost(d):
    """Modular multiplications used by square-and-multiply for exponent d."""
    return max(d.bit_length() - 1, 0) + max(bin(d).count("1") - 1, 0)


def attacker_tests(N, e, k=3):
    """The attacker's own test messages (coprime to N) and their encryptions."""
    ms = []
    for m in range(2, N - 1):
        if gcd(m, N) == 1:
            ms.append(m)
            if len(ms) == k:
                break
    return ms, [pow(m, e, N) for m in ms]


# === D0: naive. Every d from 1 up, full pow() for every test message. =========
def d0_naive(N, e):
    ms, cs = attacker_tests(N, e)
    ops = 0
    for d in range(1, N):
        hits = [pow(c, d, N) == m for m, c in zip(ms, cs)]  # no short-circuit on purpose
        ops += pow_cost(d) * len(ms)
        if all(hits):
            return d, ops
    return None, ops


# === D1: + odd d only. e*d = 1 + k*lambda(N) with lambda(N) even, so d is odd. =
def d1_odd_only(N, e):
    ms, cs = attacker_tests(N, e)
    ops = 0
    for d in range(1, N, 2):
        hits = [pow(c, d, N) == m for m, c in zip(ms, cs)]
        ops += pow_cost(d) * len(ms)
        if all(hits):
            return d, ops
    return None, ops


# === D2: + incremental powers. Keep c^d and multiply by c^2 to reach c^(d+2). ==
def d2_incremental(N, e):
    ms, cs = attacker_tests(N, e)
    step = [c * c % N for c in cs]
    cur = list(cs)  # c^1
    ops = len(ms)  # computing the c^2 values
    for d in range(1, N, 2):
        if all([x == m for x, m in zip(cur, ms)]):
            return d, ops
        cur = [x * s % N for x, s in zip(cur, step)]
        ops += len(ms)
    return None, ops


# === D3: + one test message per step, confirm the others only on a hit. =======
def d3_check_one(N, e):
    ms, cs = attacker_tests(N, e)
    m0, c0 = ms[0], cs[0]
    step0 = c0 * c0 % N
    cur = c0
    ops = 1
    for d in range(1, N, 2):
        if cur == m0:
            ops += sum(pow_cost(d) for _ in ms[1:])
            if all(pow(c, d, N) == m for m, c in zip(ms[1:], cs[1:])):
                return d, ops
        cur = cur * step0 % N
        ops += 1
    return None, ops


DSEARCH_METHODS = {
    "D0": d0_naive,
    "D1": d1_odd_only,
    "D2": d2_incremental,
    "D3": d3_check_one,
}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def check_key(key, methods=None, max_naive_bits=16):
    """Run every method on one key; raise AssertionError on a wrong result."""
    N, e = key["N"], key["e"]
    out = {}
    for name, fn in FACTOR_METHODS.items():
        if methods and name not in methods:
            continue
        if name == "F0" and N.bit_length() > max_naive_bits:
            continue
        f, ops = fn(N)
        assert f is not None and 1 < f < N and N % f == 0, (name, N, f)
        out[name] = ops
    for name, fn in DSEARCH_METHODS.items():
        if methods and name not in methods:
            continue
        if N.bit_length() > max_naive_bits:
            continue
        d, ops = fn(N, e)
        assert d is not None, (name, N)
        assert all(pow(c, d, N) == m for m, c in zip(key["messages"], key["ciphertexts"])), (name, N, d)
        out[name] = ops
    return out


def main():
    ap = argparse.ArgumentParser(description="Self-check all attacks on generated keys.")
    ap.add_argument("--bits", type=int, nargs="*", default=[4, 6, 8, 10])
    ap.add_argument("--balanced", type=int, nargs="*", default=[20, 32])
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    sets = [(b, False) for b in args.bits] + [(b, True) for b in args.balanced]
    for bits, bal in sets:
        keys = generate(bits, args.count, seed=0, n_messages=3, balanced=bal)
        totals = {}
        for k in keys:
            for name, ops in check_key(k).items():
                totals[name] = totals.get(name, 0) + ops
            f, _ = parallel_trial_division(k["N"], 4)
            assert f == f3_wheel(k["N"])[0], ("P", k["N"], f)
        avg = "  ".join(f"{n}={t / len(keys):.0f}" for n, t in totals.items())
        print(f"{bits:>3} bits{' balanced' if bal else '         '}  OK  mean ops: {avg}")


if __name__ == "__main__":
    main()
