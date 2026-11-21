import cmath
from math import gcd

import pytest

from keygen import generate
from .. import bruteforce as bf
from ..postprocess import analyse_counts, uniform_baseline


# =========================================
# Brute force: every version breaks every key
# =========================================

KEY_SETS = [(4, False), (6, False), (10, False), (20, True), (32, True)]


@pytest.mark.parametrize("bits,balanced", KEY_SETS)
def test_factoring_versions(bits, balanced):
    for key in generate(bits, 5, seed=0, n_messages=3, balanced=balanced):
        N = key["N"]
        for name, fn in bf.FACTOR_METHODS.items():
            if name == "F0" and bits > 16:
                continue
            factor, ops = fn(N)
            assert factor in (key["p"], key["q"]), (name, N, factor)
            assert ops > 0


@pytest.mark.parametrize("bits", [4, 6, 10])
def test_dsearch_versions_decrypt(bits):
    for key in generate(bits, 5, seed=0, n_messages=3):
        N = key["N"]
        for name, fn in bf.DSEARCH_METHODS.items():
            d, _ = fn(N, key["e"])
            assert all(pow(c, d, N) == m for m, c in zip(key["messages"], key["ciphertexts"])), name


def test_parallel_matches_serial():
    for key in generate(32, 3, seed=0, n_messages=1, balanced=True):
        assert bf.parallel_trial_division(key["N"], 4)[0] == bf.optimised_brute_force(key["N"])[0]


# =========================================
# Post-processing on the exact ideal distribution
# =========================================

def ideal_counts(a, N, t, shots=4096):
    r = next(r for r in range(1, N) if pow(a, r, N) == 1)
    Q = 2**t
    counts = {}
    for y in range(Q):
        p = sum(abs(sum(cmath.exp(2j * cmath.pi * k * (s / r - y / Q)) for k in range(Q)) / Q) ** 2
                for s in range(r)) / r
        if round(p * shots):
            counts[format(y, f"0{t}b")] = round(p * shots)
    return counts


@pytest.mark.parametrize("N,a,expected", [(15, 7, 0.75), (15, 2, 0.75), (21, 4, 0.0)])
def test_postprocess_ideal(N, a, expected):
    keys = generate(N.bit_length(), 3, seed=0, n_messages=3)
    t = 2 * N.bit_length()
    rec = {"N": N, "a": a, "counting_qubits": t, "shots": 4096, "counts": ideal_counts(a, N, t)}
    assert analyse_counts(rec, keys)["p_s"] == pytest.approx(expected, abs=0.01)


def test_postprocess_flags_trivial_base():
    keys = generate(4, 1, seed=0, n_messages=3)
    rec = {"N": 15, "a": 6, "counting_qubits": 4, "shots": 1, "counts": {"0000": 1}}
    assert gcd(6, 15) > 1
    assert analyse_counts(rec, keys)["trivial_gcd"]


def test_uniform_baseline_is_not_zero_for_small_n():
    keys = generate(4, 1, seed=0, n_messages=3)
    assert uniform_baseline(8, 2, 15, keys) > 0.1


# =========================================
# Euclid: the classical steps that break the key
# =========================================

def test_euclid_matches_python():
    import random
    from math import gcd as math_gcd
    from ..euclid import extended_gcd, gcd, mod_inverse
    rng = random.Random(0)
    for _ in range(2000):
        a, b = rng.randrange(1, 10**9), rng.randrange(1, 10**9)
        assert gcd(a, b) == math_gcd(a, b)
        g, x, y = extended_gcd(a, b)
        assert a * x + b * y == g == math_gcd(a, b)
        if g == 1:
            assert mod_inverse(a, b) == pow(a, -1, b)


def test_break_key_decrypts_every_key():
    from ..euclid import break_key
    for bits in (4, 5, 6, 10):
        for key in generate(bits, 10, seed=0, n_messages=3):
            d, decrypted = break_key(key["p"], key["q"], key)
            assert decrypted == key["messages"]


@pytest.mark.parametrize("bits,balanced", [(4, False), (5, False), (6, False), (12, False), (32, True)])
def test_keys_really_encrypt(bits, balanced):
    for key in generate(bits, 10, seed=0, n_messages=3, balanced=balanced):
        lam = (key["p"] - 1) * (key["q"] - 1) // gcd(key["p"] - 1, key["q"] - 1)
        assert key["e"] % lam != 1  # otherwise m^e = m for every message
        assert all(c != m for m, c in zip(key["messages"], key["ciphertexts"]))


def test_ideal_p_s_matches_known_values():
    from ..postprocess import ideal_p_s
    keys15 = generate(4, 3, seed=0, n_messages=3)
    assert ideal_p_s(15, 7, 4, keys15) == pytest.approx(0.75)   # period 4 divides 2^4: exact
    assert ideal_p_s(15, 7, 8, keys15) == pytest.approx(0.75)
    keys21 = generate(5, 3, seed=0, n_messages=3)
    assert ideal_p_s(21, 4, 10, keys21) == pytest.approx(0.0)   # odd period: never works
