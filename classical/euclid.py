"""euclid.py: Euclid's algorithm, the classical steps that turn Shor's period into the key.

After Shor finds the period r of a^x mod N:
    p = gcd(a^(r/2) - 1, N)                  Euclid's algorithm        -> factors p, q
    d = e^-1 mod (p-1)(q-1)                  extended Euclid           -> private key d
    m = c^d mod N                            decryption                -> plaintext

Written out by hand (instead of math.gcd / pow(e, -1, phi)) so every step of breaking
a key is visible in our own code.
"""


def gcd(a, b, steps=None):
    """Greatest common divisor by repeated division with remainder.

    If `steps` is a list, every division a = q*b + r is appended to it as (a, q, b, r).
    """
    a, b = abs(a), abs(b)
    while b:
        q, r = divmod(a, b)
        if steps is not None:
            steps.append((a, q, b, r))
        a, b = b, r
    return a


def extended_gcd(a, b):
    """Return (g, x, y) with a*x + b*y = g = gcd(a, b).

    Same divisions as gcd(); the coefficients x, y are updated alongside, so that at
    every step old_r = a*old_x + b*old_y still holds.
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def mod_inverse(e, m):
    """d with e*d = 1 (mod m), from extended Euclid. Raises ValueError if gcd(e, m) != 1."""
    g, x, _ = extended_gcd(e, m)
    if g != 1:
        raise ValueError(f"{e} has no inverse mod {m} (gcd = {g})")
    return x % m


def break_key(p, q, key):
    """Private key and decrypted messages for `key` (N, e, ciphertexts) from its factors."""
    N = key["N"]
    if p * q != N:
        raise ValueError(f"{p} * {q} != {N}")
    phi = (p - 1) * (q - 1)
    d = mod_inverse(key["e"], phi)
    decrypted = [pow(c, d, N) for c in key["ciphertexts"]]
    return d, decrypted


if __name__ == "__main__":
    # worked example for the report: N = 15 = 3 * 5, a = 7, period r = 4
    N, a, r, e = 15, 7, 4, 3
    steps = []
    x = pow(a, r // 2, N)
    p = gcd(x - 1, N, steps)
    print(f"a^(r/2) mod N = {a}^{r // 2} mod {N} = {x}")
    print(f"gcd({x - 1}, {N}):")
    for a_, q_, b_, r_ in steps:
        print(f"    {a_} = {q_} * {b_} + {r_}")
    print(f"factors: p = {p}, q = {N // p}")
    phi = (p - 1) * (N // p - 1)
    g, xx, yy = extended_gcd(e, phi)
    print(f"extended Euclid: {e} * {xx} + {phi} * {yy} = {g}  ->  d = {xx % phi}")
