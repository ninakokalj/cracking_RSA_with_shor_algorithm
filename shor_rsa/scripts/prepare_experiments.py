import json
import random

from pathlib import Path
from math import gcd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# bases to test for each N
BASES_PER_N = 3

# Shor picks a at random, so we do too (fixed seed for reproducibility)
BASE_SEED = 0


def load_keys():

    all_keys = []
    seeds = set()

    # find key files
    files = sorted(DATA_DIR.glob("keys_*bit.json"))

    if not files:
        raise FileNotFoundError("No key files found.")

    for file in files:

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_keys.extend(data["keys"])
        seeds.add(data["seed"])

    if len(seeds) != 1:
        raise ValueError("Files use different seeds.")

    return all_keys, seeds.pop()


def select_bases(N, count, rng):

    # a must be coprime to N (otherwise gcd(a, N) already gives a factor)
    candidates = [a for a in range(2, N - 1) if gcd(a, N) == 1]

    # random choice, not "the first ones": some bases have an odd period or
    # a^(r/2) = -1 mod N and always fail, and that failure rate is part of Shor's cost
    return sorted(rng.sample(candidates, min(count, len(candidates))))


def counting_qubits(N):

    # 2n counting qubits give enough precision for any period (n = bits of N)
    return 2 * (N - 1).bit_length()


def prepare_experiments():

    keys, seed = load_keys()

    # get distinct N values
    unique_N = sorted(set(key["N"] for key in keys))

    experiments = []

    rng = random.Random(BASE_SEED)

    for N in unique_N:

        bases = select_bases(N, BASES_PER_N, rng)

        for a in bases:

            experiments.append({
                "N": N,
                "a": a,
                "counting_qubits": counting_qubits(N)
            })

    # save data
    output = {
        "keygen_seed": seed,
        "base_seed": BASE_SEED,
        "total_keys": len(keys),
        "unique_N": len(unique_N),
        "experiments": experiments
    }

    output_file = DATA_DIR / "experiments.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"RSA keys read: {len(keys)}")
    print(f"Distinct N values: {len(unique_N)}")
    print(f"Experiments prepared: {len(experiments)}")

    print("\nExperiments:")

    for experiment in experiments:
        print(
            f"N = {experiment['N']}, "
            f"a = {experiment['a']}, "
            f"t = {experiment['counting_qubits']}"
        )


if __name__ == "__main__":
    prepare_experiments()