import json

from pathlib import Path
from math import gcd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# bases to test for each N
BASES_PER_N = 2


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


def select_bases(N, count=2):

    bases = []

    for a in range(2, N):

        # a must be coprime to N
        if gcd(a, N) == 1:
            bases.append(a)

        if len(bases) == count:
            break

    return bases


def prepare_experiments():

    keys, seed = load_keys()

    # get distinct N values
    unique_N = sorted(set(key["N"] for key in keys))

    experiments = []

    for N in unique_N:

        bases = select_bases(N, BASES_PER_N)

        for a in bases:

            experiments.append({
                "N": N,
                "a": a
            })

    # save data
    output = {
        "keygen_seed": seed,
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
            f"a = {experiment['a']}"
        )


if __name__ == "__main__":
    prepare_experiments()