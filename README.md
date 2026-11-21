# Cracking simplified RSA with Shor's algorithm

Seminar project (group 29, Kvantni celični avtomati in kvantno procesiranje, 2026/27):
Luka Mikić, Nina Kokalj.

We break simplified RSA keys (small primes) with Shor's algorithm in Qiskit and with an
optimised classical brute-force attack, and compare how both scale:

- **with key size:** classical time grows exponentially; Shor's time per attempt grows with
  the circuit's gate count;
- **with compute power:** classical gets at best a linear speedup from more CPU cores; Shor's
  number of attempts depends on the gate error rate, and qubit count only decides which keys fit.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # all tests
```

Run every command from the repository root; scripts are run as modules (`python -m ...`).

## Layout

| Path | Contents |
|---|---|
| `keygen.py` | key generator; difficulty = bit length of N (see its docstring) |
| `data/` | generated keys (`keys_*bit.json`, seed 0) and the list of Shor experiments |
| `shor_rsa/shor_circuit.py` | Shor circuit, uses the binary multiplier (`multiplier="binary"`, default) |
| `shor_rsa/arithmetic/` | modular multipliers: binary adders (`modular_multiplication_clean.py`, gates grow like n^3.6) and a general matrix (`modular_gate.py`, short only for small N) |
| `shor_rsa/scripts/` | prepare (random bases, t = 2n) and run the ideal experiments |
| `shor_rsa/experiments/noise/run_noisy.py` | error-rate sweep with depolarizing noise |
| `shor_rsa/experiments/hardware/run_ibm.py` | runs Shor on a real IBM processor (see below) |
| `shor_rsa/circuit_analysis/` | gate counts and time per attempt vs key size |
| `shor_rsa/results/` | one JSON per Shor run (ideal, noisy, hardware) and circuit analysis outputs |
| `classical/bruteforce.py` | brute-force attacks; the optimisation steps are numbered in the code |
| `classical/euclid.py` | Euclid and extended Euclid: period → factors → private key |
| `classical/postprocess.py` | measured counts → period → factors → private keys, success per shot |
| `classical/break_keys.py` | breaks every key from the Shor runs and decrypts its messages |
| `classical/scaling.py` | classical scaling experiments, comparison, noise fit and plots (`classical/results/`) |
| `report/main.tex` | report chapter (Slovenian, LaTeX); figures come from `classical/results/plots/` |

## Reproducing the results

```bash
# keys (already in data/, regenerate to check)
.venv/bin/python keygen.py 4 --count 5 --out data/keys_4bit.json
.venv/bin/python keygen.py 5 --count 5 --out data/keys_5bit.json
.venv/bin/python keygen.py 6 --count 10 --out data/keys_6bit.json

# Shor (about 2 hours in total on a laptop)
.venv/bin/python -m shor_rsa.scripts.prepare_experiments          # data/experiments.json
.venv/bin/python -m shor_rsa.scripts.run_ideal                    # shor_rsa/results/ideal/
.venv/bin/python -m shor_rsa.experiments.noise.run_noisy          # shor_rsa/results/noisy/batch/
.venv/bin/python -m shor_rsa.circuit_analysis.gate_scaling        # gates per multiplication vs key size
.venv/bin/python -m shor_rsa.circuit_analysis.time_per_attempt    # time per attempt on IBM Brisbane

# classical side and comparison
.venv/bin/python -m classical.scaling size         # brute force time vs key size (8-52 bits)
.venv/bin/python -m classical.scaling cores        # brute force speedup vs CPU cores
.venv/bin/python -m classical.scaling quantum      # success per shot of every Shor run + noise fit
.venv/bin/python -m classical.scaling theory       # time to break vs key size, crossover
.venv/bin/python -m classical.scaling plot         # classical/results/plots/*.pdf
.venv/bin/python -m classical.break_keys           # break and decrypt every key
```

`python -m classical.scaling all --quick` does a fast test run with small keys.

## Running on a real IBM quantum processor

Needs an IBM Quantum account (the free Open Plan is enough). The API key is stored in
`~/.qiskit`, never in the repository.

```bash
.venv/bin/python -m shor_rsa.experiments.hardware.run_ibm login --token <API key>
.venv/bin/python -m shor_rsa.experiments.hardware.run_ibm submit --N 15 --a 7 --t 4 --shots 4096
.venv/bin/python -m shor_rsa.experiments.hardware.run_ibm fetch    # later, once the job is done
```

The job id is saved as soon as the job is submitted (`shor_rsa/results/hardware/pending/`),
so nothing is lost while it waits in the queue. `submit --fake` checks the same pipeline
locally: it compiles for a saved snapshot of IBM Brisbane and simulates without noise.

## Conventions

- Shor results: one JSON file per run `(N, a, counting qubits)`; a run attacks every key
  with that N. Measured bitstrings are read as `y = int(bitstring, 2)`.
- A Shor run only shows quantum signal if its success per shot beats the pure-noise
  baseline that `classical/postprocess.py` reports next to it.
