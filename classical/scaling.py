"""scaling.py: scaling experiments and the classical vs quantum comparison.

Two questions, one shared unit (seconds to break a key) on one axis (bits of N):
    1. How does the cost grow with key size?
    2. How does it change with more compute power?
       classical: CPU cores (trial division split across processes)
       quantum:   gate error rate (Nina's noise sweep); qubit count only decides which keys fit

Subcommands (each writes classical/results/<name>.csv; `plot` writes classical/results/plots/*.pdf)
    size     time and ops of the optimised brute force on balanced keys, plus a log2-time fit
    cores    parallel brute force speedup for 1 .. 12 workers (52-bit keys by default)
    quantum  p_s and estimated QPU seconds from the Shor runs in shor_rsa/results,
             plus the fit p_s = b + (p0 - b) * exp(-k * eps * G) to the noise sweep
    theory   time to break a key vs key size: brute force (measured, fitted, extrapolated)
             against Shor (time per attempt compiled for IBM Brisbane, x expected attempts,
             fitted as a power of n); crossover; error rate Shor needs at each key size;
             GNFS as a literature reference line (not measured)
    plot     all figures
    all      everything above in order

Classical side = bruteforce.optimised_brute_force only. The other versions in
bruteforce.py document the optimisation steps and are not part of the results.
Shor's time per attempt comes from shor_rsa/circuit_analysis/time_per_attempt.py, which
must run first. It assumes error-free gates: the noise fit says what error rate that needs.

Examples
    .venv/bin/python -m classical.scaling all --quick
    .venv/bin/python -m classical.scaling size --bits 8 16 24 32 40 48 52
    .venv/bin/python -m classical.scaling quantum --records "shor_rsa/results/ideal/*.json"

Measured vs extrapolated: `theory` curves past the measured range are fits, labelled as
such. GNFS is calibrated to the RSA-250 record (829 bits, about 2700 core-years, 2020).
"""

import argparse
import csv
import glob
import json
import math
import os
import time
from statistics import median
from pathlib import Path

from keygen import generate
from . import bruteforce as bf
from .postprocess import analyse_counts, estimated_qpu_seconds, ideal_p_s, load_keys, KEY_FILES

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS = str(ROOT_DIR / "classical" / "results")
PLOTS = str(ROOT_DIR / "classical" / "results" / "plots")
SHOR_RESULTS = ROOT_DIR / "shor_rsa" / "results"
# Shor runs read by `quantum` (the batch folders hold the repeated runs with seeds)
RECORD_GLOBS = [SHOR_RESULTS / "ideal" / "*.json",
                SHOR_RESULTS / "noisy" / "batch" / "*.json",
                SHOR_RESULTS / "hardware" / "*.json"]
YEAR = 365.25 * 24 * 3600
GNFS_C = (64 / 9) ** (1 / 3)
GNFS_REF = (829, 2700 * YEAR)  # RSA-250: bits, core-seconds
TIME_PER_ATTEMPT = SHOR_RESULTS / "circuit_analysis" / "time_per_attempt.csv"
SUPERCOMPUTER_CORES = 1e6
GNFS_MIN_BITS = 256

CLASSICAL = bf.optimised_brute_force
CLASSICAL_NAME = "brute force"


# --------------------------------------------------------------------------- helpers

def time_call(fn, *args, min_total=0.05, min_runs=3, max_runs=1000):
    """Median wall time of fn(*args): repeat until min_total seconds or max_runs runs."""
    times, total = [], 0.0
    while (len(times) < min_runs or total < min_total) and len(times) < max_runs:
        t0 = time.perf_counter_ns()
        fn(*args)
        dt = (time.perf_counter_ns() - t0) / 1e9
        times.append(dt)
        total += dt
    return median(times)


def write_csv(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def read_csv(name):
    path = os.path.join(RESULTS, f"{name}.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def linfit(xs, ys):
    """Least squares y = slope * x + intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


# --------------------------------------------------------------------------- size

def run_size(bits_list, keys_per_bits, seed, fit_from):
    rows = []
    for bits in bits_list:
        keys = generate(bits, keys_per_bits, seed, n_messages=3, balanced=True)
        times = [time_call(CLASSICAL, k["N"]) for k in keys]
        ops = [CLASSICAL(k["N"])[1] for k in keys]
        rows.append({"bits": bits, "keys": len(keys), "median_time_s": median(times),
                     "mean_ops": sum(ops) / len(ops)})
        print(f"  {bits:>3} bits  {median(times):.3e} s")
    write_csv("size", rows)

    pts = [(int(r["bits"]), math.log2(float(r["median_time_s"])))
           for r in rows if int(r["bits"]) >= fit_from]
    if len(pts) >= 2:
        slope, icpt = linfit(*zip(*pts))
        write_csv("fit", [{"slope_log2_per_bit": slope, "intercept": icpt,
                           "from_bits": fit_from, "points": len(pts)}])
        print(f"  fit: time doubles every {1 / slope:.2f} bits (theory: 2)")


# --------------------------------------------------------------------------- cores

def _spin(n):
    """Pure CPU work with no coordination, to measure what the hardware itself allows."""
    x = 0
    for i in range(n):
        x += i % 7
    return x


def hardware_ceiling(pool, w, work=30_000_000):
    """Time for a fixed amount of perfectly parallel work on w processes."""
    t0 = time.perf_counter()
    pool.map(_spin, [work // (w * 8)] * (w * 8), chunksize=1)
    return time.perf_counter() - t0


def run_cores(bits, keys_count, seed, workers_list):
    """Brute-force speedup vs CPU cores, next to the speedup the hardware allows at all.

    Laptops throttle under sustained load and mix fast and slow cores, so the raw speedup
    depends on the machine's state; brute force / hardware ceiling does not.
    """
    if 1 not in workers_list:
        raise ValueError("the worker list must include 1 (the baseline for the speedup)")
    keys = generate(bits, keys_count, seed, n_messages=1, balanced=True)
    times = {}
    # measured in the given order (all cores first by default), speedups computed after
    for w in workers_list:
        with bf.make_pool(w) as pool:  # start-up kept out of the timing
            pool.map(abs, range(w))
            hw = hardware_ceiling(pool, w)
            t = median(time_call(bf.parallel_trial_division, k["N"], w, pool, min_total=0)
                       for k in keys)
        times[w] = (t, hw)
        print(f"  {w:>2} workers  {t:.3f} s  (independent work: {hw:.3f} s)", flush=True)
    base, hw_base = times[1]
    rows = []
    for w in sorted(times):
        t, hw = times[w]
        rows.append({"bits": bits, "workers": w, "median_time_s": t,
                     "speedup": base / t, "efficiency": base / t / w,
                     "hardware_speedup": hw_base / hw,
                     "share_of_hardware": (base / t) / (hw_base / hw),
                     "measured_order": workers_list.index(w) + 1})
        print(f"  {w:>2} workers: speedup {base / t:.2f}  (hardware allows {hw_base / hw:.2f})")
    write_csv("cores", rows)


# --------------------------------------------------------------------------- quantum

def noise_label(rec):
    """Two-qubit error rate as a number, or the backend name for a device noise model."""
    if "cx_error_probability" in rec:
        return rec["cx_error_probability"]
    if rec.get("simulator") == "ideal" or rec.get("experiment") == "ideal":
        return 0
    return rec.get("experiment") or rec.get("backend")


def run_quantum(key_glob, record_globs):
    keys = load_keys(key_glob)
    paths = sorted(p for g in record_globs for p in glob.glob(g))
    if not paths:
        print("no quantum records found; skipping")
        return
    rows = []
    for path in paths:
        rec = json.load(open(path))
        res = analyse_counts(rec, keys)
        if res["trivial_gcd"]:
            continue
        c = rec.get("circuit", {})
        rows.append({"file": os.path.basename(path), "N": rec["N"], "bits": rec["N"].bit_length(),
                     "a": rec["a"], "t": res["t"], "multiplier": rec.get("multiplier", "matrix"),
                     "backend": rec.get("backend", rec.get("simulator")),
                     "noise_level": noise_label(rec),
                     "shots": res["shots"], "p_s": res["p_s"], "p_s_uniform": res["p_s_uniform"],
                     "qubits": c.get("qubits"), "depth": c.get("depth"),
                     "two_qubit_gates": c.get("two_qubit_gates", rec.get("ecr_gates")),
                     "est_qpu_s": estimated_qpu_seconds(rec, res["p_s"]), "job_id": rec.get("job_id")})
        # error rate x gate count, and how much of the ideal signal (above pure noise) is kept
        eps = rec.get("cx_error_probability", c.get("median_two_qubit_error"))
        G = rows[-1]["two_qubit_gates"]
        p_id = ideal_p_s(rec["N"], rec["a"], res["t"], keys) if res["t"] <= 12 else None
        b = res["p_s_uniform"]
        rows[-1].update(
            eps=eps if eps is not None else "",
            eps_G=eps * G if eps is not None and G else "",
            p_ideal=p_id if p_id is not None else "",
            signal_kept=(res["p_s"] - b) / (p_id - b) if p_id is not None and b is not None
            and p_id - b > 0.05 else "")
    write_csv("quantum", rows)
    fit_noise(rows)


def fit_noise(rows):
    """Fit p_s = b + (p0 - b) * exp(-k * eps * G) to each noise sweep (same N, a, multiplier).

    b = pure-noise baseline, p0 = p_s without noise, G = two-qubit gates. k is the share of
    gate errors that actually ruin a shot (k = 1: every error does).
    """
    sweeps = {}
    for r in rows:
        if _is_float(r["noise_level"]) and r["two_qubit_gates"]:
            sweeps.setdefault((r["N"], r["a"], r["t"], r["multiplier"]), []).append(r)
    fits = []
    for (N, a, t, mult), rs in sweeps.items():
        levels = sorted({float(r["noise_level"]) for r in rs})
        if len(levels) < 3 or levels[0] != 0:
            continue
        mean = {e: sum(r["p_s"] for r in rs if float(r["noise_level"]) == e) /
                   sum(1 for r in rs if float(r["noise_level"]) == e) for e in levels}
        b, p0, G = float(rs[0]["p_s_uniform"]), mean[0.0], int(rs[0]["two_qubit_gates"])
        # use points clearly above the baseline (log of a near-zero difference is noise)
        pts = [(e * G, math.log((mean[e] - b) / (p0 - b))) for e in levels[1:]
               if mean[e] - b > 0.1 * (p0 - b)]
        if not pts:
            continue
        k = -sum(x * y for x, y in pts) / sum(x * x for x, _ in pts)  # line through origin
        fits.append({"N": N, "a": a, "t": t, "multiplier": mult, "two_qubit_gates": G,
                     "p0": p0, "baseline": b, "k": k, "points_used": len(pts),
                     "eps_half_signal": math.log(2) / (k * G)})
        print(f"  noise fit N={N} a={a}: k = {k:.3f}; signal halves at eps = "
              f"{math.log(2) / (k * G):.2e} (G = {G} two-qubit gates)")
    if fits:
        write_csv("noise_fit", fits)


# --------------------------------------------------------------------------- theory

def gnfs_seconds(bits, cores=1):
    def L(n):
        ln_n = n * math.log(2)
        return math.exp(GNFS_C * ln_n ** (1 / 3) * math.log(ln_n) ** (2 / 3))
    return GNFS_REF[1] * L(bits) / L(GNFS_REF[0]) / cores


def shor_model():
    """Shor time to break a key at n bits: power-law fit of compiled time per attempt
    (gate part) + fixed per-shot overhead, times the expected attempts 1 / mean p_s."""
    if not TIME_PER_ATTEMPT.exists():
        print(f"  {TIME_PER_ATTEMPT.name} missing: run shor_rsa.circuit_analysis.time_per_attempt")
        return None
    rows = list(csv.DictReader(open(TIME_PER_ATTEMPT)))
    ideal = [r for r in read_csv("quantum")
             if r["noise_level"] == "0" and r["multiplier"] == "binary" and r["backend"] == "ideal"]
    # random base per attempt: the average success per shot over the random bases we ran
    p_mean = sum(float(r["p_s"]) for r in ideal) / len(ideal) if ideal else 1.0
    overhead = {int(r["n"]): float(r["attempt_duration_s_est"]) - int(r["counting_qubits"])
                * float(r["mult_duration_s"]) for r in rows}
    fixed = sum(overhead.values()) / len(overhead)
    xs = [math.log(int(r["n"])) for r in rows]
    ys = [math.log(int(r["counting_qubits"]) * float(r["mult_duration_s"])) for r in rows]
    k, c = linfit(xs, ys)
    gk, gc = linfit(xs, [math.log(float(r["attempt_two_qubit_gates_est"])) for r in rows])
    return {"k": k, "c": c, "fixed_s": fixed, "p_mean": p_mean, "gates_k": gk, "gates_c": gc,
            "measured": [(int(r["n"]), float(r["attempt_duration_s_est"]) / p_mean) for r in rows],
            "max_n": max(int(r["n"]) for r in rows)}


def run_theory():
    fit = read_csv("fit")
    shor = shor_model()
    noise = read_csv("noise_fit")
    k_noise = float(noise[0]["k"]) if noise else None
    bits_axis = sorted({*range(4, 257), *range(256, 4097, 4)})  # 1-bit steps where crossovers lie
    rows = []
    for n in bits_axis:
        gnfs_ok = n >= GNFS_MIN_BITS  # below this GNFS is not used in practice, model meaningless
        row = {"bits": n,
               "gnfs_1core_s": gnfs_seconds(n) if gnfs_ok else "",
               "gnfs_supercomputer_s": gnfs_seconds(n, SUPERCOMPUTER_CORES) if gnfs_ok else "",
               "bruteforce_fit_s": "", "shor_fit_s": "", "shor_two_qubit_gates": "",
               "shor_required_eps": ""}
        if fit:
            exp2 = float(fit[0]["slope_log2_per_bit"]) * n + float(fit[0]["intercept"])
            row["bruteforce_fit_s"] = 2 ** exp2 if exp2 < 1000 else math.inf
        if shor:
            attempt = math.exp(shor["c"]) * n ** shor["k"] + shor["fixed_s"]
            row["shor_fit_s"] = attempt / shor["p_mean"]
            gates = math.exp(shor["gates_c"]) * n ** shor["gates_k"]
            row["shor_two_qubit_gates"] = gates
            if k_noise:
                # gate error rate at which half of the quantum signal survives
                row["shor_required_eps"] = math.log(2) / (k_noise * gates)
        rows.append(row)
    write_csv("theory", rows)

    if shor:
        print(f"  Shor time per attempt grows like n^{shor['k']:.2f} "
              f"(expected attempts {1 / shor['p_mean']:.2f}, error-free gates assumed)")
        for col in ("bruteforce_fit_s", "gnfs_1core_s", "gnfs_supercomputer_s"):
            cross = next((r["bits"] for r in rows if _is_float(r[col]) and _is_float(r["shor_fit_s"])
                          and float(r[col]) > float(r["shor_fit_s"])), None)
            print(f"  Shor overtakes {col[:-2]} at about {cross} bits" if cross
                  else f"  no crossover with {col[:-2]} up to 4096 bits")


# --------------------------------------------------------------------------- plots

def run_plot():
    """Figures for the report (Slovenian labels, decimal commas, sized for half a page width)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 7, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    thin = lambda x: f"{int(x):,}".replace(",", " ")  # thousands with a narrow space
    small = (3.3, 2.6)
    out = PLOTS
    os.makedirs(out, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out, name))
        plt.close(fig)

    size = read_csv("size")
    if size:
        fig, ax = plt.subplots(figsize=small)
        xs = [int(r["bits"]) for r in size]
        ax.plot(xs, [float(r["median_time_s"]) for r in size], "o", ms=4, label="izmerjeno")
        fit = read_csv("fit")
        if fit:
            f = fit[0]
            ax.plot(xs, [2 ** (float(f["slope_log2_per_bit"]) * x + float(f["intercept"])) for x in xs],
                    "--", label=f"prilagoditev: ×2 vsakih {1 / float(f['slope_log2_per_bit']):.2f} bita".replace(".", ","))
        ax.set(yscale="log", xlabel="število bitov $n$", ylabel="čas za razbitje [s]")
        ax.legend(loc="upper left")
        save(fig, "size.pdf")

    cores = read_csv("cores")
    if cores:
        w = [int(r["workers"]) for r in cores]
        fig, ax = plt.subplots(figsize=small)
        ax.plot(w, [float(r["speedup"]) for r in cores], marker="o", ms=4, label="groba sila")
        if "hardware_speedup" in cores[0]:
            ax.plot(w, [float(r["hardware_speedup"]) for r in cores], marker="s", ms=3,
                    label="neodvisno breme")
        ax.plot(w, w, ls="--", c="gray", label="linearno")
        ax.set(xlabel="število procesov", ylabel="pohitritev")
        ax.yaxis.set_major_formatter(comma)
        ax.legend(loc="upper left")
        save(fig, "cores.pdf")

    quantum = read_csv("quantum")
    numeric = [r for r in quantum if _is_float(r["noise_level"])]
    # only noise sweeps (several error rates for the same N, a), not single ideal runs
    groups = {}
    for r in numeric:
        groups.setdefault((r["N"], r["a"], r["t"]), []).append(r)
    sweeps = {k: v for k, v in groups.items() if len({r["noise_level"] for r in v}) > 2}
    nf_all = read_csv("noise_fit")
    if sweeps:
        fig, ax = plt.subplots(figsize=small)
        for (N, a, t), rs in sweeps.items():
            levels = sorted({float(r["noise_level"]) for r in rs})
            vals = [[float(r["p_s"]) for r in rs if float(r["noise_level"]) == e] for e in levels]
            means = [sum(v) / len(v) for v in vals]
            stds = [(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
                    for v, m in zip(vals, means)]
            line = ax.errorbar(levels, means, yerr=stds, marker="o", ms=4, capsize=3, ls="none",
                               label="simulacija").lines[0]
            nf = next((f for f in nf_all if (f["N"], f["a"]) == (N, a)), None)
            if nf:
                G, k, b, p0 = (float(nf[x]) for x in ("two_qubit_gates", "k", "baseline", "p0"))
                es = [0] + [10 ** (x / 20) for x in range(-120, -39)]
                ax.plot(es, [b + (p0 - b) * math.exp(-k * e * G) for e in es], ls="--",
                        c=line.get_color(), lw=1,
                        label=f"prilagoditev, $\\kappa$ = {k:.2f}".replace(".", ","))
            base = rs[0]["p_s_uniform"]
            if base:
                ax.axhline(float(base), ls=":", c="gray", label="čisti šum")
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(left=0)
        ax.set(xlabel="napaka dvokubitnih vrat $\\varepsilon$", ylabel="verjetnost uspeha $p_s$")
        ax.yaxis.set_major_formatter(comma)
        ax.legend(loc="lower left")
        save(fig, "quantum_noise.pdf")

    kept = [r for r in quantum if _is_float(r.get("eps_G")) and float(r["eps_G"]) > 0
            and _is_float(r.get("signal_kept"))]
    if kept:
        fig, ax = plt.subplots(figsize=small)
        sim = [r for r in kept if "ibm" not in str(r["backend"])]
        hw = [r for r in kept if r not in sim]

        def sci(e):
            # 0.00185 -> $1{,}85 \\cdot 10^{-3}$, 1e-05 -> $10^{-5}$ (Slovenian decimal comma)
            exp = math.floor(math.log10(e) + 1e-9)
            m = round(e / 10 ** exp, 2)
            mant = "" if m == 1 else f"{m:g}".replace(".", "{,}") + " \\cdot "
            return f"${mant}10^{{{exp}}}$"

        if sim:
            ax.scatter([float(r["eps_G"]) for r in sim], [float(r["signal_kept"]) for r in sim],
                       s=12, label="simulacija")
        if hw:
            ax.scatter([float(r["eps_G"]) for r in hw], [float(r["signal_kept"]) for r in hw],
                       s=70, marker="*", c="C1", zorder=3, label=hw[0]["backend"])
        # one label per circuit (repeated runs sit on top of each other), with its eps and G:
        # eps * G is the x axis
        for mult in sorted({r["multiplier"] for r in hw}):
            rs = [r for r in hw if r["multiplier"] == mult]
            x = sum(float(r["eps_G"]) for r in rs) / len(rs)
            y = sum(float(r["signal_kept"]) for r in rs) / len(rs)
            name = "binarno" if mult == "binary" else "matrično"
            runs = f", {len(rs)} meritev" if len(rs) > 1 else ""
            eps = sum(float(r["eps"]) for r in rs) / len(rs)
            text_xy = (95, 0.6) if x >= 10 else (0.035, 0.42)
            ax.annotate(f"{name}{runs}\n$\\varepsilon$ = {sci(eps)}\n"
                        f"$G$ = {thin(rs[0]['two_qubit_gates'])}", (x, y), xytext=text_xy,
                        ha="right" if x >= 10 else "left", va="center", fontsize="small",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
        if nf_all:
            k = float(nf_all[0]["k"])
            xs = [10 ** (x / 20) for x in range(-30, 41)]
            ax.plot(xs, [math.exp(-k * x) for x in xs], ls="--", c="gray",
                    label=f"$e^{{-\\kappa \\varepsilon G}}$, $\\kappa$ = {k:.2f}".replace(".", ","))
        ax.axhline(0, c="k", lw=0.5)
        ax.set(xscale="log", xlabel="$\\varepsilon G$", ylabel="delež ohranjenega signala $S$")
        ax.yaxis.set_major_formatter(comma)
        ax.legend(loc="lower left")
        save(fig, "signal_vs_epsG.pdf")

    theory = read_csv("theory")
    if theory:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        styles = {"bruteforce_fit_s": ("groba sila (ekstrapolacija)", "--"),
                  "shor_fit_s": ("Shor, naša izvedba, vrata brez napak (ekstrapolacija)", "--"),
                  "gnfs_1core_s": ("GNFS, 1 jedro", ":"),
                  "gnfs_supercomputer_s": ("GNFS, $10^6$ jeder", ":")}
        for col, (label, ls) in styles.items():
            pts = [(int(r["bits"]), float(r[col])) for r in theory
                   if _is_float(r.get(col)) and float(r[col]) < 1e45]
            if pts:
                ax.plot(*zip(*pts), ls=ls, label=label)
        if size:
            ax.scatter([int(r["bits"]) for r in size], [float(r["median_time_s"]) for r in size],
                       s=10, label="groba sila (izmerjeno)")
        shor = shor_model()
        if shor:
            ax.scatter(*zip(*shor["measured"]), s=14, marker="x", c="k",
                       label="Shor (ocena iz prevedenih vezij)")
        ax.set(xscale="log", yscale="log", xlabel="število bitov $n$", ylabel="čas za razbitje [s]",
               ylim=(1e-7, 1e40))
        ax.axhline(YEAR * 1.38e10, c="gray", lw=0.5)
        ax.text(5, YEAR * 1.38e10 * 3, "starost vesolja", fontsize=7, color="gray")
        # real key sizes: RSA-250 (the measured GNFS point the model is calibrated on),
        # and today's 2048- and 4096-bit keys
        for bits, name in ((GNFS_REF[0], "RSA-250"), (2048, "2048"), (4096, "4096")):
            ax.axvline(bits, c="gray", lw=0.5, ls=":")
            ax.text(bits * 0.93, 3e37, name, fontsize=6.5, color="gray", rotation=90,
                    ha="right", va="top")
        ax.scatter([GNFS_REF[0]], [GNFS_REF[1]], s=22, marker="D", c="C2", zorder=3,
                   label="RSA-250 (izmerjeno, GNFS)")
        # Shor at today's key sizes (the times are given in the report's caption)
        by_bits = {int(r["bits"]): r for r in theory}
        marks = [(b, float(by_bits[b]["shor_fit_s"])) for b in (2048, 4096)
                 if b in by_bits and _is_float(by_bits[b].get("shor_fit_s"))]
        if marks:
            ax.scatter(*zip(*marks), s=14, c="C1", zorder=3)
        ax.legend(loc="upper left", fontsize=6.5)
        save(fig, "crossover.pdf")
    print(f"plots in {out}/")


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["size", "cores", "quantum", "theory", "plot", "all"])
    ap.add_argument("--quick", action="store_true", help="small settings for a fast test run")
    ap.add_argument("--bits", type=int, nargs="*", help="size: key sizes (default 8..52 step 4)")
    ap.add_argument("--keys-per-bits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-from", type=int, default=16, help="ignore smaller keys in fits (overhead)")
    ap.add_argument("--cores-bits", type=int, default=52)
    ap.add_argument("--workers", type=int, nargs="*", default=[12, 8, 4, 2, 1],
                    help="measured in this order (default: all cores first)")
    ap.add_argument("--keys", default=KEY_FILES, help="glob of key files")
    ap.add_argument("--records", nargs="*", default=[str(g) for g in RECORD_GLOBS])
    a = ap.parse_args()

    bits = a.bits or (list(range(8, 25, 4)) if a.quick else list(range(8, 53, 4)))
    cores_bits = 40 if a.quick else a.cores_bits
    workers = [w for w in a.workers if w <= (os.cpu_count() or 1)]

    if a.what in ("size", "all"):
        run_size(bits, 2 if a.quick else a.keys_per_bits, a.seed, a.fit_from)
    if a.what in ("cores", "all"):
        run_cores(cores_bits, 1 if a.quick else 3, a.seed, workers)
    if a.what in ("quantum", "all"):
        run_quantum(a.keys, a.records)
    if a.what in ("theory", "all"):
        run_theory()
    if a.what in ("plot", "all"):
        run_plot()


if __name__ == "__main__":
    main()
