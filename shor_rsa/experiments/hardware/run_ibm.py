"""Run Shor circuits on a real IBM quantum processor (or a local fake one).

Three steps, because a job can wait in IBM's queue for minutes to hours:

    # 1. once per machine: store the API key locally (in ~/.qiskit, never in the repo)
    python -m shor_rsa.experiments.hardware.run_ibm login --token <API key> [--instance <CRN>]

    # 2. compile for the processor and submit; the job id is saved right away
    python -m shor_rsa.experiments.hardware.run_ibm submit --N 15 --a 7 --t 4 [--shots 4096]
                                                           [--backend ibm_xxx]

    # 3. later: collect finished jobs into result records
    python -m shor_rsa.experiments.hardware.run_ibm fetch

    # dry run of the whole pipeline: compiled for FakeBrisbane (a saved snapshot of the
    # real chip), then simulated locally WITHOUT noise. Simulating Brisbane's full noise
    # model for this circuit takes about a minute per shot; noise is studied in
    # experiments/noise/run_noisy.py instead.
    python -m shor_rsa.experiments.hardware.run_ibm submit --N 15 --a 7 --t 4 --fake --shots 1024

Pending jobs:  shor_rsa/results/hardware/pending/<job id>.json
Results:       shor_rsa/results/hardware/N{N}_a{a}_t{t}_{backend}_{job id}.json
The result records use the same format as the simulator runs, so classical.postprocess
and classical.break_keys work on them unchanged.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit.transpiler import generate_preset_pass_manager

from ...shor_circuit import create_shor_circuit

ROOT_DIR = Path(__file__).resolve().parents[2]  # shor_rsa
RESULTS_DIR = ROOT_DIR / "results" / "hardware"
PENDING_DIR = RESULTS_DIR / "pending"

OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 0
TWO_QUBIT_GATES = ("ecr", "cz", "cx")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def circuit_metrics(isa, backend):
    """Size and estimated duration of the compiled (ISA) circuit on this backend."""
    ops = isa.count_ops()
    try:
        duration = isa.estimate_duration(backend.target, unit="s")
    except Exception:  # some targets have no gate durations
        duration = None
    config = backend.configuration() if hasattr(backend, "configuration") else None
    # median calibrated error of the processor's two-qubit gate, for signal vs eps * G
    gate = next((g for g in TWO_QUBIT_GATES if g in backend.target.operation_names), None)
    errors = sorted(p.error for p in backend.target[gate].values()
                    if p is not None and p.error is not None and p.error < 0.5) if gate else []
    return {
        "median_two_qubit_error": errors[len(errors) // 2] if errors else None,
        "qubits": len({isa.find_bit(q).index for inst in isa.data for q in inst.qubits}),
        "depth": isa.depth(),
        "two_qubit_gates": sum(ops.get(g, 0) for g in TWO_QUBIT_GATES),
        "operations": dict(ops),
        "duration_s": duration,
        "rep_delay_s": getattr(config, "default_rep_delay", None),
    }


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()  # uses the account saved by `login`


def login(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", token=args.token, instance=args.instance,
        overwrite=True, set_as_default=True,
    )
    service = QiskitRuntimeService()
    names = [b.name for b in service.backends(operational=True, simulator=False)]
    print(f"Account saved. Available processors: {', '.join(names) or 'none'}")


def submit(args):
    qc = create_shor_circuit(args.N, args.a, args.t, multiplier=args.multiplier)

    if args.fake:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        backend = FakeBrisbane()
    else:
        service = get_service()
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(min_num_qubits=qc.num_qubits, operational=True,
                                           simulator=False))
    print(f"Processor: {backend.name}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=OPTIMIZATION_LEVEL,
                                      seed_transpiler=SEED_TRANSPILER)
    isa = pm.run(qc)
    metrics = circuit_metrics(isa, backend)
    print(f"Compiled: {metrics['qubits']} qubits, depth {metrics['depth']}, "
          f"{metrics['two_qubit_gates']} two-qubit gates, "
          f"estimated {metrics['duration_s']} s per shot")

    record = {
        "N": args.N, "a": args.a, "counting_qubits": args.t, "multiplier": args.multiplier,
        "shots": args.shots, "backend": backend.name,
        "simulator": "fake, noiseless" if args.fake else None,
        "optimization_level": OPTIMIZATION_LEVEL, "seed_transpiler": SEED_TRANSPILER,
        "submitted": now(), "circuit": metrics,
    }

    if args.fake:
        sim = AerSimulator(method="matrix_product_state")  # noiseless, see module docstring
        counts = sim.run(isa, shots=args.shots, seed_simulator=0).result().get_counts()
        record.update(job_id="local", counts=dict(sorted(counts.items())), completed=now())
        save_result(record)
        return

    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run([isa], shots=args.shots)
    record["job_id"] = job.job_id()
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    path = PENDING_DIR / f"{record['job_id']}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"Submitted job {record['job_id']} -> {path.relative_to(ROOT_DIR.parent)}")
    print("Collect it later with: python -m shor_rsa.experiments.hardware.run_ibm fetch")


def save_result(record):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = (f"N{record['N']}_a{record['a']}_t{record['counting_qubits']}_"
            f"{record['backend']}_{record['job_id']}.json")
    path = RESULTS_DIR / name
    path.write_text(json.dumps(record, indent=2))
    print(f"Result saved -> {path.relative_to(ROOT_DIR.parent)}")


def fetch(args):
    pending = sorted(PENDING_DIR.glob("*.json"))
    if not pending:
        print("No pending jobs.")
        return
    service = get_service()
    for path in pending:
        record = json.loads(path.read_text())
        job = service.job(record["job_id"])
        status = job.status()
        status = getattr(status, "name", status)
        print(f"{record['job_id']}: {status}")
        if status in ("ERROR", "CANCELLED"):
            record.update(status=status, error=str(job.error_message()))
            save_result(record)
            path.unlink()
            continue
        if status != "DONE":
            continue
        result = job.result()[0]
        counts = result.data.c.get_counts()  # classical register "c" of create_shor_circuit
        usage = job.usage() if hasattr(job, "usage") else None  # QPU seconds actually used
        record.update(counts=dict(sorted(counts.items())), completed=now(),
                      qpu_seconds=usage, status=status)
        save_result(record)
        path.unlink()


def main():
    ap = argparse.ArgumentParser(description="Run Shor circuits on IBM quantum hardware.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("login", help="save the IBM Quantum API key locally")
    p.add_argument("--token", required=True)
    p.add_argument("--instance", default=None, help="instance CRN (optional)")
    p.set_defaults(func=login)

    p = sub.add_parser("submit", help="compile and submit one Shor circuit")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--a", type=int, required=True)
    p.add_argument("--t", type=int, required=True, help="counting qubits")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--multiplier", default="binary", choices=["binary", "matrix"])
    p.add_argument("--backend", default=None, help="processor name (default: least busy)")
    p.add_argument("--fake", action="store_true",
                   help="local dry run: compile for FakeBrisbane, simulate without noise")
    p.set_defaults(func=submit)

    p = sub.add_parser("fetch", help="collect results of finished jobs")
    p.set_defaults(func=fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
