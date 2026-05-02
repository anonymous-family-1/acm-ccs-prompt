import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT = Path(__file__).parent / "run_single_experiment.py"

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-base",
    "microsoft/codebert-base",
    "microsoft/unixcoder-base",
    "distilbert-base-uncased",
]
POOLINGS = ["cls", "mean", "max"]

def exp_id(model: str, pooling: str) -> str:
    return f"{model.replace('/', '_')}__{pooling}"

def run_experiment(model: str, pooling: str, gpu: int,
                   out_base: Path, dataset: Path | None) -> tuple[str, int, str]:
    eid = exp_id(model, pooling)
    cmd = [
        sys.executable, str(SCRIPT),
        "--model-name", model,
        "--pooling",    pooling,
        "--output-dir", str(out_base),
    ]
    if dataset:
        cmd += ["--dataset", str(dataset)]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    (out_base / eid).mkdir(parents=True, exist_ok=True)

    print(f"  START  [{eid}]  GPU={gpu}", flush=True)
    result = subprocess.run(cmd, env=env)
    status = "DONE" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  {status}  [{eid}]  GPU={gpu}", flush=True)
    return eid, result.returncode, status

def main(dry_run: bool, skip_done: bool,
         dataset: Path | None, out_base: Path) -> None:

    experiments = [
        (model, pooling)
        for model   in MODELS
        for pooling in POOLINGS
    ]

    if skip_done:
        experiments = [
            (m, p) for m, p in experiments
            if not (out_base / exp_id(m, p) / "results.json").exists()
        ]

    print(f"Experiments to run: {len(experiments)} of {len(MODELS) * len(POOLINGS)}")
    print(f"Output dir : {out_base}")
    print(f"Dataset    : {dataset or 'default (bert_dataset.json)'}")
    for i, (m, p) in enumerate(experiments):
        print(f"  [{i:02d}] GPU={i%2}  {exp_id(m, p)}")

    if dry_run:
        print("\n--dry-run: exiting without launching.")
        return

    print(f"\nLaunching with 2 parallel workers (one per GPU) …\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_experiment, model, pooling, i % 2, out_base, dataset): (model, pooling)
            for i, (model, pooling) in enumerate(experiments)
        }
        failed = []
        for future in as_completed(futures):
            eid, rc, _ = future.result()
            if rc != 0:
                failed.append(eid)

    print("\n" + "=" * 55)
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
    else:
        print("All experiments completed successfully.")

    print("\n{:<50} {:>10} {:>10}".format("experiment", "val_acc", "test_acc"))
    print("-" * 72)
    import json
    for model in MODELS:
        for pooling in POOLINGS:
            rpath = out_base / exp_id(model, pooling) / "results.json"
            if rpath.exists():
                r = json.loads(rpath.read_text())
                print("{:<50} {:>10.4f} {:>10.4f}".format(
                    exp_id(model, pooling),
                    r.get("best_val_acc", float("nan")),
                    r.get("test_acc",     float("nan")),
                ))
            else:
                print("{:<50} {:>10} {:>10}".format(exp_id(model, pooling), "missing", "missing"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-done",  action="store_true")
    parser.add_argument("--dataset",    type=Path, default=None,
                        help="Path to dataset JSON (default: bert_dataset.json)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results" / "experiment_outputs",
                        help="Directory to write experiment outputs")
    args = parser.parse_args()
    main(dry_run=args.dry_run, skip_done=args.skip_done,
         dataset=args.dataset, out_base=args.output_dir)
