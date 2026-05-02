import argparse
import json
from pathlib import Path

import numpy as np

FT           = Path(__file__).parent
P1_EXT       = FT / "results" / "phase1_external_eval_results.json"
P2_EXT       = FT / "results" / "phase2_external_eval_results.json"
DEVGPT_PATH  = FT / "data" / "external_dataset.json"
DEFAULT_OUT  = FT / "results" / "threshold_analysis_results.json"

MODELS   = [
    "bert-base-uncased", "roberta-base", "microsoft/deberta-base",
    "microsoft/codebert-base", "microsoft/unixcoder-base", "distilbert-base-uncased",
]
POOLINGS = ["cls", "mean", "max"]

def eid(model, pooling):
    return f"{model.replace('/', '_')}__{pooling}"

def operating_points_from_ext(ext_results: dict) -> dict:
    ops = {}
    for e, r in ext_results.items():
        if "operating_points" in r:
            ops[e] = r["operating_points"]
    return ops

def deployment_summary(ops: dict, phase_name: str) -> dict:
    models_with_ops = {e: v for e, v in ops.items() if v}
    if not models_with_ops:
        return {}

    def best_by(key):
        valid = {e: v for e, v in models_with_ops.items()
                 if isinstance(v.get(key), float) and not np.isnan(v[key])}
        if not valid:
            return None, None
        best_e = max(valid, key=lambda e: valid[e][key])
        return best_e, valid[best_e][key]

    r_at_95p_e, r_at_95p_v = best_by("recall_at_95prec")
    p_at_90r_e, p_at_90r_v = best_by("prec_at_90recall")
    bf1_e, bf1_v           = best_by("best_f1")

    print(f"\n  [{phase_name}] Deployment operating points:")
    print(f"    Best recall@95%precision : {r_at_95p_e}  recall={r_at_95p_v:.4f}" if r_at_95p_v else "    Best recall@95%precision : N/A")
    print(f"    Best precision@90%recall : {p_at_90r_e}  prec={p_at_90r_v:.4f}" if p_at_90r_v else "    Best precision@90%recall : N/A")
    print(f"    Best F1                  : {bf1_e}  F1={bf1_v:.4f}" if bf1_v else "    Best F1 : N/A")

    return {
        "best_recall_at_95prec": {"experiment": r_at_95p_e, "recall": r_at_95p_v},
        "best_prec_at_90recall": {"experiment": p_at_90r_e, "prec":   p_at_90r_v},
        "best_f1":               {"experiment": bf1_e,      "f1":     bf1_v},
    }

def main(out_path: Path = DEFAULT_OUT,
         p1_ext: Path = P1_EXT,
         p2_ext: Path = P2_EXT) -> None:

    all_output = {}

    for phase, ext_path in [("phase1", p1_ext), ("phase2", p2_ext)]:
        if not ext_path.exists():
            print(f"  {phase}: {ext_path} not found — skipping")
            continue

        res = json.loads(ext_path.read_text())
        ops = operating_points_from_ext(res)
        summary = deployment_summary(ops, phase)
        all_output[phase] = {
            "operating_points": ops,
            "deployment_summary": summary,
        }

        print(f"\n{'='*80}")
        print(f"  {phase.upper()} — Operating Points on DevGPT")
        print(f"{'Experiment':<45} {'BestF1':>7} {'Rec@95P':>8} {'Pre@90R':>8} {'Thresh':>8}")
        print("-" * 78)
        for m in MODELS:
            for p in POOLINGS:
                e = eid(m, p)
                if e not in ops:
                    continue
                op = ops[e]
                bf1 = op.get("best_f1", float("nan"))
                r95 = op.get("recall_at_95prec", float("nan"))
                p90 = op.get("prec_at_90recall", float("nan"))
                th  = op.get("best_f1_threshold", float("nan"))
                def f(v):
                    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  N/A "
                print(f"  {e:<43} {f(bf1):>7} {f(r95):>8} {f(p90):>8} {f(th):>8}")

    if "phase1" in all_output and "phase2" in all_output:
        print(f"\n{'='*80}")
        print(f"  Phase 1 vs Phase 2 — Deployment Trade-offs")
        print(f"{'Operating Point':<30} {'Phase 1':>20} {'Phase 2':>20}")
        print("-" * 72)
        for key, label in [
            ("best_recall_at_95prec", "Recall @ 95% Precision"),
            ("best_prec_at_90recall", "Precision @ 90% Recall"),
            ("best_f1",               "Best F1"),
        ]:
            p1s = all_output["phase1"]["deployment_summary"].get(key, {})
            p2s = all_output["phase2"]["deployment_summary"].get(key, {})
            val_key = list(p1s.keys())[1] if len(p1s) > 1 else None
            p1v = p1s.get(val_key) if val_key else None
            p2v = p2s.get(val_key) if val_key else None
            p1_str = f"{p1v:.4f}" if isinstance(p1v, float) else "N/A"
            p2_str = f"{p2v:.4f}" if isinstance(p2v, float) else "N/A"
            print(f"  {label:<28} {p1_str:>20} {p2_str:>20}")

    out_path.write_text(
        json.dumps(all_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nThreshold analysis saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-external", type=Path, default=P1_EXT)
    parser.add_argument("--phase2-external", type=Path, default=P2_EXT)
    parser.add_argument("--output",          type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(out_path=args.output, p1_ext=args.phase1_external, p2_ext=args.phase2_external)
