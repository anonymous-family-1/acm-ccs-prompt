import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

FT        = Path(__file__).parent
P1_EXT      = FT / "results" / "phase1_external_eval_results.json"
P2_EXT      = FT / "results" / "phase2_external_eval_results.json"
DEVGPT_PATH = FT / "data" / "external_dataset.json"
DEFAULT_OUT = FT / "results" / "statistical_results.json"

N_BOOTSTRAP = 2000
SEED        = 42

MODELS   = [
    "bert-base-uncased", "roberta-base", "microsoft/deberta-base",
    "microsoft/codebert-base", "microsoft/unixcoder-base", "distilbert-base-uncased",
]
POOLINGS = ["cls", "mean", "max"]

def eid(model, pooling):
    return f"{model.replace('/', '_')}__{pooling}"

def load_devgpt():
    raw     = json.loads(DEVGPT_PATH.read_text(encoding="utf-8"))
    records = [r for r in raw if r.get("sensitive") is not None]
    return np.array([1 if r["sensitive"] else 0 for r in records])

def bootstrap_metrics(y_true, y_pred, y_prob, n=N_BOOTSTRAP, seed=SEED):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                  f1_score, roc_auc_score)
    rng = np.random.default_rng(seed)
    n_s = len(y_true)
    accs, f1ms, f1ss, aurocs, praucs = [], [], [], [], []
    for _ in range(n):
        idx = rng.integers(0, n_s, size=n_s)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        accs.append(accuracy_score(yt, yp))
        f1ms.append(f1_score(yt, yp, average="macro"))
        f1ss.append(f1_score(yt, yp, pos_label=1, average="binary"))
        aurocs.append(roc_auc_score(yt, ypr))
        praucs.append(average_precision_score(yt, ypr))

    def ci(arr):
        a = np.array(arr)
        return {"mean": float(np.mean(a)),
                "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5))}
    return {
        "accuracy":     ci(accs),
        "f1_macro":     ci(f1ms),
        "f1_sensitive": ci(f1ss),
        "auroc":        ci(aurocs),
        "pr_auc":       ci(praucs),
    }

def mcnemar_test(y_true, pred1, pred2):
    b = int(((pred1 == y_true) & (pred2 != y_true)).sum())
    c = int(((pred1 != y_true) & (pred2 == y_true)).sum())
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=False, correction=True)
    return {
        "b_p1_correct_p2_wrong": b,
        "c_p1_wrong_p2_correct": c,
        "statistic": float(result.statistic),
        "pvalue":    float(result.pvalue),
        "significant_at_005": bool(result.pvalue < 0.05),
    }

def main(out_path: Path = DEFAULT_OUT,
         p1_ext: Path = P1_EXT,
         p2_ext: Path = P2_EXT) -> None:
    print("Loading result files …")

    if not p1_ext.exists():
        print(f"  Missing: {p1_ext} — run evaluate_external.py first")
        return
    if not p2_ext.exists():
        print(f"  Missing: {p2_ext} — run Phase 2 pipeline first")
        return

    p1_res = json.loads(p1_ext.read_text())
    p2_res = json.loads(p2_ext.read_text())
    y_true = load_devgpt()
    print(f"DevGPT ground truth: {len(y_true)} records, "
          f"{y_true.sum()} sensitive ({y_true.mean()*100:.1f}%)")

    all_output = {
        "bootstrap_ci": {},
        "mcnemar_phase1_vs_phase2": {},
        "wilcoxon": {},
    }

    print("\n=== Bootstrap 95% CIs ===")
    p1_aurocs, p2_aurocs = [], []
    p1_f1s, p2_f1s       = [], []

    for model in MODELS:
        for pooling in POOLINGS:
            e = eid(model, pooling)

            for phase, res_dict, auroc_list, f1_list in [
                ("phase1", p1_res, p1_aurocs, p1_f1s),
                ("phase2", p2_res, p2_aurocs, p2_f1s),
            ]:
                if e not in res_dict:
                    continue
                r    = res_dict[e]
                cr   = r["classification_report"]
                ci = r.get("bootstrap_ci", {})
                if ci:
                    all_output["bootstrap_ci"].setdefault(phase, {})[e] = ci
                    auroc_list.append(r["auroc"])
                    f1_list.append(r["f1_sensitive"])
                    if phase == "phase1":
                        print(f"  [{phase}] {e:<45} "
                              f"AUROC={r['auroc']:.4f} "
                              f"[{ci.get('auroc',{}).get('ci_lo',0):.4f}, "
                              f"{ci.get('auroc',{}).get('ci_hi',0):.4f}]  "
                              f"F1-sens={r['f1_sensitive']:.4f}")

    print("\n=== McNemar's Test (Phase 1 vs Phase 2) ===")
    print(f"{'Experiment':<45} {'b':>5} {'c':>5} {'p-value':>10} {'Sig?':>6}")
    print("-" * 70)

    for model in MODELS:
        for pooling in POOLINGS:
            e = eid(model, pooling)
            if e not in p1_res or e not in p2_res:
                continue

            p1r = p1_res[e]
            p2r = p2_res[e]

            cm1 = np.array(p1r["confusion_matrix"])
            cm2 = np.array(p2r["confusion_matrix"])

            p1_acc = p1r["accuracy"]
            p2_acc = p2r["accuracy"]
            delta  = p2_acc - p1_acc
            print(f"  {e:<45} Δacc={delta:+.4f}  "
                  f"p1_auroc={p1r.get('auroc','N/A'):.4f}  "
                  f"p2_auroc={p2r.get('auroc','N/A'):.4f}  "
                  f"Δauroc={p2r.get('auroc',0)-p1r.get('auroc',0):+.4f}")

            all_output["mcnemar_phase1_vs_phase2"][e] = {
                "phase1_accuracy": p1_acc,
                "phase2_accuracy": p2_acc,
                "delta_accuracy":  delta,
                "phase1_auroc":    p1r.get("auroc"),
                "phase2_auroc":    p2r.get("auroc"),
                "delta_auroc":     (p2r.get("auroc", 0) - p1r.get("auroc", 0))
                                    if p1r.get("auroc") and p2r.get("auroc") else None,
                "phase1_f1_sensitive": p1r.get("f1_sensitive"),
                "phase2_f1_sensitive": p2r.get("f1_sensitive"),
                "delta_f1_sensitive":  (p2r.get("f1_sensitive", 0) -
                                         p1r.get("f1_sensitive", 0))
                                        if p1r.get("f1_sensitive") and
                                           p2r.get("f1_sensitive") else None,
            }

    print("\n=== Wilcoxon Signed-Rank Test (Phase 1 vs Phase 2, n=18 pairs) ===")
    paired = [(e, all_output["mcnemar_phase1_vs_phase2"][e])
              for e in all_output["mcnemar_phase1_vs_phase2"]]

    for metric_key in ["delta_accuracy", "delta_auroc", "delta_f1_sensitive"]:
        vals = [v[metric_key] for _, v in paired if v[metric_key] is not None]
        if len(vals) < 5:
            print(f"  {metric_key}: insufficient pairs ({len(vals)})")
            continue
        stat, pval = stats.wilcoxon(vals, alternative="greater")
        mean_delta = float(np.mean(vals))
        print(f"  {metric_key:<22}  mean_Δ={mean_delta:+.4f}  "
              f"W={stat:.1f}  p={pval:.4f}  "
              f"{'SIGNIFICANT' if pval < 0.05 else 'not significant'}")
        all_output["wilcoxon"][metric_key] = {
            "n_pairs":    len(vals),
            "mean_delta": mean_delta,
            "statistic":  float(stat),
            "pvalue":     float(pval),
            "significant_at_005": bool(pval < 0.05),
        }

    deltas = [v["delta_auroc"] for v in all_output["mcnemar_phase1_vs_phase2"].values()
              if v["delta_auroc"] is not None]
    if deltas:
        print(f"\nPhase 2 vs Phase 1 (AUROC on DevGPT, {len(deltas)} models):")
        print(f"  Mean Δ AUROC = {np.mean(deltas):+.4f}")
        print(f"  Range: [{min(deltas):+.4f}, {max(deltas):+.4f}]")
        print(f"  Models improved: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    out_path.write_text(
        json.dumps(all_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nStatistical results saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-external", type=Path, default=P1_EXT)
    parser.add_argument("--phase2-external", type=Path, default=P2_EXT)
    parser.add_argument("--output",          type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(out_path=args.output, p1_ext=args.phase1_external, p2_ext=args.phase2_external)
