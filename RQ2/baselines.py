import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FT              = Path(__file__).parent
DATASET_PATH    = FT / "bert_dataset.json"
DEVGPT_PATH     = FT / "data/external_dataset.json"
DEFAULT_OUT     = FT / "results" / "baselines_results.json"

SEED = 42

KEYWORD_PATTERNS = [
    r"(?i)(api[_\-\s]?key|access[_\-]?token|bearer\s+[a-zA-Z0-9+/]{20,}|"
    r"Authorization:\s*Bearer|secret[_\-]?key|client[_\-]?secret)",
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"(?i)(first[\s_]?name|last[\s_]?name|date[\s_]?of[\s_]?birth|"
    r"home[\s_]?address|social[\s_]?security)",
    r"(?i)(password\s*[=:]\s*\S+|db[_\-]?pass|"
    r"mongodb(\+srv)?://[^/\s]+:[^/\s]+@|"
    r"postgresql://[^/\s]+:[^/\s]+@|mysql://[^/\s]+:[^/\s]+@|"
    r"connection[_\-]?string|jdbc:[a-z]+://)",
    r"-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY",
    r"(?i)(private[_\-]?key|-----BEGIN CERTIFICATE)",
    r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3})\b",
    r"(?i)(internal[_\-]?hostname|corp\.|\.internal\b|\.local\b)",
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",
    r"\b5[1-5][0-9]{14}\b",
    r"\b(?:0x)?[0-9a-fA-F]{40}\b",
    r"(?i)(account[_\-]?number|routing[_\-]?number|iban\b|swift\b)",
    r"(?i)(GITHUB_TOKEN|TRAVIS_CI|CIRCLE_CI|JENKINS_|AWS_SECRET|"
    r"DOCKER_HUB_PASSWORD|NPM_TOKEN|PYPI_TOKEN)",
]

_compiled = [re.compile(p) for p in KEYWORD_PATTERNS]

def keyword_predict(texts):
    preds, scores = [], []
    for text in texts:
        hits = sum(1 for pat in _compiled if pat.search(text))
        preds.append(1 if hits > 0 else 0)
        scores.append(min(hits / 3.0, 1.0))
    return np.array(preds), np.array(scores)

def metrics(y_true, y_pred, y_prob):
    n = len(y_true)
    pos = int(y_true.sum())
    cr = classification_report(y_true, y_pred,
                                target_names=["not_sensitive", "sensitive"],
                                output_dict=True)
    has_both = len(np.unique(y_true)) == 2
    return {
        "n_records":    n,
        "n_sensitive":  pos,
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "f1_macro":     float(f1_score(y_true, y_pred, average="macro")),
        "f1_sensitive": float(f1_score(y_true, y_pred, pos_label=1, average="binary")),
        "auroc":        float(roc_auc_score(y_true, y_prob)) if has_both else None,
        "pr_auc":       float(average_precision_score(y_true, y_prob)) if has_both else None,
        "mcc":          float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": cr,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def print_metrics(name, split, m):
    auroc_s = f"{m['auroc']:.4f}" if m.get("auroc") else "N/A"
    print(f"  [{name}] {split}: acc={m['accuracy']:.4f}  "
          f"f1_mac={m['f1_macro']:.4f}  f1_sens={m['f1_sensitive']:.4f}  "
          f"auroc={auroc_s}  mcc={m['mcc']:.4f}")

def main(dataset_path: Path = DATASET_PATH, out_path: Path = DEFAULT_OUT,
         eval_data: Path = DEVGPT_PATH) -> None:

    raw        = json.loads(dataset_path.read_text(encoding="utf-8"))
    texts_all  = [d["prompt_text"] for d in raw]
    labels_all = [d["label"]       for d in raw]
    strat_keys = [d["strat_key"]   for d in raw]

    tr_x, te_x, tr_y, te_y, tr_sk, _ = train_test_split(
        texts_all, labels_all, strat_keys,
        test_size=0.20, random_state=SEED, stratify=strat_keys,
    )
    tr_x, _, tr_y, _, tr_sk, _ = train_test_split(
        tr_x, tr_y, tr_sk,
        test_size=0.125, random_state=SEED, stratify=tr_sk,
    )
    print(f"In-dist  train={len(tr_x)}  test={len(te_x)}")

    devgpt_raw = json.loads(eval_data.read_text(encoding="utf-8"))
    dg_records = [r for r in devgpt_raw if r.get("sensitive") is not None]
    dg_texts   = [r["prompt_text"] for r in dg_records]
    dg_labels  = [1 if r["sensitive"] else 0 for r in dg_records]
    print(f"DevGPT   n={len(dg_records)}  "
          f"pos={sum(dg_labels)} ({sum(dg_labels)/len(dg_labels)*100:.1f}%)")

    all_results = {}

    print("\n=== Majority Class Baseline ===")
    maj_label = int(np.array(tr_y).sum() > len(tr_y) / 2)

    for split_name, te_texts, te_y_arr in [
        ("in_dist", te_x,    np.array(te_y)),
        ("devgpt",  dg_texts, np.array(dg_labels)),
    ]:
        preds = np.full(len(te_y_arr), maj_label)
        probs = np.full(len(te_y_arr), float(maj_label))
        m = metrics(te_y_arr, preds, probs)
        print_metrics("majority", split_name, m)
        all_results.setdefault("majority_class", {})[split_name] = m

    print("\n=== Keyword / Regex Baseline ===")
    for split_name, te_texts, te_y_arr in [
        ("in_dist", te_x,    np.array(te_y)),
        ("devgpt",  dg_texts, np.array(dg_labels)),
    ]:
        preds, scores = keyword_predict(te_texts)
        m = metrics(te_y_arr, preds, scores)
        print_metrics("keyword_regex", split_name, m)
        all_results.setdefault("keyword_regex", {})[split_name] = m

    print("\n=== TF-IDF (word 1-2gram) + LR ===")
    pipe_word = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   max_features=100_000, sublinear_tf=True)),
        ("lr",    LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced", random_state=SEED)),
    ])
    pipe_word.fit(tr_x, tr_y)

    for split_name, te_texts, te_y_arr in [
        ("in_dist", te_x,    np.array(te_y)),
        ("devgpt",  dg_texts, np.array(dg_labels)),
    ]:
        preds = pipe_word.predict(te_texts)
        probs = pipe_word.predict_proba(te_texts)[:, 1]
        m = metrics(te_y_arr, preds, probs)
        print_metrics("tfidf_word_lr", split_name, m)
        all_results.setdefault("tfidf_word_lr", {})[split_name] = m

    print("\n=== TF-IDF (char 3-5gram) + LR ===")
    pipe_char = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   max_features=100_000, sublinear_tf=True)),
        ("lr",    LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced", random_state=SEED)),
    ])
    pipe_char.fit(tr_x, tr_y)

    for split_name, te_texts, te_y_arr in [
        ("in_dist", te_x,    np.array(te_y)),
        ("devgpt",  dg_texts, np.array(dg_labels)),
    ]:
        preds = pipe_char.predict(te_texts)
        probs = pipe_char.predict_proba(te_texts)[:, 1]
        m = metrics(te_y_arr, preds, probs)
        print_metrics("tfidf_char_lr", split_name, m)
        all_results.setdefault("tfidf_char_lr", {})[split_name] = m

    print(f"\n{'='*90}")
    print(f"{'Baseline':<22} {'Split':<10} {'Acc':>7} {'F1-mac':>7} {'F1-sens':>8} "
          f"{'AUROC':>7} {'PR-AUC':>7} {'MCC':>7}")
    print("-" * 90)
    for bname, splits in all_results.items():
        for sname, m in splits.items():
            auroc_s  = f"{m['auroc']:.4f}" if m['auroc'] else "  N/A "
            prauc_s  = f"{m['pr_auc']:.4f}" if m['pr_auc'] else "  N/A "
            print(f"{bname:<22} {sname:<10} {m['accuracy']:>7.4f} "
                  f"{m['f1_macro']:>7.4f} {m['f1_sensitive']:>8.4f} "
                  f"{auroc_s:>7} {prauc_s:>7} {m['mcc']:>7.4f}")

    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nBaseline results saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=Path, default=DATASET_PATH)
    parser.add_argument("--eval-data", type=Path, default=DEVGPT_PATH)
    parser.add_argument("--output",    type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(dataset_path=args.dataset, out_path=args.output, eval_data=args.eval_data)
