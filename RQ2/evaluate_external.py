import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

DATASET_PATH     = Path(__file__).parent / "data/external_dataset.json"
DEFAULT_EXP_DIR  = Path(__file__).parent / "results" / "experiment_outputs"
DEFAULT_OUT_PATH = Path(__file__).parent / "results" / "external_eval_results.json"

MAX_LEN    = 256
BATCH_SIZE = 64
N_BOOTSTRAP = 1000
EXPERIMENTS = [
    ("bert-base-uncased",          "cls"),
    ("bert-base-uncased",          "mean"),
    ("bert-base-uncased",          "max"),
    ("roberta-base",               "cls"),
    ("roberta-base",               "mean"),
    ("roberta-base",               "max"),
    ("microsoft/deberta-base",     "cls"),
    ("microsoft/deberta-base",     "mean"),
    ("microsoft/deberta-base",     "max"),
    ("microsoft/codebert-base",    "cls"),
    ("microsoft/codebert-base",    "mean"),
    ("microsoft/codebert-base",    "max"),
    ("microsoft/unixcoder-base",   "cls"),
    ("microsoft/unixcoder-base",   "mean"),
    ("microsoft/unixcoder-base",   "max"),
    ("distilbert-base-uncased",    "cls"),
    ("distilbert-base-uncased",    "mean"),
    ("distilbert-base-uncased",    "max"),
]

class EvalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels             = torch.tensor(labels, dtype=torch.long)
        self.has_token_type_ids = "token_type_ids" in self.encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }
        if self.has_token_type_ids:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        return item

class PooledClassifier(nn.Module):
    def __init__(self, encoder, pooling, hidden_size):
        super().__init__()
        self.encoder    = encoder
        self.pooling    = pooling
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)

    def pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        masked = last_hidden_state.masked_fill(mask == 0, float("-inf"))
        return masked.max(dim=1).values

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled  = self.dropout(self.pool(outputs.last_hidden_state, attention_mask))
        return self.classifier(pooled)

def load_model(model_name: str, pooling: str, device, out_base: Path):
    eid        = f"{model_name.replace('/', '_')}__{pooling}"
    model_dir  = out_base / eid / "best_model"
    encoder    = AutoModel.from_pretrained(model_dir)
    hidden_size = encoder.config.hidden_size
    model      = PooledClassifier(encoder, pooling, hidden_size)
    head       = torch.load(model_dir / "classification_head.pt", map_location="cpu",
                            weights_only=False)
    model.classifier.load_state_dict(head["classifier"])
    model.dropout.load_state_dict(head["dropout"])
    return model.to(device)

def predict_with_probs(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels_b = batch.pop("labels").to(device)
            batch    = {k: v.to(device) for k, v in batch.items()}
            logits   = model(**batch)
            probs    = torch.softmax(logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def bootstrap_ci(y_true, y_pred, y_prob, n=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    metrics = {"accuracy": [], "f1_macro": [], "f1_sensitive": [],
                "auroc": [], "pr_auc": []}

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["f1_macro"].append(f1_score(yt, yp, average="macro"))
        metrics["f1_sensitive"].append(f1_score(yt, yp, pos_label=1, average="binary"))
        metrics["auroc"].append(roc_auc_score(yt, ypr))
        metrics["pr_auc"].append(average_precision_score(yt, ypr))

    ci = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        ci[k] = {
            "mean":  float(np.mean(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
        }
    return ci

def operating_points(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1s[:-1])
    best_f1_thresh = float(thresholds[best_f1_idx])
    best_f1_val    = float(f1s[best_f1_idx])

    mask95 = precisions[:-1] >= 0.95
    recall_at_95p = float(recalls[:-1][mask95].max()) if mask95.any() else float("nan")

    mask90 = recalls[:-1] >= 0.90
    prec_at_90r = float(precisions[:-1][mask90].max()) if mask90.any() else float("nan")

    return {
        "best_f1_threshold": best_f1_thresh,
        "best_f1":           best_f1_val,
        "recall_at_95prec":  recall_at_95p,
        "prec_at_90recall":  prec_at_90r,
    }

def main(exp_dir: Path = DEFAULT_EXP_DIR,
         out_path: Path = DEFAULT_OUT_PATH,
         eval_data: Path = DATASET_PATH) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw     = json.loads(eval_data.read_text(encoding="utf-8"))
    records = [r for r in raw if r.get("sensitive") is not None]
    print(f"External dataset: {len(records)} valid records")

    texts  = [r["prompt_text"] for r in records]
    labels = [1 if r["sensitive"] else 0 for r in records]

    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  Sensitive (1): {pos}  Not sensitive (0): {neg}  "
          f"Base rate: {pos/len(labels)*100:.1f}%")

    all_results = {}

    for model_name, pooling in EXPERIMENTS:
        eid = f"{model_name.replace('/', '_')}__{pooling}"
        model_dir = exp_dir / eid / "best_model"
        if not model_dir.exists():
            print(f"  SKIP {eid} (no checkpoint)")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {eid}")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        ds     = EvalDataset(texts, labels, tokenizer, MAX_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

        model              = load_model(model_name, pooling, device, exp_dir)
        y_true, y_pred, y_prob = predict_with_probs(model, loader, device)

        acc     = accuracy_score(y_true, y_pred)
        f1_mac  = f1_score(y_true, y_pred, average="macro")
        f1_sens = f1_score(y_true, y_pred, pos_label=1, average="binary")
        auroc   = roc_auc_score(y_true, y_prob)
        pr_auc  = average_precision_score(y_true, y_prob)
        mcc     = matthews_corrcoef(y_true, y_pred)
        cr      = classification_report(y_true, y_pred,
                                        target_names=["not_sensitive", "sensitive"],
                                        output_dict=True)
        cm      = confusion_matrix(y_true, y_pred)
        ci      = bootstrap_ci(y_true, y_pred, y_prob)
        ops     = operating_points(y_true, y_prob)

        print(f"  Accuracy : {acc:.4f}  AUROC: {auroc:.4f}  PR-AUC: {pr_auc:.4f}  MCC: {mcc:.4f}")
        print(f"  F1-macro : {f1_mac:.4f}  F1-sensitive: {f1_sens:.4f}")
        print(f"  Recall@95%Prec: {ops['recall_at_95prec']:.4f}  "
              f"Prec@90%Rec: {ops['prec_at_90recall']:.4f}")
        print(classification_report(y_true, y_pred,
              target_names=["not_sensitive", "sensitive"]))
        print(f"  95% CI AUROC: [{ci['auroc']['ci_lo']:.4f}, {ci['auroc']['ci_hi']:.4f}]")
        print(f"  Confusion matrix:\n{cm}")

        all_results[eid] = {
            "experiment_id":         eid,
            "model_name":            model_name,
            "pooling":               pooling,
            "eval_dataset":          str(eval_data.name),
            "total_records":         len(records),
            "n_sensitive":           int(pos),
            "n_not_sensitive":       int(neg),
            "accuracy":              float(acc),
            "f1_macro":              float(f1_mac),
            "f1_sensitive":          float(f1_sens),
            "auroc":                 float(auroc),
            "pr_auc":                float(pr_auc),
            "mcc":                   float(mcc),
            "classification_report": cr,
            "confusion_matrix":      cm.tolist(),
            "bootstrap_ci":          ci,
            "operating_points":      ops,
        }

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"SUMMARY — External DevGPT ({len(records)} records, base rate {pos/len(labels)*100:.1f}%)")
    print(f"{'='*80}")
    print(f"{'Experiment':<45} {'Acc':>7} {'F1-mac':>7} {'F1-sens':>8} {'AUROC':>7} {'PR-AUC':>7} {'MCC':>7}")
    print("-" * 90)
    for eid, res in all_results.items():
        print(f"{eid:<45} {res['accuracy']:>7.4f} {res['f1_macro']:>7.4f} "
              f"{res['f1_sensitive']:>8.4f} {res['auroc']:>7.4f} "
              f"{res['pr_auc']:>7.4f} {res['mcc']:>7.4f}")

    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--eval-data",      type=Path, default=DATASET_PATH)
    parser.add_argument("--output",         type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    main(exp_dir=args.experiment_dir, out_path=args.output, eval_data=args.eval_data)
