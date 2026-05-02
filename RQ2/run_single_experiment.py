import argparse
import json
import random
from collections import Counter
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DEFAULT_DATASET = Path(__file__).parent / "bert_dataset.json"
DEFAULT_OUT     = Path(__file__).parent / "results" / "experiment_outputs"

MAX_LEN    = 256
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
SEED       = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels              = torch.tensor(labels, dtype=torch.long)
        self.has_token_type_ids  = "token_type_ids" in self.encodings

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
    def __init__(self, model_name: str, pooling: str, num_labels: int = 2):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.pooling    = pooling
        hidden_size     = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        masked = last_hidden_state.masked_fill(mask == 0, float("-inf"))
        return masked.max(dim=1).values

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled  = self.dropout(self.pool(outputs.last_hidden_state, attention_mask))
        logits  = self.classifier(pooled)
        loss    = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return loss, logits

def make_weighted_sampler(strat_keys: list[str]) -> WeightedRandomSampler:
    freq   = Counter(strat_keys)
    weights = [1.0 / freq[k] for k in strat_keys]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels_b = batch.pop("labels").to(device)
            batch    = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(**batch)
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    return accuracy_score(y_true, y_pred), y_true, y_pred, y_prob

def main(model_name: str, pooling: str,
         dataset_path: Path, out_base: Path) -> None:
    set_seed(SEED)

    safe_name = model_name.replace("/", "_")
    exp_id    = f"{safe_name}__{pooling}"
    out_dir   = out_base / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = open(out_dir / "train.log", "w", buffering=1)

    def p(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"[{exp_id}] device={device}  model={model_name}  pooling={pooling}")

    raw       = json.loads(dataset_path.read_text(encoding="utf-8"))
    texts     = [d["prompt_text"] for d in raw]
    labels    = [d["label"]       for d in raw]
    strat_keys = [d["strat_key"]  for d in raw]

    tr_x, te_x, tr_y, te_y, tr_sk, te_sk = train_test_split(
        texts, labels, strat_keys,
        test_size=0.20, random_state=SEED, stratify=strat_keys,
    )
    tr_x, va_x, tr_y, va_y, tr_sk, va_sk = train_test_split(
        tr_x, tr_y, tr_sk,
        test_size=0.125, random_state=SEED, stratify=tr_sk,
    )
    p(f"  train={len(tr_x)}  val={len(va_x)}  test={len(te_x)}")

    for split_name, sk_list in [("train", tr_sk), ("val", va_sk), ("test", te_sk)]:
        dist = Counter(sk_list)
        p(f"  {split_name} strat distribution: {dict(sorted(dist.items()))}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds  = PromptDataset(tr_x, tr_y, tokenizer, MAX_LEN)
    val_ds    = PromptDataset(va_x, va_y, tokenizer, MAX_LEN)
    test_ds   = PromptDataset(te_x, te_y, tokenizer, MAX_LEN)

    sampler      = make_weighted_sampler(tr_sk)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,   num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,     num_workers=2, pin_memory=True)

    model         = PooledClassifier(model_name, pooling).to(device)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = len(train_loader) * EPOCHS
    scheduler     = get_linear_schedule_with_warmup(
                        optimizer, int(total_steps * 0.1), total_steps)

    best_val_acc = 0.0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            labels_b = batch.pop("labels").to(device)
            batch    = {k: v.to(device) for k, v in batch.items()}
            loss, _  = model(**batch, labels=labels_b)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 or step == len(train_loader):
                p(f"  [{exp_id}] epoch {epoch}/{EPOCHS}  "
                  f"step {step}/{len(train_loader)}  loss={total_loss/step:.4f}")

        avg_loss = total_loss / len(train_loader)
        val_acc, _, _, _ = evaluate(model, val_loader, device)
        p(f"[{exp_id}] epoch {epoch} done — loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_dir = out_dir / "best_model"
            model_save_dir.mkdir(exist_ok=True)
            model.encoder.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            torch.save(
                {
                    "classifier": model.classifier.state_dict(),
                    "dropout":    model.dropout.state_dict(),
                    "pooling":    pooling,
                    "model_name": model_name,
                },
                model_save_dir / "classification_head.pt",
            )
            p(f"  [{exp_id}] saved best model (val_acc={val_acc:.4f})")

    p(f"\n[{exp_id}] Loading best checkpoint for test eval …")
    best_encoder = AutoModel.from_pretrained(out_dir / "best_model")
    best_model   = PooledClassifier.__new__(PooledClassifier)
    nn.Module.__init__(best_model)
    best_model.encoder    = best_encoder
    best_model.pooling    = pooling
    best_model.dropout    = nn.Dropout(0.1)
    best_model.classifier = nn.Linear(best_encoder.config.hidden_size, 2)
    head = torch.load(
        out_dir / "best_model" / "classification_head.pt", map_location="cpu"
    )
    best_model.classifier.load_state_dict(head["classifier"])
    best_model.dropout.load_state_dict(head["dropout"])
    best_model = best_model.to(device)

    test_acc, y_true, y_pred, y_prob = evaluate(best_model, test_loader, device)

    auroc  = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    mcc    = matthews_corrcoef(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_sen = f1_score(y_true, y_pred, pos_label=1, average="binary")

    p(f"\n{'='*55}")
    p(f"[{exp_id}] TEST ACCURACY: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    p(f"[{exp_id}] AUROC: {auroc:.4f}  PR-AUC: {pr_auc:.4f}  MCC: {mcc:.4f}")
    p(f"[{exp_id}] F1-macro: {f1_mac:.4f}  F1-sensitive: {f1_sen:.4f}")
    p(f"{'='*55}")
    p(classification_report(y_true, y_pred, target_names=["not_sensitive", "sensitive"]))
    p(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    results = {
        "experiment_id":  exp_id,
        "model_name":     model_name,
        "pooling":        pooling,
        "max_len":        MAX_LEN,
        "batch_size":     BATCH_SIZE,
        "epochs":         EPOCHS,
        "lr":             LR,
        "seed":           SEED,
        "split":          {"train": len(tr_x), "val": len(va_x), "test": len(te_x)},
        "best_val_acc":   best_val_acc,
        "test_acc":       test_acc,
        "auroc":          float(auroc),
        "pr_auc":         float(pr_auc),
        "mcc":            float(mcc),
        "f1_macro":       float(f1_mac),
        "f1_sensitive":   float(f1_sen),
        "history":        history,
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["not_sensitive", "sensitive"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    p(f"[{exp_id}] results saved → {out_dir / 'results.json'}")
    log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  required=True)
    parser.add_argument("--pooling",     required=True, choices=["cls", "mean", "max"])
    parser.add_argument("--dataset",     type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.model_name, args.pooling, args.dataset, args.output_dir)
