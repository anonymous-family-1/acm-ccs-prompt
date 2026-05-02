from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redacted.craft.reconstruction_attack import (
    aggregate_attempts,
    aggregate_by_type,
    attack_result,
    linkage_attack,
)
from redacted.craft.transform import naive_mask, transform_text, TransformResult

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run adversarial reconstruction attack: CRAFT vs naive_mask."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attack-model", default="qwen2.5-coder:32b",
                   help="LLM used for reconstruction attempts (should differ from answer/judge model).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-alive", default="24h")
    return p.parse_args()

def _manifest_hash(records: list[dict]) -> str:
    encoded = json.dumps([r["index"] for r in records], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

def _save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def _log(path: Path | None, line: str) -> None:
    if path:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

def _summarize_row(craft_attempts, naive_attempts) -> dict:
    return {
        "craft": aggregate_attempts(craft_attempts),
        "naive_mask": aggregate_attempts(naive_attempts),
    }

def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]

    if args.start_index:
        records = records[args.start_index :]
    if args.end_index is not None:
        records = records[: max(0, args.end_index - args.start_index)]
    if args.limit is not None:
        records = records[: args.limit]

    mhash = _manifest_hash(records)
    log_path = args.log_file or args.output.with_suffix(args.output.suffix + ".log")

    rows: list[dict] = []
    craft_results_all: list[TransformResult] = []
    naive_results_all: list[TransformResult] = []
    processed: set[int] = set()

    if args.resume:
        state = _load(args.output)
        if state and state.get("manifest_hash") == mhash:
            rows = list(state.get("rows", []))
            processed = {int(r["index"]) for r in rows}
            _log(log_path, f"resume rows={len(rows)}")
        else:
            _log(log_path, "resume: no matching prior state")
    else:
        _log(log_path, f"start manifest={args.manifest.name} total={len(records)} attack_model={args.attack_model}")

    total = len(records)
    for idx, record in enumerate(records, start=1):
        if int(record["index"]) in processed:
            continue
        original = record["prompt_text"]
        try:
            craft_r = transform_text(original)
            naive_r = naive_mask(original)

            craft_attempts = attack_result(args.attack_model, original, craft_r, args.keep_alive)
            naive_attempts = attack_result(args.attack_model, original, naive_r, args.keep_alive)

            craft_results_all.append(craft_r)
            naive_results_all.append(naive_r)

            row = {
                "index": record["index"],
                "artifact_kind": craft_r.artifact_kind,
                "operators_applied": craft_r.operators_applied,
                "craft_summary": aggregate_attempts(craft_attempts),
                "naive_summary": aggregate_attempts(naive_attempts),
                "craft_attempts": [a.to_dict() for a in craft_attempts],
                "naive_attempts": [a.to_dict() for a in naive_attempts],
            }
        except Exception as exc:
            _log(log_path, f"error index={record['index']} {type(exc).__name__}: {exc}")
            craft_results_all.append(TransformResult(original, "unknown", [], {}, {}, {}))
            naive_results_all.append(TransformResult(original, "unknown", [], {}, {}, {}))
            row = {
                "index": record["index"],
                "artifact_kind": record.get("artifact_kind", "unknown"),
                "operators_applied": [],
                "craft_summary": aggregate_attempts([]),
                "naive_summary": aggregate_attempts([]),
                "craft_attempts": [],
                "naive_attempts": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(row)
        print(f"processed={idx}/{total} index={record['index']} artifact={row['artifact_kind']}")
        _log(log_path, (
            f"processed={idx}/{total} index={record['index']} artifact={row['artifact_kind']} "
            f"craft_exact={row['craft_summary'].get('exact_match_rate', 0):.3f} "
            f"naive_exact={row['naive_summary'].get('exact_match_rate', 0):.3f}"
        ))

        if len(rows) % args.save_every == 0:
            _save(args.output, _build_output(args, mhash, rows, craft_results_all, naive_results_all))
            _log(log_path, f"checkpoint rows={len(rows)}")

    output = _build_output(args, mhash, rows, craft_results_all, naive_results_all)
    _save(args.output, output)
    _log(log_path, f"complete rows={len(rows)}")
    print(json.dumps(output["aggregate"], indent=2))

def _build_output(
    args: argparse.Namespace,
    mhash: str,
    rows: list[dict],
    craft_results: list[TransformResult],
    naive_results: list[TransformResult],
) -> dict:
    all_craft_attempts = [a for r in rows for a in r.get("craft_attempts", [])]
    all_naive_attempts = [a for r in rows for a in r.get("naive_attempts", [])]

    craft_by_kind: dict[str, list] = defaultdict(list)
    naive_by_kind: dict[str, list] = defaultdict(list)
    for row in rows:
        kind = row["artifact_kind"]
        craft_by_kind[kind].extend(row.get("craft_attempts", []))
        naive_by_kind[kind].extend(row.get("naive_attempts", []))

    craft_by_op: dict[str, list] = defaultdict(list)
    for row in rows:
        key = "+".join(row.get("operators_applied", ["unknown"]))
        craft_by_op[key].extend(row.get("craft_attempts", []))

    from redacted.craft.reconstruction_attack import AttackAttempt
    linkage = linkage_attack(craft_results) if craft_results else {}
    naive_linkage = linkage_attack(naive_results) if naive_results else {}

    return {
        "manifest": args.manifest.name,
        "manifest_hash": mhash,
        "attack_model": args.attack_model,
        "aggregate": {
            "craft": aggregate_attempts([
                _dict_to_attempt(a) for a in all_craft_attempts
            ]),
            "naive_mask": aggregate_attempts([
                _dict_to_attempt(a) for a in all_naive_attempts
            ]),
        },
        "by_artifact_kind": {
            "craft": {k: aggregate_attempts([_dict_to_attempt(a) for a in v])
                      for k, v in sorted(craft_by_kind.items())},
            "naive_mask": {k: aggregate_attempts([_dict_to_attempt(a) for a in v])
                           for k, v in sorted(naive_by_kind.items())},
        },
        "by_operator": {
            k: aggregate_attempts([_dict_to_attempt(a) for a in v])
            for k, v in sorted(craft_by_op.items())
        },
        "linkage_attack": {
            "craft": linkage,
            "naive_mask": naive_linkage,
        },
        "rows": rows,
    }

def _dict_to_attempt(d: dict):
    from redacted.craft.reconstruction_attack import AttackAttempt
    return AttackAttempt(
        placeholder=d.get("placeholder", ""),
        placeholder_type=d.get("placeholder_type", ""),
        original_value=d.get("original_value", ""),
        reconstructed_value=d.get("reconstructed_value", ""),
        confidence=d.get("confidence", "low"),
        exact_match=bool(d.get("exact_match", False)),
        format_match=bool(d.get("format_match", False)),
        category_match=bool(d.get("category_match", False)),
    )

if __name__ == "__main__":
    main()
