from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded multi-baseline eval outputs.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()

def _summarize(rows: list[dict], baseline: str) -> dict:
    key = f"craft_vs_{baseline}"
    counts = Counter(r["judgments"][key]["winner"] for r in rows if key in r.get("judgments", {}))
    decisive = counts.get("A", 0) + counts.get("B", 0)
    return {
        "count": len(rows),
        "craft_wins": counts.get("A", 0),
        f"{baseline}_wins": counts.get("B", 0),
        "ties": counts.get("TIE", 0),
        "craft_win_rate_no_ties": counts.get("A", 0) / decisive if decisive else 0.0,
    }

def main() -> None:
    args = parse_args()
    merged_rows: list[dict] = []
    meta: dict = {}

    for path in args.inputs:
        shard = json.loads(path.read_text(encoding="utf-8"))
        if not meta:
            meta = {k: shard[k] for k in ("manifest", "answer_model", "judge_model", "llm_direct_model") if k in shard}
        merged_rows.extend(shard.get("rows", []))

    merged_rows.sort(key=lambda r: r["index"])

    baselines = ["naive_mask", "presidio", "llm_direct"]
    by_artifact: dict[str, list[dict]] = defaultdict(list)
    for row in merged_rows:
        by_artifact[row["artifact_kind"]].append(row)

    out = {
        **meta,
        "total_rows": len(merged_rows),
        "summary": {b: _summarize(merged_rows, b) for b in baselines},
        "by_artifact": {
            kind: {b: _summarize(kind_rows, b) for b in baselines}
            for kind, kind_rows in sorted(by_artifact.items())
        },
        "rows": merged_rows,
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {len(merged_rows)} rows → {args.output}")
    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()
