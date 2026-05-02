from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redacted.craft.baselines import presidio_sanitize, spacy_sanitize
from redacted.craft.evaluate import aggregate, score_row
from redacted.craft.transform import (
    ABLATION_ABSTRACT_ONLY,
    ABLATION_SUPPRESS_ONLY,
    naive_mask,
    transform_text,
    transform_text_ablation,
)
from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRAFT ablation study (algorithmic, no LLM).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()

def _no_key_aware(text: str):
    from redacted.craft.transform import CRAFT_PIPELINES
    from redacted.craft.model import Op
    from redacted.craft.operators import PlaceholderBank, apply_abstract, apply_canonicalize
    from redacted.craft.artifacts import build_artifact, HIGH_RISK_PATTERNS
    from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps
    from redacted.craft.transform import TransformResult
    import re

    spans = resolve_overlaps(find_sensitive_spans(text))
    artifact = build_artifact(text, spans)
    pipeline = CRAFT_PIPELINES.get(artifact.kind, CRAFT_PIPELINES["mixed_artifact"])
    ops = pipeline.ops
    bank = PlaceholderBank()

    if not ops:
        return TransformResult(text=text, artifact_kind=artifact.kind, operators_applied=[],
                               placeholders={}, operator_counts={}, artifact_counts={artifact.kind: 1})

    if Op.CANONICALIZE in ops:
        result = re.sub(r"/home/[^/\s\"'`\n]+", "/home/<USER>", text)
        result = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"'`\n]+", r"C:\\Users\\<USER>", result)
        new_spans = resolve_overlaps(find_sensitive_spans(result))
        out: list[str] = []
        cursor = 0
        for span in new_spans:
            out.append(result[cursor:span.start])
            context_before = result[max(0, span.start - 60):span.start]
            from redacted.craft.operators import _is_value_position, _is_benign_scalar
            if _is_value_position(context_before) and _is_benign_scalar(span.text):
                out.append(span.text)
            else:
                out.append(apply_abstract(span, bank))
            cursor = span.end
        out.append(result[cursor:])
        rewritten = "".join(out)
        return TransformResult(text=rewritten, artifact_kind=artifact.kind,
                               operators_applied=["CANONICALIZE"],
                               placeholders=bank.all_placeholders(),
                               operator_counts={"CANONICALIZE": 1},
                               artifact_counts={artifact.kind: 1})

    return transform_text(text)

def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.limit:
        records = records[:args.limit]
    total = len(records)

    policies = {
        "craft_full":     lambda t: transform_text(t),
        "abstract_only":  lambda t: transform_text_ablation(t, ABLATION_ABSTRACT_ONLY),
        "suppress_only":  lambda t: transform_text_ablation(t, ABLATION_SUPPRESS_ONLY),
        "no_key_aware":   _no_key_aware,
        "naive_mask":     naive_mask,
        "presidio":       presidio_sanitize,
        "spacy_ner":      spacy_sanitize,
    }

    rows: dict[str, list] = {p: [] for p in policies}
    by_artifact: dict[str, dict[str, list]] = defaultdict(lambda: {p: [] for p in policies})

    for idx, record in enumerate(records, start=1):
        original = record["prompt_text"]
        artifact_kind = record.get("artifact_kind", "unknown")

        for policy_name, fn in policies.items():
            result = fn(original)
            s = score_row(original, result)
            rows[policy_name].append(s)
            by_artifact[artifact_kind][policy_name].append(s)

        if idx % 100 == 0:
            print(f"processed {idx}/{total}")

    output = {
        "manifest": args.manifest.name,
        "total": total,
        "policies": {p: aggregate(rows[p]) for p in policies},
        "by_artifact": {
            kind: {p: aggregate(kind_rows[p]) for p in policies}
            for kind, kind_rows in sorted(by_artifact.items())
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'Policy':<18}  {'privacy':>8}  {'utility':>8}  {'pareto':>8}")
    print("-" * 50)
    for p, agg in output["policies"].items():
        print(f"{p:<18}  {agg['privacy_score']:8.4f}  {agg['utility_score']:8.4f}  {agg['pareto_score']:8.4f}")
    print()
    print(json.dumps(output["policies"], indent=2))

if __name__ == "__main__":
    main()
