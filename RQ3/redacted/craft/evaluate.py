from __future__ import annotations

import math
import random
from collections import Counter

from redacted.craft.detectors import (
    HIGH_RISK_PATTERNS,
    config_keys,
    exception_tokens,
    find_sensitive_spans,
    frame_count,
    residual_matches,
    resolve_overlaps,
    safe_token_set,
    strip_sensitive_spans,
)
from redacted.craft.detectors import Span

from .transform import CRAFT_PIPELINES, TransformResult, naive_mask, transform_text

def exact_leaks(spans: list[Span], transformed: str) -> int:
    return sum(1 for s in spans if s.text and s.text in transformed)

def high_risk_exact_leaks(spans: list[Span], transformed: str) -> int:
    return sum(1 for s in spans if s.pattern in HIGH_RISK_PATTERNS and s.text and s.text in transformed)

def leakage_rate(spans: list[Span], transformed: str) -> float:
    return exact_leaks(spans, transformed) / len(spans) if spans else 0.0

def structural_risk(counter: Counter[str]) -> float:
    return sum(count * (1.0 if p in HIGH_RISK_PATTERNS else 0.15) for p, count in counter.items())

def type_disclosure_rate(result: TransformResult) -> float:
    if result.operators_applied == ["NAIVE_MASK"]:
        return 0.0

    total_hr = result.high_risk_span_count
    if total_hr > 0:
        return (result.high_risk_suppress_count * 0.3 + result.high_risk_abstract_count * 0.7) / total_hr

    suppress_count = result.operator_counts.get("SUPPRESS", 0)
    abstract_count = result.operator_counts.get("ABSTRACT", 0)
    total = suppress_count + abstract_count
    if total == 0:
        return 0.0
    return (suppress_count * 0.3 + abstract_count * 0.7) / total

def linkage_score(result: TransformResult) -> float:
    ph_tokens = [tok for tok in result.text.split() if tok.startswith("<") and tok.endswith(">")]
    if not ph_tokens:
        return 0.0
    counts = Counter(ph_tokens)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ph_tokens)

def recall_score(source: set[str], candidate: set[str]) -> float:
    return len(source & candidate) / len(source) if source else 1.0

def utility_proxy(original: str, transformed: str, spans: list[Span]) -> dict[str, float]:
    src_safe = safe_token_set(strip_sensitive_spans(original, spans))
    src_exc = exception_tokens(original)
    src_keys = config_keys(original)
    src_frames = frame_count(original)
    t_safe = safe_token_set(transformed)
    t_exc = exception_tokens(transformed)
    t_keys = config_keys(transformed)
    t_frames = frame_count(transformed)
    return {
        "safe_token_recall": recall_score(src_safe, t_safe),
        "exception_recall": recall_score(src_exc, t_exc),
        "config_key_recall": recall_score(src_keys, t_keys),
        "frame_recall": min(t_frames / src_frames, 1.0) if src_frames else 1.0,
        "line_ratio": min((transformed.count("\n") + 1) / max(1, original.count("\n") + 1), 1.0),
        "length_ratio": len(transformed) / max(1, len(original)),
    }

def score_row(original: str, result: TransformResult) -> dict[str, float]:
    spans = resolve_overlaps(find_sensitive_spans(original))
    residual = residual_matches(result.text)

    metrics: dict[str, float] = {
        "residual_matches": float(sum(residual.values())),
        "high_risk_residual_matches": float(
            sum(c for p, c in residual.items() if p in HIGH_RISK_PATTERNS)
        ),
        "exact_leaks": float(exact_leaks(spans, result.text)),
        "high_risk_exact_leaks": float(high_risk_exact_leaks(spans, result.text)),
        "leakage_rate": leakage_rate(spans, result.text),
        "structural_risk": structural_risk(residual),
        "type_disclosure_rate": type_disclosure_rate(result),
        "linkage_score": linkage_score(result),
    }
    metrics.update(utility_proxy(original, result.text, spans))

    hr_leaks = metrics["high_risk_exact_leaks"]
    hr_residual = metrics["high_risk_residual_matches"]
    tdr = metrics["type_disclosure_rate"]
    sr = metrics["structural_risk"]

    metrics["privacy_score"] = 1.0 / (1.0 + hr_leaks * 5.0 + hr_residual * 2.0 + sr + tdr * 0.3)

    metrics["utility_score"] = (
        0.20 * metrics["safe_token_recall"]
        + 0.25 * metrics["exception_recall"]
        + 0.15 * metrics["config_key_recall"]
        + 0.30 * metrics["frame_recall"]
        + 0.10 * metrics["line_ratio"]
    )

    metrics["pareto_score"] = 0.55 * metrics["privacy_score"] + 0.45 * metrics["utility_score"]
    return metrics

def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}

def bootstrap_ci(
    rows: list[dict],
    metric: str,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(rows)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(rows)[metric] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    lo = max(0, int(n_bootstrap * alpha / 2))
    hi = min(n_bootstrap - 1, int(n_bootstrap * (1.0 - alpha / 2)))
    return (samples[lo], samples[hi])

def aggregate_with_ci(
    rows: list[dict],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    n = len(rows)
    keys = list(rows[0].keys())
    rng = random.Random(seed)

    boot_sums: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        sample = [rng.choice(rows) for _ in range(n)]
        for k in keys:
            boot_sums[k].append(sum(r[k] for r in sample) / n)

    result: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = sorted(boot_sums[k])
        lo = max(0, int(n_bootstrap * 0.025))
        hi = min(n_bootstrap - 1, int(n_bootstrap * 0.975))
        result[k] = {
            "mean": sum(r[k] for r in rows) / n,
            "ci_lower": vals[lo],
            "ci_upper": vals[hi],
        }
    return result

def two_proportion_ztest(
    x1: int, n1: int,
    x2: int, n2: int,
) -> dict[str, float]:
    if n1 == 0 or n2 == 0:
        return {"p1": 0.0, "p2": 0.0, "z_stat": 0.0, "p_value": 1.0, "significant_at_05": False}
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"p1": p1, "p2": p2, "z_stat": 0.0, "p_value": 1.0, "significant_at_05": False}
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {
        "p1": round(p1, 6),
        "p2": round(p2, 6),
        "z_stat": round(z, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
        "n1": n1, "n2": n2, "x1": x1, "x2": x2,
    }

def mcnemar_test(
    rows_a: list[dict],
    rows_b: list[dict],
    metric: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(rows_a) != len(rows_b):
        raise ValueError("rows_a and rows_b must have the same length")
    b = sum(1 for a, bb in zip(rows_a, rows_b) if a[metric] > threshold and bb[metric] <= threshold)
    c = sum(1 for a, bb in zip(rows_a, rows_b) if a[metric] <= threshold and bb[metric] > threshold)
    ties = len(rows_a) - b - c
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "a_wins": b, "b_wins": c, "ties": ties}
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {"chi2": chi2, "p_value": p_value, "a_wins": b, "b_wins": c, "ties": ties}

def pareto_frontier(policies: dict[str, dict[str, float]]) -> dict:
    dominated: set[str] = set()
    dominance_pairs: list[tuple[str, str]] = []

    names = list(policies.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, ua = policies[a]["privacy_score"], policies[a]["utility_score"]
            pb, ub = policies[b]["privacy_score"], policies[b]["utility_score"]
            if pa >= pb and ua >= ub and (pa > pb or ua > ub):
                dominated.add(b)
                dominance_pairs.append((a, b))
            elif pb >= pa and ub >= ua and (pb > pa or ub > ua):
                dominated.add(a)
                dominance_pairs.append((b, a))

    non_dominated = [n for n in names if n not in dominated]
    return {
        "non_dominated": non_dominated,
        "dominated": sorted(dominated),
        "dominance_pairs": dominance_pairs,
        "note": (
            "Non-dominated policies offer distinct privacy–utility trade-offs; "
            "no single policy is strictly better than another on both axes. "
            "Dominated policies are strictly worse on ≥1 axis and no better on any."
        ),
    }

def evaluate_manifest(records: list[dict]) -> dict[str, dict[str, float]]:
    craft_rows: list[dict[str, float]] = []
    naive_rows: list[dict[str, float]] = []

    for record in records:
        original = record["prompt_text"]
        craft_result = transform_text(original)
        naive_result = naive_mask(original)
        craft_rows.append(score_row(original, craft_result))
        naive_rows.append(score_row(original, naive_result))

    return {
        "craft": aggregate(craft_rows),
        "naive_mask": aggregate(naive_rows),
    }
