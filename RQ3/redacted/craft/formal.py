from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .model import Op

if TYPE_CHECKING:
    from redacted.craft.detectors import Span
    from .model import TransformResult

VALUE_SPACE_LOG2: dict[str, float] = {
    "AWS_ACCESS_KEY":       math.log2(36 ** 16),
    "API_KEY_OPENAI":       math.log2(62 ** 48),
    "GITHUB_TOKEN":         math.log2(62 ** 36),
    "JWT_TOKEN":            math.log2(62 ** 80),
    "SLACK_TOKEN":          math.log2(62 ** 24),
    "STRIPE_KEY":           math.log2(62 ** 24),
    "CRYPTO_PRIVATE_KEY":   256.0,
    "DB_PASS":              math.log2(95 ** 12),
    "DB_USER":              math.log2(26 ** 8),
    "EMAIL":                math.log2(26 ** 8 * 10 ** 6),
    "CARD_NUMBER":          math.log2(10 ** 16),
    "ETH_ADDRESS":          160.0,
    "IBAN":                 math.log2(10 ** 20),
    "PRIVATE_HOST":         math.log2(26 ** 10),
    "SECRET_VALUE":         math.log2(95 ** 16),
}

_DEFAULT_VALUE_SPACE_LOG2: float = math.log2(62 ** 20)

TYPE_VOCABULARY_SIZE: int = 25
TYPE_ENTROPY_BITS: float = math.log2(TYPE_VOCABULARY_SIZE)

@dataclass(frozen=True)
class OperatorPrivacyBound:
    op: Op
    exact_recovery_upper_bound: float
    format_recovery_upper_bound: float
    type_disclosure_weight: float
    linkage_deterministic: bool
    bits_leaked: float

OPERATOR_PRIVACY_BOUNDS: dict[Op, OperatorPrivacyBound] = {
    Op.SUPPRESS: OperatorPrivacyBound(
        op=Op.SUPPRESS,
        exact_recovery_upper_bound=2.0 ** (-_DEFAULT_VALUE_SPACE_LOG2),
        format_recovery_upper_bound=0.50,
        type_disclosure_weight=0.3,
        linkage_deterministic=False,
        bits_leaked=TYPE_ENTROPY_BITS,
    ),
    Op.ABSTRACT: OperatorPrivacyBound(
        op=Op.ABSTRACT,
        exact_recovery_upper_bound=2.0 ** (-_DEFAULT_VALUE_SPACE_LOG2),
        format_recovery_upper_bound=0.70,
        type_disclosure_weight=0.7,
        linkage_deterministic=True,
        bits_leaked=TYPE_ENTROPY_BITS + math.log2(10),
    ),
    Op.SUMMARIZE: OperatorPrivacyBound(
        op=Op.SUMMARIZE,
        exact_recovery_upper_bound=0.0,
        format_recovery_upper_bound=0.0,
        type_disclosure_weight=0.0,
        linkage_deterministic=False,
        bits_leaked=0.0,
    ),
    Op.CANONICALIZE: OperatorPrivacyBound(
        op=Op.CANONICALIZE,
        exact_recovery_upper_bound=2.0 ** (-_DEFAULT_VALUE_SPACE_LOG2),
        format_recovery_upper_bound=0.50,
        type_disclosure_weight=0.5,
        linkage_deterministic=True,
        bits_leaked=TYPE_ENTROPY_BITS + 1.0,
    ),
    Op.GENERALIZE: OperatorPrivacyBound(
        op=Op.GENERALIZE,
        exact_recovery_upper_bound=2.0 ** (-math.log2(10 ** 6)),
        format_recovery_upper_bound=0.30,
        type_disclosure_weight=0.2,
        linkage_deterministic=True,
        bits_leaked=math.log2(5),
    ),
}

def exact_recovery_upper_bound(ph_type: str) -> float:
    return 2.0 ** (-VALUE_SPACE_LOG2.get(ph_type, _DEFAULT_VALUE_SPACE_LOG2))

@dataclass
class DetectionConditionedBound:
    op: Op
    detector_recall: float
    epsilon_operator: float
    epsilon_conditioned: float
    bits_leaked_upper_bound: float

    @classmethod
    def compute(cls, op: Op, detector_recall: float) -> "DetectionConditionedBound":
        bound = OPERATOR_PRIVACY_BOUNDS[op]
        eps_op = bound.exact_recovery_upper_bound
        eps_cond = detector_recall * eps_op + (1.0 - detector_recall) * 1.0
        bits = bound.bits_leaked + (1.0 - detector_recall) * _DEFAULT_VALUE_SPACE_LOG2
        return cls(
            op=op,
            detector_recall=detector_recall,
            epsilon_operator=eps_op,
            epsilon_conditioned=eps_cond,
            bits_leaked_upper_bound=bits,
        )

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "detector_recall": self.detector_recall,
            "epsilon_operator": self.epsilon_operator,
            "epsilon_conditioned": self.epsilon_conditioned,
            "bits_leaked_upper_bound": self.bits_leaked_upper_bound,
        }

def operator_bounds_table(detector_recall: float = 1.0) -> list[dict]:
    return [
        DetectionConditionedBound.compute(op, detector_recall).to_dict()
        for op in OPERATOR_PRIVACY_BOUNDS
    ]

def verify_suppress_guarantee(
    result_text: str,
    original_spans: list["Span"],
) -> dict[str, object]:
    from .artifacts import HIGH_RISK_PATTERNS
    hr_spans = [s for s in original_spans if s.pattern in HIGH_RISK_PATTERNS]
    violations = [s for s in hr_spans if s.text and s.text in result_text]
    return {
        "high_risk_spans_checked": len(hr_spans),
        "violations": len(violations),
        "violation_rate": len(violations) / len(hr_spans) if hr_spans else 0.0,
        "theorem_1_holds": len(violations) == 0,
        "violation_prefixes": [s.text[:8] + "…" for s in violations],
    }

def verify_abstract_linkage_determinism(
    results: list["TransformResult"],
) -> dict[str, object]:
    from collections import defaultdict
    value_to_phs: dict[str, set] = defaultdict(set)
    ph_to_values: dict[str, set] = defaultdict(set)
    for result in results:
        for ph, val in result.placeholders.items():
            value_to_phs[val].add(ph)
            ph_to_values[ph].add(val)
    multi_ph = {v: sorted(phs) for v, phs in value_to_phs.items() if len(phs) > 1}
    multi_val = {ph: sorted(vals) for ph, vals in ph_to_values.items() if len(vals) > 1}
    n = len(value_to_phs)
    return {
        "unique_values_seen": n,
        "values_with_multiple_placeholders": len(multi_ph),
        "placeholders_mapping_multiple_values": len(multi_val),
        "within_corpus_determinism_rate": 1.0 - len(multi_ph) / max(1, n),
        "theorem_2_note": (
            "Cross-session linkage (Theorem 2) holds whenever the same value appears "
            "in two prompts processed by the same PlaceholderBank instance. "
            "Across independent sessions, indices differ but type labels match, "
            "enabling partial linkage via type co-occurrence."
        ),
    }

def compute_mutual_information_estimate(
    attempts: list,
) -> dict[str, float]:
    if not attempts:
        return {"i_exact_lower_bound_bits": 0.0, "i_format_lower_bound_bits": 0.0}
    total = len(attempts)
    exact_rate = sum(a.exact_match for a in attempts) / total
    format_rate = sum(a.format_match for a in attempts) / total
    i_exact = -math.log2(1.0 - exact_rate) if exact_rate < 1.0 else float("inf")
    i_format = -math.log2(1.0 - format_rate) if format_rate < 1.0 else float("inf")
    return {
        "exact_match_rate": exact_rate,
        "format_match_rate": format_rate,
        "i_exact_lower_bound_bits": i_exact,
        "i_format_lower_bound_bits": i_format,
    }

if __name__ == "__main__":
    import json

    recall_levels = [1.0, 0.99, 0.95, 0.90]
    output = {
        f"r={r:.2f}": operator_bounds_table(r) for r in recall_levels
    }
    print(json.dumps(output, indent=2))
