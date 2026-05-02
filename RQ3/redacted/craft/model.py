from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

class Op(str, Enum):
    SUPPRESS = "SUPPRESS"
    ABSTRACT = "ABSTRACT"
    SUMMARIZE = "SUMMARIZE"
    CANONICALIZE = "CANONICALIZE"
    GENERALIZE = "GENERALIZE"

OPERATOR_SPECS: dict[str, dict[str, object]] = {
    Op.SUPPRESS: {
        "preserves": ("semantic_type",),
        "removes": ("value", "format", "structure", "context"),
        "privacy_level": "maximum",
    },
    Op.ABSTRACT: {
        "preserves": ("semantic_type", "cardinality", "relative_position"),
        "removes": ("value", "exact_format"),
        "privacy_level": "high",
    },
    Op.SUMMARIZE: {
        "preserves": ("exception_type", "frame_count", "method_signatures", "file_names", "line_numbers"),
        "removes": ("variable_values", "message_details", "full_class_hierarchy", "local_state"),
        "privacy_level": "high",
    },
    Op.CANONICALIZE: {
        "preserves": ("key_names", "structure", "benign_scalar_values", "value_types"),
        "removes": ("sensitive_values", "user_identifiers", "credentials", "internal_hosts"),
        "privacy_level": "medium",
    },
    Op.GENERALIZE: {
        "preserves": ("path_tail", "path_depth", "separator_style", "file_extension"),
        "removes": ("absolute_root", "username", "hostname", "intermediate_directories"),
        "privacy_level": "high",
    },
}

@dataclass(frozen=True)
class Artifact:
    kind: str
    start: int
    end: int
    text: str
    spans: tuple = ()
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class OperatorPipeline:
    artifact_kind: str
    ops: tuple[Op, ...]
    rationale: str

@dataclass
class TransformResult:
    text: str
    artifact_kind: str
    operators_applied: list[str]
    placeholders: dict[str, str]
    operator_counts: dict[str, int]
    artifact_counts: dict[str, int]
    high_risk_span_count: int = 0
    high_risk_suppress_count: int = 0
    high_risk_abstract_count: int = 0
