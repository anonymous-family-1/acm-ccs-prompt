from __future__ import annotations

from collections import Counter

from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps

from .artifacts import HIGH_RISK_PATTERNS, build_artifact
from .model import Op, OperatorPipeline, TransformResult
from .operators import (
    PlaceholderBank,
    apply_abstract,
    apply_canonicalize,
    apply_generalize,
    apply_summarize,
    apply_suppress,
)

CRAFT_PIPELINES: dict[str, OperatorPipeline] = {
    "stack_trace": OperatorPipeline(
        "stack_trace",
        (Op.SUMMARIZE,),
        "Structural extraction: preserves exception chain and frame signatures; removes all runtime values.",
    ),
    "config_blob": OperatorPipeline(
        "config_blob",
        (Op.CANONICALIZE,),
        "Key-preserving redaction: retains keys and benign scalars; key-name-aware abstraction of secrets.",
    ),
    "secret_blob": OperatorPipeline(
        "secret_blob",
        (Op.SUPPRESS, Op.ABSTRACT),
        "Tiered redaction: HIGH_RISK spans suppressed to zero information; remainder typed-abstracted.",
    ),
    "network_artifact": OperatorPipeline(
        "network_artifact",
        (Op.ABSTRACT,),
        "URI decomposition: scheme and host class preserved; credentials and internal hosts abstracted.",
    ),
    "filesystem_trace": OperatorPipeline(
        "filesystem_trace",
        (Op.GENERALIZE,),
        "Path-suffix preservation: last two components retained; root, username, and intermediates removed.",
    ),
    "identifier_blob": OperatorPipeline(
        "identifier_blob",
        (Op.ABSTRACT,),
        "Typed abstraction: PII semantic class preserved; all identifying values removed.",
    ),
    "mixed_artifact": OperatorPipeline(
        "mixed_artifact",
        (Op.SUPPRESS, Op.ABSTRACT),
        "Conservative tiered redaction: HIGH_RISK spans suppressed; all other spans typed-abstracted.",
    ),
    "clean_text": OperatorPipeline(
        "clean_text",
        (),
        "No transformation: no sensitive content detected.",
    ),
}

def _uniform_pipeline(op: Op, label: str) -> dict[str, OperatorPipeline]:
    kinds = list(CRAFT_PIPELINES.keys())
    return {
        kind: OperatorPipeline(kind, (op,), f"{label}: applied uniformly regardless of artifact kind.")
        for kind in kinds
        if kind != "clean_text"
    } | {"clean_text": CRAFT_PIPELINES["clean_text"]}

ABLATION_ABSTRACT_ONLY = _uniform_pipeline(Op.ABSTRACT, "ABSTRACT-only ablation")
ABLATION_SUPPRESS_ONLY = _uniform_pipeline(Op.SUPPRESS, "SUPPRESS-only ablation")

def transform_text(text: str, pipeline: OperatorPipeline | None = None) -> TransformResult:
    spans = resolve_overlaps(find_sensitive_spans(text))
    artifact = build_artifact(text, spans)

    if pipeline is None:
        pipeline = CRAFT_PIPELINES.get(artifact.kind, CRAFT_PIPELINES["mixed_artifact"])

    ops = pipeline.ops
    op_counts: Counter[str] = Counter()
    hr_total = sum(1 for s in spans if s.pattern in HIGH_RISK_PATTERNS)
    hr_suppress = 0
    hr_abstract = 0

    if not ops:
        return TransformResult(
            text=text,
            artifact_kind=artifact.kind,
            operators_applied=[],
            placeholders={},
            operator_counts={},
            artifact_counts={artifact.kind: 1},
            high_risk_span_count=hr_total,
        )

    bank = PlaceholderBank()

    if Op.SUMMARIZE in ops:
        rewritten = apply_summarize(text)
        op_counts[Op.SUMMARIZE.value] += 1

    elif Op.CANONICALIZE in ops:
        rewritten = apply_canonicalize(text, spans, bank)
        op_counts[Op.CANONICALIZE.value] += 1

    elif Op.GENERALIZE in ops:
        rewritten = apply_generalize(text, spans, bank)
        op_counts[Op.GENERALIZE.value] += 1

    else:
        net_mode = (artifact.kind == "network_artifact")
        out: list[str] = []
        cursor = 0
        for span in spans:
            out.append(text[cursor : span.start])
            if Op.SUPPRESS in ops and span.pattern in HIGH_RISK_PATTERNS:
                out.append(apply_suppress(span))
                op_counts[Op.SUPPRESS.value] += 1
                hr_suppress += 1
            else:
                out.append(apply_abstract(span, bank, network_mode=net_mode))
                op_counts[Op.ABSTRACT.value] += 1
                if span.pattern in HIGH_RISK_PATTERNS:
                    hr_abstract += 1
            cursor = span.end
        out.append(text[cursor:])
        rewritten = "".join(out)

    return TransformResult(
        text=rewritten,
        artifact_kind=artifact.kind,
        operators_applied=[op.value for op in ops],
        placeholders=bank.all_placeholders(),
        operator_counts=dict(op_counts),
        artifact_counts={artifact.kind: 1},
        high_risk_span_count=hr_total,
        high_risk_suppress_count=hr_suppress,
        high_risk_abstract_count=hr_abstract,
    )

def transform_text_ablation(text: str, pipelines: dict[str, OperatorPipeline]) -> TransformResult:
    spans = resolve_overlaps(find_sensitive_spans(text))
    artifact = build_artifact(text, spans)
    pipeline = pipelines.get(artifact.kind, pipelines.get("mixed_artifact", CRAFT_PIPELINES["mixed_artifact"]))
    return transform_text(text, pipeline=pipeline)

def naive_mask(text: str) -> TransformResult:
    spans = resolve_overlaps(find_sensitive_spans(text))
    artifact = build_artifact(text, spans)
    hr_count = sum(1 for s in spans if s.pattern in HIGH_RISK_PATTERNS)
    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        out.append("<REDACTED>")
        cursor = span.end
    out.append(text[cursor:])
    return TransformResult(
        text="".join(out),
        artifact_kind=artifact.kind,
        operators_applied=["NAIVE_MASK"],
        placeholders={},
        operator_counts={"NAIVE_MASK": len(spans)},
        artifact_counts={artifact.kind: 1},
        high_risk_span_count=hr_count,
    )
