from __future__ import annotations

import json
import os
import re
from urllib import request

from .transform import TransformResult, naive_mask

_presidio_analyzer = None
_presidio_anonymizer = None

def _get_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()
    return _presidio_analyzer, _presidio_anonymizer

def presidio_sanitize(text: str) -> TransformResult:
    analyzer, anonymizer = _get_presidio()
    try:
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return TransformResult(
                text=text,
                artifact_kind="clean_text",
                operators_applied=["PRESIDIO"],
                placeholders={},
                operator_counts={"PRESIDIO": 0},
                artifact_counts={"presidio": 1},
            )
        from presidio_anonymizer.entities import OperatorConfig
        entity_types = {r.entity_type for r in results}
        operators = {
            et: OperatorConfig("replace", {"new_value": f"<{et}>"})
            for et in entity_types
        }
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return TransformResult(
            text=anonymized.text,
            artifact_kind="presidio",
            operators_applied=["PRESIDIO"],
            placeholders={},
            operator_counts={"PRESIDIO": len(results)},
            artifact_counts={"presidio": 1},
        )
    except Exception as exc:
        return TransformResult(
            text=text,
            artifact_kind="presidio_error",
            operators_applied=["PRESIDIO_ERROR"],
            placeholders={},
            operator_counts={"PRESIDIO_ERROR": 1},
            artifact_counts={"presidio_error": 1},
        )

_SPACY_SENSITIVE_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "NORP",
})

_spacy_nlp = None

def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])
    return _spacy_nlp

def spacy_sanitize(text: str) -> TransformResult:
    nlp = _get_spacy()
    try:
        doc = nlp(text[:100_000])
    except Exception:
        return TransformResult(
            text=text,
            artifact_kind="spacy_error",
            operators_applied=["SPACY_ERROR"],
            placeholders={},
            operator_counts={"SPACY_ERROR": 1},
            artifact_counts={"spacy_error": 1},
        )

    sensitive_ents = [e for e in doc.ents if e.label_ in _SPACY_SENSITIVE_LABELS]
    if not sensitive_ents:
        return TransformResult(
            text=text,
            artifact_kind="spacy_ner",
            operators_applied=["SPACY_NER"],
            placeholders={},
            operator_counts={"SPACY_NER": 0},
            artifact_counts={"spacy_ner": 1},
        )

    result = text
    redacted = 0
    for ent in sorted(sensitive_ents, key=lambda e: e.start_char, reverse=True):
        placeholder = f"<{ent.label_}>"
        result = result[: ent.start_char] + placeholder + result[ent.end_char :]
        redacted += 1

    return TransformResult(
        text=result,
        artifact_kind="spacy_ner",
        operators_applied=["SPACY_NER"],
        placeholders={},
        operator_counts={"SPACY_NER": redacted},
        artifact_counts={"spacy_ner": 1},
    )

LLM_SANITIZE_SYSTEM = """You are a privacy-preserving assistant.
Sanitize the technical prompt below by replacing sensitive VALUES with short typed placeholders.

Rules:
- Keep ALL structural information: error type names, config key names, code structure, file extensions, stack frame method names.
- Replace only the actual sensitive values: passwords, API keys/tokens, IP addresses, hostnames, full file paths (keep only filename), usernames, email addresses, credit card numbers.
- Use concise typed placeholders: <PASSWORD>, <API_KEY>, <IP_ADDRESS>, <HOSTNAME>, <FILE_PATH>, <USERNAME>, <EMAIL>, <TOKEN>.
- Do NOT add explanation. Return ONLY the sanitized text.
"""

def _ollama(model: str, system: str, prompt: str, keep_alive: str, host: str, timeout: int = 120) -> str:
    host = host.rstrip("/")
    payload = {
        "model": model, "system": system, "prompt": prompt,
        "stream": False, "keep_alive": keep_alive,
        "options": {"temperature": 0},
    }
    req = request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return str(json.load(resp).get("response", "")).strip()

def llm_direct_sanitize(
    text: str,
    model: str = "qwen2.5-coder:7b",
    keep_alive: str = "24h",
    host: str = "http://127.0.0.1:11434",
) -> TransformResult:
    try:
        sanitized = _ollama(model, LLM_SANITIZE_SYSTEM, text, keep_alive, host)
        if not sanitized or len(sanitized) < 5:
            sanitized = text
    except Exception:
        sanitized = text
    return TransformResult(
        text=sanitized,
        artifact_kind="llm_direct",
        operators_applied=["LLM_DIRECT"],
        placeholders={},
        operator_counts={"LLM_DIRECT": 1},
        artifact_counts={"llm_direct": 1},
    )
