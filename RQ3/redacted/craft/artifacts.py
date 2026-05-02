from __future__ import annotations

from collections import Counter

from redacted.craft.detectors import CONFIG_KEY_RE, EXCEPTION_RE, JAVA_FRAME_RE, PY_FRAME_RE
from redacted.craft.detectors import Span

from .model import Artifact

HIGH_RISK_PATTERNS: frozenset[str] = frozenset({
    "api_key_openai", "github_token", "jwt_token", "slack_token", "stripe_key",
    "twilio_sid", "sendgrid_key", "google_api_key", "telegram_bot_token",
    "discord_token", "discord_webhook", "npm_token", "digitalocean_token",
    "shopify_token", "gitlab_token", "mapbox_token", "aws_access_key",
    "gcp_credentials", "azure_connection", "private_key_header", "ssh_public_key",
    "certificate", "docker_compose_secret", "commented_credential", "ci_secret_var",
    "crypto_private_key", "ansible_vault",
})

NETWORK_PATTERNS: frozenset[str] = frozenset({
    "url_with_credentials", "db_connection_string", "jdbc_url",
    "connection_string_kv", "internal_ip", "proprietary_endpoint",
    "firebase_url", "docker_registry", "kubernetes_secret",
})

CREDENTIAL_PATTERNS: frozenset[str] = frozenset({
    "api_key_openai", "github_token", "jwt_token", "slack_token", "stripe_key",
    "twilio_sid", "sendgrid_key", "google_api_key", "telegram_bot_token",
    "discord_token", "discord_webhook", "npm_token", "digitalocean_token",
    "shopify_token", "gitlab_token", "mapbox_token", "aws_access_key",
    "gcp_credentials", "private_key_header", "ssh_public_key", "certificate",
    "crypto_private_key", "docker_compose_secret", "commented_credential",
    "ci_secret_var", "ansible_vault", "azure_connection",
})

PATH_PATTERNS: frozenset[str] = frozenset({"windows_path", "unix_path"})

IDENTIFIER_PATTERNS: frozenset[str] = frozenset({
    "email_address", "credit_card", "iban", "wallet_address_eth",
})

def classify_artifact_kind(text: str, spans: list[Span]) -> tuple[str, dict[str, object]]:
    counts: Counter[str] = Counter(span.pattern for span in spans)
    lines = text.splitlines()

    frame_hits = sum(1 for line in lines if JAVA_FRAME_RE.match(line) or PY_FRAME_RE.match(line))
    exc_hits = len(EXCEPTION_RE.findall(text))
    key_hits = len(CONFIG_KEY_RE.findall(text))

    has_network = any(p in NETWORK_PATTERNS for p in counts)
    has_credential = any(p in CREDENTIAL_PATTERNS for p in counts)
    has_path = any(p in PATH_PATTERNS for p in counts)
    has_identifier = any(p in IDENTIFIER_PATTERNS for p in counts)
    has_stack = bool(counts.get("stack_trace") or frame_hits >= 2 or (exc_hits >= 1 and len(lines) >= 3))

    metadata: dict[str, object] = {
        "frame_hits": frame_hits,
        "exception_hits": exc_hits,
        "config_key_hits": key_hits,
        "pattern_counts": dict(counts),
        "total_spans": len(spans),
    }

    sensitive_kind_count = sum([
        has_stack, has_network, has_credential, has_path, has_identifier, key_hits >= 3,
    ])

    if has_stack:
        return "stack_trace", metadata

    if sensitive_kind_count >= 2:
        return "mixed_artifact", metadata

    if has_network:
        return "network_artifact", metadata

    if key_hits >= 3:
        return ("secret_blob" if has_credential else "config_blob"), metadata

    if has_credential:
        return "secret_blob", metadata

    if has_identifier:
        return "identifier_blob", metadata

    if has_path:
        return "filesystem_trace", metadata

    if not spans:
        return "clean_text", metadata

    return "mixed_artifact", metadata

def build_artifact(text: str, spans: list[Span]) -> Artifact:
    kind, metadata = classify_artifact_kind(text, spans)
    return Artifact(kind=kind, start=0, end=len(text), text=text, spans=tuple(spans), metadata=metadata)
