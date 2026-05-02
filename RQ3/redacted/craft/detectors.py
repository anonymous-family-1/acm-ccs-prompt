from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ._patterns import PATTERNS

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    pattern: str
    text: str

JAVA_FRAME_RE = re.compile(r"^\s*at\s+([A-Za-z0-9_.$<>]+)\(([A-Za-z0-9_.$<>-]+):(\d+)\)\s*$")
PY_FRAME_RE   = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in ([A-Za-z0-9_.$<>-]+))?')
EXCEPTION_RE  = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Exception|Error|Traceback))\b")
CONFIG_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]{1,63})\s*[:=]", re.MULTILINE)
SAFE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{2,}")

HIGH_RISK_PATTERNS = {
    "api_key_openai", "github_token", "jwt_token", "slack_token", "stripe_key",
    "twilio_sid", "sendgrid_key", "google_api_key", "telegram_bot_token",
    "discord_token", "discord_webhook", "npm_token", "digitalocean_token",
    "shopify_token", "gitlab_token", "mapbox_token", "aws_access_key",
    "gcp_credentials", "azure_connection", "private_key_header", "ssh_public_key",
    "certificate", "url_with_credentials", "db_connection_string", "jdbc_url",
    "internal_ip", "email_address", "credit_card", "iban", "wallet_address_eth",
    "crypto_private_key", "docker_compose_secret", "commented_credential",
    "ci_secret_var",
}

def find_sensitive_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for pattern_name, regex in PATTERNS.items():
        for match in regex.finditer(text):
            spans.append(Span(match.start(), match.end(), pattern_name, match.group(0)))
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    return spans

def resolve_overlaps(spans: list[Span]) -> list[Span]:
    chosen: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if any(not (span.end <= kept.start or span.start >= kept.end) for kept in chosen):
            continue
        chosen.append(span)
    chosen.sort(key=lambda s: s.start)
    return chosen

def safe_token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in SAFE_TOKEN_RE.finditer(text)}

def exception_tokens(text: str) -> set[str]:
    return {m.group(1) for m in EXCEPTION_RE.finditer(text)}

def config_keys(text: str) -> set[str]:
    return {m.group(1) for m in CONFIG_KEY_RE.finditer(text)}

def frame_count(text: str) -> int:
    total = 0
    for line in text.splitlines():
        if JAVA_FRAME_RE.match(line) or PY_FRAME_RE.match(line):
            total += 1
        elif "JAVA_FRAME method=" in line or "PY_FRAME file=" in line:
            total += 1
    return total

def char_mask(text: str, spans: list[Span]) -> list[bool]:
    masked = [False] * len(text)
    for span in spans:
        for idx in range(span.start, span.end):
            masked[idx] = True
    return masked

def strip_sensitive_spans(text: str, spans: list[Span]) -> str:
    masked = char_mask(text, spans)
    return "".join(ch for idx, ch in enumerate(text) if not masked[idx])

def residual_matches(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for span in resolve_overlaps(find_sensitive_spans(text)):
        counter[span.pattern] += 1
    return counter
