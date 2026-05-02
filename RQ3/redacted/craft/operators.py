from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from urllib.parse import urlsplit

from redacted.craft.detectors import EXCEPTION_RE, JAVA_FRAME_RE, PY_FRAME_RE
from redacted.craft.detectors import Span

from .artifacts import CREDENTIAL_PATTERNS, HIGH_RISK_PATTERNS, NETWORK_PATTERNS, PATH_PATTERNS

_BENIGN_KEYWORD_RE = re.compile(
    r"^(true|false|yes|no|null|none|enabled|disabled|on|off|\d{1,10}(\.\d+)?|[0-9a-f]{1,8})$",
    re.IGNORECASE,
)
_BENIGN_ENUM_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|secret|token|api[_\-]?key|apikey|auth|credential|private[_\-]?key|access[_\-]?key|client[_\-]?secret)",
    re.IGNORECASE,
)

_CRED_PREFIX_RE = re.compile(r"\b(sk-|ghp_|ghs_|AKIA|eyJ[A-Za-z0-9]|xox[bpoa]-)")

GENERIC_TYPES: dict[str, str] = {
    "api_key_openai": "API_KEY_OPENAI", "github_token": "GITHUB_TOKEN",
    "jwt_token": "JWT_TOKEN", "slack_token": "SLACK_TOKEN",
    "stripe_key": "STRIPE_KEY", "twilio_sid": "TWILIO_SID",
    "sendgrid_key": "SENDGRID_KEY", "google_api_key": "GOOGLE_API_KEY",
    "firebase_url": "FIREBASE_URL", "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "discord_token": "DISCORD_TOKEN", "discord_webhook": "DISCORD_WEBHOOK",
    "npm_token": "NPM_TOKEN", "digitalocean_token": "DIGITALOCEAN_TOKEN",
    "shopify_token": "SHOPIFY_TOKEN", "gitlab_token": "GITLAB_TOKEN",
    "mapbox_token": "MAPBOX_TOKEN", "aws_access_key": "AWS_ACCESS_KEY",
    "gcp_credentials": "GCP_SERVICE_ACCOUNT", "azure_connection": "AZURE_CONNECTION",
    "private_key_header": "PRIVATE_KEY", "ssh_public_key": "SSH_PUBLIC_KEY",
    "certificate": "CERTIFICATE", "internal_ip": "IP_PRIVATE",
    "docker_registry": "DOCKER_REGISTRY", "kubernetes_secret": "K8S_SECRET",
    "docker_compose_secret": "COMPOSE_SECRET", "ansible_vault": "ANSIBLE_VAULT",
    "email_address": "EMAIL", "credit_card": "CARD_NUMBER",
    "iban": "IBAN", "wallet_address_eth": "ETH_ADDRESS",
    "crypto_private_key": "CRYPTO_PRIVATE_KEY",
    "commented_credential": "COMMENTED_CREDENTIAL",
    "proprietary_endpoint": "PRIVATE_ENDPOINT", "ci_secret_var": "CI_SECRET",
    "windows_path": "WIN_PATH", "unix_path": "UNIX_PATH",
    "db_connection_string": "DB_CONN", "jdbc_url": "JDBC_URL",
    "connection_string_kv": "CONN_STR", "url_with_credentials": "URL_WITH_CREDS",
    "stack_trace": "STACK_TRACE",
}

class PlaceholderBank:

    def __init__(self) -> None:
        self._salt: str = os.urandom(16).hex()
        self._value_to_ph: dict[tuple[str, str], str] = {}
        self._ph_set: set[str] = set()
        self._type_counter: Counter[str] = Counter()

    def get(self, placeholder_type: str, value: str) -> str:
        key = (placeholder_type, value)
        if key in self._value_to_ph:
            return self._value_to_ph[key]
        self._type_counter[placeholder_type] += 1
        digest = hashlib.sha256(
            f"{self._salt}:{placeholder_type}:{value}".encode()
        ).hexdigest()
        idx, attempt = digest[:4], 0
        while f"<{placeholder_type}:{idx}>" in self._ph_set:
            attempt += 1
            idx = hashlib.sha256(
                f"{self._salt}:{placeholder_type}:{value}:{attempt}".encode()
            ).hexdigest()[:4]
        ph = f"<{placeholder_type}:{idx}>"
        self._value_to_ph[key] = ph
        self._ph_set.add(ph)
        return ph

    def all_placeholders(self) -> dict[str, str]:
        return {ph: kv[1] for kv, ph in self._value_to_ph.items()}

    def type_counts(self) -> dict[str, int]:
        return dict(self._type_counter)

    def repeated_placeholders(self) -> int:
        counts: Counter[str] = Counter(self._value_to_ph.values())
        return sum(c - 1 for c in counts.values() if c > 1)

def apply_suppress(span: Span) -> str:
    label = GENERIC_TYPES.get(span.pattern, span.pattern.upper())
    return f"<SUPPRESSED:{label}>"

def apply_abstract(span: Span, bank: PlaceholderBank, network_mode: bool = False) -> str:
    pattern, value = span.pattern, span.text
    if pattern == "internal_ip":
        return _abstract_ip(value, bank, network_mode=network_mode)
    if pattern in PATH_PATTERNS:
        return _abstract_path(value, bank, windows=(pattern == "windows_path"))
    if pattern in {"db_connection_string", "jdbc_url", "url_with_credentials",
                   "firebase_url", "proprietary_endpoint"}:
        return _abstract_uri(value, bank, network_mode=network_mode)
    if pattern == "connection_string_kv":
        return _abstract_conn_kv(value, bank)
    ph_type = GENERIC_TYPES.get(pattern, pattern.upper())
    return bank.get(ph_type, value)

def _abstract_ip(value: str, bank: PlaceholderBank, network_mode: bool = False) -> str:
    if network_mode:
        parts = value.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + ".x"
        return "<IP>"
    if re.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}", value):
        return bank.get("IP_10", value)
    if re.fullmatch(r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}", value):
        return bank.get("IP_172", value)
    if re.fullmatch(r"192\.168\.\d{1,3}\.\d{1,3}", value):
        return bank.get("IP_192_168", value)
    return bank.get("IP_PRIVATE", value)

def _abstract_path(value: str, bank: PlaceholderBank, windows: bool) -> str:
    parts = [p for p in value.replace("\\", "/").split("/") if p]
    if not parts:
        return bank.get("PATH", value)
    user_home = ("Users" in parts[:3]) if windows else (parts[0] == "home" and len(parts) > 1)
    keep = parts[-2:] if len(parts) >= 2 else parts[-1:]
    tail = "/".join(keep)
    root_class = ("WIN_HOME" if user_home else "WIN_ROOT") if windows else ("UNIX_HOME" if user_home else "UNIX_ROOT")
    ph = bank.get(root_class, "/".join(parts[: len(parts) - len(keep)]))
    return f"{ph}/…/{tail}" if len(parts) > len(keep) else f"{ph}/{tail}"

def _render_host(host: str, bank: PlaceholderBank, network_mode: bool = False) -> str:
    lowered = host.lower()
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        return "<LOCALHOST>"
    if re.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}", host):
        if network_mode:
            parts = host.split(".")
            return ".".join(parts[:3]) + ".x"
        return bank.get("IP_10", host)
    if re.fullmatch(r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}", host):
        if network_mode:
            parts = host.split(".")
            return ".".join(parts[:3]) + ".x"
        return bank.get("IP_172", host)
    if re.fullmatch(r"192\.168\.\d{1,3}\.\d{1,3}", host):
        if network_mode:
            parts = host.split(".")
            return ".".join(parts[:3]) + ".x"
        return bank.get("IP_192_168", host)
    if lowered.endswith((".internal", ".private", ".local", ".corp", ".lan")):
        if network_mode:
            tld = "." + lowered.rsplit(".", 1)[1]
            return f"<HOST>{tld}"
        return bank.get("PRIVATE_HOST", host)
    return host

def _abstract_uri(value: str, bank: PlaceholderBank, network_mode: bool = False) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        scheme = value.split("://", 1)[0].upper() if "://" in value else "URI"
        return f"<{scheme}_MALFORMED>"
    scheme = (parts.scheme or "uri").upper()
    host_raw = parts.netloc.rsplit("@", 1)[-1].split(":", 1)[0]
    host = _render_host(host_raw, bank, network_mode=network_mode)
    path_part = parts.path if parts.path not in {"", "/"} else ""
    return f"<{scheme}>://{host}{path_part}"

def _abstract_conn_kv(value: str, bank: PlaceholderBank) -> str:
    parts: list[str] = []
    for item in value.split(";"):
        piece = item.strip()
        if not piece or "=" not in piece:
            continue
        key, raw_val = piece.split("=", 1)
        low, val = key.strip().lower(), raw_val.strip()
        if low in {"password", "pwd"}:
            parts.append(f"{key.strip()}={bank.get('DB_PASS', val)}")
        elif low in {"user id", "uid", "username"}:
            parts.append(f"{key.strip()}={bank.get('DB_USER', val)}")
        elif low in {"server", "data source"}:
            parts.append(f"{key.strip()}={_render_host(val, bank)}")
        elif low in {"database", "initial catalog"}:
            safe = val if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", val) else bank.get("DB_NAME", val)
            parts.append(f"{key.strip()}={safe}")
        else:
            parts.append(f"{key.strip()}={val}")
    return ";".join(parts)

_PATH_IN_MSG_RE = re.compile(
    r"(/(?:home|usr|etc|var|opt|srv|tmp|root|app|workspace|data|mnt|run|proc|Users)"
    r"[^\s\"'`\n,){;\\]{1,300})"
    r"|([A-Za-z]:\\[^\s\"'`\n,){;]{1,300})"
)
_PRIVATE_IP_IN_MSG_RE = re.compile(
    r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3})\b"
)

_KW_VALUE_RE = re.compile(
    r"""(\b(?:password|passwd|secret|token|api[_\-]?key|apikey|auth(?:_?token)?|credential|private[_\-]?key)\s*=\s*)"""
    r"""(["\']?)([^"\')\s,\n]{4,})\2""",
    re.IGNORECASE,
)

def _sanitize_source_line(line: str) -> str:
    from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps
    spans = resolve_overlaps(find_sensitive_spans(line))
    if spans:
        parts: list[str] = []
        cursor = 0
        for span in spans:
            parts.append(line[cursor:span.start])
            parts.append(f"<{span.pattern.upper()}>")
            cursor = span.end
        parts.append(line[cursor:])
        line = "".join(parts)
    line = _PATH_IN_MSG_RE.sub("<PATH_REDACTED>", line)
    line = _PRIVATE_IP_IN_MSG_RE.sub("<IP_PRIVATE>", line)
    line = _KW_VALUE_RE.sub(lambda m: m.group(1) + "<SECRET_VALUE>", line)
    return line

def _sanitize_exc_message(msg: str) -> str:
    if not msg:
        return msg
    msg = _PATH_IN_MSG_RE.sub("<PATH_REDACTED>", msg)
    msg = _PRIVATE_IP_IN_MSG_RE.sub("<IP_PRIVATE>", msg)
    msg = _CRED_PREFIX_RE.sub("<CREDENTIAL_REDACTED>", msg)
    return msg[:300] + ("..." if len(msg) > 300 else "")

def _sanitize_frame_path(filepath: str) -> str:
    parts = [p for p in filepath.replace("\\", "/").split("/") if p]
    if len(parts) <= 2:
        return filepath
    keep = "/".join(parts[-2:])
    if parts[0] == "home" and len(parts) > 2:
        return f"<UNIX_HOME>/…/{keep}"
    if parts[0] == "Users" and len(parts) > 2:
        return f"<WIN_HOME>/…/{keep}"
    return f"/{parts[0]}/…/{keep}"

def apply_summarize(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skip_locals = False
    prev_was_frame = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out.append("")
            prev_was_frame = False
            continue

        if stripped.lower() in ("local variables:", "locals:"):
            skip_locals = True
            continue
        if skip_locals:
            if not (line.startswith(" ") or line.startswith("\t")):
                skip_locals = False
            else:
                continue

        if stripped in (
            "Traceback (most recent call last):",
            "During handling of the above exception, another exception occurred:",
            "The above exception was the direct cause of the following exception:",
        ):
            out.append(line)
            prev_was_frame = False
            continue

        if JAVA_FRAME_RE.match(line):
            out.append(line)
            prev_was_frame = True
            continue

        if PY_FRAME_RE.match(line):
            out.append(line)
            prev_was_frame = True
            continue

        if prev_was_frame and re.match(r"^\s{4}", line):
            if stripped.startswith(("raise ", "throw ", "assert ")):
                exc_m = EXCEPTION_RE.search(stripped)
                kw = stripped.split()[0]
                out.append(f"    {kw} {exc_m.group(1)}(...)" if exc_m else f"    {kw}(...)")
            else:
                out.append(_sanitize_source_line(line))
            prev_was_frame = False
            continue

        prev_was_frame = False

        exc_m = EXCEPTION_RE.search(stripped)
        if exc_m:
            exc_name = exc_m.group(1)
            idx = stripped.index(exc_name)
            prefix = stripped[:idx]
            rest = stripped[idx + len(exc_name):].strip().lstrip(":").strip()
            sanitized = _sanitize_exc_message(rest)
            out.append(f"{prefix}{exc_name}: {sanitized}" if sanitized else f"{prefix}{exc_name}")
            continue

        redacted = _PATH_IN_MSG_RE.sub("<PATH_REDACTED>", line)
        redacted = _PRIVATE_IP_IN_MSG_RE.sub("<IP_PRIVATE>", redacted)
        out.append(redacted)

    return "\n".join(out)

def apply_canonicalize(text: str, spans: list[Span], bank: PlaceholderBank) -> str:
    result = re.sub(r"/home/[^/\s\"'`\n]+", "/home/<USER>", text)
    result = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"'`\n]+", r"C:\\Users\\<USER>", result)

    from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps
    new_spans = resolve_overlaps(find_sensitive_spans(result))

    out: list[str] = []
    cursor = 0
    for span in new_spans:
        out.append(result[cursor : span.start])
        context_before = result[max(0, span.start - 60) : span.start]
        if _is_value_position(context_before) and _is_benign_scalar(span.text):
            out.append(span.text)
        else:
            out.append(apply_abstract(span, bank))
        cursor = span.end
    out.append(result[cursor:])
    intermediate = "".join(out)

    return _redact_sensitive_keys(intermediate, bank)

def _is_value_position(context: str) -> bool:
    return bool(re.search(r"[:=]\s*$", context.rstrip()))

def _is_benign_scalar(value: str) -> bool:
    stripped = value.strip()
    return bool(_BENIGN_KEYWORD_RE.match(stripped)) or bool(_BENIGN_ENUM_RE.match(stripped))

def _redact_sensitive_keys(text: str, bank: PlaceholderBank) -> str:
    def _replace_line(m: re.Match) -> str:
        indent, key, sep, value = m.group(1), m.group(2), m.group(3), m.group(4)
        stripped_val = value.strip().strip("\"'")
        if _SENSITIVE_KEY_RE.search(key) and stripped_val and not _is_benign_scalar(stripped_val):
            ph = bank.get("SECRET_VALUE", stripped_val)
            if value.strip().startswith('"'):
                return f"{indent}{key}{sep}\"{ph}\""
            if value.strip().startswith("'"):
                return f"{indent}{key}{sep}'{ph}'"
            return f"{indent}{key}{sep}{ph}"
        return m.group(0)

    pattern = re.compile(
        r"^(\s*)([A-Za-z_][A-Za-z0-9_.\-]{0,63})\s*([=:])\s*(.+)$",
        re.MULTILINE,
    )
    return pattern.sub(_replace_line, text)

_PATH_CONT_RE = re.compile(r"^([/\\][^\s\n\"'`<>()\[\]]{1,512})")

def apply_generalize(text: str, spans: list[Span], bank: PlaceholderBank, keep_tail: int = 2) -> str:
    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor : span.start])
        if span.pattern in PATH_PATTERNS:
            cont = _PATH_CONT_RE.match(text[span.end :])
            full_path = span.text + (cont.group(1) if cont else "")
            end = span.end + (len(cont.group(1)) if cont else 0)
            out.append(_generalize_path(full_path, span.pattern == "windows_path", bank, keep_tail))
            cursor = end
        else:
            out.append(apply_abstract(span, bank))
            cursor = span.end
    out.append(text[cursor:])
    return "".join(out)

def _generalize_path(value: str, windows: bool, bank: PlaceholderBank, keep_tail: int) -> str:
    sep = "\\" if windows else "/"
    parts = [p for p in value.replace("\\", "/").split("/") if p]
    if not parts:
        return bank.get("PATH", value)

    if windows and len(parts) > 2 and parts[1].lower() == "users":
        root_class, skip = "WIN_HOME", 3
    elif not windows and len(parts) > 1 and parts[0] == "home":
        root_class, skip = "UNIX_HOME", 2
    elif not windows and parts[0] in {"etc", "var", "usr", "opt", "srv", "proc"}:
        root_class, skip = f"SYS_{parts[0].upper()}", 1
    else:
        root_class, skip = ("WIN_ROOT" if windows else "UNIX_ROOT"), 1

    tail_start = max(skip, len(parts) - keep_tail)
    tail = parts[tail_start:]
    root_ph = bank.get(root_class, sep.join(parts[:skip]))

    if not tail:
        return root_ph

    tail_str = sep.join(tail)
    if tail_start > skip:
        return f"{root_ph}{sep}…{sep}{tail_str}"
    return f"{root_ph}{sep}{tail_str}"
