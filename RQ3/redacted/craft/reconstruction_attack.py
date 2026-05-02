from __future__ import annotations

import json
import os
import random
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from urllib import request

_RANDOM = random.SystemRandom()

from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps
from redacted.craft.detectors import Span

from .transform import TransformResult

FORMAT_VALIDATORS: dict[str, re.Pattern] = {
    "API_KEY_OPENAI":   re.compile(r"^sk-[A-Za-z0-9]{20,}$"),
    "GITHUB_TOKEN":     re.compile(r"^(ghp_|ghs_|github_pat_)[A-Za-z0-9_]{10,}$"),
    "AWS_ACCESS_KEY":   re.compile(r"^AKIA[A-Z0-9]{16}$"),
    "JWT_TOKEN":        re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"),
    "SLACK_TOKEN":      re.compile(r"^xox[bpoa]-[A-Za-z0-9-]+$"),
    "STRIPE_KEY":       re.compile(r"^(sk|pk)_(live|test)_[A-Za-z0-9]{24,}$"),
    "UNIX_HOME":        re.compile(r"^/home/[a-z][a-z0-9_-]{0,31}(/.*)?$"),
    "UNIX_ROOT":        re.compile(r"^/[a-z][a-z0-9/._-]+$"),
    "WIN_HOME":         re.compile(r"^[A-Z]:\\Users\\[A-Za-z][A-Za-z0-9 _-]*"),
    "WIN_ROOT":         re.compile(r"^[A-Z]:\\[A-Za-z0-9 _\\.-]+$"),
    "IP_10":            re.compile(r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
    "IP_172":           re.compile(r"^172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}$"),
    "IP_192_168":       re.compile(r"^192\.168\.\d{1,3}\.\d{1,3}$"),
    "IP_PRIVATE":       re.compile(r"^(10|172|192\.168)\.\d+\.\d+\.\d+$"),
    "EMAIL":            re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE),
    "PRIVATE_HOST":     re.compile(r"^[a-z][a-z0-9.-]*\.(internal|local|corp|lan|private)$"),
    "DB_PASS":          re.compile(r"^.{6,}$"),
    "DB_USER":          re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,31}$"),
    "SECRET_VALUE":     re.compile(r"^.{4,}$"),
    "CARD_NUMBER":      re.compile(r"^\d{13,19}$"),
    "IBAN":             re.compile(r"^[A-Z]{2}\d{2}[A-Za-z0-9]{1,30}$"),
    "ETH_ADDRESS":      re.compile(r"^0x[0-9a-fA-F]{40}$"),
    "CRYPTO_PRIVATE_KEY": re.compile(r"^[A-Za-z0-9+/=]{40,}$"),
}

CATEGORY_BUCKETS: dict[str, str] = {
    "API_KEY_OPENAI": "api_token", "GITHUB_TOKEN": "api_token",
    "SLACK_TOKEN": "api_token", "STRIPE_KEY": "api_token",
    "AWS_ACCESS_KEY": "api_token", "GITLAB_TOKEN": "api_token",
    "TELEGRAM_BOT_TOKEN": "api_token", "DISCORD_TOKEN": "api_token",
    "PRIVATE_KEY": "cryptographic_key", "SSH_PUBLIC_KEY": "cryptographic_key",
    "CERTIFICATE": "cryptographic_key", "CRYPTO_PRIVATE_KEY": "cryptographic_key",
    "JWT_TOKEN": "auth_token",
    "UNIX_HOME": "filesystem_path", "UNIX_ROOT": "filesystem_path",
    "WIN_HOME": "filesystem_path", "WIN_ROOT": "filesystem_path",
    "IP_10": "network_address", "IP_172": "network_address",
    "IP_192_168": "network_address", "IP_PRIVATE": "network_address",
    "PRIVATE_HOST": "network_address",
    "EMAIL": "pii_identifier",
    "CARD_NUMBER": "pii_financial", "IBAN": "pii_financial",
    "DB_PASS": "credential", "DB_USER": "credential", "SECRET_VALUE": "credential",
    "ETH_ADDRESS": "crypto_identifier",
}

TYPED_ATTACK_SYSTEM = """You are a red-team security researcher evaluating prompt sanitization strength.

You have a sanitized prompt with typed placeholder tokens like <AWS_ACCESS_KEY:1>, <UNIX_HOME:2>, <SECRET_VALUE:3>.
Each token encodes the semantic type and an index.

Your task: attempt to reconstruct the EXACT original sensitive value for each unique placeholder.
Use every available signal:
1. The type label (e.g., AWS_ACCESS_KEY tells you the format is AKIA[A-Z0-9]{16})
2. Context in the surrounding prompt text
3. Common real-world values for that type

Return JSON only — one entry per unique placeholder token seen in the prompt:
{
  "attempts": [
    {
      "placeholder": "<EXACT_PLACEHOLDER_TOKEN>",
      "reconstructed_value": "your best guess at the exact original value",
      "confidence": "high|medium|low"
    }
  ]
}
"""

SUPPRESSED_ATTACK_SYSTEM = """You are a red-team security researcher evaluating prompt sanitization strength.

You have a sanitized prompt with suppressed placeholder tokens like <SUPPRESSED:AWS_ACCESS_KEY>.
These tokens tell you the type but reveal zero information about the actual value.

Your task: attempt to reconstruct the original value for each suppressed token.
Use only:
1. The type name (determines the format)
2. Context clues in the surrounding text

Return JSON only — one entry per unique suppressed token:
{
  "attempts": [
    {
      "placeholder": "<EXACT_SUPPRESSED_TOKEN>",
      "reconstructed_value": "your best guess at the original value",
      "confidence": "high|medium|low"
    }
  ]
}
"""

BLIND_ATTACK_SYSTEM = """You are a red-team security researcher evaluating prompt sanitization strength.

You have a sanitized prompt where ALL sensitive values have been replaced with the identical token <REDACTED>.
No type information is available. You must infer what each <REDACTED> was from context alone.

Return JSON only — number attempts by order of appearance of <REDACTED> in the text:
{
  "attempts": [
    {
      "index": 1,
      "reconstructed_value": "your best guess for the 1st <REDACTED>",
      "confidence": "high|medium|low"
    }
  ]
}
"""

def _ollama(model: str, system: str, prompt: str, keep_alive: str, timeout: int = 300) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
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

def _parse_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = text.rfind("{")
    while start != -1:
        try:
            return json.loads(text[start:].strip())
        except json.JSONDecodeError:
            start = text.rfind("{", 0, start)
    return {"attempts": []}

@dataclass
class AttackAttempt:
    placeholder: str
    placeholder_type: str
    original_value: str
    reconstructed_value: str
    confidence: str
    exact_match: bool
    format_match: bool
    category_match: bool

    def to_dict(self) -> dict:
        return asdict(self)

def _ph_type(placeholder: str) -> str:
    inner = placeholder.strip("<>")
    if inner.startswith("SUPPRESSED:"):
        return inner.split(":", 1)[1]
    if ":" in inner:
        return inner.rsplit(":", 1)[0]
    return inner

def _format_match(ph_type: str, value: str) -> bool:
    v = value.strip()
    validator = FORMAT_VALIDATORS.get(ph_type)
    if validator:
        return bool(validator.match(v))
    return bool(v)

def _category_match(ph_type: str, guess: str) -> bool:
    expected_bucket = CATEGORY_BUCKETS.get(ph_type)
    if not expected_bucket:
        return False
    g = guess.strip().lower()
    if expected_bucket == "api_token":
        return bool(re.search(r"[A-Za-z0-9_-]{20,}", g))
    if expected_bucket == "filesystem_path":
        return g.startswith("/") or bool(re.match(r"[a-z]:\\", g, re.IGNORECASE))
    if expected_bucket == "network_address":
        return bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", g) or re.search(r"\.(internal|local|corp|lan)", g))
    if expected_bucket == "pii_identifier":
        return "@" in g
    if expected_bucket == "auth_token":
        parts = g.split(".")
        return len(parts) == 3 and all(p for p in parts)
    if expected_bucket == "credential":
        return len(g) >= 4
    return False

def _score_attempt(placeholder: str, original: str, guess: str, confidence: str) -> AttackAttempt:
    ph_type = _ph_type(placeholder)
    exact = guess.strip().lower() == original.strip().lower()
    fmt = _format_match(ph_type, guess)
    cat = _category_match(ph_type, guess)
    return AttackAttempt(
        placeholder=placeholder,
        placeholder_type=ph_type,
        original_value=original,
        reconstructed_value=guess,
        confidence=confidence,
        exact_match=exact,
        format_match=fmt,
        category_match=cat,
    )

def _typed_attack(
    model: str,
    sanitized_text: str,
    placeholders: dict[str, str],
    keep_alive: str,
    suppressed: bool = False,
) -> list[AttackAttempt]:
    if not placeholders:
        return []
    system = SUPPRESSED_ATTACK_SYSTEM if suppressed else TYPED_ATTACK_SYSTEM
    prompt = f"Sanitized prompt:\n{sanitized_text[:6000]}"
    try:
        raw = _ollama(model, system, prompt, keep_alive)
        parsed = _parse_json(raw)
    except Exception:
        return []
    attempts_raw = parsed.get("attempts", [])
    attempt_map: dict[str, dict] = {}
    for a in attempts_raw:
        ph = str(a.get("placeholder", "")).strip()
        if ph:
            attempt_map[ph] = a
    results: list[AttackAttempt] = []
    for ph, orig in placeholders.items():
        a = attempt_map.get(ph, {})
        guess = str(a.get("reconstructed_value", ""))
        confidence = str(a.get("confidence", "low"))
        results.append(_score_attempt(ph, orig, guess, confidence))
    return results

def _blind_attack(
    model: str,
    sanitized_text: str,
    original_spans: list[Span],
    keep_alive: str,
) -> list[AttackAttempt]:
    n = sanitized_text.count("<REDACTED>")
    if n == 0:
        return []
    prompt = f"Sanitized prompt:\n{sanitized_text[:6000]}\n\nThere are {n} <REDACTED> token(s) in order."
    try:
        raw = _ollama(model, BLIND_ATTACK_SYSTEM, prompt, keep_alive)
        parsed = _parse_json(raw)
    except Exception:
        return []
    attempts_raw = sorted(parsed.get("attempts", []), key=lambda a: int(a.get("index", 0)))
    results: list[AttackAttempt] = []
    for i, span in enumerate(original_spans):
        a = attempts_raw[i] if i < len(attempts_raw) else {}
        guess = str(a.get("reconstructed_value", ""))
        confidence = str(a.get("confidence", "low"))
        results.append(_score_attempt("<REDACTED>", span.text, guess, confidence))
    return results

def attack_result(
    model: str,
    original: str,
    result: TransformResult,
    keep_alive: str,
) -> list[AttackAttempt]:
    is_naive = result.operators_applied == ["NAIVE_MASK"]
    is_suppress_heavy = result.operator_counts.get("SUPPRESS", 0) > result.operator_counts.get("ABSTRACT", 0)

    if is_naive:
        spans = resolve_overlaps(find_sensitive_spans(original))
        return _blind_attack(model, result.text, spans, keep_alive)

    suppressed_phs = {ph: v for ph, v in result.placeholders.items() if ph.startswith("<SUPPRESSED:")}
    typed_phs = {ph: v for ph, v in result.placeholders.items() if not ph.startswith("<SUPPRESSED:")}

    attempts: list[AttackAttempt] = []
    if typed_phs:
        attempts.extend(_typed_attack(model, result.text, typed_phs, keep_alive, suppressed=False))
    if suppressed_phs:
        attempts.extend(_typed_attack(model, result.text, suppressed_phs, keep_alive, suppressed=True))
    return attempts

def linkage_attack(
    policy_results: list[TransformResult],
) -> dict[str, float]:
    ph_to_prompts: dict[str, list[int]] = {}
    for idx, result in enumerate(policy_results):
        for ph in result.placeholders:
            ph_to_prompts.setdefault(ph, []).append(idx)

    linked_pairs = 0
    suppressed_linked_pairs = 0
    total_pairs = 0
    n = len(policy_results)
    if n < 2:
        return {"linked_pair_rate": 0.0, "suppressed_link_rate": 0.0}

    for ph, prompt_indices in ph_to_prompts.items():
        if len(prompt_indices) >= 2:
            k = len(prompt_indices)
            pairs = k * (k - 1) // 2
            if ph.startswith("<SUPPRESSED:"):
                suppressed_linked_pairs += pairs
            else:
                linked_pairs += pairs

    total_pairs = n * (n - 1) // 2
    return {
        "linked_pair_rate": linked_pairs / total_pairs if total_pairs else 0.0,
        "suppressed_link_rate": suppressed_linked_pairs / total_pairs if total_pairs else 0.0,
    }

def aggregate_attempts(attempts: list[AttackAttempt]) -> dict[str, float]:
    if not attempts:
        return {
            "total_attempts": 0,
            "exact_match_rate": 0.0,
            "format_match_rate": 0.0,
            "category_match_rate": 0.0,
            "high_confidence_exact_rate": 0.0,
        }
    total = len(attempts)
    high_conf = [a for a in attempts if a.confidence == "high"]
    return {
        "total_attempts": total,
        "exact_match_rate": sum(a.exact_match for a in attempts) / total,
        "format_match_rate": sum(a.format_match for a in attempts) / total,
        "category_match_rate": sum(a.category_match for a in attempts) / total,
        "high_confidence_exact_rate": (
            sum(a.exact_match for a in high_conf) / len(high_conf) if high_conf else 0.0
        ),
    }

def aggregate_by_type(attempts: list[AttackAttempt]) -> dict[str, dict[str, float]]:
    by_type: dict[str, list[AttackAttempt]] = {}
    for a in attempts:
        by_type.setdefault(a.placeholder_type, []).append(a)
    return {t: aggregate_attempts(aa) for t, aa in sorted(by_type.items())}

_ALNUM = string.ascii_letters + string.digits

def _generate_format_valid(ph_type: str) -> str:
    if ph_type == "API_KEY_OPENAI":
        return "sk-" + "".join(_RANDOM.choices(_ALNUM, k=48))
    if ph_type == "GITHUB_TOKEN":
        return "ghp_" + "".join(_RANDOM.choices(_ALNUM, k=36))
    if ph_type == "AWS_ACCESS_KEY":
        return "AKIA" + "".join(_RANDOM.choices(string.ascii_uppercase + string.digits, k=16))
    if ph_type == "JWT_TOKEN":
        b64 = _ALNUM + "-_"
        return (
            "".join(_RANDOM.choices(b64, k=36))
            + "." + "".join(_RANDOM.choices(b64, k=24))
            + "." + "".join(_RANDOM.choices(b64, k=43))
        )
    if ph_type == "SLACK_TOKEN":
        d = string.digits
        return "xoxb-" + "".join(_RANDOM.choices(d, k=11)) + "-" + "".join(_RANDOM.choices(d, k=11)) + "-" + "".join(_RANDOM.choices(_ALNUM, k=24))
    if ph_type == "STRIPE_KEY":
        return "sk_live_" + "".join(_RANDOM.choices(_ALNUM, k=24))
    if ph_type in ("IP_10", "IP_PRIVATE"):
        return f"10.{_RANDOM.randint(0, 255)}.{_RANDOM.randint(0, 255)}.{_RANDOM.randint(1, 254)}"
    if ph_type == "IP_172":
        return f"172.{_RANDOM.randint(16, 31)}.{_RANDOM.randint(0, 255)}.{_RANDOM.randint(1, 254)}"
    if ph_type == "IP_192_168":
        return f"192.168.{_RANDOM.randint(0, 255)}.{_RANDOM.randint(1, 254)}"
    if ph_type == "EMAIL":
        name = "".join(_RANDOM.choices(string.ascii_lowercase, k=6))
        return f"{name}@example.com"
    if ph_type == "UNIX_HOME":
        user = "".join(_RANDOM.choices(string.ascii_lowercase, k=6))
        return f"/home/{user}"
    if ph_type == "UNIX_ROOT":
        root = _RANDOM.choice(["etc", "var", "opt", "usr"])
        sub = "".join(_RANDOM.choices(string.ascii_lowercase, k=6))
        return f"/{root}/{sub}"
    if ph_type == "WIN_HOME":
        user = "".join(_RANDOM.choices(string.ascii_letters, k=6))
        return f"C:\\Users\\{user}"
    if ph_type == "WIN_ROOT":
        sub = "".join(_RANDOM.choices(string.ascii_letters, k=6))
        return f"C:\\{sub}"
    if ph_type == "PRIVATE_HOST":
        name = "".join(_RANDOM.choices(string.ascii_lowercase, k=8))
        return f"{name}.internal"
    if ph_type in ("DB_PASS", "SECRET_VALUE"):
        chars = _ALNUM + "!@#$%^&*"
        return "".join(_RANDOM.choices(chars, k=16))
    if ph_type == "DB_USER":
        return "".join(_RANDOM.choices(string.ascii_lowercase, k=8))
    if ph_type == "ETH_ADDRESS":
        return "0x" + "".join(_RANDOM.choices("0123456789abcdef", k=40))
    if ph_type == "IBAN":
        cc = _RANDOM.choice(["DE", "GB", "FR", "ES"])
        return cc + str(_RANDOM.randint(10, 99)) + "".join(_RANDOM.choices(string.digits, k=16))
    if ph_type == "CARD_NUMBER":
        return "".join(_RANDOM.choices(string.digits, k=16))
    if ph_type == "CRYPTO_PRIVATE_KEY":
        return "".join(_RANDOM.choices(_ALNUM + "+/", k=64)) + "=="
    return "".join(_RANDOM.choices(_ALNUM, k=16))

def oracle_format_attack(result: "TransformResult") -> list[AttackAttempt]:
    attempts: list[AttackAttempt] = []
    for ph, orig in result.placeholders.items():
        ph_type = _ph_type(ph)
        guess = _generate_format_valid(ph_type)
        attempts.append(_score_attempt(ph, orig, guess, "high"))
    return attempts

def aggregate_oracle_comparison(
    oracle_attempts: list[AttackAttempt],
    llm_attempts: list[AttackAttempt],
) -> dict:
    oracle = aggregate_attempts(oracle_attempts)
    llm = aggregate_attempts(llm_attempts)
    return {
        "oracle_upper_bound": oracle,
        "llm_attacker": llm,
        "format_match_gap": oracle.get("format_match_rate", 0.0) - llm.get("format_match_rate", 0.0),
        "exact_match_gap": oracle.get("exact_match_rate", 0.0) - llm.get("exact_match_rate", 0.0),
    }
