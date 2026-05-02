from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PATTERNS = {
    "api_key_openai": re.compile(r"sk-(proj-|svcacct-)?[A-Za-z0-9]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "jwt_token": re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    "stripe_key": re.compile(r"(sk|pk)_(test|live)_[A-Za-z0-9]{24,}"),
    "twilio_sid": re.compile(r"AC[A-F0-9]{32}"),
    "sendgrid_key": re.compile(r"SG\.[A-Za-z0-9\-_]{22,}\.[A-Za-z0-9\-_]{43,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "firebase_url": re.compile(r"https://[a-z0-9\-]+\.firebaseio\.com", re.IGNORECASE),
    "telegram_bot_token": re.compile(r"\d{8,10}:[A-Za-z0-9\-_]{35}"),
    "discord_token": re.compile(r"[MN][A-Za-z0-9]{23}\.[A-Za-z0-9\-_]{6}\.[A-Za-z0-9\-_]{27}"),
    "discord_webhook": re.compile(r"https://discord(app)?\.com/api/webhooks/\d+/[A-Za-z0-9\-_]+", re.IGNORECASE),
    "npm_token": re.compile(r"npm_[A-Za-z0-9]{36}"),
    "digitalocean_token": re.compile(r"dop_v1_[a-f0-9]{64}"),
    "shopify_token": re.compile(r"shpat_[a-fA-F0-9]{32}"),
    "gitlab_token": re.compile(r"glpat-[A-Za-z0-9\-_]{20}"),
    "mapbox_token": re.compile(r"pk\.eyJ1[A-Za-z0-9\-_\.]+"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "gcp_credentials": re.compile(r'"type"\s*:\s*"service_account"', re.IGNORECASE),
    "azure_connection": re.compile(r"DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]+", re.IGNORECASE),
    "private_key_header": re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "ssh_public_key": re.compile(r"ssh-(rsa|dss|ed25519|ecdsa)\s+[A-Za-z0-9+/=]{100,}"),
    "certificate": re.compile(r"-----BEGIN CERTIFICATE-----"),
    "url_with_credentials": re.compile(r"https?://[^:@\s]+:[^@\s]{4,}@[^\s]+", re.IGNORECASE),
    "db_connection_string": re.compile(r'(mongodb(\+srv)?|postgres(ql)?|mysql|redis|mssql|sqlite|oracle):\/\/[^\s"\'<>]+', re.IGNORECASE),
    "jdbc_url": re.compile(r'jdbc:[a-z]+://[^\s"\']+', re.IGNORECASE),
    "connection_string_kv": re.compile(r"(Server|Data Source)\s*=\s*[^;]+;\s*(Database|Initial Catalog)\s*=\s*[^;]+", re.IGNORECASE),
    "internal_ip": re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"),
    "docker_registry": re.compile(r'[a-z0-9\-\.]+\.(azurecr\.io|ecr\.[a-z0-9\-]+\.amazonaws\.com|gcr\.io)/[^\s"\']+', re.IGNORECASE),
    "kubernetes_secret": re.compile(r"kind:\s*Secret", re.IGNORECASE),
    "stack_trace": re.compile(r'(Traceback \(most recent call last\)|at [a-zA-Z\.]+\([A-Za-z]+\.java:\d+\)|File "[^"]+", line \d+)'),
    "unix_path": re.compile(r"(/home/[^\s/]+|/var/www|/etc/[^\s]+|/usr/local/[^\s]+)"),
    "windows_path": re.compile(r'[A-Za-z]:\\[^\s"\'<>]{5,}'),
    "docker_compose_secret": re.compile(r"(MYSQL_ROOT_PASSWORD|POSTGRES_PASSWORD|MONGO_INITDB_ROOT_PASSWORD)\s*[:=]\s*\S+"),
    "ansible_vault": re.compile(r"\$ANSIBLE_VAULT;\d+\.\d+"),
    "email_address": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
    "wallet_address_eth": re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
    "crypto_private_key": re.compile(r"\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b"),
    "commented_credential": re.compile(r"(#|//|/\*)\s*(password|api.?key|secret|token)\s*[:=]\s*\S{4,}", re.IGNORECASE),
    "proprietary_endpoint": re.compile(r"https?://[a-z0-9\-]+\.(internal|private|local|corp|lan)(:\d+)?(/[^\s]*)?", re.IGNORECASE),
    "ci_secret_var": re.compile(r"(TRAVIS|CIRCLE|JENKINS|GITHUB_ACTIONS|GITLAB_CI)[_\-]?(TOKEN|SECRET|KEY|PASSWORD)\s*[:=]\s*\S+", re.IGNORECASE),
}

CATEGORY_MAP = {
    "API Keys & Tokens": [
        "api_key_openai", "github_token", "jwt_token", "slack_token",
        "stripe_key", "twilio_sid", "sendgrid_key", "google_api_key",
        "firebase_url", "telegram_bot_token", "discord_token", "discord_webhook",
        "npm_token", "digitalocean_token", "shopify_token", "gitlab_token",
        "mapbox_token",
    ],
    "Cloud Credentials": ["aws_access_key", "gcp_credentials", "azure_connection"],
    "Cryptographic Material": [
        "private_key_header", "ssh_public_key", "certificate",
        "url_with_credentials", "crypto_private_key",
    ],
    "Database Credentials": ["db_connection_string", "jdbc_url", "connection_string_kv"],
    "Internal Infrastructure": [
        "internal_ip", "docker_registry", "kubernetes_secret", "proprietary_endpoint",
    ],
    "System Internals / Stack Traces": ["stack_trace", "unix_path", "windows_path"],
    "PII": ["email_address", "credit_card", "iban"],
    "Financial / Crypto": ["wallet_address_eth", "crypto_private_key"],
    "CI/CD & DevOps Secrets": [
        "docker_compose_secret", "ansible_vault", "ci_secret_var", "commented_credential",
    ],
}

PATTERN_TO_CATEGORY = {
    pattern_name: category
    for category, pattern_names in CATEGORY_MAP.items()
    for pattern_name in pattern_names
}

@dataclass
class Finding:
    dataset: str
    prompt_index: int
    prompt_text: str
    sensitive: bool
    categories: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

def scan_prompt(text: str) -> tuple[bool, list[str], list[str], list[str]]:
    matched_patterns: list[str] = []
    evidence: list[str] = []

    for pattern_name, regex in PATTERNS.items():
        matches = regex.findall(text)
        if not matches:
            continue

        matched_patterns.append(pattern_name)
        for match in matches[:3]:
            snippet = match if isinstance(match, str) else match[0]
            evidence.append(f"[{pattern_name}] {snippet[:120]}")

    if not matched_patterns:
        return False, [], [], []

    categories = sorted(
        {PATTERN_TO_CATEGORY[name] for name in matched_patterns if name in PATTERN_TO_CATEGORY}
    )
    return True, categories, matched_patterns, evidence

def load_prompts(path: Path) -> tuple[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("dataset", "unknown"), payload.get("unique_prompts", [])

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "unique_prompts.json"
    output_path = base_dir / "regex_identified_codechat_sensitive_prompts.json"

    dataset_name, prompts = load_prompts(input_path)
    findings: list[Finding] = []

    for index, prompt in enumerate(prompts):
        sensitive, categories, matched_patterns, evidence = scan_prompt(prompt)
        if not sensitive:
            continue
        findings.append(
            Finding(
                dataset=dataset_name,
                prompt_index=index,
                prompt_text=prompt,
                sensitive=sensitive,
                categories=categories,
                matched_patterns=matched_patterns,
                evidence=evidence,
            )
        )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(finding) for finding in findings], handle, ensure_ascii=False, indent=2)

    category_counts: dict[str, int] = {}
    for finding in findings:
        for category in finding.categories:
            category_counts[category] = category_counts.get(category, 0) + 1

    print(f"Scanned {len(prompts)} unique prompts from {dataset_name}")
    print(f"Sensitive prompts: {len(findings)}")
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{category}: {count}")
    print(output_path)

if __name__ == "__main__":
    main()
