from __future__ import annotations

import json
from dataclasses import dataclass, field

from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps

SENSITIVE_FIXTURES: dict[str, list[str]] = {
    "api_key_openai": [
        "OPENAI_API_KEY=sk-testTESTtestTESTtestTESTtestTESTtestTESTtestTEST00",
        "Authorization: Bearer sk-projTESTtestTESTtestTESTtestTESTtestTESTtestTE",
        "client = OpenAI(api_key='sk-oldTESTtestTESTtestTESTtestTESTtestTESTtest00')",
    ],
    "github_token": [
        "GITHUB_TOKEN=ghp_TestTokenABCDEFGHIJKLMNOPQRSTUVWXYZ01",
        "Authorization: token ghp_AnotherTestGHPTokenABCDEFGHIJKLMNOPQ",
        "x-github-token: ghp_YetAnotherTestTokenABCDEFGHIJKLMNOPQRSTU",
    ],
    "aws_access_key": [
        "AWS_ACCESS_KEY_ID=AKIATESTONLY12345678",
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
        "[default]\naws_access_key_id=AKIAI44QH8DHBEXAMPLE",
    ],
    "jwt_token": [
        "token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiaWF0IjoxNjAwMDAwMDAwfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.eyJ1c2VyIjoidGVzdCIsImV4cCI6OTk5OTk5OTk5OX0.abc123defTESTsignature",
    ],
    "slack_token": [
        "SLACK_TOKEN=xoxb-111111111111-222222222222-testTESTtestTESTtest1234",
        "token = 'xoxp-333333333333-444444444444-555555555555-testTESTtestTESTtest'",
    ],
    "stripe_key": [
        "STRIPE_SECRET_KEY=sk_live_testTESTtestTESTtestTESTtestTESTtest12",
        "stripe.api_key = 'sk_test_testTESTtestTESTtestTESTtestTESTtest12'",
    ],
    "email_address": [
        "contact alice@example-corp.com for access",
        "user: bob.smith+tag@subdomain.example.org",
        "FROM: noreply@test.internal",
    ],
    "credit_card": [
        "card: 4111111111111111",
        "payment_card_number=5500005555555559",
        "visa: 4242424242424242",
    ],
    "iban": [
        "IBAN: DE89370400440532013000",
        "bank_account=GB29NWBK60161331926819",
    ],
    "wallet_address_eth": [
        "wallet: 0xAbCdEf0123456789AbCdEf0123456789AbCdEf01",
        "to: 0x1234567890ABCDEFabcdef1234567890ABCDEF12",
    ],
    "internal_ip": [
        "server: 10.0.1.42",
        "host=192.168.100.200",
        "backend: 172.16.0.1",
    ],
    "unix_path": [
        "config = /home/alice/projects/myapp/config.py",
        "include /etc/nginx/sites-enabled/default",
        "cert_file = /etc/ssl/certs/myapp.crt",
    ],
    "windows_path": [
        r"path = C:\Users\Bob\AppData\Local\Temp\config.ini",
        r"binary = C:\Program Files\MyApp\bin\run.exe",
    ],
    "db_connection_string": [
        "postgresql://testuser:testpass@db.internal:5432/mydb",
        "mysql://root:s3cr3t@127.0.0.1:3306/prod_db",
    ],
    "private_key_header": [
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAtestTESTtestTESTtestTEST\n-----END RSA PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDtest\n-----END PRIVATE KEY-----",
    ],
    "ssh_public_key": [
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDWvjFu9bnpCLtxqR3VlGXhmPLsT4sNaRL1YhEgOQkZp2mXtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtest alice@example.com",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTESTtestTEST deploy@ci",
    ],
}

ARTIFACT_KIND_FIXTURES: list[tuple[str, list[str], str]] = [
    (
        "config_blob",
        ["CANONICALIZE"],
        "DB_HOST=localhost\nDB_PORT=5432\nMAX_RETRIES=3\nLOG_LEVEL=INFO\nTIMEOUT=30",
    ),
    (
        "config_blob",
        ["CANONICALIZE"],
        "server.port=8080\nspring.datasource.url=jdbc:h2:mem:testdb\nlogging.level.root=WARN\napp.name=myservice\nmax.connections=100",
    ),
    (
        "secret_blob",
        ["SUPPRESS", "ABSTRACT"],
        "AWS_ACCESS_KEY_ID=AKIATESTONLY12345678\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    (
        "secret_blob",
        ["SUPPRESS", "ABSTRACT"],
        "api_key=sk-testTESTtestTESTtestTESTtestTESTtestTESTtestTEST00\nbase_url=https://api.openai.com",
    ),
    (
        "identifier_blob",
        ["ABSTRACT"],
        "Please process refund for alice@example-corp.com, card: 4111111111111111",
    ),
    (
        "identifier_blob",
        ["ABSTRACT"],
        "Transfer EUR 500 to IBAN: DE89370400440532013000 for bob.smith@subdomain.example.org",
    ),
    (
        "clean_text",
        [],
        "def process_batch(items: list[str]) -> dict:\n    return {item: len(item) for item in items}",
    ),
    (
        "clean_text",
        [],
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.status = 'paid';",
    ),
]

KEY_NAME_AWARE_FIXTURES: list[tuple[str, str]] = [
    ("db_password = hunter2", "db_password"),
    ("api_key = mykey123", "api_key"),
    ("client_secret = abc-def-ghi", "client_secret"),
    ("AUTH_TOKEN = Bearer localdevtoken", "AUTH_TOKEN"),
    ("private_key = /run/secrets/app.key", "private_key"),
    ("passwd = changeme", "passwd"),
    ("access_key = dev-key-00001", "access_key"),
    ("credential = user:pass@internal", "credential"),
]

BENIGN_FIXTURES: list[str] = [
    "The function returns a list of integers sorted in ascending order.",
    "SELECT * FROM users WHERE id = 42 ORDER BY created_at DESC;",
    "version: 3.14.1\nmax_retries: 5\nenabled: true\nlog_level: INFO",
    "pip install requests==2.31.0",
    "def process_batch(items: list[str], timeout: int = 30) -> dict:",
    "at com.example.app.MainClass.run(MainClass.java:42)",
    "File 'app/views.py', line 123, in handle_request",
    "Error: Connection refused on port 8080",
    "2024-01-15 12:34:56.789 INFO  Starting application",
    "import os\nimport sys\nimport json\nfrom pathlib import Path",
    "npm install --save-dev typescript@5.0.0",
    "docker run -it --rm ubuntu:22.04 bash",
    "git commit -m 'fix: resolve null pointer in user service'",
    "kubectl get pods --namespace=production",
    "curl -X GET https://api.example.com/v2/status",
]

@dataclass
class DetectorCoverageReport:
    recall_by_type: dict[str, float] = field(default_factory=dict)
    detected_by_type: dict[str, int] = field(default_factory=dict)
    total_by_type: dict[str, int] = field(default_factory=dict)
    missed_examples_by_type: dict[str, list[str]] = field(default_factory=dict)
    overall_recall: float = 0.0
    false_positive_rate: float = 0.0
    false_positive_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        weak_types = [t for t, r in self.recall_by_type.items() if r < 1.0]
        return {
            "overall_recall": self.overall_recall,
            "false_positive_rate": self.false_positive_rate,
            "recall_by_type": dict(sorted(self.recall_by_type.items())),
            "weak_coverage_types": weak_types,
            "missed_examples": {
                t: v[:2] for t, v in self.missed_examples_by_type.items() if v
            },
            "false_positive_examples": self.false_positive_examples[:5],
            "privacy_implication": _privacy_implication(self.overall_recall),
        }

def _privacy_implication(recall: float) -> dict:
    miss_rate = 1.0 - recall
    return {
        "missed_span_rate": miss_rate,
        "epsilon_suppress_conditioned": miss_rate,
        "note": (
            f"At recall={recall:.3f}, {miss_rate * 100:.1f}% of HIGH_RISK spans escape "
            "detection and appear verbatim in the sanitized output, leaking fully. "
            "This is the dominant term in ε_cond = r·ε_op + (1−r)·1.0."
        ),
    }

def measure_detector_recall(
    fixtures: dict[str, list[str]] | None = None,
) -> DetectorCoverageReport:
    if fixtures is None:
        fixtures = SENSITIVE_FIXTURES

    detected_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}
    missed_by_type: dict[str, list[str]] = {}

    for expected_type, examples in fixtures.items():
        detected = 0
        missed: list[str] = []
        for text in examples:
            spans = resolve_overlaps(find_sensitive_spans(text))
            if any(s.pattern == expected_type for s in spans):
                detected += 1
            else:
                missed.append(text)
        detected_by_type[expected_type] = detected
        total_by_type[expected_type] = len(examples)
        missed_by_type[expected_type] = missed

    total_fixtures = sum(total_by_type.values())
    total_detected = sum(detected_by_type.values())

    return DetectorCoverageReport(
        recall_by_type={
            t: detected_by_type[t] / total_by_type[t] if total_by_type[t] else 0.0
            for t in fixtures
        },
        detected_by_type=detected_by_type,
        total_by_type=total_by_type,
        missed_examples_by_type=missed_by_type,
        overall_recall=total_detected / total_fixtures if total_fixtures else 0.0,
    )

def measure_detector_precision(benign: list[str] | None = None) -> DetectorCoverageReport:
    if benign is None:
        benign = BENIGN_FIXTURES

    fp_examples: list[str] = []
    fp_count = 0
    for text in benign:
        spans = resolve_overlaps(find_sensitive_spans(text))
        if spans:
            fp_count += 1
            patterns = [s.pattern for s in spans]
            fp_examples.append(f"{text[:60]}… → {patterns}")

    return DetectorCoverageReport(
        false_positive_rate=fp_count / len(benign) if benign else 0.0,
        false_positive_examples=fp_examples,
    )

def measure_spacy_coverage(
    fixtures: dict[str, list[str]] | None = None,
) -> dict:
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])

    if fixtures is None:
        fixtures = SENSITIVE_FIXTURES

    detected_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}

    for pattern_type, examples in fixtures.items():
        detected = 0
        for text in examples:
            doc = nlp(text[:10_000])
            if doc.ents:
                detected += 1
        detected_by_type[pattern_type] = detected
        total_by_type[pattern_type] = len(examples)

    total_fixtures = sum(total_by_type.values())
    total_detected = sum(detected_by_type.values())

    return {
        "overall_recall": total_detected / total_fixtures if total_fixtures else 0.0,
        "recall_by_type": {
            t: detected_by_type[t] / total_by_type[t] if total_by_type[t] else 0.0
            for t in fixtures
        },
        "note": (
            "spaCy en_core_web_lg recall appears ~41% but is misleading: "
            "for aws_access_key it detects the variable name 'AWS_ACCESS_KEY_ID' (labelled ORG) "
            "while leaving the credential value AKIA... verbatim. "
            "For email_address, ssh public keys (content-wise), api_key_openai, github_token, "
            "jwt_token, slack_token, and private_key_header recall is 0%. "
            "NER-based approaches cannot detect structured technical credentials by design — "
            "they require format-aware patterns, not entity recognition."
        ),
    }

def measure_key_name_aware_redaction(
    fixtures: list[tuple[str, str]] | None = None,
) -> dict:
    from .transform import CRAFT_PIPELINES, transform_text

    if fixtures is None:
        fixtures = KEY_NAME_AWARE_FIXTURES

    config_blob_pipeline = CRAFT_PIPELINES["config_blob"]
    hits: list[str] = []
    misses: list[str] = []

    for text, key in fixtures:
        result = transform_text(text, pipeline=config_blob_pipeline)
        if "<SECRET_VALUE:" in result.text:
            hits.append(key)
        else:
            misses.append(f"{key!r}: got {result.text!r}")

    total = len(fixtures)
    return {
        "total": total,
        "redacted": len(hits),
        "redaction_rate": len(hits) / total if total else 0.0,
        "missed_keys": misses,
        "note": (
            "Key-name-aware redaction catches sensitive values whose KEY NAME matches "
            "a credential pattern even when the value itself is not format-valid for any regex. "
            "These cases are missed by span detection alone and represent defense-in-depth."
        ),
    }

def measure_artifact_kind_coverage(
    fixtures: list[tuple[str, list[str], str]] | None = None,
) -> dict:
    from redacted.craft.detectors import find_sensitive_spans, resolve_overlaps
    from .artifacts import build_artifact
    from .transform import transform_text

    if fixtures is None:
        fixtures = ARTIFACT_KIND_FIXTURES

    hits: list[dict] = []
    misses: list[dict] = []

    for expected_kind, expected_ops, text in fixtures:
        spans = resolve_overlaps(find_sensitive_spans(text))
        art = build_artifact(text, spans)
        result = transform_text(text)
        kind_ok = art.kind == expected_kind
        ops_ok = result.operators_applied == expected_ops
        entry = {
            "expected_kind": expected_kind,
            "got_kind": art.kind,
            "expected_ops": expected_ops,
            "got_ops": result.operators_applied,
            "kind_correct": kind_ok,
            "ops_correct": ops_ok,
            "text_preview": text[:60],
        }
        (hits if (kind_ok and ops_ok) else misses).append(entry)

    total = len(fixtures)
    return {
        "total": total,
        "correct": len(hits),
        "accuracy": len(hits) / total if total else 0.0,
        "misses": misses,
        "note": (
            "Validates end-to-end artifact-kind classification and operator routing "
            "for kinds not represented in the held-out evaluation corpus."
        ),
    }

def full_coverage_report(
    fixtures: dict[str, list[str]] | None = None,
    benign: list[str] | None = None,
    key_name_fixtures: list[tuple[str, str]] | None = None,
    artifact_kind_fixtures: list[tuple[str, list[str], str]] | None = None,
) -> dict:
    recall = measure_detector_recall(fixtures)
    precision = measure_detector_precision(benign)
    key_name = measure_key_name_aware_redaction(key_name_fixtures)
    artifact_kinds = measure_artifact_kind_coverage(artifact_kind_fixtures)
    report = recall.to_dict()
    report["false_positive_rate"] = precision.false_positive_rate
    report["false_positive_examples"] = precision.false_positive_examples[:5]
    report["key_name_aware_redaction"] = key_name
    report["artifact_kind_coverage"] = artifact_kinds
    return report

if __name__ == "__main__":
    print(json.dumps(full_coverage_report(), indent=2))
