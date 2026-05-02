from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redacted.craft.transform import naive_mask, transform_text

ANSWER_SYSTEM = """You are evaluating a technical user prompt.

Return JSON only with:
{
  "primary_issue": "...",
  "likely_cause": "...",
  "next_action": "..."
}
"""

PAIRWISE_JUDGE_SYSTEM = """You are comparing two answers derived from transformed technical prompts.

The reference answer comes from the original (unsanitized) prompt.

Choose which candidate better preserves:
- the same main issue
- the same likely cause
- a useful next action

Return JSON only:
{
  "winner": "A",
  "reason": "short explanation"
}

Candidate A is CRAFT; Candidate B is naive_mask.
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise task benchmark: CRAFT vs naive_mask.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--answer-model", default="qwen2.5-coder:7b")
    p.add_argument("--judge-model", default="qwen2.5-coder:32b")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-alive", default="24h")
    return p.parse_args()

def ollama_generate(model: str, system: str, prompt: str, keep_alive: str, timeout: int = 300) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model, "system": system, "prompt": prompt,
        "stream": False, "format": "json", "keep_alive": keep_alive,
        "options": {"temperature": 0},
    }
    req = request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    return str(body.get("response", "")).strip()

def extract_json(text: str) -> dict:
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
    raise ValueError("No JSON found in output")

def _coerce_answer(text: str) -> dict:
    out = {"primary_issue": "unknown", "likely_cause": "unknown", "next_action": "unknown"}
    mapping = {
        "primary_issue": ("primary_issue", "issue", "problem", "error"),
        "likely_cause": ("likely_cause", "cause", "root cause", "reason"),
        "next_action": ("next_action", "next action", "action", "fix", "resolution"),
    }
    remaining = []
    for line in [l.strip(" -*\t") for l in text.splitlines() if l.strip()]:
        low = line.lower()
        matched = False
        for field, keys in mapping.items():
            if any(low.startswith(k) for k in keys):
                out[field] = line.split(":", 1)[1].strip() if ":" in line else line
                matched = True
                break
        if not matched:
            remaining.append(line)
    for field in ("primary_issue", "likely_cause", "next_action"):
        if out[field] == "unknown" and remaining:
            out[field] = remaining.pop(0)
    return out

def _coerce_judge(text: str) -> dict:
    low = text.lower()
    winner = "TIE"
    if re.search(r"\bwinner\b\s*[:=-]?\s*a\b", low) or re.search(r"\bcandidate a\b", low):
        winner = "A"
    elif re.search(r"\bwinner\b\s*[:=-]?\s*b\b", low) or re.search(r"\bcandidate b\b", low):
        winner = "B"
    reason = text.strip().splitlines()[0].strip()[:200] if text.strip() else "fallback"
    return {"winner": winner, "reason": reason}

def answer_prompt(model: str, prompt_text: str, keep_alive: str) -> dict:
    last_err, last_text = None, ""
    for _ in range(3):
        try:
            last_text = ollama_generate(model, ANSWER_SYSTEM, prompt_text, keep_alive)
            return extract_json(last_text)
        except Exception as exc:
            last_err = exc
    if last_text:
        return _coerce_answer(last_text)
    raise last_err or ValueError("No answer produced")

def judge_pairwise(
    model: str, original: str, reference: dict,
    candidate_a: dict, candidate_b: dict, keep_alive: str,
) -> dict:
    prompt = (
        f"Original prompt:\n{original[:4000]}\n\n"
        f"Reference answer:\n{json.dumps(reference, ensure_ascii=False)}\n\n"
        f"Candidate A (CRAFT):\n{json.dumps(candidate_a, ensure_ascii=False)}\n\n"
        f"Candidate B (naive_mask):\n{json.dumps(candidate_b, ensure_ascii=False)}\n"
    )
    last_err, last_text, parsed = None, "", None
    for _ in range(3):
        try:
            last_text = ollama_generate(model, PAIRWISE_JUDGE_SYSTEM, prompt, keep_alive)
            parsed = extract_json(last_text)
            break
        except Exception as exc:
            last_err = exc
    if parsed is None:
        parsed = _coerce_judge(last_text) if last_text else (_ for _ in ()).throw(last_err or ValueError("No judge output"))
    winner = str(parsed.get("winner", "TIE")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return {"winner": winner, "reason": str(parsed.get("reason", ""))}

def stable_manifest_hash(manifest: dict) -> str:
    encoded = json.dumps(
        {"name": manifest.get("name"), "selected_count": manifest.get("selected_count"),
         "records": [r["index"] for r in manifest.get("records", [])]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def save_state(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def load_state(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def append_log(path: Path | None, line: str) -> None:
    if path:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["winner"] for row in rows)
    decisive = counts.get("A", 0) + counts.get("B", 0)
    return {
        "count": len(rows),
        "craft_wins": counts.get("A", 0),
        "naive_mask_wins": counts.get("B", 0),
        "ties": counts.get("TIE", 0),
        "craft_win_rate_no_ties": counts.get("A", 0) / decisive if decisive else 0.0,
    }

def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.start_index:
        records = records[args.start_index :]
    if args.end_index is not None:
        records = records[: max(0, args.end_index - args.start_index)]
    if args.limit is not None:
        records = records[: args.limit]

    manifest_hash = stable_manifest_hash({
        "name": manifest.get("name"), "selected_count": len(records), "records": records,
    })
    log_path = args.log_file or args.output.with_suffix(args.output.suffix + ".log")

    rows: list[dict] = []
    by_artifact: dict[str, list[dict]] = defaultdict(list)
    processed: set[int] = set()

    if args.resume:
        state = load_state(args.output)
        if state and state.get("manifest_hash") == manifest_hash:
            rows = list(state.get("rows", []))
            processed = {int(r["index"]) for r in rows}
            for r in rows:
                by_artifact[r["artifact_kind"]].append(r)
            append_log(log_path, f"resume rows={len(rows)} summary={json.dumps(summarize(rows))}")
        else:
            append_log(log_path, "resume requested but no matching prior state found")
    else:
        append_log(log_path, f"start manifest={args.manifest.name} total={len(records)} answer_model={args.answer_model} judge_model={args.judge_model}")

    for idx, record in enumerate(records, start=1):
        if int(record["index"]) in processed:
            continue
        original = record["prompt_text"]
        try:
            reference = answer_prompt(args.answer_model, original, args.keep_alive)
            craft_result = transform_text(original)
            naive_result = naive_mask(original)
            craft_answer = answer_prompt(args.answer_model, craft_result.text, args.keep_alive)
            naive_answer = answer_prompt(args.answer_model, naive_result.text, args.keep_alive)
            judged = judge_pairwise(args.judge_model, original, reference, craft_answer, naive_answer, args.keep_alive)
            artifact_kind = craft_result.artifact_kind
        except Exception as exc:
            append_log(log_path, f"error index={record['index']} error={type(exc).__name__}: {exc}")
            judged = {"winner": "TIE", "reason": f"fallback: {type(exc).__name__}"}
            reference = craft_answer = naive_answer = {"primary_issue": "unknown", "likely_cause": "unknown", "next_action": "unknown"}
            artifact_kind = record.get("artifact_kind", "unknown")

        row = {
            "index": record["index"],
            "artifact_kind": artifact_kind,
            "winner": judged["winner"],
            "reason": judged["reason"],
            "reference_answer": reference,
            "craft_answer": craft_answer,
            "naive_answer": naive_answer,
        }
        rows.append(row)
        by_artifact[artifact_kind].append(row)
        print(f"processed={idx}/{len(records)}")
        append_log(log_path, f"processed={idx}/{len(records)} index={record['index']} artifact={artifact_kind} winner={row['winner']}")

        if len(rows) % args.save_every == 0:
            payload = {
                "manifest": args.manifest.name, "manifest_hash": manifest_hash,
                "answer_model": args.answer_model, "judge_model": args.judge_model,
                "summary": summarize(rows),
                "by_artifact": {k: summarize(v) for k, v in sorted(by_artifact.items())},
                "rows": rows,
            }
            save_state(args.output, payload)
            append_log(log_path, f"checkpoint rows={len(rows)} summary={json.dumps(payload['summary'])}")

    payload = {
        "manifest": args.manifest.name, "manifest_hash": manifest_hash,
        "answer_model": args.answer_model, "judge_model": args.judge_model,
        "summary": summarize(rows),
        "by_artifact": {k: summarize(v) for k, v in sorted(by_artifact.items())},
        "rows": rows,
    }
    save_state(args.output, payload)
    append_log(log_path, f"complete rows={len(rows)} summary={json.dumps(payload['summary'])}")
    print(json.dumps(payload["summary"], indent=2))

if __name__ == "__main__":
    main()
