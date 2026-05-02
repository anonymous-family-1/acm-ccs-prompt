from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redacted.craft.baselines import llm_direct_sanitize, presidio_sanitize, spacy_sanitize
from redacted.craft.transform import naive_mask, transform_text

ANSWER_SYSTEM = """You are evaluating a technical user prompt.

Return JSON only:
{
  "primary_issue": "...",
  "likely_cause": "...",
  "next_action": "..."
}
"""

JUDGE_SYSTEM = """You are comparing two answers derived from sanitized versions of a technical prompt.

The reference answer comes from the original unsanitized prompt.

Choose which candidate better preserves:
- the same primary issue
- the same likely cause
- a useful next action

Return JSON only:
{
  "winner": "A",
  "reason": "one sentence"
}

Candidate A is CRAFT. Candidate B is the baseline.
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRAFT multi-baseline task pairwise evaluation.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--answer-model", default="qwen2.5-coder:32b")
    p.add_argument("--judge-model", default="qwen2.5-coder:32b")
    p.add_argument("--llm-direct-model", default="qwen2.5-coder:7b")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-alive", default="24h")
    return p.parse_args()

def _ollama(model: str, system: str, prompt: str, host: str, keep_alive: str, timeout: int = 300) -> str:
    import urllib.request
    host = host.rstrip("/")
    payload = {
        "model": model, "system": system, "prompt": prompt,
        "stream": False, "format": "json", "keep_alive": keep_alive,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    return {}

def _coerce_answer(text: str) -> dict:
    out = {"primary_issue": "unknown", "likely_cause": "unknown", "next_action": "unknown"}
    for line in [l.strip(" -*\t") for l in text.splitlines() if l.strip()]:
        low = line.lower()
        for field, keys in [
            ("primary_issue", ("primary_issue", "issue", "problem", "error")),
            ("likely_cause", ("likely_cause", "cause", "root cause", "reason")),
            ("next_action", ("next_action", "next action", "action", "fix")),
        ]:
            if any(low.startswith(k) for k in keys):
                out[field] = line.split(":", 1)[1].strip() if ":" in line else line
                break
    return out

def _answer(model: str, prompt_text: str, host: str, keep_alive: str) -> dict:
    last_err, last_text = None, ""
    for _ in range(3):
        try:
            last_text = _ollama(model, ANSWER_SYSTEM, prompt_text[:6000], host, keep_alive)
            parsed = _parse_json(last_text)
            if parsed:
                return parsed
        except Exception as exc:
            last_err = exc
    if last_text:
        return _coerce_answer(last_text)
    return {"primary_issue": "error", "likely_cause": str(last_err), "next_action": "unknown"}

def _judge(model: str, original: str, reference: dict, craft: dict, baseline: dict, host: str, keep_alive: str) -> dict:
    prompt = (
        f"Original prompt:\n{original[:3000]}\n\n"
        f"Reference answer:\n{json.dumps(reference, ensure_ascii=False)}\n\n"
        f"Candidate A (CRAFT):\n{json.dumps(craft, ensure_ascii=False)}\n\n"
        f"Candidate B (baseline):\n{json.dumps(baseline, ensure_ascii=False)}\n"
    )
    last_err, last_text = None, ""
    for _ in range(3):
        try:
            last_text = _ollama(model, JUDGE_SYSTEM, prompt, host, keep_alive)
            parsed = _parse_json(last_text)
            if parsed:
                break
        except Exception as exc:
            last_err = exc
    else:
        parsed = {}
    if not parsed and last_text:
        low = last_text.lower()
        winner = "TIE"
        if re.search(r"\bwinner\b\s*[:=-]?\s*a\b", low):
            winner = "A"
        elif re.search(r"\bwinner\b\s*[:=-]?\s*b\b", low):
            winner = "B"
        parsed = {"winner": winner, "reason": last_text.strip().splitlines()[0][:200]}
    winner = str(parsed.get("winner", "TIE")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return {"winner": winner, "reason": str(parsed.get("reason", ""))}

def _summarize(rows: list[dict], baseline: str) -> dict:
    key = f"craft_vs_{baseline}"
    counts = Counter(r["judgments"][key]["winner"] for r in rows if key in r.get("judgments", {}))
    decisive = counts.get("A", 0) + counts.get("B", 0)
    return {
        "count": len(rows),
        "craft_wins": counts.get("A", 0),
        f"{baseline}_wins": counts.get("B", 0),
        "ties": counts.get("TIE", 0),
        "craft_win_rate_no_ties": counts.get("A", 0) / decisive if decisive else 0.0,
    }

def _manifest_hash(records: list[dict]) -> str:
    encoded = json.dumps([r["index"] for r in records], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

def _save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def _log(path: Path | None, line: str) -> None:
    if path:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

def _build_payload(args, mhash: str, rows: list[dict]) -> dict:
    by_artifact: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_artifact[row["artifact_kind"]].append(row)

    baselines = ["naive_mask", "presidio", "llm_direct"]
    return {
        "manifest": args.manifest.name,
        "manifest_hash": mhash,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "llm_direct_model": args.llm_direct_model,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "summary": {b: _summarize(rows, b) for b in baselines},
        "by_artifact": {
            kind: {b: _summarize(kind_rows, b) for b in baselines}
            for kind, kind_rows in sorted(by_artifact.items())
        },
        "rows": rows,
    }

def main() -> None:
    args = parse_args()
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.start_index:
        records = records[args.start_index:]
    if args.end_index is not None:
        records = records[: max(0, args.end_index - args.start_index)]

    mhash = _manifest_hash(records)
    log_path = args.log_file or args.output.with_suffix(args.output.suffix + ".log")
    total = len(records)

    rows: list[dict] = []
    processed: set[int] = set()

    if args.resume:
        state = _load(args.output)
        if state and state.get("manifest_hash") == mhash:
            rows = list(state.get("rows", []))
            processed = {int(r["index"]) for r in rows}
            _log(log_path, f"resume rows={len(rows)}")
            print(f"Resumed from {len(rows)} rows.")
        else:
            _log(log_path, "resume: no matching state found, starting fresh")
    else:
        _log(log_path, (
            f"start manifest={args.manifest.name} total={total} "
            f"start={args.start_index} end={args.end_index} "
            f"host={host} answer={args.answer_model} judge={args.judge_model}"
        ))

    for idx, record in enumerate(records, start=1):
        if int(record["index"]) in processed:
            continue

        original = record["prompt_text"]
        artifact_kind = record.get("artifact_kind", "unknown")

        try:
            craft_r = transform_text(original)
            naive_r = naive_mask(original)
            presidio_r = presidio_sanitize(original)
            llm_r = llm_direct_sanitize(
                original,
                model=args.llm_direct_model,
                keep_alive=args.keep_alive,
                host=host,
            )

            ref_ans = _answer(args.answer_model, original, host, args.keep_alive)
            craft_ans = _answer(args.answer_model, craft_r.text, host, args.keep_alive)
            naive_ans = _answer(args.answer_model, naive_r.text, host, args.keep_alive)
            presidio_ans = _answer(args.answer_model, presidio_r.text, host, args.keep_alive)
            llm_ans = _answer(args.answer_model, llm_r.text, host, args.keep_alive)

            j_naive = _judge(args.judge_model, original, ref_ans, craft_ans, naive_ans, host, args.keep_alive)
            j_presidio = _judge(args.judge_model, original, ref_ans, craft_ans, presidio_ans, host, args.keep_alive)
            j_llm = _judge(args.judge_model, original, ref_ans, craft_ans, llm_ans, host, args.keep_alive)

            row = {
                "index": record["index"],
                "artifact_kind": craft_r.artifact_kind,
                "craft_operators": craft_r.operators_applied,
                "sanitized_texts": {
                    "craft": craft_r.text,
                    "naive_mask": naive_r.text,
                    "presidio": presidio_r.text,
                    "llm_direct": llm_r.text,
                },
                "answers": {
                    "reference": ref_ans,
                    "craft": craft_ans,
                    "naive_mask": naive_ans,
                    "presidio": presidio_ans,
                    "llm_direct": llm_ans,
                },
                "judgments": {
                    "craft_vs_naive_mask": j_naive,
                    "craft_vs_presidio": j_presidio,
                    "craft_vs_llm_direct": j_llm,
                },
            }
        except Exception as exc:
            _log(log_path, f"error index={record['index']} {type(exc).__name__}: {exc}")
            empty_ans = {"primary_issue": "error", "likely_cause": str(exc), "next_action": "unknown"}
            tie = {"winner": "TIE", "reason": f"error: {type(exc).__name__}"}
            row = {
                "index": record["index"],
                "artifact_kind": artifact_kind,
                "craft_operators": [],
                "sanitized_texts": {},
                "answers": {"reference": empty_ans, "craft": empty_ans,
                            "naive_mask": empty_ans, "presidio": empty_ans, "llm_direct": empty_ans},
                "judgments": {
                    "craft_vs_naive_mask": tie,
                    "craft_vs_presidio": tie,
                    "craft_vs_llm_direct": tie,
                },
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(row)
        jn = row["judgments"]["craft_vs_naive_mask"]["winner"]
        jp = row["judgments"]["craft_vs_presidio"]["winner"]
        jl = row["judgments"]["craft_vs_llm_direct"]["winner"]
        print(f"[{idx}/{total}] idx={record['index']} kind={row['artifact_kind']} "
              f"vs_naive={jn} vs_presidio={jp} vs_llm={jl}")
        _log(log_path, f"processed={idx}/{total} index={record['index']} artifact={row['artifact_kind']} "
                       f"naive={jn} presidio={jp} llm={jl}")

        if len(rows) % args.save_every == 0:
            _save(args.output, _build_payload(args, mhash, rows))
            _log(log_path, f"checkpoint rows={len(rows)}")

    _save(args.output, _build_payload(args, mhash, rows))
    _log(log_path, f"complete rows={len(rows)}")

    payload = _build_payload(args, mhash, rows)
    print("\n=== SUMMARY ===")
    print(json.dumps(payload["summary"], indent=2))

if __name__ == "__main__":
    main()
