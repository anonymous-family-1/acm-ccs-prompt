from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _run(script: str, argv: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "redacted" / "craft" / script), *argv]
    return subprocess.call(cmd)

def main() -> None:
    parser = argparse.ArgumentParser(description="CRAFT pipeline CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ae = sub.add_parser("auto-eval", help="Automatic scoring: CRAFT vs naive_mask vs Presidio.")
    ae.add_argument("--manifest", required=True)
    ae.add_argument("--output", required=True)

    abl = sub.add_parser("ablation-eval", help="Ablation study: six design variants, algorithmic scoring.")
    abl.add_argument("--manifest", required=True)
    abl.add_argument("--output", required=True)
    abl.add_argument("--limit", type=int)

    te = sub.add_parser("task-eval", help="LLM pairwise benchmark: CRAFT vs naive_mask.")
    te.add_argument("--manifest", required=True)
    te.add_argument("--output", required=True)
    te.add_argument("--answer-model", default="qwen2.5-coder:7b")
    te.add_argument("--judge-model", default="qwen2.5-coder:32b")
    te.add_argument("--limit")
    te.add_argument("--start-index", default="0")
    te.add_argument("--end-index")
    te.add_argument("--resume", action="store_true")
    te.add_argument("--save-every", default="10")
    te.add_argument("--log-file")
    te.add_argument("--keep-alive", default="24h")

    me = sub.add_parser("multi-eval", help="Multi-baseline task pairwise: CRAFT vs naive_mask, Presidio, llm_direct.")
    me.add_argument("--manifest", required=True)
    me.add_argument("--output", required=True)
    me.add_argument("--answer-model", default="qwen2.5-coder:32b")
    me.add_argument("--judge-model", default="qwen2.5-coder:32b")
    me.add_argument("--llm-direct-model", default="qwen2.5-coder:7b")
    me.add_argument("--start-index", default="0")
    me.add_argument("--end-index")
    me.add_argument("--resume", action="store_true")
    me.add_argument("--save-every", default="10")
    me.add_argument("--log-file")
    me.add_argument("--keep-alive", default="24h")

    mg = sub.add_parser("merge-eval", help="Merge sharded multi-eval outputs.")
    mg.add_argument("--inputs", nargs="+", required=True)
    mg.add_argument("--output", required=True)

    re_p = sub.add_parser("recon-eval", help="Adversarial reconstruction attack: CRAFT vs naive_mask.")
    re_p.add_argument("--manifest", required=True)
    re_p.add_argument("--output", required=True)
    re_p.add_argument("--attack-model", default="qwen2.5-coder:32b")
    re_p.add_argument("--limit")
    re_p.add_argument("--start-index", default="0")
    re_p.add_argument("--end-index")
    re_p.add_argument("--resume", action="store_true")
    re_p.add_argument("--save-every", default="10")
    re_p.add_argument("--log-file")
    re_p.add_argument("--keep-alive", default="24h")

    args = parser.parse_args()

    if args.cmd == "auto-eval":
        raise SystemExit(_run("auto_eval.py", ["--manifest", args.manifest, "--output", args.output]))

    if args.cmd == "ablation-eval":
        argv = ["--manifest", args.manifest, "--output", args.output]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        raise SystemExit(_run("ablation_eval.py", argv))

    if args.cmd == "task-eval":
        argv = [
            "--manifest", args.manifest, "--output", args.output,
            "--answer-model", args.answer_model, "--judge-model", args.judge_model,
            "--start-index", args.start_index, "--save-every", args.save_every,
            "--keep-alive", args.keep_alive,
        ]
        if args.limit:
            argv += ["--limit", args.limit]
        if args.end_index:
            argv += ["--end-index", args.end_index]
        if args.resume:
            argv.append("--resume")
        if args.log_file:
            argv += ["--log-file", args.log_file]
        raise SystemExit(_run("task_pairwise_eval.py", argv))

    if args.cmd == "multi-eval":
        argv = [
            "--manifest", args.manifest, "--output", args.output,
            "--answer-model", args.answer_model, "--judge-model", args.judge_model,
            "--llm-direct-model", args.llm_direct_model,
            "--start-index", args.start_index, "--save-every", args.save_every,
            "--keep-alive", args.keep_alive,
        ]
        if args.end_index:
            argv += ["--end-index", args.end_index]
        if args.resume:
            argv.append("--resume")
        if args.log_file:
            argv += ["--log-file", args.log_file]
        raise SystemExit(_run("multi_baseline_eval.py", argv))

    if args.cmd == "merge-eval":
        raise SystemExit(_run("merge_baseline_eval.py",
                               ["--inputs", *args.inputs, "--output", args.output]))

    if args.cmd == "recon-eval":
        argv = [
            "--manifest", args.manifest, "--output", args.output,
            "--attack-model", args.attack_model,
            "--start-index", args.start_index, "--save-every", args.save_every,
            "--keep-alive", args.keep_alive,
        ]
        if args.limit:
            argv += ["--limit", args.limit]
        if args.end_index:
            argv += ["--end-index", args.end_index]
        if args.resume:
            argv.append("--resume")
        if args.log_file:
            argv += ["--log-file", args.log_file]
        raise SystemExit(_run("reconstruction_eval.py", argv))

if __name__ == "__main__":
    main()
