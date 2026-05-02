# CRAFT: Context-Routing Artifact-Faithful Transformations

## Installation

```bash
pip install -e .
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Running the pipeline

### Privacy-utility scores (no LLM needed)

```bash
python -m redacted.craft.auto_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/auto_eval.json \
  --n-bootstrap 2000
```

### Ablation study (no LLM needed)

```bash
python -m redacted.craft.ablation_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/ablation_eval.json
```

### LLM pairwise benchmark (requires Ollama + qwen2.5-coder:32b)

```bash
python -m redacted.craft.multi_baseline_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/multi_eval_part1.json \
  --start 0 --end 500 \
  --answer-model qwen2.5-coder:32b \
  --judge-model qwen2.5-coder:32b \
  --host http://localhost:11434

python -m redacted.craft.multi_baseline_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/multi_eval_part2.json \
  --start 500 --end 1000 \
  --answer-model qwen2.5-coder:32b \
  --judge-model qwen2.5-coder:32b \
  --host http://localhost:11434

python -m redacted.craft.merge_baseline_eval \
  results/multi_eval_part1.json \
  results/multi_eval_part2.json \
  --output results/multi_eval_merged.json
```

### Adversarial reconstruction (requires Ollama + any instruction model)

```bash
python -m redacted.craft.reconstruction_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/recon_eval.json \
  --attack-model gpt-oss:20b \
  --host http://localhost:11434 \
  --resume --save-every 10
```
