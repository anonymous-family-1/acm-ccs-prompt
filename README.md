# ACM CCS Prompt Artifact

This repository contains code and data artifacts for three research-question tracks centered on sensitive information in LLM prompts:

- `RQ1`: labeled datasets and annotation artifacts.
- `RQ2`: prompt-sensitivity classification baselines, training, and evaluation.
- `RQ3`: CRAFT, a privacy-preserving prompt sanitization pipeline and its evaluation code.

The repository is organized so each `RQ*` directory can be used independently.

## Repository Layout

```text
.
├── RQ1/   # datasets and annotation files
├── RQ2/   # classification experiments and statistical analysis
└── RQ3/   # CRAFT sanitization package, data, and evaluation pipelines
```

## RQ1: Datasets and Annotation Artifacts

`RQ1` contains the manually reviewed and source datasets used for prompt-sensitivity analysis.

Files:

- `codechat_manually_judged_prompts.json`: manual labels for the CodeChat dataset after regex filtering.
- `edevgpt_prompt_sensitive_only.json`: manual labels for the sensitive EDevGPT subset kept in the artifact.
- `edev_gpt_unique_prompts_dataset.json`: source EDevGPT prompt corpus.
- `author1.json` and `author2.json`: 500-example annotation files for inter-rater agreement calculations.

See [RQ1/README.md] for the file-level notes already included with the artifact.

## RQ2: Prompt-Sensitivity Classification

`RQ2` contains the training and evaluation pipeline for binary prompt-sensitivity classification.

Included data:

- `data/phase1_dataset.json`
- `data/phase2_dataset.json`
- `data/external_dataset.json`

Included scripts:

- `baselines.py`: majority-class, regex/keyword, TF-IDF word, and TF-IDF character baselines.
- `run_single_experiment.py`: train one encoder/pooling configuration.
- `run_all_experiments.py`: launch all model/pooling combinations.
- `evaluate_external.py`: evaluate trained checkpoints on the external dataset.
- `statistical_tests.py`: bootstrap confidence intervals and significance testing across phases.
- `threshold_analysis.py`: operating-point and deployment-threshold summary.

Models evaluated by `run_all_experiments.py`:

- `bert-base-uncased`
- `roberta-base`
- `microsoft/deberta-base`
- `microsoft/codebert-base`
- `microsoft/unixcoder-base`
- `distilbert-base-uncased`

Each model is run with `cls`, `mean`, and `max` pooling.

### RQ2 Setup

```bash
cd RQ2
pip install -r requirements.txt
```

### RQ2 Reproduction

Run baselines:

```bash
python3 baselines.py \
  --dataset data/phase1_dataset.json \
  --eval-data data/external_dataset.json \
  --output results/baselines_results.json
```

Run all Phase 1 experiments and evaluate externally:

```bash
python3 run_all_experiments.py \
  --dataset data/phase1_dataset.json \
  --output-dir results/phase1_experiment_outputs

python3 evaluate_external.py \
  --experiment-dir results/phase1_experiment_outputs \
  --eval-data data/external_dataset.json \
  --output results/phase1_external_eval_results.json
```

Run all Phase 2 experiments and evaluate externally:

```bash
python3 run_all_experiments.py \
  --dataset data/phase2_dataset.json \
  --output-dir results/phase2_experiment_outputs

python3 evaluate_external.py \
  --experiment-dir results/phase2_experiment_outputs \
  --eval-data data/external_dataset.json \
  --output results/phase2_external_eval_results.json
```

Run statistical comparison and threshold analysis:

```bash
python3 statistical_tests.py \
  --phase1-external results/phase1_external_eval_results.json \
  --phase2-external results/phase2_external_eval_results.json \
  --output results/statistical_results.json

python3 threshold_analysis.py \
  --phase1-external results/phase1_external_eval_results.json \
  --phase2-external results/phase2_external_eval_results.json \
  --output results/threshold_analysis_results.json
```

`run_all_experiments.py` schedules two parallel workers and maps them onto GPU IDs `0` and `1` through `CUDA_VISIBLE_DEVICES`. If you do not have a two-GPU setup, use `run_single_experiment.py` directly or adapt the launcher.

See [RQ2/README.md] for the original reproduction outline.

## RQ3: CRAFT Sanitization Pipeline

`RQ3` packages CRAFT (`Context-Routing Artifact-Faithful Transformations`) for privacy-preserving LLM prompt sanitization.

The implementation lives under `RQ3/redacted/craft/` and exposes:

- `transform_text`: the main CRAFT transformation entry point.
- `naive_mask`: a baseline sanitizer.
- `CRAFT_PIPELINES`: artifact-specific transformation pipelines.
- evaluation modules for automatic scoring, ablations, pairwise judging, and reconstruction attacks.

Key modules:

- `auto_eval.py`: automatic privacy/utility scoring against `naive_mask`, Presidio, and spaCy NER.
- `ablation_eval.py`: algorithmic ablation study.
- `multi_baseline_eval.py`: pairwise LLM benchmark against multiple baselines.
- `reconstruction_eval.py`: adversarial reconstruction attack evaluation.
- `merge_baseline_eval.py`: merge sharded multi-baseline outputs.
- `cli.py`: thin CLI wrapper around the pipeline scripts.

### RQ3 Setup

```bash
cd RQ3
pip install -e .
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### RQ3 Reproduction

Automatic privacy/utility evaluation:

```bash
python -m redacted.craft.auto_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/auto_eval.json \
  --n-bootstrap 2000
```

Ablation study:

```bash
python -m redacted.craft.ablation_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/ablation_eval.json
```

LLM pairwise benchmark with sharded output:

```bash
export OLLAMA_HOST=http://localhost:11434

python -m redacted.craft.multi_baseline_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/multi_eval_part1.json \
  --start-index 0 --end-index 500 \
  --answer-model qwen2.5-coder:32b \
  --judge-model qwen2.5-coder:32b

python -m redacted.craft.multi_baseline_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/multi_eval_part2.json \
  --start-index 500 --end-index 1000 \
  --answer-model qwen2.5-coder:32b \
  --judge-model qwen2.5-coder:32b

python -m redacted.craft.merge_baseline_eval \
  --inputs results/multi_eval_part1.json results/multi_eval_part2.json \
  --output results/multi_eval_merged.json
```

Adversarial reconstruction evaluation:

```bash
export OLLAMA_HOST=http://localhost:11434

python -m redacted.craft.reconstruction_eval \
  --manifest data/spar_pp_task_test_1000_manifest.json \
  --output results/recon_eval.json \
  --attack-model gpt-oss:20b \
  --resume --save-every 10
```

The LLM-based RQ3 evaluations expect an Ollama-compatible local model endpoint for the configured models and read the endpoint from `OLLAMA_HOST`.

See [RQ3/README.md] for the directory-specific command list.

## Notes

- There is no single top-level dependency file for the whole repository; install dependencies per `RQ2` and `RQ3`.
- `RQ1` is data-only.
- Many scripts write outputs into `results/` directories that are not committed in this repository.
