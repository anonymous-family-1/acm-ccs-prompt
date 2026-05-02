# Reproduction

## Setup

```bash
pip install torch transformers scikit-learn numpy scipy statsmodels
```

---

## Steps

### 1. Baselines

```bash
python3 baselines.py \
    --dataset   data/phase1_dataset.json \
    --eval-data data/external_dataset.json \
    --output    results/baselines_results.json
```

### 2. Train Phase 1 and Evaluate

```bash
python3 run_all_experiments.py \
    --dataset    data/phase1_dataset.json \
    --output-dir results/phase1_experiment_outputs

python3 evaluate_external.py \
    --experiment-dir results/phase1_experiment_outputs \
    --eval-data      data/external_dataset.json \
    --output         results/phase1_external_eval_results.json
```

### 3. Train Phase 2 and Evaluate

```bash
python3 run_all_experiments.py \
    --dataset    data/phase2_dataset.json \
    --output-dir results/phase2_experiment_outputs

python3 evaluate_external.py \
    --experiment-dir results/phase2_experiment_outputs \
    --eval-data      data/external_dataset.json \
    --output         results/phase2_external_eval_results.json
```

### 4. Statistical Tests

```bash
python3 statistical_tests.py \
    --phase1-external results/phase1_external_eval_results.json \
    --phase2-external results/phase2_external_eval_results.json \
    --output          results/statistical_results.json
```

### 5. Threshold Analysis

```bash
python3 threshold_analysis.py \
    --phase1-external results/phase1_external_eval_results.json \
    --phase2-external results/phase2_external_eval_results.json \
    --output          results/threshold_analysis_results.json
```
