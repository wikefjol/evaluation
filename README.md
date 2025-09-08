# Training Evaluation Scripts

This directory contains scripts to analyze and verify training results from k-fold experiments.

## Setup on HPC

1. Copy this directory to HPC:
```bash
scp -r evaluation alvis:workspace/
```

2. Or clone if using git:
```bash
ssh alvis
cd workspace
git clone <repo-url> evaluation
```

3. Run verification:
```bash
ssh alvis
cd workspace/evaluation
setup_env  # Load all required modules
python scripts/verify_training.py
```

## Scripts

- `verify_training.py`: Main verification script that analyzes all 70 models
  - Checks completeness (all results.json files exist)
  - Verifies checkpoints (15 epochs per model)  
  - Analyzes learning progression
  - Identifies suspicious training patterns

## Expected Output

The script will show:
- ✅/⚠️ indicators for each check
- Distribution of best epochs across models
- Models with minimal learning (potential issues)
- Models with good learning progression
- Summary statistics

## Interpreting Results

**Red flags:**
- Many models with best_epoch = 0
- Loss changes < 0.01 (minimal learning)
- Missing checkpoints
- Accuracy not improving from epoch 0 to 14

**Good signs:**
- Best epochs distributed across 1-14
- Loss decreasing significantly
- Accuracy improving over training
- All 15 checkpoints present