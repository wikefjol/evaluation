# Post-Hoc Evaluation Plan

## Overview
Generate comprehensive prediction DataFrames from trained k-fold models for flexible post-hoc analysis. Compare single-rank ensemble predictions vs hierarchical model predictions.

## Trained Model Structure (/mimer)

```
/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/
└── exp1_sequence_fold/
    ├── debug_5genera_10fold/           # Demo dataset (current)
    │   └── models/standard/
    │       ├── fold_1/
    │       │   ├── single_species/
    │       │   │   ├── checkpoint_epoch_14.pt
    │       │   │   ├── results.json
    │       │   │   └── training_history.json
    │       │   ├── single_genus/
    │       │   ├── single_family/
    │       │   ├── single_class/
    │       │   ├── single_order/
    │       │   ├── single_phylum/
    │       │   └── hierarchical/
    │       ├── fold_2/...
    │       └── fold_10/...
    └── full_dataset/                   # Future full dataset
        └── models/standard/
            └── fold_*/...
```

## Prediction Generation Strategy

### Code Structure
```
evaluation/
├── configs/
│   ├── demo_5genera.yaml              # Points to debug_5genera_10fold models
│   └── full_dataset.yaml              # Points to full_dataset models (future)
├── scripts/
│   ├── generate_fold_predictions.py   # Single entry point
│   └── inference_utils.py             # Shared model loading/inference
└── predictions/                       # Local temp storage if needed
```

### Configuration Files

**demo_5genera.yaml:**
```yaml
dataset_name: "debug_5genera_10fold"
base_path: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold"
models_path: "{base_path}/models/standard"
data_path: "{base_path}/data/standard.csv"
label_encoders_path: "{base_path}/data/label_encoders_standard_fold{fold}.json"
output_path: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/predictions/demo_5genera"
num_folds: 10
model_types:
  - single_species
  - single_genus
  - single_family
  - single_class
  - single_order
  - single_phylum
  - hierarchical
```

**full_dataset.yaml:** (Future)
```yaml
dataset_name: "full_dataset"
base_path: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/full_dataset"
# ... similar structure
```

### Single Entry Point Usage

**Command Structure:**
```bash
cd /cephyr/users/filbern/Alvis/workspace/evaluation
setup_env

# Generate individual prediction files (70 total for demo)
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_species
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_genus
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_family
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_class
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_order
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model single_phylum
python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold 1 --model hierarchical
# Repeat for folds 2-10...
```

**Batch Generation:**
```bash
# Generate all 70 prediction files
for fold in {1..10}; do
  for model in single_species single_genus single_family single_class single_order single_phylum hierarchical; do
    python scripts/generate_fold_predictions.py --config configs/demo_5genera.yaml --fold $fold --model $model
  done
done
```

### Output Structure (/mimer)

**Individual Prediction Files:**
```
/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/predictions/demo_5genera/
├── fold_1_single_species.parquet
├── fold_1_single_genus.parquet
├── fold_1_single_family.parquet
├── fold_1_single_class.parquet
├── fold_1_single_order.parquet
├── fold_1_single_phylum.parquet
├── fold_1_hierarchical.parquet
├── fold_2_single_species.parquet
├── ...
└── fold_10_hierarchical.parquet
```

**Final Combined DataFrames:**
```bash
# Separate aggregation step (creates master DataFrames)
python scripts/combine_predictions.py --config configs/demo_5genera.yaml

# Creates:
# /mimer/.../predictions/demo_5genera/single_rank_predictions.parquet
# /mimer/.../predictions/demo_5genera/hierarchical_predictions.parquet
```

## DataFrame Schema

**Both DataFrames have identical structure:**

| Column | Type | Description |
|--------|------|-------------|
| `fold` | int | Fold number (1-10) |
| `sequence_id` | str | Unique sequence identifier |
| `sequence` | str | DNA sequence |
| `sequence_length` | int | Sequence length in bp |
| `true_phylum` | str | Ground truth phylum |
| `true_class` | str | Ground truth class |
| `true_order` | str | Ground truth order |
| `true_family` | str | Ground truth family |
| `true_genus` | str | Ground truth genus |
| `true_species` | str | Ground truth species |
| `pred_phylum` | str | Predicted phylum |
| `pred_class` | str | Predicted class |
| `pred_order` | str | Predicted order |
| `pred_family` | str | Predicted family |
| `pred_genus` | str | Predicted genus |
| `pred_species` | str | Predicted species |
| `conf_phylum` | float | Confidence score (0-1) |
| `conf_class` | float | Confidence score (0-1) |
| `conf_order` | float | Confidence score (0-1) |
| `conf_family` | float | Confidence score (0-1) |
| `conf_genus` | float | Confidence score (0-1) |
| `conf_species` | float | Confidence score (0-1) |
| `top10_phylum` | str | JSON: Top-10 predictions with probs |
| `top10_class` | str | JSON: Top-10 predictions with probs |
| `top10_order` | str | JSON: Top-10 predictions with probs |
| `top10_family` | str | JSON: Top-10 predictions with probs |
| `top10_genus` | str | JSON: Top-10 predictions with probs |
| `top10_species` | str | JSON: Top-10 predictions with probs |
| `entropy_phylum` | float | Prediction entropy |
| `entropy_class` | float | Prediction entropy |
| `entropy_order` | float | Prediction entropy |
| `entropy_family` | float | Prediction entropy |
| `entropy_genus` | float | Prediction entropy |
| `entropy_species` | float | Prediction entropy |
| `model_type` | str | "single_rank_ensemble" or "hierarchical" |
| `prediction_timestamp` | str | When inference was generated |

## Key Research Questions Enabled

1. **Taxonomic Consistency:** Do single-rank models make consistent predictions across levels?
2. **Hierarchical Advantage:** Does the hierarchical model maintain better taxonomic consistency?
3. **Confidence Patterns:** How do confidence scores differ between model types?
4. **Error Patterns:** Where do models disagree and why?
5. **Level-Specific Performance:** Which taxonomic levels are most/least predictable?

## Implementation Status

- [x] Training pipeline (70 models completed)
- [x] DataFrame schema design
- [x] Code structure planning
- [ ] Configuration files
- [ ] Main prediction generation script
- [ ] Inference utilities
- [ ] Combination/aggregation script
- [ ] Testing with demo dataset
- [ ] Full dataset scaling

## Next Steps

1. Implement `generate_fold_predictions.py` 
2. Create configuration files
3. Test with single fold-model combination
4. Scale to all 70 predictions
5. Implement aggregation script
6. Begin post-hoc analysis