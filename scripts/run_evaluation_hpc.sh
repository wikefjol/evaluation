#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --output=/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/logs/evaluation_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/logs/evaluation_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=mig
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

# Setup environment
source ~/setup_env.sh

# Parse arguments
EXPERIMENT_TYPE=${1:-"exp1_sequence_fold"}
DATASET_SIZE=${2:-"debug_5genera_10fold"}
UNION_TYPE=${3:-"standard"}
FOLD=${4:-1}

echo "Starting evaluation for:"
echo "  Experiment: $EXPERIMENT_TYPE"
echo "  Dataset: $DATASET_SIZE"
echo "  Union: $UNION_TYPE"
echo "  Fold: $FOLD"

# Navigate to evaluation directory
cd /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/evaluation

# Run evaluation
python scripts/evaluate_models.py \
    --experiment-type $EXPERIMENT_TYPE \
    --dataset-size $DATASET_SIZE \
    --union-type $UNION_TYPE \
    --folds $FOLD

echo "Evaluation complete!"