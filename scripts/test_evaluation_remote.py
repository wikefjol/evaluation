#!/usr/bin/env python3
"""
test_evaluation_remote.py

Test the evaluation module by fetching a model and test data from remote
and running evaluation locally.
"""

import subprocess
import tempfile
from pathlib import Path
import sys
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_remote_files():
    """Fetch necessary files from remote for testing."""
    
    # Create temporary directory structure
    temp_dir = Path("/tmp/fungal_eval_test")
    temp_dir.mkdir(exist_ok=True)
    
    exp_dir = temp_dir / "experiments" / "exp1_sequence_fold" / "debug_5genera_10fold"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Fetching models and data from remote...")
    
    # Fetch a sample model (hierarchical fold 1 standard)
    models_dir = exp_dir / "models" / "standard" / "fold_1" / "hierarchical"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Use rsync to fetch the model
    cmd = [
        "rsync", "-avz",
        "alvis:/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard/fold_1/hierarchical/checkpoint_best.pt",
        str(models_dir / "checkpoint_best.pt")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to fetch hierarchical model: {result.stderr}")
        return None
    
    # Fetch single species model
    models_dir_species = exp_dir / "models" / "standard" / "fold_1" / "single_species"
    models_dir_species.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "rsync", "-avz",
        "alvis:/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard/fold_1/single_species/checkpoint_best.pt",
        str(models_dir_species / "checkpoint_best.pt")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to fetch single species model: {result.stderr}")
        return None
    
    # Fetch test data
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "rsync", "-avz",
        "alvis:/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/data/standard.csv",
        str(data_dir / "standard.csv")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to fetch test data: {result.stderr}")
        return None
    
    logger.info(f"Files fetched successfully to {temp_dir}")
    return temp_dir


def run_evaluation(base_dir):
    """Run the evaluation on fetched files."""
    
    # Update path to include evaluation module
    sys.path.append('/Users/filipberntsson/workspace/evaluation/src')
    from evaluator import ComparativeEvaluator
    
    experiment_dir = base_dir / "experiments" / "exp1_sequence_fold" / "debug_5genera_10fold"
    
    logger.info("Initializing evaluator...")
    evaluator = ComparativeEvaluator(experiment_dir, fold=1, union_type='standard')
    
    logger.info("Loading models...")
    evaluator.load_models()
    
    logger.info("Loading test data...")
    evaluator.load_test_data()
    
    # Only evaluate on a small subset for testing
    evaluator.test_data = evaluator.test_data.head(100)
    logger.info(f"Using {len(evaluator.test_data)} sequences for test")
    
    logger.info("Running evaluation...")
    results = evaluator.evaluate_fold(batch_size=16)
    
    # Save results
    output_dir = base_dir / "evaluation_results"
    evaluator.save_results(results, output_dir)
    
    logger.info("\nResults Summary:")
    for level, metrics in results['metrics']['per_level_accuracy'].items():
        hier_acc = metrics.get('hierarchical')
        single_acc = metrics.get('single')
        
        if hier_acc is not None and single_acc is not None:
            logger.info(f"{level:10s}: Hierarchical={hier_acc:.3f}, Single={single_acc:.3f}")
        elif hier_acc is not None:
            logger.info(f"{level:10s}: Hierarchical={hier_acc:.3f}, Single=N/A")
        elif single_acc is not None:
            logger.info(f"{level:10s}: Hierarchical=N/A, Single={single_acc:.3f}")
    
    return results


def main():
    """Main execution."""
    
    # Fetch files from remote
    temp_dir = fetch_remote_files()
    
    if temp_dir:
        try:
            # Run evaluation
            results = run_evaluation(temp_dir)
            
            logger.info(f"\nEvaluation complete! Results saved to {temp_dir}/evaluation_results")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Optionally clean up
            # shutil.rmtree(temp_dir)
            logger.info(f"Test files remain at: {temp_dir}")


if __name__ == "__main__":
    main()