#!/usr/bin/env python3
"""
evaluate_models.py

Main script to evaluate trained models and compare hierarchical vs single-rank performance.
Processes specified folds and unions, generating predictions and metrics.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from evaluator import ComparativeEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_experiment(
    experiment_type: str,
    dataset_size: str,
    union_type: str,
    folds: List[int],
    base_dir: Path
) -> Dict[str, Any]:
    """
    Evaluate models for a specific experiment configuration.
    
    Args:
        experiment_type: 'exp1_sequence_fold' or 'exp2_species_fold'
        dataset_size: 'debug_5genera_10fold' or 'full_10fold'
        union_type: 'standard' or 'conservative'
        folds: List of fold numbers to evaluate
        base_dir: Base experiments directory
        
    Returns:
        Dictionary containing aggregated results
    """
    experiment_dir = base_dir / experiment_type / dataset_size
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return {}
    
    logger.info(f"Evaluating {union_type} union for {experiment_type}/{dataset_size}")
    
    all_results = {}
    
    for fold in folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating fold {fold}")
        logger.info(f"{'='*60}")
        
        evaluator = ComparativeEvaluator(experiment_dir, fold, union_type)
        
        # Check if models exist for this fold
        models_exist = (experiment_dir / 'models' / union_type / f'fold_{fold}').exists()
        if not models_exist:
            logger.warning(f"No models found for fold {fold}, skipping")
            continue
        
        # Evaluate the fold
        try:
            fold_results = evaluator.evaluate_fold()
            
            # Save results for this fold
            output_dir = base_dir / 'evaluation' / f'{union_type}_{experiment_type}_{dataset_size}' / f'fold_{fold}'
            evaluator.save_results(fold_results, output_dir)
            
            all_results[f'fold_{fold}'] = fold_results['metrics']
            
        except Exception as e:
            logger.error(f"Error evaluating fold {fold}: {e}")
            continue
    
    # Calculate average metrics across folds
    if all_results:
        avg_metrics = calculate_average_metrics(all_results)
        
        # Save aggregated results
        output_path = base_dir / 'evaluation' / f'{union_type}_{experiment_type}_{dataset_size}' / 'aggregated_metrics.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'per_fold': all_results,
                'average': avg_metrics
            }, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Average metrics across folds:")
        logger.info(f"{'='*60}")
        for level, metrics in avg_metrics['per_level_accuracy'].items():
            logger.info(f"{level:10s}: Hierarchical={metrics['hierarchical']:.3f}, "
                       f"Single={metrics['single']:.3f}")
    
    return all_results


def calculate_average_metrics(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate average metrics across multiple folds."""
    avg_metrics = {'per_level_accuracy': {}}
    
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    
    for level in taxonomic_levels:
        hier_accuracies = []
        single_accuracies = []
        
        for fold_results in all_results.values():
            if 'per_level_accuracy' in fold_results and level in fold_results['per_level_accuracy']:
                hier_acc = fold_results['per_level_accuracy'][level].get('hierarchical')
                single_acc = fold_results['per_level_accuracy'][level].get('single')
                
                if hier_acc is not None:
                    hier_accuracies.append(hier_acc)
                if single_acc is not None:
                    single_accuracies.append(single_acc)
        
        avg_metrics['per_level_accuracy'][level] = {
            'hierarchical': sum(hier_accuracies) / len(hier_accuracies) if hier_accuracies else None,
            'single': sum(single_accuracies) / len(single_accuracies) if single_accuracies else None
        }
    
    return avg_metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--experiment-type', type=str, default='exp1_sequence_fold',
                       choices=['exp1_sequence_fold', 'exp2_species_fold'],
                       help='Experiment type to evaluate')
    parser.add_argument('--dataset-size', type=str, default='debug_5genera_10fold',
                       choices=['debug_5genera_10fold', 'full_10fold'],
                       help='Dataset size to evaluate')
    parser.add_argument('--union-type', type=str, default='standard',
                       choices=['standard', 'conservative'],
                       help='Union type to evaluate')
    parser.add_argument('--folds', type=int, nargs='+', default=[1],
                       help='Fold numbers to evaluate')
    parser.add_argument('--local-test', action='store_true',
                       help='Run in local test mode')
    
    args = parser.parse_args()
    
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    # Determine base directory
    if args.local_test or 'LOCAL_TEST' in os.environ:
        logger.info("Running in LOCAL_TEST mode")
        base_dir = Path("/Users/filipberntsson/workspace/experiments")
    else:
        experiments_dir = os.getenv('EXPERIMENTS_DIR')
        if not experiments_dir:
            raise ValueError("EXPERIMENTS_DIR must be set in .env")
        base_dir = Path(experiments_dir)
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Evaluating {args.union_type} union for {args.experiment_type}/{args.dataset_size}")
    logger.info(f"Folds to evaluate: {args.folds}")
    
    # Run evaluation
    results = evaluate_experiment(
        args.experiment_type,
        args.dataset_size,
        args.union_type,
        args.folds,
        base_dir
    )
    
    if results:
        logger.info("\nEvaluation complete!")
        logger.info(f"Results saved to {base_dir / 'evaluation'}")
    else:
        logger.error("No results generated")


if __name__ == "__main__":
    main()