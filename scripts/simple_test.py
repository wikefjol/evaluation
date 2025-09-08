#!/usr/bin/env python3
"""
Simple test to verify evaluation module works
"""

import sys
import os
from pathlib import Path
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
base_dir = Path("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification")
sys.path.append(str(base_dir / "evaluation" / "src"))
sys.path.append(str(base_dir / "training" / "src"))

def test_model_loading():
    """Test loading a single model"""
    
    # Path to a model we know exists
    model_path = base_dir / "experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard/fold_1/hierarchical/checkpoint_best.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return False
    
    logger.info(f"Loading checkpoint from {model_path}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    logger.info("Checkpoint keys: " + str(checkpoint.keys()))
    
    # Check config
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info(f"Model type: {config['model'].get('classification_type', 'unknown')}")
        logger.info(f"Taxonomic levels: {config['model'].get('taxonomic_levels', [])}")
    
    # Check label encoders
    if 'label_encoders' in checkpoint:
        logger.info(f"Label encoders for: {list(checkpoint['label_encoders'].keys())}")
    
    # Check vocab
    if 'vocab' in checkpoint:
        logger.info(f"Vocab size: {len(checkpoint['vocab'])}")
    
    logger.info("Model checkpoint loaded successfully!")
    return True

def test_evaluation():
    """Test the full evaluation pipeline"""
    from evaluator import ComparativeEvaluator
    
    experiment_dir = base_dir / "experiments/exp1_sequence_fold/debug_5genera_10fold"
    
    logger.info("Creating evaluator...")
    evaluator = ComparativeEvaluator(experiment_dir, fold=1, union_type='standard')
    
    logger.info("Loading models...")
    evaluator.load_models()
    
    logger.info("Loading test data...")
    test_df = evaluator.load_test_data()
    logger.info(f"Loaded {len(test_df)} test sequences")
    
    # Test on just a few sequences
    evaluator.test_data = test_df.head(10)
    
    logger.info("Running evaluation on 10 sequences...")
    results = evaluator.evaluate_fold(batch_size=5)
    
    logger.info("Evaluation complete!")
    
    # Show results
    for level in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
        metrics = results['metrics']['per_level_accuracy'].get(level, {})
        hier = metrics.get('hierarchical', 'N/A')
        single = metrics.get('single', 'N/A')
        logger.info(f"{level:10s}: Hier={hier}, Single={single}")
    
    return True

if __name__ == "__main__":
    logger.info("Testing model loading...")
    if test_model_loading():
        logger.info("\nTesting full evaluation...")
        test_evaluation()
    else:
        logger.error("Model loading failed")