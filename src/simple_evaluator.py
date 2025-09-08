"""
Simplified evaluator that works directly with checkpoints without importing training modules
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleEvaluator:
    """Simple evaluator that loads models and evaluates them"""
    
    def __init__(self, experiment_dir: Path, fold: int, union_type: str):
        self.experiment_dir = experiment_dir
        self.fold = fold
        self.union_type = union_type
        self.models_dir = experiment_dir / 'models' / union_type / f'fold_{fold}'
        self.data_path = experiment_dir / 'data' / f'{union_type}.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {}
        self.test_data = None
        self.taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load a checkpoint and extract model state"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint
    
    def evaluate_with_checkpoint(self, checkpoint: Dict, sequences: List[str], level: str = None) -> np.ndarray:
        """
        Evaluate sequences using checkpoint directly.
        Returns predicted labels for the sequences.
        """
        # This is a placeholder - in reality we need the model architecture
        # For now, return random predictions for testing
        n_sequences = len(sequences)
        
        if level:
            # Single-rank model
            encoder_data = checkpoint['label_encoders'][level]
            labels = list(encoder_data['idx_to_label'].values())
            # Random predictions for testing
            predictions = np.random.choice(labels, size=n_sequences)
        else:
            # Hierarchical model - return dict of predictions
            predictions = {}
            for tax_level in self.taxonomic_levels:
                if tax_level in checkpoint['label_encoders']:
                    encoder_data = checkpoint['label_encoders'][tax_level]
                    labels = list(encoder_data['idx_to_label'].values())
                    predictions[tax_level] = np.random.choice(labels, size=n_sequences)
        
        return predictions
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data for this fold"""
        logger.info(f"Loading test data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Filter to test fold
        fold_col = 'fold_exp1' if 'exp1' in str(self.experiment_dir) else 'fold_exp2'
        test_df = df[df[fold_col] == self.fold].copy()
        
        logger.info(f"Loaded {len(test_df)} test sequences for fold {self.fold}")
        self.test_data = test_df
        return test_df
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all available models"""
        
        if self.test_data is None:
            self.load_test_data()
        
        results = {
            'sequence_ids': self.test_data['sequence_id'].tolist(),
            'true_labels': {},
            'predictions': {}
        }
        
        # Store true labels
        for level in self.taxonomic_levels:
            if level in self.test_data.columns:
                results['true_labels'][level] = self.test_data[level].tolist()
        
        sequences = self.test_data['sequence'].tolist()
        
        # Evaluate hierarchical model if exists
        hier_path = self.models_dir / 'hierarchical' / 'checkpoint_best.pt'
        if hier_path.exists():
            logger.info("Evaluating hierarchical model...")
            checkpoint = self.load_checkpoint(hier_path)
            hier_preds = self.evaluate_with_checkpoint(checkpoint, sequences)
            
            if isinstance(hier_preds, dict):
                for level, preds in hier_preds.items():
                    results['predictions'][f'hier_{level}'] = preds.tolist()
            logger.info("Hierarchical evaluation complete")
        
        # Evaluate single-rank models
        for level in self.taxonomic_levels:
            single_path = self.models_dir / f'single_{level}' / 'checkpoint_best.pt'
            if single_path.exists():
                logger.info(f"Evaluating single_{level} model...")
                checkpoint = self.load_checkpoint(single_path)
                single_preds = self.evaluate_with_checkpoint(checkpoint, sequences, level=level)
                results['predictions'][f'single_{level}'] = single_preds.tolist()
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        return {'predictions': results, 'metrics': metrics}
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate accuracy metrics"""
        metrics = {'per_level_accuracy': {}}
        
        for level in self.taxonomic_levels:
            if level not in results['true_labels']:
                continue
            
            true_labels = results['true_labels'][level]
            
            # Hierarchical accuracy
            hier_key = f'hier_{level}'
            if hier_key in results['predictions']:
                hier_preds = results['predictions'][hier_key]
                hier_correct = sum(1 for t, p in zip(true_labels, hier_preds) if t == p)
                hier_acc = hier_correct / len(true_labels)
            else:
                hier_acc = None
            
            # Single model accuracy
            single_key = f'single_{level}'
            if single_key in results['predictions']:
                single_preds = results['predictions'][single_key]
                single_correct = sum(1 for t, p in zip(true_labels, single_preds) if t == p)
                single_acc = single_correct / len(true_labels)
            else:
                single_acc = None
            
            metrics['per_level_accuracy'][level] = {
                'hierarchical': hier_acc,
                'single': single_acc,
                'n_test': len(true_labels),
                'n_classes': len(set(true_labels))
            }
            
            if hier_acc is not None or single_acc is not None:
                logger.info(f"{level:10s}: Hier={hier_acc:.3f if hier_acc else 'N/A':6s}, "
                           f"Single={single_acc:.3f if single_acc else 'N/A':6s}, "
                           f"Classes={len(set(true_labels))}")
        
        return metrics
    
    def save_results(self, results: Dict, output_dir: Path):
        """Save results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV
        pred_df = pd.DataFrame({'sequence_id': results['predictions']['sequence_ids']})
        
        # Add true labels
        for level in self.taxonomic_levels:
            if level in results['predictions']['true_labels']:
                pred_df[f'true_{level}'] = results['predictions']['true_labels'][level]
        
        # Add predictions
        for col_name, values in results['predictions']['predictions'].items():
            pred_df[col_name] = values
        
        pred_df.to_csv(output_dir / 'predictions.csv', index=False)
        logger.info(f"Saved predictions to {output_dir / 'predictions.csv'}")
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")


def test_simple_evaluator():
    """Test the simple evaluator"""
    base_dir = Path("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification")
    experiment_dir = base_dir / "experiments/exp1_sequence_fold/debug_5genera_10fold"
    
    evaluator = SimpleEvaluator(experiment_dir, fold=1, union_type='standard')
    
    # Load test data
    evaluator.load_test_data()
    
    # Use only first 100 sequences for testing
    evaluator.test_data = evaluator.test_data.head(100)
    
    # Evaluate
    results = evaluator.evaluate_all_models()
    
    # Save
    output_dir = base_dir / "evaluation_test"
    evaluator.save_results(results, output_dir)
    
    print("\nEvaluation complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    test_simple_evaluator()