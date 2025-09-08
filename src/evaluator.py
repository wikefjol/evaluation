"""
evaluator.py

Main evaluation module for comparing hierarchical vs single-rank models.
Loads trained models and generates predictions for test sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model components from training module
import sys
import os

# Add training src to path - works both locally and on HPC
if os.path.exists('/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/training/src'):
    sys.path.append('/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/training/src')
elif os.path.exists('/Users/filipberntsson/workspace/training/src'):
    sys.path.append('/Users/filipberntsson/workspace/training/src')
else:
    # Try relative path
    training_src = Path(__file__).parent.parent.parent / 'training' / 'src'
    if training_src.exists():
        sys.path.append(str(training_src))

from model import SequenceClassificationModel
from heads import HierarchicalClassificationHead, SingleClassificationHead
from preprocessing import KmerTokenizer


class ModelEvaluator:
    """Loads and evaluates a single model (hierarchical or single-rank)."""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoders = None
        self.config = None
        self.model_type = None
        
    def load_model(self) -> None:
        """Load model, configuration, and label encoders from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load configuration
        self.config = checkpoint['config']
        
        # Load label encoders
        self.label_encoders = checkpoint.get('label_encoders', {})
        
        # Load vocabulary and create tokenizer
        vocab = checkpoint.get('vocab', {})
        self.tokenizer = KmerTokenizer(
            k=self.config['preprocessing']['kmer_size'],
            stride=self.config['preprocessing']['stride'],
            vocab_size=len(vocab) if vocab else 256
        )
        # Set the loaded vocab
        if vocab:
            self.tokenizer.vocab = vocab
            self.tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Determine model type
        if self.config['model']['classification_type'] == 'hierarchical':
            self.model_type = 'hierarchical'
            taxonomic_levels = self.config['model']['taxonomic_levels']
        else:
            self.model_type = 'single'
            taxonomic_levels = self.config['model']['taxonomic_levels']
        
        # Create model based on configuration
        if self.model_type == 'hierarchical':
            num_classes_dict = {}
            for level in taxonomic_levels:
                if level in self.label_encoders:
                    encoder = self.label_encoders[level]
                    if isinstance(encoder, dict):
                        num_classes_dict[level] = encoder.get('num_classes', len(encoder.get('label_to_idx', {})))
                    else:
                        num_classes_dict[level] = len(encoder.classes_)
            # For hierarchical, create the full model with hierarchical head
            self.model = SequenceClassificationModel(
                config=self.config['model'],
                num_labels_dict=num_classes_dict,
                classification_type='hierarchical'
            )
        else:
            level = taxonomic_levels[0]
            encoder = self.label_encoders[level]
            if isinstance(encoder, dict):
                num_classes = encoder.get('num_classes', len(encoder.get('label_to_idx', {})))
            else:
                num_classes = len(encoder.classes_)
            # For single-rank, create model with single head
            self.model = SequenceClassificationModel(
                config=self.config['model'],
                num_labels=num_classes,
                classification_type='single'
            )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded {self.model_type} model for levels: {taxonomic_levels}")
    
    def predict_batch(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a batch of sequences.
        
        Returns:
            Dictionary mapping taxonomic level to predictions array
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize sequences
        tokens = [self.tokenizer.tokenize(seq) for seq in sequences]
        
        # Pad sequences
        max_length = self.config['preprocessing']['max_length']
        padded_tokens = []
        attention_masks = []
        
        for token_seq in tokens:
            if len(token_seq) > max_length:
                token_seq = token_seq[:max_length]
            
            attention_mask = [1] * len(token_seq)
            padding_length = max_length - len(token_seq)
            
            if padding_length > 0:
                token_seq = token_seq + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            padded_tokens.append(token_seq)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_tokens, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
        
        # Get predictions
        predictions = {}
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            if self.model_type == 'hierarchical':
                # Outputs should be a dict of logits for each level
                for level in self.config['model']['taxonomic_levels']:
                    if level in outputs:
                        logits = outputs[level]
                        pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
                        pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                        # Decode predictions
                        encoder = self.label_encoders[level]
                        if isinstance(encoder, dict):
                            idx_to_label = encoder.get('idx_to_label', {})
                            decoded_labels = [idx_to_label.get(idx, f'UNKNOWN_{idx}') for idx in pred_indices]
                        else:
                            decoded_labels = encoder.inverse_transform(pred_indices)
                        
                        predictions[level] = {
                            'indices': pred_indices,
                            'probabilities': pred_probs,
                            'labels': decoded_labels
                        }
            else:
                # Single-rank model returns single logits tensor
                level = self.config['model']['taxonomic_levels'][0]
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get(level))
                else:
                    logits = outputs
                    
                pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
                pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                # Decode predictions
                encoder = self.label_encoders[level]
                if isinstance(encoder, dict):
                    idx_to_label = encoder.get('idx_to_label', {})
                    decoded_labels = [idx_to_label.get(idx, f'UNKNOWN_{idx}') for idx in pred_indices]
                else:
                    decoded_labels = encoder.inverse_transform(pred_indices)
                    
                predictions[level] = {
                    'indices': pred_indices,
                    'probabilities': pred_probs,
                    'labels': decoded_labels
                }
        
        return predictions


class ComparativeEvaluator:
    """Compares hierarchical vs single-rank models across all taxonomic levels."""
    
    def __init__(self, experiment_dir: Path, fold: int, union_type: str):
        self.experiment_dir = experiment_dir
        self.fold = fold
        self.union_type = union_type
        self.models_dir = experiment_dir / 'models' / union_type / f'fold_{fold}'
        self.data_path = experiment_dir / 'data' / f'{union_type}.csv'
        
        self.hierarchical_model = None
        self.single_models = {}
        self.test_data = None
        
        self.taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        
    def load_models(self) -> None:
        """Load all models for this fold."""
        logger.info(f"Loading models for {self.union_type} fold {self.fold}")
        
        # Load hierarchical model
        hier_checkpoint = self.models_dir / 'hierarchical' / 'checkpoint_best.pt'
        if hier_checkpoint.exists():
            self.hierarchical_model = ModelEvaluator(hier_checkpoint)
            self.hierarchical_model.load_model()
            logger.info("Loaded hierarchical model")
        else:
            logger.warning(f"Hierarchical model not found at {hier_checkpoint}")
        
        # Load single-rank models
        for level in self.taxonomic_levels:
            single_checkpoint = self.models_dir / f'single_{level}' / 'checkpoint_best.pt'
            if single_checkpoint.exists():
                self.single_models[level] = ModelEvaluator(single_checkpoint)
                self.single_models[level].load_model()
                logger.info(f"Loaded single_{level} model")
            else:
                logger.warning(f"Single_{level} model not found at {single_checkpoint}")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data for this fold."""
        logger.info(f"Loading test data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Determine which fold column to use based on experiment type
        if 'exp1' in str(self.experiment_dir):
            fold_col = 'fold_exp1'
        else:
            fold_col = 'fold_exp2'
        
        # Filter to test fold
        test_df = df[df[fold_col] == self.fold].copy()
        logger.info(f"Loaded {len(test_df)} test sequences for fold {self.fold}")
        
        self.test_data = test_df
        return test_df
    
    def evaluate_fold(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate all models on the test fold.
        
        Returns:
            Dictionary containing predictions and metrics
        """
        if self.test_data is None:
            self.load_test_data()
        
        if not self.hierarchical_model and not self.single_models:
            self.load_models()
        
        # Initialize results storage
        results = {
            'sequence_ids': self.test_data['sequence_id'].tolist(),
            'true_labels': {},
            'hierarchical_predictions': {},
            'single_predictions': {}
        }
        
        # Store true labels
        for level in self.taxonomic_levels:
            results['true_labels'][level] = self.test_data[level].tolist()
        
        # Process in batches
        sequences = self.test_data['sequence'].tolist()
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(sequences)} sequences in {n_batches} batches")
        
        # Initialize prediction storage
        for level in self.taxonomic_levels:
            results['hierarchical_predictions'][level] = []
            results['single_predictions'][level] = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # Get hierarchical predictions
            if self.hierarchical_model:
                hier_preds = self.hierarchical_model.predict_batch(batch_sequences)
                for level in self.taxonomic_levels:
                    if level in hier_preds:
                        results['hierarchical_predictions'][level].extend(
                            hier_preds[level]['labels'].tolist()
                        )
                    else:
                        # Pad with None if level not predicted
                        results['hierarchical_predictions'][level].extend(
                            [None] * len(batch_sequences)
                        )
            
            # Get single-rank predictions
            for level in self.taxonomic_levels:
                if level in self.single_models:
                    single_preds = self.single_models[level].predict_batch(batch_sequences)
                    results['single_predictions'][level].extend(
                        single_preds[level]['labels'].tolist()
                    )
                else:
                    # Pad with None if model not available
                    results['single_predictions'][level].extend(
                        [None] * len(batch_sequences)
                    )
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{n_batches} batches")
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        return {
            'predictions': results,
            'metrics': metrics
        }
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics for hierarchical vs single models."""
        metrics = {
            'per_level_accuracy': {},
            'per_level_counts': {}
        }
        
        for level in self.taxonomic_levels:
            true_labels = results['true_labels'][level]
            
            # Hierarchical accuracy
            hier_preds = results['hierarchical_predictions'][level]
            if any(p is not None for p in hier_preds):
                hier_correct = sum(
                    1 for t, p in zip(true_labels, hier_preds)
                    if p is not None and t == p
                )
                hier_total = sum(1 for p in hier_preds if p is not None)
                hier_accuracy = hier_correct / hier_total if hier_total > 0 else 0
            else:
                hier_accuracy = None
            
            # Single model accuracy
            single_preds = results['single_predictions'][level]
            if any(p is not None for p in single_preds):
                single_correct = sum(
                    1 for t, p in zip(true_labels, single_preds)
                    if p is not None and t == p
                )
                single_total = sum(1 for p in single_preds if p is not None)
                single_accuracy = single_correct / single_total if single_total > 0 else 0
            else:
                single_accuracy = None
            
            metrics['per_level_accuracy'][level] = {
                'hierarchical': hier_accuracy,
                'single': single_accuracy
            }
            
            metrics['per_level_counts'][level] = {
                'n_classes': len(set(true_labels)),
                'n_test_sequences': len(true_labels)
            }
            
            logger.info(f"{level}: Hierarchical={hier_accuracy:.3f if hier_accuracy else 'N/A'}, "
                       f"Single={single_accuracy:.3f if single_accuracy else 'N/A'}")
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'sequence_id': results['predictions']['sequence_ids']
        })
        
        # Add true labels
        for level in self.taxonomic_levels:
            predictions_df[f'true_{level}'] = results['predictions']['true_labels'][level]
        
        # Add predictions
        for level in self.taxonomic_levels:
            predictions_df[f'hier_{level}'] = results['predictions']['hierarchical_predictions'][level]
            predictions_df[f'single_{level}'] = results['predictions']['single_predictions'][level]
        
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Save metrics as JSON
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")