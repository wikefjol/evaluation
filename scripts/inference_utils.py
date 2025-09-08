"""
Inference utilities for model loading and prediction generation
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader
import logging

# Add training src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "training" / "src"))

from data import LabelEncoder, load_fold_data, collate_fn_filter_none
from preprocessing import create_tokenizer
from model import create_model

logger = logging.getLogger(__name__)


def load_label_encoders(encoder_path: str) -> Dict[str, LabelEncoder]:
    """Load label encoders from JSON file"""
    with open(encoder_path, 'r') as f:
        encoder_dicts = json.load(f)
    
    # Convert dictionaries back to LabelEncoder objects
    label_encoders = {}
    for level, enc_dict in encoder_dicts.items():
        label_encoders[level] = LabelEncoder.from_dict(enc_dict)
    
    return label_encoders


def load_trained_model(checkpoint_path: str, config: Dict, use_best: bool = True):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model directory
        config: Training configuration 
        use_best: Whether to use best checkpoint or final epoch
        
    Returns:
        model, tokenizer, label_encoders (if available)
    """
    model_dir = Path(checkpoint_path)
    
    # Determine which checkpoint to load
    if use_best:
        # Load best checkpoint based on results.json
        results_path = model_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            best_epoch = results.get('best_epoch', 14)
            checkpoint_file = f"checkpoint_epoch_{best_epoch}.pt"
        else:
            logger.warning(f"No results.json found in {model_dir}, using final checkpoint")
            checkpoint_file = "checkpoint_epoch_14.pt"
    else:
        checkpoint_file = "checkpoint_epoch_14.pt"
    
    checkpoint_path = model_dir / checkpoint_file
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    model_config = checkpoint['config']
    
    # Create tokenizer exactly as in training
    # Use saved tokenizer vocab if available
    if 'tokenizer_vocab' in checkpoint and checkpoint['tokenizer_vocab'] is not None:
        from preprocessing import KmerTokenizer
        tokenizer = KmerTokenizer(
            k=model_config['preprocessing']['kmer_size'],
            stride=model_config['preprocessing']['stride']
        )
        tokenizer.vocab = checkpoint['tokenizer_vocab']
    else:
        # Fallback: create tokenizer as in training
        sequences_for_vocab = []  # Empty for exhaustive k-mer vocab
        tokenizer = create_tokenizer(model_config, sequences_for_vocab)
    
    # Determine model architecture
    is_hierarchical = model_config['model'].get('classification_type') == 'hierarchical'
    
    if is_hierarchical:
        # Hierarchical model
        taxonomic_levels = model_config['model']['taxonomic_levels']
        label_encoders = checkpoint.get('label_encoders', {})
        
        # Get num_classes from label encoders
        num_classes_per_level = []
        for level in taxonomic_levels:
            if level in label_encoders:
                if isinstance(label_encoders[level], dict):
                    num_classes_per_level.append(label_encoders[level]['num_classes'])
                else:
                    num_classes_per_level.append(len(label_encoders[level].label_to_index))
            else:
                raise ValueError(f"Label encoder for {level} not found in checkpoint")
        
        # Create hierarchical model
        model = create_model(
            vocab_size=len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else model_config['model']['vocab_size'],
            num_classes=num_classes_per_level,
            config=model_config
        )
        
    else:
        # Single-rank model
        label_encoders = checkpoint.get('label_to_idx', {})
        num_classes = len(label_encoders) if label_encoders else checkpoint.get('num_classes', 1000)
        
        model = create_model(
            vocab_size=len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else model_config['model']['vocab_size'],
            num_classes=num_classes,
            config=model_config
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, checkpoint.get('label_encoders', {}), model_config


def run_inference_single_rank(model, tokenizer, sequences: List[str], labels: List[str],
                              sequence_ids: List[str], label_encoder: LabelEncoder,
                              config: Dict) -> List[Dict]:
    """
    Run inference for single-rank model
    
    Returns:
        List of prediction dictionaries
    """
    from data import FungalSequenceDataset
    
    # Create dataset 
    dataset = FungalSequenceDataset(
        sequences=sequences,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config['preprocessing']['max_length'],
        label_encoder=label_encoder
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_filter_none
    )
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    predictions = []
    batch_idx = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:  # Filtered by collate_fn
                continue
                
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-10 predictions
            top10_probs, top10_indices = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)
            
            # Process each sequence in batch
            for i in range(logits.size(0)):
                seq_idx = batch_idx * config.get('batch_size', 32) + i
                if seq_idx >= len(sequence_ids):
                    break
                    
                # Top-1 prediction
                pred_idx = torch.argmax(probs[i]).item()
                pred_label = label_encoder.decode(pred_idx)
                confidence = probs[i, pred_idx].item()
                
                # Top-10 predictions
                top10_labels = []
                for j in range(top10_indices.size(-1)):
                    idx = top10_indices[i, j].item()
                    label = label_encoder.decode(idx)
                    prob = top10_probs[i, j].item()
                    if label is not None:
                        top10_labels.append([label, prob])
                
                # Calculate entropy
                entropy = -torch.sum(probs[i] * torch.log(probs[i] + 1e-8)).item()
                
                predictions.append({
                    'sequence_id': sequence_ids[seq_idx],
                    'prediction': pred_label,
                    'confidence': confidence,
                    'top10': top10_labels,
                    'entropy': entropy
                })
            
            batch_idx += 1
    
    return predictions


def run_inference_hierarchical(model, tokenizer, sequences: List[str], val_df: pd.DataFrame,
                               sequence_ids: List[str], label_encoders: Dict[str, LabelEncoder],
                               taxonomic_levels: List[str], config: Dict) -> List[Dict]:
    """
    Run inference for hierarchical model
    
    Returns:
        List of prediction dictionaries with all taxonomic levels
    """
    from data import HierarchicalFungalDataset
    
    # Create dataset
    dataset = HierarchicalFungalDataset(
        val_df,
        tokenizer,
        max_length=config['preprocessing']['max_length'],
        taxonomic_levels=taxonomic_levels,
        label_encoders=label_encoders
    )
    
    # Create data loader  
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_filter_none
    )
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    predictions = []
    batch_idx = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass - returns dict with logits for each level
            outputs = model(input_ids, attention_mask)
            
            # Process each taxonomic level
            level_predictions = {}
            for level_idx, level in enumerate(taxonomic_levels):
                logits = outputs[level_idx]  # Logits for this level
                probs = F.softmax(logits, dim=-1)
                
                # Get top-10 for this level
                top10_probs, top10_indices = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)
                
                level_predictions[level] = {
                    'probs': probs,
                    'top10_probs': top10_probs,
                    'top10_indices': top10_indices
                }
            
            # Process each sequence in batch
            for i in range(input_ids.size(0)):
                seq_idx = batch_idx * config.get('batch_size', 32) + i
                if seq_idx >= len(sequence_ids):
                    break
                
                seq_predictions = {
                    'sequence_id': sequence_ids[seq_idx]
                }
                
                # Extract predictions for each taxonomic level
                for level in taxonomic_levels:
                    level_data = level_predictions[level]
                    encoder = label_encoders[level]
                    
                    # Top-1 prediction
                    pred_idx = torch.argmax(level_data['probs'][i]).item()
                    pred_label = encoder.decode(pred_idx)
                    confidence = level_data['probs'][i, pred_idx].item()
                    
                    # Top-10 predictions
                    top10_labels = []
                    for j in range(level_data['top10_indices'].size(-1)):
                        idx = level_data['top10_indices'][i, j].item()
                        label = encoder.decode(idx)
                        prob = level_data['top10_probs'][i, j].item()
                        if label is not None:
                            top10_labels.append([label, prob])
                    
                    # Entropy
                    entropy = -torch.sum(level_data['probs'][i] * torch.log(level_data['probs'][i] + 1e-8)).item()
                    
                    seq_predictions[f'{level}_prediction'] = pred_label
                    seq_predictions[f'{level}_confidence'] = confidence
                    seq_predictions[f'{level}_top10'] = top10_labels
                    seq_predictions[f'{level}_entropy'] = entropy
                
                predictions.append(seq_predictions)
            
            batch_idx += 1
    
    return predictions