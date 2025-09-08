#!/usr/bin/env python3
"""
Generate predictions for a single fold-model combination
Main entry point for post-hoc evaluation pipeline
"""

import argparse
import yaml
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
import os

# Add training src to path for both direct imports and checkpoint loading
training_src_path = str(Path(__file__).parent.parent.parent / "training" / "src")
sys.path.append(training_src_path)
# Also add as 'src' for checkpoint compatibility
sys.path.append(str(Path(__file__).parent.parent.parent / "training"))

from data import load_fold_data, LabelEncoder
from inference_utils import load_trained_model, run_inference_single_rank, run_inference_hierarchical

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand base_path templates (but not fold-specific ones)
    for key, value in config.items():
        if isinstance(value, str) and '{base_path}' in value and '{fold}' not in value:
            config[key] = value.format(base_path=config['base_path'])
    
    return config


def load_fold_validation_data(config: dict, fold: int):
    """Load validation data for specific fold"""
    # Load fold data
    train_df, val_df = load_fold_data(
        Path(config['data_path']), 
        fold, 
        config['fold_type']
    )
    
    logger.info(f"Loaded fold {fold} validation data: {len(val_df)} sequences")
    
    # Extract sequences and labels
    sequences = val_df['sequence'].tolist()
    sequence_ids = val_df['sequence_id'].tolist()
    
    # Get all taxonomic levels
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    true_labels = {}
    for level in taxonomic_levels:
        if level in val_df.columns:
            true_labels[level] = val_df[level].tolist()
    
    return sequences, sequence_ids, true_labels, val_df


def generate_predictions(config: dict, fold: int, model_type: str, use_best: bool = True):
    """
    Generate predictions for a single fold-model combination
    
    Args:
        config: Configuration dictionary
        fold: Fold number (1-10)
        model_type: Model type (e.g., 'single_species', 'hierarchical')
        use_best: Whether to use best checkpoint or final
    """
    logger.info(f"Generating predictions for fold {fold}, model {model_type}")
    
    # Load validation data
    sequences, sequence_ids, true_labels, val_df = load_fold_validation_data(config, fold)
    
    # Load model checkpoint
    model_dir = Path(config['models_path']) / f"fold_{fold}" / model_type
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load trained model
    model, tokenizer, checkpoint_encoders, model_config = load_trained_model(
        str(model_dir), config, use_best=use_best
    )
    
    logger.info(f"Loaded model from {model_dir}")
    
    # Determine if hierarchical
    is_hierarchical = model_type == 'hierarchical'
    
    if is_hierarchical:
        # Load global label encoders (no fold-specific path)
        encoder_path = config['label_encoders_path'].replace('{base_path}', config['base_path'])
        with open(encoder_path, 'r') as f:
            encoder_dicts = json.load(f)
        
        label_encoders = {}
        for level, enc_dict in encoder_dicts.items():
            label_encoders[level] = LabelEncoder.from_dict(enc_dict)
        
        taxonomic_levels = model_config['model']['taxonomic_levels']
        
        # Run hierarchical inference
        predictions = run_inference_hierarchical(
            model, tokenizer, sequences, val_df, sequence_ids,
            label_encoders, taxonomic_levels, config
        )
        
    else:
        # Single-rank model
        target_level = model_type.split('_')[1]  # Extract 'species' from 'single_species'
        
        # Load global label encoders (no fold-specific path)
        encoder_path = config['label_encoders_path'].replace('{base_path}', config['base_path'])
        with open(encoder_path, 'r') as f:
            encoder_dicts = json.load(f)
        
        if target_level not in encoder_dicts:
            raise ValueError(f"Label encoder for {target_level} not found")
        
        label_encoder = LabelEncoder.from_dict(encoder_dicts[target_level])
        
        # Get labels for this level
        if target_level in true_labels:
            level_labels = true_labels[target_level]
        else:
            raise ValueError(f"True labels for {target_level} not found in data")
        
        # Run single-rank inference
        predictions = run_inference_single_rank(
            model, tokenizer, sequences, level_labels, sequence_ids,
            label_encoder, config
        )
    
    # Convert to DataFrame format
    df_data = []
    
    for i, pred in enumerate(predictions):
        if i >= len(sequence_ids):
            break
            
        seq_id = sequence_ids[i]
        sequence = sequences[i]
        
        row = {
            'fold': fold,
            'sequence_id': seq_id,
            'sequence': sequence,
            'sequence_length': len(sequence),
            'model_type': 'hierarchical' if is_hierarchical else 'single_rank_ensemble',
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        # Add true labels
        for level in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
            if level in true_labels:
                row[f'true_{level}'] = true_labels[level][i] if i < len(true_labels[level]) else None
            else:
                row[f'true_{level}'] = None
        
        # Add predictions
        if is_hierarchical:
            # Hierarchical predictions - all levels
            for level in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
                row[f'pred_{level}'] = pred.get(f'{level}_prediction', None)
                row[f'conf_{level}'] = pred.get(f'{level}_confidence', None)
                row[f'top10_{level}'] = json.dumps(pred.get(f'{level}_top10', []))
                row[f'entropy_{level}'] = pred.get(f'{level}_entropy', None)
        else:
            # Single-rank prediction - only target level
            target_level = model_type.split('_')[1]
            for level in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
                if level == target_level:
                    row[f'pred_{level}'] = pred.get('prediction', None)
                    row[f'conf_{level}'] = pred.get('confidence', None)
                    row[f'top10_{level}'] = json.dumps(pred.get('top10', []))
                    row[f'entropy_{level}'] = pred.get('entropy', None)
                else:
                    # Other levels are None for single-rank models
                    row[f'pred_{level}'] = None
                    row[f'conf_{level}'] = None
                    row[f'top10_{level}'] = json.dumps([])
                    row[f'entropy_{level}'] = None
        
        df_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Save to parquet
    output_dir = Path(config['output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"fold_{fold}_{model_type}.parquet"
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved {len(df)} predictions to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for single fold-model combination")
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-10)')
    parser.add_argument('--model', required=True, help='Model type (e.g., single_species, hierarchical)')
    parser.add_argument('--use_best', action='store_true', default=True, help='Use best checkpoint (default)')
    parser.add_argument('--use_final', action='store_true', help='Use final checkpoint instead of best')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate arguments
    if args.fold < 1 or args.fold > config['num_folds']:
        raise ValueError(f"Fold must be between 1 and {config['num_folds']}")
    
    if args.model not in config['model_types']:
        raise ValueError(f"Model type must be one of: {config['model_types']}")
    
    # Determine checkpoint preference
    use_best = not args.use_final
    
    try:
        output_file = generate_predictions(config, args.fold, args.model, use_best=use_best)
        logger.info(f"Successfully generated predictions: {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise


if __name__ == "__main__":
    main()