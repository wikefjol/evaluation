#!/usr/bin/env python3
"""Check checkpoint structure"""

import sys
import torch
from pathlib import Path
import pickle

base_dir = Path("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification")
checkpoint_path = base_dir / "experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard/fold_1/hierarchical/checkpoint_best.pt"

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check label encoders structure
if 'label_encoders' in checkpoint:
    encoders = checkpoint['label_encoders']
    print("Label encoders type:", type(encoders))
    
    if isinstance(encoders, dict):
        for level, encoder in encoders.items():
            print(f"\n{level}:")
            print(f"  Type: {type(encoder)}")
            if hasattr(encoder, 'classes_'):
                print(f"  Classes: {encoder.classes_[:5]}... (showing first 5)")
                print(f"  Num classes: {len(encoder.classes_)}")
            elif isinstance(encoder, dict):
                print(f"  Dict keys: {list(encoder.keys())[:5]}...")
                print(f"  Dict size: {len(encoder)}")