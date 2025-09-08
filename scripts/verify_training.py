#!/usr/bin/env python3
"""
Training verification script for k-fold experiments
Analyzes all 70 models to verify training worked correctly
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict, Counter

def load_results(results_path):
    """Load results.json file safely"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return None

def verify_checkpoints(model_dir):
    """Verify all 15 checkpoints exist (epoch_0.pt to epoch_14.pt)"""
    expected_checkpoints = [f"checkpoint_epoch_{i}.pt" for i in range(15)]
    existing_checkpoints = []
    
    for checkpoint in expected_checkpoints:
        checkpoint_path = model_dir / checkpoint
        if checkpoint_path.exists():
            existing_checkpoints.append(checkpoint)
    
    return len(existing_checkpoints), expected_checkpoints

def analyze_training_progression(training_history):
    """Analyze if model actually learned during training"""
    if not training_history or len(training_history) == 0:
        return None
    
    first_epoch = training_history[0]
    last_epoch = training_history[-1]
    
    # Check loss progression
    first_train_loss = first_epoch.get('train_loss', 0)
    last_train_loss = last_epoch.get('train_loss', 0)
    first_val_loss = first_epoch.get('val_loss', 0)
    last_val_loss = last_epoch.get('val_loss', 0)
    
    # Check accuracy progression
    first_train_acc = first_epoch.get('train_overall_accuracy', 0)
    last_train_acc = last_epoch.get('train_overall_accuracy', 0)
    first_val_acc = first_epoch.get('val_overall_accuracy', 0)
    last_val_acc = last_epoch.get('val_overall_accuracy', 0)
    
    return {
        'train_loss_change': first_train_loss - last_train_loss,
        'val_loss_change': first_val_loss - last_val_loss,
        'train_acc_change': last_train_acc - first_train_acc,
        'val_acc_change': last_val_acc - first_val_acc,
        'epochs_completed': len(training_history),
        'first_epoch': first_epoch,
        'last_epoch': last_epoch
    }

def main():
    # Base path to experiments
    base_path = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard"
    
    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        print("Please update the path in the script")
        return
    
    print("=" * 60)
    print("     TRAINING VERIFICATION REPORT")
    print("=" * 60)
    
    # Find all results.json files
    results_pattern = f"{base_path}/fold_*/*/results.json"
    results_files = glob.glob(results_pattern)
    
    print(f"Found {len(results_files)} results files")
    
    if len(results_files) != 70:
        print(f"⚠️  Expected 70 results files, found {len(results_files)}")
    
    # Analysis containers
    best_epoch_dist = Counter()
    checkpoint_issues = []
    learning_issues = []
    successful_models = []
    model_summaries = []
    
    for results_file in sorted(results_files):
        # Parse model info from path
        path_parts = Path(results_file).parts
        fold_name = path_parts[-3]  # fold_X
        model_type = path_parts[-2]  # single_species, hierarchical, etc.
        model_dir = Path(results_file).parent
        
        # Load results
        results = load_results(results_file)
        if not results:
            continue
        
        # Verify checkpoints
        checkpoint_count, expected = verify_checkpoints(model_dir)
        
        # Get key metrics
        best_epoch = results.get('best_epoch', 'unknown')
        best_val_acc = results.get('best_val_accuracy', 0)
        final_epoch = results.get('final_epoch', 'unknown')
        training_history = results.get('training_history', [])
        
        # Analyze progression
        progression = analyze_training_progression(training_history)
        
        # Track best epoch distribution
        if isinstance(best_epoch, int):
            best_epoch_dist[best_epoch] += 1
        
        # Model summary
        summary = {
            'fold': fold_name,
            'model_type': model_type,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'final_epoch': final_epoch,
            'checkpoints': f"{checkpoint_count}/15",
            'progression': progression
        }
        model_summaries.append(summary)
        
        # Check for issues
        if checkpoint_count < 15:
            checkpoint_issues.append(f"{fold_name}/{model_type}: only {checkpoint_count}/15 checkpoints")
        
        if progression and progression['val_loss_change'] < 0.01:  # Loss barely improved
            learning_issues.append({
                'model': f"{fold_name}/{model_type}",
                'val_loss_change': progression['val_loss_change'],
                'val_acc_change': progression['val_acc_change'],
                'best_epoch': best_epoch
            })
        elif progression and progression['val_loss_change'] > 0.1:  # Good improvement
            successful_models.append(f"{fold_name}/{model_type}")
    
    # Print completeness check
    print(f"\n📊 COMPLETENESS CHECK:")
    print(f"   Results files: {len(results_files)}/70 {'✅' if len(results_files) == 70 else '⚠️'}")
    
    # Print checkpoint verification
    print(f"\n🗂️  CHECKPOINT VERIFICATION:")
    if checkpoint_issues:
        print(f"   Issues found: {len(checkpoint_issues)}")
        for issue in checkpoint_issues[:10]:  # Show first 10
            print(f"     - {issue}")
        if len(checkpoint_issues) > 10:
            print(f"     ... and {len(checkpoint_issues) - 10} more")
    else:
        print(f"   All models have 15 checkpoints ✅")
    
    # Print best epoch distribution
    print(f"\n🎯 BEST EPOCH DISTRIBUTION:")
    epoch_0_count = best_epoch_dist.get(0, 0)
    total_models = sum(best_epoch_dist.values())
    
    if epoch_0_count > total_models * 0.5:
        print(f"   ⚠️  {epoch_0_count}/{total_models} models ({epoch_0_count/total_models*100:.1f}%) have best_epoch = 0")
        print("       This suggests training issues!")
    else:
        print(f"   ✅ Best epoch distribution looks reasonable")
    
    print("   Distribution:")
    for epoch in sorted(best_epoch_dist.keys())[:15]:  # Show first 15 epochs
        count = best_epoch_dist[epoch]
        percentage = count / total_models * 100
        bar = "█" * int(percentage / 5)  # Simple bar chart
        print(f"     Epoch {epoch:2d}: {count:2d} models ({percentage:4.1f}%) {bar}")
    
    # Print learning analysis
    print(f"\n📈 LEARNING ANALYSIS:")
    print(f"   Models with minimal learning: {len(learning_issues)}")
    print(f"   Models with good learning: {len(successful_models)}")
    
    if learning_issues:
        print("\n   ⚠️  MODELS WITH MINIMAL LEARNING:")
        for issue in learning_issues[:15]:  # Show first 15
            print(f"     {issue['model']}: "
                  f"loss_change={issue['val_loss_change']:.3f}, "
                  f"acc_change={issue['val_acc_change']:.4f}, "
                  f"best_epoch={issue['best_epoch']}")
        if len(learning_issues) > 15:
            print(f"     ... and {len(learning_issues) - 15} more")
    
    # Sample of successful models
    if successful_models:
        print(f"\n   ✅ SAMPLE OF SUCCESSFUL MODELS:")
        for model in successful_models[:10]:
            print(f"     {model}")
        if len(successful_models) > 10:
            print(f"     ... and {len(successful_models) - 10} more")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(successful_models)} successful, {len(learning_issues)} problematic, {len(checkpoint_issues)} incomplete")
    print("=" * 60)

if __name__ == "__main__":
    main()