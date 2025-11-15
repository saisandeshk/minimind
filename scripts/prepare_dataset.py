#!/usr/bin/env python3
"""
Prepare dataset from a mixture configuration.

This script loads a mixture config and generates train/validation JSONL files.

Usage:
    python scripts/prepare_dataset.py --config config/data/pretrain/phase1/default.yaml
    python scripts/prepare_dataset.py --config config/data/pretrain/phase1/default.yaml --output_dir custom_output/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.mixer import DatasetMixer


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from mixture configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to mixture YAML config file")
    parser.add_argument("--output_dir", type=str, default="dataset/",
                        help="Output directory for JSONL files")
    parser.add_argument("--train_only", action="store_true",
                        help="Only prepare training set (skip validation)")
    parser.add_argument("--val_only", action="store_true",
                        help="Only prepare validation set (skip training)")
    
    args = parser.parse_args()
    
    # Load mixer
    print(f"\n{'='*70}")
    print(f"Loading configuration: {args.config}")
    print(f"{'='*70}\n")
    
    mixer = DatasetMixer.from_yaml(args.config)
    
    # Validate mixture
    validation = mixer.validate_mixture()
    print(f"Mixture validation: {validation}")
    
    if not validation['is_valid']:
        print("\n❌ Error: Invalid mixture ratios! They must sum to 1.0")
        sys.exit(1)
    
    # Generate output filenames
    from pathlib import Path
    config_name = Path(args.config).stem
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_file = os.path.join(args.output_dir, f"{mixer.config.phase}_{config_name}_train.jsonl")
    val_file = os.path.join(args.output_dir, f"{mixer.config.phase}_{config_name}_val.jsonl")
    
    # Prepare train set
    if not args.val_only:
        print(f"\nPreparing training set...")
        mixer.prepare_dataset(train_file, split="train")
    
    # Prepare validation set
    if not args.train_only:
        print(f"\nPreparing validation set...")
        mixer.prepare_dataset(val_file, split="validation")
    
    print(f"\n{'='*70}")
    print("✅ Dataset preparation complete!")
    print(f"{'='*70}\n")
    
    if not args.val_only:
        print(f"Training set:   {train_file}")
    if not args.train_only:
        print(f"Validation set: {val_file}")
    
    print("\nYou can now train with (If pre-train):")
    print(f"  python trainer/train_pretrain.py \\")
    print(f"      --data_config {args.config} \\")
    print(f"      --use_prepared \\")
    print(f"      --epochs 1")
    print()

    print("\nYou can now train with (If mid-train):")
    print(f"  python trainer/train_midtrain.py \\")
    print(f"      --data_config {args.config} \\")
    print(f"      --from weight out/pretrain_512.pth")
    print(f"      --use_prepared \\")
    print(f"      --epochs 1\\")
    print(f"      --batch_size 16\\")
    print(f"      --learning_rate 2e-4\\")
    print(f"      --use_wandb\\")
    print(f"      --wandb_project "miniGPT-midtrain-experiment"\\")
    print()


if __name__ == "__main__":
    main()
