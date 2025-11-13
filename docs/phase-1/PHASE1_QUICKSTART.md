# Phase 1 Quick Start Guide

**Status:** ✅ Implementation Complete  
**Date:** 2025-11-12

---

## 🎯 What We Built

A modular dataset mixing system that:
- Loads datasets from HuggingFace (and other sources)
- Applies quality/length filters
- Mixes multiple datasets by specified ratios
- Generates preprocessed JSONL files
- Works seamlessly with existing MiniMind training

---

## 🚀 Quick Start (5 minutes)

### Step 1: Test the Pipeline

```bash
cd /home/saisandeshk/llm/miniGPT

# Run end-to-end test
python scripts/test_mixer_pipeline.py
```

This will:
1. Load `config/data/pretrain/phase1/default.yaml`
2. Download TinyStories from HuggingFace
3. Apply filters
4. Generate a test JSONL file
5. Load it with PretrainDataset
6. Verify everything works

### Step 2: Prepare Your Dataset

```bash
# Prepare train + validation sets
python scripts/prepare_dataset.py \
    --config config/data/pretrain/phase1/default.yaml \
    --output_dir dataset/

# This creates:
#   dataset/pretrain_phase1_default_train.jsonl
#   dataset/pretrain_phase1_default_val.jsonl
```

### Step 3: Train!

```bash
# Train with the prepared dataset
python trainer/train_pretrain.py \
         --data_config config/data/pretrain/phase1/default.yaml \
         --use_prepared --epochs 1 --batch_size 16 --device cuda:0 \
         --use_wandb --wandb_project "miniGPT-test"
```

---

## 📁 Key Files

### Configuration
- `config/data/pretrain/phase1/default.yaml` - Default mixture (TinyStories)
- `config/data/README.md` - Detailed configuration guide

### Core Implementation
- `dataset/mixer.py` - Main mixing engine
- `dataset/loader.py` - Dataset loading (HuggingFace, JSONL, etc.)
- `dataset/filters.py` - Quality and length filters

### Tools
- `scripts/prepare_dataset.py` - CLI tool for dataset preparation
- `scripts/test_mixer_pipeline.py` - End-to-end test

### Tests
- `tests/test_mixer.py` - Mixer unit tests
- `tests/test_filters.py` - Filter unit tests
- `tests/test_loader.py` - Loader unit tests

---

## 🧪 Running Tests

```bash
cd /home/saisandeshk/llm/miniGPT

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_mixer.py -v

# Run with detailed output
python -m pytest tests/test_mixer.py -v -s
```

---

## 📝 Creating Custom Mixtures

1. Copy the default config:
```bash
cp config/data/pretrain/phase1/default.yaml \
   config/data/pretrain/phase1/my_mixture.yaml
```

2. Edit `my_mixture.yaml`:
```yaml
metadata:
  phase: "pretrain_phase1"
  description: "My custom mixture"
  total_tokens: 100_000_000
  max_seq_length: 512

datasets:
  # Dataset 1: 60%
  - name: "tinystories"
    source: "roneneldan/TinyStories"
    mix_ratio: 0.6
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
    max_samples: 100000
    
  # Dataset 2: 40%
  - name: "another_dataset"
    source: "username/dataset-name"
    mix_ratio: 0.4
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
```

3. Prepare and train:
```bash
python scripts/prepare_dataset.py --config config/data/pretrain/phase1/my_mixture.yaml
python trainer/train_pretrain.py --data_config config/data/pretrain/phase1/my_mixture.yaml --use_prepared --epochs 1
```

---

## 🔍 Inspecting Generated Data

```bash
# Look at first 5 samples
head -5 dataset/pretrain_phase1_default_train.jsonl

# Count samples
wc -l dataset/pretrain_phase1_default_train.jsonl

# Check file size
ls -lh dataset/pretrain_phase1_default_train.jsonl

# Verify source distribution (requires jq)
cat dataset/pretrain_phase1_default_train.jsonl | jq -r '.source' | sort | uniq -c
```

---

## ⚙️ Configuration Options

### train_pretrain.py Arguments

```bash
# New arguments for mixer support:
--data_config PATH       Path to mixture YAML config
--use_prepared           Skip re-preparation of dataset

# Existing arguments still work:
--data_path PATH         Direct JSONL file path (original mode)
--epochs N               Number of epochs
--batch_size N           Batch size per GPU
--learning_rate LR       Learning rate
--device DEVICE          cuda:0, cuda:1, or cpu
--accumulation_steps N   Gradient accumulation steps
```

### Mixture YAML Options

```yaml
metadata:
  phase: "pretrain_phase1"      # Phase identifier
  total_tokens: 50_000_000      # Estimated tokens
  max_seq_length: 512           # Max sequence length

datasets:
  - name: "identifier"          # Dataset name
    source: "hf/dataset"        # HF name or local path
    mix_ratio: 1.0              # Proportion (must sum to 1.0)
    format: "huggingface"       # huggingface, jsonl, parquet, arrow
    text_field: "text"          # Field with text data
    splits: ["train"]           # Which splits to use
    max_samples: 100000         # Optional: limit samples
    
    filters:
      - type: "length"          # Filter by length
        min_length: 50
        max_length: 2000
      
      - type: "quality"         # Filter by quality score
        min_score: 0.0

validation:
  ratio: 0.05                   # 5% for validation
  seed: 42                      # Random seed
```

---

## 🐛 Troubleshooting

### Issue: "Failed to load HuggingFace dataset"
**Solution:** Check dataset name, internet connection, or try:
```python
from datasets import load_dataset
load_dataset("roneneldan/TinyStories", split="train")
```

### Issue: "Invalid mixture ratios"
**Solution:** Ensure all `mix_ratio` values sum to 1.0:
```python
from dataset.mixer import DatasetMixer
mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/default.yaml")
print(mixer.validate_mixture())
```

### Issue: Out of memory during preparation
**Solution:** Reduce `max_samples` in YAML config temporarily

### Issue: Tokenizer not found
**Solution:** Make sure `model/` directory exists with tokenizer files

---

## 📊 Example Output

When preparing a dataset, you'll see:

```
======================================================================
Preparing train dataset...
Output: dataset/pretrain_phase1_default_train.jsonl
======================================================================

📦 Loading dataset: tinystories
   Source: roneneldan/TinyStories
   Initial size: 2,119,719 samples
   Applying 2 filter(s)...
   Filters applied: 2,119,719 → 2,119,719 samples (0 removed, 0.0%)
   Limiting to 100000 samples
   ✅ Final size: 100,000 samples

🔀 Mixing datasets...

📊 Mixture composition:
   tinystories: 95,000 samples (100.0%)

🔀 Shuffling 95,000 samples...
💾 Saving to dataset/pretrain_phase1_default_train.jsonl...

======================================================================
✅ Dataset saved successfully!
   File: dataset/pretrain_phase1_default_train.jsonl
   Samples: 95,000
   Size: 12.34 MB
======================================================================
```

---

## 📚 Documentation

For more details, see:
- `docs/PHASE1_UPDATED_APPROACH.md` - Detailed technical design
- `docs/IMPLEMENTATION_GUIDE.md` - Step-by-step implementation
- `config/data/README.md` - Configuration reference

---

## ✅ Checklist Before Phase 2

- [ ] Run `scripts/test_mixer_pipeline.py` successfully
- [ ] Run unit tests with `pytest tests/`
- [ ] Prepare dataset with default config
- [ ] Train for at least 1 epoch successfully
- [ ] Create 1-2 custom mixture configs
- [ ] Verify mixture ratios in output

---

## 🎉 Success!

You now have a working modular dataset mixing pipeline! 

**Next:** Move to Phase 2 (Model Architecture Modularity) when ready.

---

**Questions?** Check the documentation or test scripts for examples.
