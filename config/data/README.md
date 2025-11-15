# Dataset Mixture Configurations

This directory contains YAML configuration files for defining dataset mixtures used in different training phases.

## Directory Structure

```
config/data/
├── pretrain/
│   ├── phase1/
│   │   └── default.yaml       # Default pretraining mixture
│   │   └── general.yaml  
│   │   └── reasoning.yaml  
│   └── phase2/
├── midtrain/
│   ├── phase1/
│   │   └── default.yaml  # Default midtraining mixture
│   │   └── code_heavy.yaml  
│   │   └── math_heavy.yaml  
│   └── phase2/
└── posttrain/
    ├── sft/
    ├── dpo/
    ├── ppo/
    └── rlaif/
```

## YAML Configuration Format

Each mixture configuration file defines:

```yaml
metadata:
  phase: "pretrain_phase1"
  description: "Human-readable description"
  total_tokens: 50_000_000  # Estimated total tokens
  max_seq_length: 512
  version: "1.0"

datasets:
  - name: "dataset_identifier"
    source: "huggingface/dataset-name"  # or local path
    mix_ratio: 1.0  # Proportion (must sum to 1.0 across all datasets)
    format: "huggingface"  # or jsonl, parquet, arrow
    text_field: "text"  # Field containing text data
    splits: ["train"]  # Which splits to use
    max_samples: 100000  # Optional: limit samples
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 2000
      
      - type: "quality"
        min_score: 0.5

validation:
  ratio: 0.05  # 5% for validation
  seed: 42
  stratified: true
```

## Usage

### 1. Prepare Dataset from Config

```python
from dataset.mixer import DatasetMixer

# Load configuration
mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/default.yaml")

# Generate JSONL files
train_file = mixer.prepare_dataset(
    output_file="dataset/pretrain_phase1_default_train.jsonl",
    split="train"
)

val_file = mixer.prepare_dataset(
    output_file="dataset/pretrain_phase1_default_val.jsonl",
    split="validation"
)
```

### 2. Train with Mixture Config

```bash
# Option 1: Auto-prepare dataset and train
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --epochs 1 \
    --batch_size 32

# Option 2: Use pre-prepared dataset (skip re-preparation)
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --use_prepared \
    --epochs 1 \
    --batch_size 32
```

### 3. Traditional Mode (Direct JSONL)

```bash
# Still works for backward compatibility
python trainer/train_pretrain.py \
    --data_path dataset/my_data.jsonl \
    --epochs 1 \
    --batch_size 32
```

## Creating New Mixtures

1. Copy an existing YAML file as a template
2. Modify the datasets section:
   - Add/remove datasets
   - Adjust mix_ratio values (must sum to 1.0)
   - Configure filters as needed
3. Test with a small max_samples first
4. Prepare the full dataset when ready

### Example: Multi-Dataset Mixture

```yaml
datasets:
  # 60% TinyStories
  - name: "tinystories"
    source: "roneneldan/TinyStories"
    mix_ratio: 0.6
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
    
  # 30% Custom data
  - name: "custom"
    source: "dataset/my_custom.jsonl"
    mix_ratio: 0.3
    format: "jsonl"
    text_field: "text"
    
  # 10% Another dataset
  - name: "other"
    source: "some/other-dataset"
    mix_ratio: 0.1
    format: "huggingface"
    text_field: "content"
```

## Available Filters

### Length Filter
Filters by text length (character count):
```yaml
- type: "length"
  min_length: 50
  max_length: 2000
```

### Quality Filter
Filters by heuristic quality score (0.0 to 1.0):
```yaml
- type: "quality"
  min_score: 0.5
```

Quality scoring considers:
- Punctuation presence
- Word diversity
- Repetition
- Sentence structure

### Language Filter (placeholder)
```yaml
- type: "language"
  languages: ["en"]
```

Note: Currently a placeholder. Implement with langdetect if needed.

## Output Format

Generated JSONL files contain:

```json
{"text": "Sample text content...", "source": "dataset_name"}
{"text": "Another sample...", "source": "dataset_name"}
```

The `source` field tracks which dataset each sample came from, useful for analysis.

## Tips

1. **Start small**: Use `max_samples` to test with a subset first
2. **Validate ratios**: Ensure mix_ratio values sum to 1.0
3. **Check output**: Inspect generated JSONL files before full training
4. **Monitor disk space**: JSONL files can be large (~same size as source data)
5. **Reuse prepared data**: Use `--use_prepared` to skip re-preparation

## Troubleshooting

### "Invalid mixture ratios"
- Check that all `mix_ratio` values sum to 1.0
- Use the validator: `mixer.validate_mixture()`

### "Failed to load HuggingFace dataset"
- Verify the dataset name/path is correct
- Check internet connection for HF datasets
- Try loading manually to see full error

### Out of memory during preparation
- Reduce `max_samples` temporarily
- Process datasets one at a time
- Use streaming mode (future enhancement)

## Future Enhancements

- [ ] Streaming support for very large datasets
- [ ] Advanced deduplication (MinHash, SimHash)
- [ ] Actual language detection integration
- [ ] Token-level mixing (instead of sample-level)
- [ ] Curriculum learning support
- [ ] Quality model-based filtering
