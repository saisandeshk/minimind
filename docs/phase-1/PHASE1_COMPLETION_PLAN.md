# Phase 1 Completion Plan: Full Training Lifecycle Pipeline

**Version:** 1.0  
**Date:** 2025-11-14  
**Status:** 📝 Planning → Awaiting Approval

---

## 🎯 Objective

Complete Phase 1 by extending dataset mixture + training script support to cover the **entire LLM training lifecycle**:
- ✅ **Pre-training** (DONE)
- 🔨 **Mid-training** (TODO)
- 🔨 **Post-training** (TODO): SFT, DPO, PPO, GRPO, SPO, RLAIF

---

## 📊 Current Status

### ✅ What's Working (Pre-training)
- Dataset mixer loads from YAML configs
- Filters (length, quality) work correctly
- Pre-training script enhanced with:
  - WandB logging with comprehensive metrics
  - Evaluation loop with validation set
  - MFU (Model FLOPs Utilization) tracking
  - Gradient norm monitoring
  - Throughput metrics (tokens/s, samples/s)
  - Checkpoint resume support
- Tested on TinyStories dataset successfully

### 🔨 What Needs Work
1. **Mid-training support** (data mixtures + training logic)
2. **Post-training data configs** (SFT, DPO, PPO, etc.)
3. **Post-training script enhancements** (same level as pretrain)
4. **Consistent interface** across all training stages

---

## 📋 Implementation Plan

### **Task 1: Mid-Training Support** ⏰ ~2-3 hours

Mid-training is a continuation of pre-training with:
- Different data mixture (more refined, domain-specific)
- Learning rate re-warmup and decay
- Replay of some pre-training data to prevent forgetting

#### 1.1 Create Mid-Training Data Configs

**Files to create:**
```
config/data/midtrain/
├── phase1/
│   ├── default.yaml          # Default mid-training mixture
│   ├── code_heavy.yaml        # Code-focused mixture
│   └── reasoning_heavy.yaml   # Math/reasoning mixture
└── README.md                  # Documentation
```

**YAML Schema (midtrain example):**
```yaml
metadata:
  phase: "midtrain_phase1"
  description: "Mid-training with reasoning and code focus"
  total_tokens: 50_000_000    # 50M tokens
  max_seq_length: 512
  version: "1.0"

datasets:
  # 30% from original pretrain data (replay)
  - name: "tinystories_replay"
    source: "roneneldan/TinyStories"
    mix_ratio: 0.3
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
    max_samples: 30000
    
  # 40% code-focused data
  - name: "code_data"
    source: "dataset/code_samples.jsonl"  # Local placeholder
    mix_ratio: 0.4
    format: "jsonl"
    text_field: "text"
    
  # 30% reasoning data
  - name: "reasoning_data"
    source: "dataset/reasoning_samples.jsonl"  # Local placeholder
    mix_ratio: 0.3
    format: "jsonl"
    text_field: "text"

validation:
  ratio: 0.05
  seed: 42
```

#### 1.2 Modify `train_pretrain.py` for Mid-Training

**Changes needed:**
1. Add `--is_midtraining` flag
2. Add learning rate re-warmup logic
3. Track phase in checkpoint metadata
4. Update logging to indicate mid-training phase

**Code additions (~100 lines):**

```python
# In argument parser
parser.add_argument('--is_midtraining', default=0, type=int, choices=[0, 1], 
                    help="Mid-training mode (0=pretrain, 1=midtrain)")
parser.add_argument('--midtrain_warmup_ratio', type=float, default=0.05,
                    help="Warmup ratio for mid-training (fraction of total steps)")

# Modify learning rate schedule for mid-training
def get_midtrain_lr(step, max_steps, learning_rate, warmup_steps):
    """
    Learning rate schedule for mid-training with re-warmup.
    
    1. Warmup from learning_rate/10 to learning_rate
    2. Cosine decay to learning_rate/10
    """
    if step < warmup_steps:
        # Linear warmup
        return (learning_rate / 10) + (learning_rate * 0.9 * step / warmup_steps)
    else:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (learning_rate / 10) + (learning_rate * 0.9 * coeff)

# In train_epoch function, modify LR calculation:
if args.is_midtraining:
    warmup_steps = int(args.midtrain_warmup_ratio * args.epochs * iters)
    lr = get_midtrain_lr(
        epoch * iters + step, 
        args.epochs * iters, 
        args.learning_rate,
        warmup_steps
    )
else:
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
```

**Files to modify:**
- `trainer/train_pretrain.py` (~100 new lines, rename consideration)

**Alternative approach:** Create `train_midtrain.py` as a copy with mid-training logic (RECOMMENDED)
- Pro: Cleaner separation, easier to maintain
- Con: Some code duplication
- **Decision:** Create separate file for clarity

---

### **Task 2: Post-Training Data Configs** ⏰ ~1-2 hours

Create YAML configs for all post-training stages.

#### 2.1 SFT (Supervised Fine-Tuning) Configs

**Files to create:**
```
config/data/posttrain/sft/
├── general.yaml      # General instruction following
├── code.yaml         # Code generation
├── reasoning.yaml    # Chain-of-thought reasoning
└── README.md
```

**Example: `general.yaml`**
```yaml
metadata:
  phase: "sft_general"
  description: "General instruction fine-tuning"
  total_tokens: 10_000_000    # 10M tokens
  max_seq_length: 512
  version: "1.0"

datasets:
  - name: "sft_conversations"
    source: "dataset/sft_mini_512.jsonl"  # Existing file
    mix_ratio: 1.0
    format: "jsonl"
    text_field: "conversations"  # Different field!
    
    filters:
      - type: "length"
        min_length: 10
        max_length: 2000

validation:
  ratio: 0.05
  seed: 42
```

#### 2.2 DPO (Direct Preference Optimization) Configs

**Files to create:**
```
config/data/posttrain/dpo/
├── helpfulness.yaml
├── safety.yaml
└── README.md
```

**Example: `helpfulness.yaml`**
```yaml
metadata:
  phase: "dpo_helpfulness"
  description: "Preference optimization for helpfulness"
  total_tokens: 5_000_000
  max_seq_length: 512
  version: "1.0"

datasets:
  - name: "preference_pairs"
    source: "dataset/dpo.jsonl"  # Existing file
    mix_ratio: 1.0
    format: "jsonl"
    text_field: "chosen"  # Special handling needed
    
validation:
  ratio: 0.05
  seed: 42
```

#### 2.3 RLAIF Configs (PPO/GRPO/SPO)

**Files to create:**
```
config/data/posttrain/rlaif/
├── ppo.yaml
├── grpo.yaml
├── spo.yaml
└── README.md
```

**Example: `ppo.yaml`**
```yaml
metadata:
  phase: "rlaif_ppo"
  description: "PPO training dataset"
  total_tokens: 2_000_000
  max_seq_length: 512
  version: "1.0"

datasets:
  - name: "rlaif_prompts"
    source: "dataset/rlaif-mini.jsonl"  # Existing file
    mix_ratio: 1.0
    format: "jsonl"
    text_field: "conversations"

validation:
  ratio: 0.05
  seed: 42
```

---

### **Task 3: Enhance Post-Training Scripts** ⏰ ~4-6 hours

Update all post-training scripts to match `train_pretrain.py` quality level.

#### 3.1 Core Features to Add (Common to All)

Each script needs:
1. ✅ Dataset mixer support (`--data_config`, `--use_prepared`)
2. ✅ Comprehensive WandB logging
3. ✅ Evaluation loop with metrics
4. ✅ Throughput tracking (tokens/s, MFU)
5. ✅ Gradient norm monitoring
6. ✅ Checkpoint resume support
7. ✅ Better documentation and comments

#### 3.2 Scripts to Update

**Priority 1 (Essential):**
1. `train_full_sft.py` - Most important post-training
2. `train_dpo.py` - RLHF baseline
3. `train_ppo.py` - Online RLAIF

**Priority 2 (Nice to have):**
4. `train_grpo.py` - Advanced RLAIF
5. `train_spo.py` - Cutting-edge RLAIF
6. `train_lora.py` - Parameter-efficient fine-tuning
7. `train_distillation.py` - Model distillation
8. `train_distill_reason.py` - Reasoning distillation

#### 3.3 Enhancement Template

Use this template for each script:

```python
# ===== STANDARD ADDITIONS =====

# 1. Add imports
from pathlib import Path
from dataset.mixer import DatasetMixer

# 2. Add argument parser section
parser.add_argument("--data_config", type=str, default=None, 
                    help="Path to dataset mixture YAML config")
parser.add_argument("--use_prepared", action="store_true", 
                    help="Use pre-prepared JSONL")
parser.add_argument("--eval_interval", type=int, default=500, 
                    help="Evaluation interval (steps)")
parser.add_argument("--eval_batches", type=int, default=100, 
                    help="Number of eval batches")

# 3. Add dataset preparation logic
if args.data_config:
    mixer = DatasetMixer.from_yaml(args.data_config)
    validation = mixer.validate_mixture()
    # ... [same as train_pretrain.py]

# 4. Add evaluation function (if not present)
def evaluate(model, eval_loader, device, autocast_ctx, max_batches=None):
    # ... [copy from train_pretrain.py, adapt for specific loss]

# 5. Enhance logging in train_epoch
if wandb:
    wandb.log({
        "train/loss": current_loss,
        "train/learning_rate": current_lr,
        "train/grad_norm": grad_norm,
        "train/tokens_per_second": tokens_per_sec,
        "train/mfu_percent": mfu,
        # ... more metrics
    }, step=global_step)

# 6. Add eval loop in train_epoch
if eval_loader and args.eval_interval > 0:
    if step % args.eval_interval == 0:
        eval_metrics = evaluate(...)
        Logger(f"Eval loss: {eval_metrics['eval/loss']:.6f}")
```

---

### **Task 4: Update Dataset Loader for Post-Training** ⏰ ~1 hour

#### 4.1 Enhance `dataset/loader.py`

Add support for loading SFT/DPO/RLAIF formats:

```python
def load_single_dataset(config: DatasetConfig) -> List[Dict[str, Any]]:
    """Load a single dataset based on config."""
    # ... existing code ...
    
    # Add format detection for different training phases
    if config.format == "jsonl":
        # Detect data format based on fields
        first_sample = samples[0] if samples else {}
        
        if 'conversations' in first_sample:
            # SFT format
            pass
        elif 'chosen' in first_sample and 'rejected' in first_sample:
            # DPO format - special handling
            pass
        elif 'text' in first_sample:
            # Pretrain/midtrain format
            pass
```

#### 4.2 Update `dataset/mixer.py`

Add phase-aware processing:

```python
def prepare_dataset(self, output_file: str, split: str = "train") -> str:
    """Generate mixed JSONL file."""
    # ... existing code ...
    
    # Phase-specific validation
    if self.config.phase.startswith('sft'):
        # Validate conversation format
        pass
    elif self.config.phase.startswith('dpo'):
        # Validate preference pair format
        pass
```

---

### **Task 5: Documentation and Testing** ⏰ ~1-2 hours

#### 5.1 Create Example Configs

For each training phase, create:
- `default.yaml` - Working example
- `README.md` - Explanation and usage

#### 5.2 Create Test Mixtures

Small test configs for validation:
- `config/data/midtrain/phase1/test.yaml` (1K samples)
- `config/data/posttrain/sft/test.yaml` (500 samples)
- `config/data/posttrain/dpo/test.yaml` (500 samples)
- `config/data/posttrain/rlaif/test.yaml` (500 samples)

#### 5.3 Update Master Documentation

Create `docs/phase-1/LIFECYCLE_TRAINING_GUIDE.md`:
- Complete training pipeline walkthrough
- Pre-training → Mid-training → Post-training flow
- Example commands for each stage
- Data mixture recommendations
- Troubleshooting section

#### 5.4 Create Integration Tests

**File:** `tests/test_full_lifecycle.py`

```python
def test_pretrain_to_midtrain():
    """Test pre-training then mid-training."""
    # 1. Run pretrain for 10 steps
    # 2. Load checkpoint
    # 3. Run midtrain for 10 steps
    # 4. Verify loss decreases

def test_midtrain_to_sft():
    """Test mid-training then SFT."""
    # Similar flow

def test_sft_to_dpo():
    """Test SFT then DPO."""
    # Similar flow
```

---

## 📁 Final Directory Structure

```
miniGPT/
├── config/data/
│   ├── pretrain/phase1/
│   │   ├── default.yaml ✅
│   │   └── test.yaml 🔨
│   ├── midtrain/phase1/
│   │   ├── default.yaml 🔨
│   │   ├── code_heavy.yaml 🔨
│   │   ├── reasoning_heavy.yaml 🔨
│   │   ├── test.yaml 🔨
│   │   └── README.md 🔨
│   └── posttrain/
│       ├── sft/
│       │   ├── general.yaml 🔨
│       │   ├── code.yaml 🔨
│       │   ├── reasoning.yaml 🔨
│       │   ├── test.yaml 🔨
│       │   └── README.md 🔨
│       ├── dpo/
│       │   ├── helpfulness.yaml 🔨
│       │   ├── safety.yaml 🔨
│       │   ├── test.yaml 🔨
│       │   └── README.md 🔨
│       └── rlaif/
│           ├── ppo.yaml 🔨
│           ├── grpo.yaml 🔨
│           ├── spo.yaml 🔨
│           ├── test.yaml 🔨
│           └── README.md 🔨
│
├── trainer/
│   ├── train_pretrain.py ✅
│   ├── train_midtrain.py 🔨 (NEW - copy of pretrain with modifications)
│   ├── train_full_sft.py 🔨 (ENHANCE)
│   ├── train_dpo.py 🔨 (ENHANCE)
│   ├── train_ppo.py 🔨 (ENHANCE)
│   ├── train_grpo.py 🔨 (ENHANCE - optional)
│   ├── train_spo.py 🔨 (ENHANCE - optional)
│   └── train_lora.py 🔨 (ENHANCE - optional)
│
├── dataset/
│   ├── mixer.py ✅ (minor updates for phase detection)
│   ├── loader.py ✅ (minor updates for format detection)
│   └── lm_dataset.py ✅ (no changes needed)
│
├── docs/phase-1/
│   ├── PHASE1_COMPLETION_PLAN.md 🔨 (THIS FILE)
│   ├── LIFECYCLE_TRAINING_GUIDE.md 🔨 (NEW)
│   └── TRAINING_EXAMPLES.md 🔨 (NEW)
│
└── tests/
    └── test_full_lifecycle.py 🔨 (NEW)
```

Legend:
- ✅ Complete
- 🔨 To be created/modified

---

## 🎯 Implementation Order (Recommended)

### Week 1: Foundation (Tasks 1-2)
**Day 1-2:** Mid-training
- Create mid-train data configs
- Create `train_midtrain.py`
- Test mid-training pipeline

**Day 3:** Post-training configs
- Create all YAML configs for SFT/DPO/RLAIF
- Write comprehensive READMEs

### Week 2: Enhancement (Task 3)
**Day 4-5:** Priority 1 scripts
- Enhance `train_full_sft.py`
- Enhance `train_dpo.py`

**Day 6:** Priority 1 continued
- Enhance `train_ppo.py`

**Day 7:** Priority 2 (optional)
- Enhance `train_grpo.py`, `train_spo.py`

### Week 3: Polish (Tasks 4-5)
**Day 8:** Dataset updates
- Update loader.py for format detection
- Update mixer.py for phase validation

**Day 9-10:** Documentation & Testing
- Write lifecycle training guide
- Create integration tests
- Test full pipeline end-to-end

---

## 🧪 Testing Strategy

### Unit Tests
- Test each config loads correctly
- Test mixer handles different phases
- Test each training script loads data

### Integration Tests
```bash
# Test full lifecycle (small scale)
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/test.yaml \
    --epochs 1 --batch_size 4 --device cuda:0

python trainer/train_midtrain.py \
    --data_config config/data/midtrain/phase1/test.yaml \
    --from_weight pretrain_512.pth \
    --epochs 1 --batch_size 4 --device cuda:0

python trainer/train_full_sft.py \
    --data_config config/data/posttrain/sft/test.yaml \
    --from_weight pretrain_512.pth \
    --epochs 1 --batch_size 4 --device cuda:0

python trainer/train_dpo.py \
    --data_config config/data/posttrain/dpo/test.yaml \
    --from_weight full_sft_512.pth \
    --epochs 1 --batch_size 4 --device cuda:0
```

### End-to-End Test
Run complete pipeline on TinyStories:
1. Pretrain → checkpoint
2. Midtrain → checkpoint
3. SFT → checkpoint
4. DPO → final model
5. Verify loss decreases at each stage
6. Verify model generates coherent text

---

## ✅ Success Criteria

Phase 1 is complete when:

1. **All Training Stages Supported**
   - ✅ Pre-training works with mixer
   - ✅ Mid-training works with mixer
   - ✅ SFT works with mixer
   - ✅ DPO works with mixer
   - ✅ PPO works with mixer

2. **Consistent Interface**
   - All scripts accept `--data_config` and `--use_prepared`
   - All scripts have evaluation loops
   - All scripts log to WandB consistently
   - All scripts track same metrics (loss, LR, throughput, MFU)

3. **Documentation Complete**
   - README for each config directory
   - Lifecycle training guide written
   - Example commands documented
   - Troubleshooting guide available

4. **Tested End-to-End**
   - Can run pretrain → midtrain → SFT → DPO successfully
   - Checkpoints load correctly between stages
   - Loss decreases as expected
   - Model quality improves

5. **Ready for Phase 2**
   - Code is clean and well-documented
   - Easy to add new model architectures
   - Easy to add new dataset mixtures
   - Foundation solid for scaling

---

## 📊 Effort Estimation

| Task | Estimated Time | Priority |
|------|---------------|----------|
| 1. Mid-training | 2-3 hours | HIGH |
| 2. Post-training configs | 1-2 hours | HIGH |
| 3.1 Enhance SFT script | 1-2 hours | HIGH |
| 3.2 Enhance DPO script | 1-2 hours | HIGH |
| 3.3 Enhance PPO script | 1-2 hours | HIGH |
| 3.4 Enhance GRPO/SPO scripts | 2-3 hours | MEDIUM |
| 4. Dataset loader updates | 1 hour | MEDIUM |
| 5. Documentation | 2 hours | HIGH |
| 5. Integration tests | 1 hour | MEDIUM |

**Total: ~15-20 hours** (2-3 days of focused work)

**With optional tasks: ~20-25 hours**

---

## 🚨 Potential Challenges

### Challenge 1: DPO/RLAIF Data Format Complexity
**Problem:** These formats have paired data (chosen/rejected) which is different from simple text.

**Solution:** 
- Update mixer to handle paired formats
- Add special processing in loader.py
- Document format requirements clearly

### Challenge 2: Learning Rate Schedules Vary
**Problem:** Different stages use different LR schedules.

**Solution:**
- Keep LR logic in each individual script
- Document recommended LR for each stage
- Add LR schedule visualization to WandB

### Challenge 3: Code Duplication
**Problem:** Similar logic across all training scripts.

**Solution (for Phase 2):**
- Extract common training logic to `trainer/trainer_base.py`
- Create base trainer class that all scripts inherit from
- Keep this simple for Phase 1, refactor in Phase 2

### Challenge 4: Missing Real Datasets
**Problem:** Some configs reference placeholder datasets that don't exist.

**Solution:**
- Use TinyStories as fallback for testing
- Document where to get real datasets
- Add data preparation scripts later

---

## 📝 Example Usage After Completion

### Complete Training Pipeline

```bash
# Step 1: Pre-training (2B tokens)
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --epochs 1 --batch_size 32 --learning_rate 5e-4 \
    --use_wandb --wandb_project "miniGPT-full-pipeline"

# Step 2: Mid-training (500M tokens, code + reasoning focus)
python trainer/train_midtrain.py \
    --data_config config/data/midtrain/phase1/code_heavy.yaml \
    --from_weight pretrain_512.pth \
    --epochs 1 --batch_size 32 --learning_rate 2e-4 \
    --use_wandb --wandb_project "miniGPT-full-pipeline"

# Step 3: SFT (instruction tuning)
python trainer/train_full_sft.py \
    --data_config config/data/posttrain/sft/general.yaml \
    --from_weight pretrain_512.pth \
    --epochs 2 --batch_size 16 --learning_rate 5e-7 \
    --use_wandb --wandb_project "miniGPT-full-pipeline"

# Step 4: DPO (preference alignment)
python trainer/train_dpo.py \
    --data_config config/data/posttrain/dpo/helpfulness.yaml \
    --from_weight full_sft_512.pth \
    --epochs 1 --batch_size 16 --learning_rate 5e-7 \
    --use_wandb --wandb_project "miniGPT-full-pipeline"

# Step 5: PPO (optional, online RLAIF)
python trainer/train_ppo.py \
    --data_config config/data/posttrain/rlaif/ppo.yaml \
    --from_weight dpo_512.pth \
    --epochs 1 --batch_size 16 --learning_rate 5e-7 \
    --use_wandb --wandb_project "miniGPT-full-pipeline"
```

---

## 🎉 Next Steps After Phase 1

Once Phase 1 is complete:

1. **Phase 2: Model Architecture Modularity**
   - Add Llama architecture
   - Add model registry system
   - Support architecture swapping via config

2. **Phase 3: Unified Training Interface**
   - Create single `train.py` that handles all stages
   - Config-driven model + data + training selection

3. **Phase 4: Experiment Management**
   - Add experiment tracking database
   - Add visualization dashboards
   - Add hyperparameter optimization

4. **Phase 5: Scale to 7B**
   - Add FSDP/DeepSpeed
   - Optimize data loading pipeline
   - Prepare large-scale datasets
   - Train final SOTA model

---

## 📞 Review Checklist

Before starting implementation, please review:

- [ ] Are the YAML schemas appropriate?
- [ ] Is the task breakdown clear?
- [ ] Are priorities correct (SFT/DPO/PPO vs GRPO/SPO)?
- [ ] Is the timeline realistic?
- [ ] Should we create `train_midtrain.py` or modify `train_pretrain.py`?
- [ ] Any additional features needed?
- [ ] Any concerns about the approach?

---

**Status:** 📝 **AWAITING APPROVAL**

Please review and provide feedback. Once approved, we'll begin implementation in the order specified above.

---

**Document Version:** 1.0  
**Author:** AI Assistant  
**Last Updated:** 2025-11-14
