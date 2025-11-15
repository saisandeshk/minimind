# Phase 1 Completion: Quick Reference

**Status:** Planning Phase  
**Main Plan:** See [PHASE1_COMPLETION_PLAN.md](./PHASE1_COMPLETION_PLAN.md)

---

## 🎯 What We're Building

Extend dataset mixer + training script support to **entire training lifecycle**:

```
✅ Pre-training  (DONE)
    ↓
🔨 Mid-training  (TODO)
    ↓
🔨 SFT           (TODO)
    ↓
🔨 DPO/PPO/RLAIF (TODO)
```

---

## 📋 Task Summary

| # | Task | Time | Priority | Status |
|---|------|------|----------|--------|
| 1 | Mid-training support | 2-3h | HIGH | 🔨 TODO |
| 2 | Post-training configs (YAML) | 1-2h | HIGH | 🔨 TODO |
| 3 | Enhance SFT/DPO/PPO scripts | 4-6h | HIGH | 🔨 TODO |
| 4 | Update dataset loader | 1h | MEDIUM | 🔨 TODO |
| 5 | Documentation & tests | 2-3h | HIGH | 🔨 TODO |

**Total:** ~15-20 hours (2-3 days)

---

## 🚀 Implementation Order

### Day 1: Mid-Training
1. Create `config/data/midtrain/phase1/default.yaml`
2. Create `trainer/train_midtrain.py` (copy train_pretrain.py + add LR re-warmup)
3. Test: `python trainer/train_midtrain.py --data_config ... --epochs 1`

### Day 2: Post-Training Configs
4. Create all YAML configs in `config/data/posttrain/`
   - `sft/general.yaml`
   - `dpo/helpfulness.yaml`
   - `rlaif/ppo.yaml`
5. Write README.md for each directory

### Day 3-4: Enhance Scripts (Priority 1)
6. Enhance `trainer/train_full_sft.py` (add mixer, wandb, eval, metrics)
7. Enhance `trainer/train_dpo.py` (same enhancements)
8. Enhance `trainer/train_ppo.py` (same enhancements)

### Day 5: Polish
9. Update `dataset/loader.py` (format detection for SFT/DPO)
10. Write `docs/phase-1/LIFECYCLE_TRAINING_GUIDE.md`
11. Create `tests/test_full_lifecycle.py`
12. Test end-to-end: pretrain → midtrain → sft → dpo

---

## 📝 Key Files to Create

### Configs
```
config/data/
├── midtrain/phase1/
│   └── default.yaml 🔨
├── posttrain/sft/
│   └── general.yaml 🔨
├── posttrain/dpo/
│   └── helpfulness.yaml 🔨
└── posttrain/rlaif/
    └── ppo.yaml 🔨
```

### Training Scripts
```
trainer/
├── train_midtrain.py 🔨 (NEW)
├── train_full_sft.py 🔨 (ENHANCE)
├── train_dpo.py 🔨 (ENHANCE)
└── train_ppo.py 🔨 (ENHANCE)
```

### Documentation
```
docs/phase-1/
└── LIFECYCLE_TRAINING_GUIDE.md 🔨 (NEW)
```

---

## 🎯 Enhancement Template

For each training script, add:

1. **Argument parser:**
   ```python
   parser.add_argument("--data_config", type=str, default=None)
   parser.add_argument("--use_prepared", action="store_true")
   parser.add_argument("--eval_interval", type=int, default=500)
   ```

2. **Dataset preparation:**
   ```python
   if args.data_config:
       mixer = DatasetMixer.from_yaml(args.data_config)
       train_file = mixer.prepare_dataset(...)
   ```

3. **Evaluation function:**
   ```python
   def evaluate(model, eval_loader, ...):
       # Copy from train_pretrain.py
   ```

4. **Enhanced logging:**
   ```python
   if wandb:
       wandb.log({
           "train/loss": loss,
           "train/lr": lr,
           "train/tokens_per_second": tokens_per_sec,
           "train/mfu_percent": mfu,
           # ... more metrics
       })
   ```

5. **Eval loop in training:**
   ```python
   if step % args.eval_interval == 0:
       eval_metrics = evaluate(...)
       Logger(f"Eval loss: {eval_metrics['eval/loss']}")
   ```

---

## 🧪 Testing Commands

```bash
# Pre-training (already works)
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --use_prepared --epochs 1 --batch_size 16 --device cuda:0

# Mid-training (to implement)
python trainer/train_midtrain.py \
    --data_config config/data/midtrain/phase1/default.yaml \
    --from_weight pretrain_512.pth \
    --use_prepared --epochs 1 --batch_size 16 --device cuda:0

# SFT (to enhance)
python trainer/train_full_sft.py \
    --data_config config/data/posttrain/sft/general.yaml \
    --from_weight pretrain_512.pth \
    --use_prepared --epochs 1 --batch_size 16 --device cuda:0

# DPO (to enhance)
python trainer/train_dpo.py \
    --data_config config/data/posttrain/dpo/helpfulness.yaml \
    --from_weight full_sft_512.pth \
    --use_prepared --epochs 1 --batch_size 16 --device cuda:0
```

---

## ✅ Success Criteria

Phase 1 complete when:
- [ ] All stages have YAML configs
- [ ] All scripts support `--data_config`
- [ ] All scripts have evaluation loops
- [ ] All scripts log to WandB consistently
- [ ] Can run full pipeline: pretrain → midtrain → sft → dpo
- [ ] Documentation complete
- [ ] Tested end-to-end

---

## 💡 Key Decisions

1. **Create `train_midtrain.py`** (don't modify train_pretrain.py)
   - Cleaner separation
   - Easier maintenance
   - Worth minor duplication

2. **Enhance scripts one by one** (don't refactor to base class yet)
   - Faster implementation
   - Keep Phase 1 simple
   - Refactor in Phase 2

3. **Priority: SFT → DPO → PPO** (skip GRPO/SPO for now)
   - Cover essential post-training methods
   - Add advanced methods later if needed

---

## 🚨 Watch Out For

1. **DPO data format** - Has `chosen` and `rejected` fields (not just `text`)
2. **Mid-training LR** - Needs re-warmup, not just continuation
3. **Eval metrics** - DPO loss calculation different from pretrain
4. **Dataset sizes** - Make test configs small for fast iteration

---

## 📞 Questions to Resolve

Before starting:
1. Should mid-training be in train_pretrain.py or separate? → **SEPARATE**
2. Priority on GRPO/SPO enhancement? → **SKIP FOR NOW**
3. Need train_lora.py enhancement? → **PHASE 2**
4. Timeline: 2-3 days realistic? → **YES**

---

## 📚 Reference

- **Main plan:** [PHASE1_COMPLETION_PLAN.md](./PHASE1_COMPLETION_PLAN.md)
- **Current status:** [VERIFICATION_REPORT.md](./VERIFICATION_REPORT.md)
- **Quick start:** [PHASE1_QUICKSTART.md](./PHASE1_QUICKSTART.md)
- **Implementation guide:** [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)

---

**Ready to start once approved!** 🚀
