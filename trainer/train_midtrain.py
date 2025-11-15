import os
import sys
from pathlib import Path

# Package path manipulation to ensure proper imports from parent directory
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


def get_midtrain_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int) -> float:
    """
    Learning rate schedule for mid-training with re-warmup.
    
    Mid-training continues from a pretrained model but with different data.
    We use a lower starting LR with re-warmup to avoid destabilizing the model.
    
    Schedule:
    1. Warmup: linear from learning_rate/10 to learning_rate over warmup_steps
    2. Decay: cosine decay from learning_rate to learning_rate/10
    
    Args:
        step: Current training step
        max_steps: Total training steps
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
    
    Returns:
        Learning rate for current step
    """
    import math
    
    min_lr = learning_rate / 10  # Minimum LR is 10% of peak
    
    if step < warmup_steps:
        # Linear warmup from min_lr to learning_rate
        return min_lr + (learning_rate - min_lr) * step / warmup_steps
    else:
        # Cosine decay from learning_rate to min_lr
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine_decay


def calculate_mfu(model_flops_per_token, tokens_per_sec, device):
    """
    Calculate Model FLOPs Utilization (MFU) - measures hardware efficiency.
    
    Args:
        model_flops_per_token: FLOPs required to process one token
        tokens_per_sec: Actual throughput in tokens/second
        device: Device type ('cuda', 'cpu', etc.)
    
    Returns:
        MFU as a percentage (0-100)
    """
    import torch

    # Peak FLOPS for different GPUs (in FLOPs for bfloat16)
    peak_flops = {
        'A100': 312e12,       # A100 80GB
        'H100': 989e12,       # H100 80GB
        'V100': 125e12,       # V100 32GB
        '4090': 165e12,       # RTX 4090
        'RTX6000_BLACKWELL': 550e12,  # RTX PRO 6000 Blackwell Max-Q
        'default': 125e12     # Conservative estimate
    }

    # Try to detect GPU type
    if 'cuda' in str(device):
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if 'A100' in gpu_name:
                peak = peak_flops['A100']
            elif 'H100' in gpu_name:
                peak = peak_flops['H100']
            elif 'V100' in gpu_name:
                peak = peak_flops['V100']
            elif '4090' in gpu_name:
                peak = peak_flops['4090']
            elif '6000' in gpu_name and ('Blackwell' in gpu_name or 'PRO' in gpu_name):
                peak = peak_flops['RTX6000_BLACKWELL']
            else:
                peak = peak_flops['default']
        except:
            peak = peak_flops['default']
    else:
        return 0.0  # MFU not applicable for CPU

    achieved_flops = model_flops_per_token * tokens_per_sec
    mfu = (achieved_flops / peak) * 100
    return mfu



def evaluate(model, eval_loader, device, autocast_ctx, max_batches=None):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: The model to evaluate
        eval_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        autocast_ctx: Automatic mixed precision context
        max_batches: Maximum number of batches to evaluate (None for all)
    
    Returns:
        dict: Dictionary containing eval metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (X, Y, loss_mask) in enumerate(eval_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            with autocast_ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                
                # Apply mask and sum
                batch_loss = (loss * loss_mask).sum().item()
                batch_tokens = loss_mask.sum().item()
                
                total_loss += batch_loss
                total_tokens += batch_tokens
                num_batches += 1
    
    eval_time = time.time() - eval_start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    model.train()
    
    return {
        'eval/loss': avg_loss,
        'eval/runtime': eval_time,
        'eval/samples_per_second': (num_batches * eval_loader.batch_size) / eval_time if eval_time > 0 else 0,
        'eval/steps_per_second': num_batches / eval_time if eval_time > 0 else 0,
        'eval/num_batches': num_batches,
        'eval/total_tokens': total_tokens,
    }


def train_epoch(
    epoch: int, 
    loader: DataLoader, 
    iters: int, 
    start_step: int = 0, 
    wandb=None,
    eval_loader=None,
    warmup_steps: int = 0
) -> None:
    """
    Train the model for a single epoch during the MID-TRAINING phase.
    
    Mid-training is a continuation of pre-training with different data mixtures,
    typically focusing on specific domains (code, reasoning, etc.) while replaying
    some pre-training data to prevent catastrophic forgetting.
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches of mid-training data
        iters: Total number of iterations in this epoch
        start_step: Starting step number if resuming from checkpoint
        wandb: Weights & Biases logger instance (or None)
        eval_loader: DataLoader for evaluation (optional)
        warmup_steps: Number of steps for LR re-warmup
    
    Mid-training Characteristics:
        - Continues from pretrained checkpoint
        - Uses specialized data mixtures (code, math, etc.)
        - Includes replay of pretrain data (30%) to prevent forgetting
        - Lower learning rate with re-warmup schedule
        - Same causal language modeling objective
    
    Key Training Features:
        - Learning rate re-warmup from LR/10 → LR
        - Cosine LR decay after warmup
        - Mixed precision training
        - Gradient accumulation
        - Distributed training support
        - Periodic evaluation and checkpointing
    """
    # Debug: Print wandb status at start
    if is_main_process():
        Logger(f"[DEBUG] train_epoch started - wandb={'enabled' if wandb else 'disabled'}")
    
    # Initialize loss function, timers, and tracking variables
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    epoch_start_time = time.time()
    total_tokens_seen = 0
    grad_norm = 0.0
    
    # Calculate model FLOPs per token (approximate for transformer)
    # Formula: 6 * n_params (2 for forward, 4 for backward)
    n_params = sum(p.numel() for p in model.parameters())
    model_flops_per_token = 6 * n_params
    
    # Iterate through data batches, starting from the specified step
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # Move all tensors to the target device (GPU/CPU)
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # Calculate learning rate with mid-training re-warmup schedule
        global_step = epoch * iters + step
        max_steps = args.epochs * iters
        lr = get_midtrain_lr(global_step, max_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with automatic mixed precision
        with autocast_ctx:
            res = model(X)
            
            # Compute per-token cross-entropy loss
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # Apply loss mask to ignore padding tokens and normalize
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # Add auxiliary loss (e.g., from MoE load balancing if enabled)
            loss += res.aux_loss
            
            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Track tokens processed in this batch
        batch_tokens = loss_mask.sum().item()
        total_tokens_seen += batch_tokens

        # Update weights only every accumulation_steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients before clipping and compute gradient norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

            # Optimizer step and zero gradients
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients and cache for memory efficiency
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Logging and monitoring
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # Scale back for accurate reporting
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60  # Estimated time remaining
            
            # Calculate throughput metrics
            tokens_per_sec = total_tokens_seen / spend_time if spend_time > 0 else 0
            samples_per_sec = (step * args.batch_size) / spend_time if spend_time > 0 else 0
            steps_per_sec = step / spend_time if spend_time > 0 else 0
            
            # Calculate MFU (Model FLOPs Utilization)
            mfu = calculate_mfu(model_flops_per_token, tokens_per_sec, args.device)
            
            # Global step across all epochs
            global_step = epoch * iters + step
            
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} lr:{current_lr:.12f} '
                f'grad_norm:{grad_norm:.4f} tokens/s:{tokens_per_sec:.0f} '
                f'MFU:{mfu:.2f}% epoch_Time:{eta_min}min'
            )
            
            # Run evaluation if enabled and at eval interval
            eval_metrics = {}
            if eval_loader and args.eval_interval > 0 and (step % args.eval_interval == 0 or step == iters - 1):
                if is_main_process():
                    Logger("Running evaluation...")
                eval_metrics = evaluate(model, eval_loader, args.device, autocast_ctx, max_batches=args.eval_batches)
                if is_main_process():
                    Logger(f"Eval loss: {eval_metrics['eval/loss']:.6f}, Eval runtime: {eval_metrics['eval/runtime']:.2f}s")
            
            # Log to wandb with comprehensive metrics
            if wandb:
                log_dict = {
                    # Training metrics
                    "train/loss": current_loss,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": grad_norm,
                    "train/epoch": epoch + (step / iters),
                    "train/global_step": global_step,
                    
                    # Throughput metrics
                    "train/tokens_per_second": tokens_per_sec,
                    "train/samples_per_second": samples_per_sec,
                    "train/steps_per_second": steps_per_sec,
                    "train/num_input_tokens_seen": total_tokens_seen,
                    
                    # Efficiency metrics
                    "train/mfu_percent": mfu,
                    "train/eta_minutes": eta_min,
                    
                    # System metrics
                    "system/epoch_time": spend_time / 60,  # in minutes
                }
                
                # Add eval metrics if available
                log_dict.update(eval_metrics)
                
                wandb.log(log_dict, step=global_step)

        # Model checkpointing (only on main process in distributed training)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # Determine filename suffix based on architecture
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract state dict (handle DDP wrapper)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # Save in half precision to reduce storage requirements
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # Save full checkpoint with optimizer state for training resumption
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main MID-TRAINING script for MiniMind Language Model.
    
    Mid-training continues from a pretrained checkpoint with specialized data mixtures
    to enhance specific capabilities (coding, reasoning, etc.) while maintaining
    general knowledge through data replay.
    
    Training Pipeline:
        1. Load pretrained model checkpoint
        2. Prepare mid-training dataset mixture (code + math + replay)
        3. Train with LR re-warmup and cosine decay
        4. Save periodic checkpoints and logs
    
    Typical Usage:
        # Mid-train from pretrained checkpoint
        python train_midtrain.py \
            --data_config config/data/midtrain/phase1/default.yaml \
            --from_weight pretrain \
            --epochs 1 --learning_rate 2e-4 --warmup_ratio 0.05
        
        # Resume mid-training from checkpoint
        python train_midtrain.py --from_resume 1 --epochs 2
    
    Key Differences from Pre-training:
        - MUST load from pretrained checkpoint (--from_weight required)
        - Lower learning rate (recommend 1e-4 to 3e-4 vs 5e-4 for pretrain)
        - LR re-warmup enabled (--warmup_ratio)
        - Specialized data mixtures with replay
    """
    parser = argparse.ArgumentParser(description="MiniMind Mid-Training")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="out", help="Model save directory")
    parser.add_argument('--save_weight', default='midtrain', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate (recommend 1e-4 to 3e-4 for mid-training)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio (fraction of total steps for LR re-warmup)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (steps)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval (steps), 0 to disable")
    parser.add_argument("--eval_batches", type=int, default=100, help="Number of batches to use for evaluation")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=512, type=int, help="Maximum sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Data and initialization
    parser.add_argument("--data_path", type=str, default="dataset/midtrain_phase1.jsonl", help="Mid-training data path (JSONL file)")
    parser.add_argument("--data_config", type=str, default=None, help="Path to dataset mixture YAML config (alternative to --data_path)")
    parser.add_argument("--use_prepared", action="store_true", help="Use pre-prepared JSONL from data_config (skip re-preparation)")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="Base weight for mid-training (REQUIRED - typically 'pretrain' or checkpoint path)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect & resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="miniGPT-test-midtrain", help="wandb project name")
    
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    # Set up distributed training if available (multi-GPU)
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # Set random seed for reproducibility, different per process
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Configure directories, model parameters, check checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize model configuration
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # Check for existing checkpoint if resuming training
    ckp_data = lm_checkpoint(
        lm_config, weight=args.save_weight, save_dir='checkpoints'
    ) if args.from_resume == 1 else None
    
    # ========== 3. Set up mixed precision training ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # No-op context for CPU, autocast for CUDA
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. Configure wandb logging ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_module
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"miniGPT-Midtrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LearningRate-{args.learning_rate}"
        )
        wandb_run = wandb_module.init(
            project=args.wandb_project, 
            name=wandb_run_name, 
            id=wandb_id, 
            resume=resume
        )
        wandb = wandb_module  # Use the module for logging
        print(f"✅ Wandb initialized: {wandb_run.name} (ID: {wandb_run.id})") 
    
    # ========== 5. Initialize model, dataset, and optimizer ==========
    # Mid-training MUST start from pretrained checkpoint
    if args.from_weight == 'none':
        raise ValueError(
            "❌ Mid-training requires a pretrained checkpoint!\n"
            "   Use --from_weight pretrain (or your checkpoint path)\n"
            "   Mid-training continues from pretrained weights with specialized data."
        )
    
    # Initialize model and tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    if is_main_process():
        Logger(f"✅ Loaded pretrained model from: {args.from_weight}")
    
    # Initialize pretraining dataset (large-scale text corpus)
    # Support two modes:
    # 1. Direct JSONL file (--data_path)
    # 2. Dataset mixture config (--data_config) with automatic preparation
    if args.data_config:
        from dataset.mixer import DatasetMixer
        
        if is_main_process():
            print(f"Using dataset mixture config: {args.data_config}")
        
        # Load mixer configuration
        mixer = DatasetMixer.from_yaml(args.data_config)
        
        # Validate mixture ratios
        validation = mixer.validate_mixture()
        if is_main_process():
            print(f"Mixture validation: {validation}")
        
        if not validation['is_valid']:
            raise ValueError("Invalid mixture ratios! They must sum to 1.0")
        
        # Generate output filenames based on config
        config_name = Path(args.data_config).stem  # e.g., "default"
        train_jsonl = f"dataset/{mixer.config.phase}_{config_name}_train.jsonl"
        val_jsonl = f"dataset/{mixer.config.phase}_{config_name}_val.jsonl"
        
        # Prepare dataset (or skip if already prepared)
        if not args.use_prepared or not os.path.exists(train_jsonl):
            if is_main_process():
                print("Preparing dataset from mixture config...")
                train_jsonl = mixer.prepare_dataset(
                    output_file=train_jsonl,
                    split="train"
                )
                # Also prepare validation if eval is enabled
                if args.eval_interval > 0 and not os.path.exists(val_jsonl):
                    mixer.prepare_dataset(
                        output_file=val_jsonl,
                        split="validation"
                    )
            
            # Wait for main process to finish preparation
            if dist.is_initialized():
                dist.barrier()
        else:
            if is_main_process():
                print(f"Using pre-prepared dataset: {train_jsonl}")
        
        # Load the prepared JSONL files
        train_ds = PretrainDataset(train_jsonl, tokenizer, max_length=args.max_seq_len) #type: ignore
        
        # Load validation dataset if eval is enabled and file exists
        eval_ds = None
        if args.eval_interval > 0 and os.path.exists(val_jsonl):
            eval_ds = PretrainDataset(val_jsonl, tokenizer, max_length=args.max_seq_len) #type: ignore
            if is_main_process():
                print(f"Loaded {len(eval_ds)} validation samples from {val_jsonl}")
        
        if is_main_process():
            print(f"Loaded {len(train_ds)} training samples from {train_jsonl}")
    else:
        # Original mode: direct JSONL file
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len) #type: ignore
        eval_ds = None  # No eval in original mode
        if is_main_process():
            print(f"Loaded {len(train_ds)} training samples from {args.data_path}")
    
    # Set up distributed sampler for multi-GPU training
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Create evaluation loader if eval dataset exists
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Initialize AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore training state from checkpoint if resuming ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap model with DistributedDataParallel ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization across GPUs
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore 
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Calculate warmup steps for mid-training ==========
    total_steps = args.epochs * len(train_ds) // (args.batch_size * args.accumulation_steps)
    if dist.is_initialized():
        total_steps = total_steps // dist.get_world_size()
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    if is_main_process():
        Logger(f"Mid-training with LR re-warmup: {warmup_steps} warmup steps out of {total_steps} total steps")
        Logger(f"LR schedule: {args.learning_rate/10:.2e} → {args.learning_rate:.2e} (warmup) → {args.learning_rate/10:.2e} (decay)")
    
    # ========== 9. Start main training loop ==========
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler to ensure proper data shuffling
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Handle checkpoint resumption for first epoch
        if epoch == start_epoch and start_step > 0:
            # Skip already processed batches
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps, '
                f'starting from step {start_step + 1}'
            )
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb, eval_loader, warmup_steps)
        else:
            # Standard dataloader for normal training
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), 0, wandb, eval_loader, warmup_steps)