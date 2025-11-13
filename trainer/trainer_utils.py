"""
Training utility functions collection
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model.model_minimind import MiniMindForCausalLM


def is_main_process():
    """
    Check if the current process is the main process in distributed training.
    
    Returns:
        bool: True if this is the main process (rank 0) or not in distributed mode,
              False otherwise.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    Print content only from the main process.
    
    Args:
        content: The content to be logged (will be converted to string).
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    Calculate learning rate using cosine annealing schedule.
    
    Implements a cosine decay from initial lr to lr/10 over total_steps.
    Formula: lr/10 + 0.5 * lr * (1 + cos(pi * current_step / total_steps))
    
    Args:
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        lr (float): Initial learning rate.
    
    Returns:
        float: The calculated learning rate for the current step.
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """
    Initialize distributed training mode.
    
    Checks if RANK environment variable is set to determine if running in
    Distributed Data Parallel (DDP) mode. If not set, returns 0 for single GPU training.
    
    Returns:
        int: Local rank of the process (0 for main process or non-DDP mode).
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # Non-DDP mode

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    Set random seeds for reproducibility across all used libraries.
    
    Sets seeds for random, numpy, torch (CPU and all GPUs), and configures
    cudnn to ensure deterministic behavior.
    
    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    Save or load model checkpoints for training resumption or inference.
    
    This function operates in two modes:
    1. Save mode (when model is provided): Saves model weights and training state
    2. Load mode (when model is None): Loads checkpoint data if available
    
    Args:
        lm_config: Language model configuration object.
        weight (str): Type of weight to save/load (e.g., 'full_sft', 'pretrain').
        model (torch.nn.Module, optional): Model to save. If None, loads checkpoint.
        optimizer (torch.optim.Optimizer, optional): Optimizer whose state to save.
        epoch (int): Current training epoch.
        step (int): Current training step.
        wandb: Weights & Biases object for experiment tracking.
        save_dir (str): Directory to save/load checkpoints.
        **kwargs: Additional items to save (e.g., schedulers, scalers).
                  Objects with state_dict() method are automatically handled.
    
    Returns:
        dict or None: In load mode, returns checkpoint dictionary if found, else None.
                      In save mode, returns None.
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # Save mode: Save model and training state
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # Extract wandb run ID if available
        wandb_id = None
        if wandb:
            try:
                # wandb.run is the active run object
                if hasattr(wandb, 'run') and wandb.run is not None:
                    wandb_id = wandb.run.id
            except:
                pass

        # Prepare full resume checkpoint with model, optimizer, and metadata
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(), #type: ignore 
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # Handle additional kwargs (e.g., scheduler, grad_scaler)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        # Atomic save to prevent corruption
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:
        # Load mode: Load checkpoint if exists
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            
            # Adjust step count if world size changed (e.g., different number of GPUs)
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU count changed ({saved_ws}→{current_ws}), step automatically adjusted to {ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='model', save_dir='out', device='cuda'):
    """
    Initialize model and tokenizer from pretrained weights or scratch.
    
    Args:
        lm_config: Language model configuration object.
        from_weight (str): Weight type to load ('pretrain', 'full_sft', or 'none').
        tokenizer_path (str): Path to tokenizer directory.
        save_dir (str): Directory containing model weights.
        device (str): Device to load model on ('cuda', 'cpu', etc.).
    
    Returns:
        tuple: (model, tokenizer) where model is on the specified device.
    """
    # Load tokenizer directly from local files
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, 'tokenizer.json'))
    
    # Set special tokens if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '<|endoftext|>'
    if tokenizer.eos_token is None:
        tokenizer.eos_token = '<|endoftext|>'
    if tokenizer.bos_token is None:
        tokenizer.bos_token = '<|endoftext|>'
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    Logger(f'Trainable parameters of loaded Model: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} million')
    return model.to(device), tokenizer #type: ignore 


class SkipBatchSampler(Sampler):
    """
    A custom sampler that skips the first N batches for resuming training mid-epoch.
    
    This sampler wraps a base sampler and yields batches of indices, skipping the
    initial batches specified by skip_batches. Useful for resuming training from
    a specific batch within an epoch.
    
    Args:
        sampler (torch.utils.data.Sampler): Base sampler providing sample indices.
        batch_size (int): Number of samples per batch.
        skip_batches (int): Number of batches to skip at the beginning.
    """
    
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        Iterate over the sampler, yielding batches of indices while skipping initial batches.
        
        Yields:
            list: Batch of indices (list of integers).
        """
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # Yield remaining samples if any, after skipping is complete
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        Calculate the effective number of batches after skipping.
        
        Returns:
            int: Number of batches remaining after skipping.
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)