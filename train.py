"""Training loop for attention clustering experiments.

Uses Muon optimizer for 2D hidden weights, AdamW for embeddings/head/norms.
Supports DroPE (removing RoPE after initial training + recalibration).
"""

import math
import os
import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig, TrainConfig, ExperimentConfig
from model.transformer import Transformer
from data.synthetic import create_dataloaders


def get_lr(step: int, config: TrainConfig, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return base_lr * (step + 1) / config.warmup_steps

    progress = (step - config.warmup_steps) / max(1, config.num_steps - config.warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_optimizer(model: Transformer, config: TrainConfig):
    """Set up Muon + AdamW optimizer.

    Falls back to pure AdamW if muon-optimizer is not installed.
    """
    param_groups = model.get_param_groups()

    try:
        from muon import SingleDeviceMuonWithAuxAdam

        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                dict(
                    params=param_groups["muon_params"],
                    use_muon=True,
                    lr=config.muon_lr,
                    weight_decay=config.weight_decay,
                ),
                dict(
                    params=param_groups["adam_params"],
                    use_muon=False,
                    lr=config.adam_lr,
                    betas=(0.9, 0.95),
                    weight_decay=config.weight_decay,
                ),
            ]
        )
        optimizer_name = "Muon+AdamW"
    except ImportError as e:
        print(f"WARNING: muon import failed ({e}), falling back to AdamW")
        optimizer = torch.optim.AdamW(
            [
                dict(
                    params=param_groups["muon_params"],
                    lr=config.adam_lr,
                    weight_decay=config.weight_decay,
                ),
                dict(
                    params=param_groups["adam_params"],
                    lr=config.adam_lr,
                    betas=(0.9, 0.95),
                    weight_decay=config.weight_decay,
                ),
            ]
        )
        optimizer_name = "AdamW (fallback)"

    return optimizer, optimizer_name


def update_lr(optimizer, step: int, config: TrainConfig):
    """Update learning rates for all parameter groups."""
    for group in optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = get_lr(step, config, base_lr)


def evaluate(model: Transformer, val_loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    """Evaluate model on validation set, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            targets = input_ids[:, 1:].contiguous()
            result = model(input_ids[:, :-1])
            logits = result["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(
    config: ExperimentConfig,
    resume_from: Optional[str] = None,
) -> dict:
    """Main training function.

    Args:
        config: Full experiment configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        dict with training metrics and paths
    """
    # Setup
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.train.dtype, torch.float32)

    # Seed everything
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)

    # Data
    train_loader, val_loader, generator = create_dataloaders(
        config.data, batch_size=config.train.batch_size
    )

    # Model
    model = Transformer(config.model).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"  Architecture: {'GQA' if config.model.is_gqa else 'MHA'}")
    print(f"  QK-Norm: {config.model.qk_norm}")
    print(f"  RoPE: {config.model.use_rope}")

    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile: enabled")

    # Optimizer
    optimizer, optimizer_name = setup_optimizer(model, config.train)
    print(f"  Optimizer: {optimizer_name}")

    # Store initial LRs before any scheduling
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"  Resumed from step {start_step}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "name": config.name,
            "model": vars(config.model),
            "data": vars(config.data),
            "train": vars(config.train),
        }, f, indent=2, default=str)

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    log_history = []
    best_val_loss = float("inf")

    print(f"\nTraining {config.name} for {config.train.num_steps} steps...")
    print("-" * 60)

    t0 = time.time()
    for step in range(start_step, config.train.num_steps):
        # Get batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)

        # Forward: predict next token
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            targets = input_ids[:, 1:].contiguous()
            result = model(input_ids[:, :-1])
            logits = result["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if config.train.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        # LR schedule
        update_lr(optimizer, step, config.train)
        optimizer.step()

        # Logging
        if step % config.train.log_every == 0:
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            entry = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": grad_norm_val,
                "lr": current_lr,
                "elapsed": elapsed,
            }
            log_history.append(entry)
            tokens_per_sec = (step + 1) * config.train.batch_size * (config.data.seq_len - 1) / max(elapsed, 1e-6)
            print(
                f"  step {step:>6d} | loss {loss.item():.4f} | "
                f"grad_norm {grad_norm_val:.3f} | lr {current_lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s"
            )

        # Evaluation
        if step > 0 and step % config.train.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  >>> val_loss {val_loss:.4f}")
            log_history[-1]["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, output_dir / "best.pt")

        # Checkpointing
        if step > 0 and step % config.train.save_every == 0:
            save_checkpoint(model, optimizer, step, output_dir / f"step_{step}.pt")

    # Final evaluation and save
    val_loss = evaluate(model, val_loader, device)
    print(f"\nFinal val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
    save_checkpoint(model, optimizer, config.train.num_steps, output_dir / "final.pt")

    # Save training log
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    total_time = time.time() - t0
    print(f"Training complete in {total_time:.1f}s")

    return {
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
        "output_dir": str(output_dir),
    }


def save_checkpoint(model, optimizer, step, path):
    """Save a training checkpoint."""
    # Unwrap compiled model if necessary
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )


def drope_recalibrate(
    config: ExperimentConfig,
    checkpoint_path: str,
    recal_steps: int = 100,
) -> dict:
    """DroPE recalibration: load a RoPE-trained model, drop RoPE, fine-tune briefly.

    Args:
        config: Experiment config (should have use_rope=True originally)
        checkpoint_path: Path to the RoPE-trained checkpoint
        recal_steps: Number of recalibration steps (~1% of training)

    Returns:
        dict with recalibration metrics
    """
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

    # Load original model with RoPE
    model = Transformer(config.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # Drop RoPE
    model.disable_rope()
    print(f"DroPE: disabled RoPE, recalibrating for {recal_steps} steps...")

    # Create recalibration config
    drope_name = config.name.replace("rope", "drope")
    recal_config = config.variant(
        name=drope_name,
        **{"train.num_steps": recal_steps, "train.warmup_steps": 10, "model.use_rope": False},
    )

    # Small learning rate for recalibration
    recal_train = recal_config.train
    recal_train.muon_lr = config.train.muon_lr * 0.1
    recal_train.adam_lr = config.train.adam_lr * 0.1

    # Data (same as original, to avoid distribution shift during recal)
    train_loader, val_loader, _ = create_dataloaders(
        config.data, batch_size=config.train.batch_size
    )

    optimizer, _ = setup_optimizer(model, recal_train)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Recalibration loop
    model.train()
    train_iter = iter(train_loader)

    for step in range(recal_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets = input_ids[:, 1:].contiguous()
        result = model(input_ids[:, :-1])
        logits = result["logits"]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        update_lr(optimizer, step, recal_train)
        optimizer.step()

        if step % 10 == 0:
            print(f"  recal step {step:>4d} | loss {loss.item():.4f}")

    # Final eval
    val_loss = evaluate(model, val_loader, device)
    print(f"  DroPE recal done | val_loss {val_loss:.4f}")

    # Save
    output_dir = Path(config.output_dir) / drope_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, recal_steps, output_dir / "drope_final.pt")

    return {"val_loss": val_loss, "output_dir": str(output_dir)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train attention clustering experiment")
    parser.add_argument("--name", default="debug", help="Experiment name")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gqa", action="store_true", help="Use GQA instead of MHA")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE (DroPE)")
    parser.add_argument("--topic-ordered", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-train-sequences", type=int, default=50000)
    args = parser.parse_args()

    ordering = "topic_ordered" if args.topic_ordered else "shuffled"
    cfg = ExperimentConfig(
        name=args.name,
        model=ModelConfig(
            num_kv_heads=4 if args.gqa else 8,
            use_rope=not args.no_rope,
        ),
        data=DataConfig(
            ordering=ordering,
            num_train_sequences=args.num_train_sequences,
        ),
        train=TrainConfig(
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )

    result = train(cfg)
    print(f"\nResult: {json.dumps(result, indent=2)}")
