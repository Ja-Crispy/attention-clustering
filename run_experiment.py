"""Run the full attention clustering experiment matrix.

Trains models across all conditions (MHA/GQA x shuffled/topic_ordered x RoPE/DroPE)
and runs mech interp analysis on each.

Usage:
    # Full matrix (8 conditions)
    python run_experiment.py --full

    # Quick debug run
    python run_experiment.py --debug

    # Single condition
    python run_experiment.py --name mha_shuffled_rope

    # Analysis only (skip training)
    python run_experiment.py --analyze-only --checkpoint results/mha_shuffled_rope/final.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch

from config import (
    ModelConfig,
    DataConfig,
    TrainConfig,
    ExperimentConfig,
    experiment_matrix,
)
from train import train, drope_recalibrate
from model.transformer import Transformer
from data.synthetic import (
    create_dataloaders,
    generate_raw_data,
    create_dataloaders_from_cache,
    MicroLanguageGenerator,
)
from analysis.attention import analyze_model
from analysis.probes import run_probing_experiment


def run_single_experiment(
    config: ExperimentConfig,
    skip_drope: bool = False,
    dataloaders: tuple | None = None,
) -> dict:
    """Run a single experiment: train + analyze.

    For RoPE conditions, also runs DroPE recalibration and analysis.

    Args:
        config: Experiment configuration
        skip_drope: Skip DroPE recalibration
        dataloaders: Optional (train_loader, val_loader) to skip data generation
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")

    # Phase 1: Train
    train_result = train(config, dataloaders=dataloaders)

    # Phase 2: Analyze
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = Transformer(config.model).to(device)
    ckpt_path = Path(train_result["output_dir"]) / "final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # Reuse val_loader from dataloaders if available
    if dataloaders is not None:
        _, val_loader = dataloaders
    else:
        _, val_loader, _ = create_dataloaders(config.data, batch_size=config.train.batch_size)

    print(f"\nRunning mech interp analysis for {config.name}...")
    analysis = analyze_model(model, val_loader, device, num_topics=config.data.num_topics)

    print(f"\nRunning linear probes for {config.name}...")
    probes = run_probing_experiment(model, val_loader, device)

    result = {
        "name": config.name,
        "train": train_result,
        "analysis": {
            "entropy": [e.tolist() for e in analysis["entropy"]],
            "within_ratio": [w.tolist() for w in analysis["within_ratio"]],
            "cross_ratio": [c.tolist() for c in analysis["cross_ratio"]],
            "specialization": [s.tolist() for s in analysis["specialization"]],
        },
        "probes": probes,
    }

    # Phase 3: DroPE recalibration (only for RoPE-trained models)
    if config.model.use_rope and not skip_drope:
        print(f"\nRunning DroPE recalibration for {config.name}...")
        drope_result = drope_recalibrate(config, str(ckpt_path), dataloaders=dataloaders)

        # Analyze DroPE'd model
        drope_name = config.name.replace("rope", "drope")
        drope_ckpt_path = Path(drope_result["output_dir"]) / "drope_final.pt"
        drope_model = Transformer(config.model).to(device)
        drope_ckpt = torch.load(drope_ckpt_path, map_location=device, weights_only=False)
        drope_model.load_state_dict(drope_ckpt["model"])
        drope_model.disable_rope()

        print(f"\nRunning mech interp analysis for {drope_name}...")
        drope_analysis = analyze_model(
            drope_model, val_loader, device, num_topics=config.data.num_topics
        )
        drope_probes = run_probing_experiment(drope_model, val_loader, device)

        result["drope"] = {
            "train": drope_result,
            "analysis": {
                "entropy": [e.tolist() for e in drope_analysis["entropy"]],
                "within_ratio": [w.tolist() for w in drope_analysis["within_ratio"]],
                "cross_ratio": [c.tolist() for c in drope_analysis["cross_ratio"]],
                "specialization": [s.tolist() for s in drope_analysis["specialization"]],
            },
            "probes": drope_probes,
        }

    # Save result
    output_path = Path(config.output_dir) / config.name / "experiment_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_full_matrix(debug: bool = False) -> list[dict]:
    """Run the complete experimental matrix."""
    if debug:
        # Small debug configs
        base_model = ModelConfig(num_layers=2, hidden_size=128, head_dim=32, num_q_heads=4)
        base_data = DataConfig(
            num_train_sequences=1000, num_val_sequences=200, seq_len=128
        )
        base_train = TrainConfig(num_steps=100, batch_size=16, log_every=10, save_every=50)
    else:
        base_model = ModelConfig()
        base_data = DataConfig()
        base_train = TrainConfig()

    # Generate raw data ONCE — shared across all conditions
    print("Generating shared dataset (one-time)...")
    train_tokens, train_labels, val_tokens, val_labels, generator = generate_raw_data(base_data)
    print(f"  Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    # Only run RoPE conditions (DroPE is derived from them)
    conditions = []
    for attn_name, num_kv in [("mha", base_model.num_q_heads), ("gqa", base_model.num_q_heads // 2)]:
        for ordering in ["shuffled", "topic_ordered"]:
            name = f"{attn_name}_{ordering}_rope"
            cfg = ExperimentConfig(
                name=name,
                model=ModelConfig(
                    **{**vars(base_model), "num_kv_heads": num_kv, "use_rope": True}
                ),
                data=DataConfig(**{**vars(base_data), "ordering": ordering}),
                train=base_train,
            )
            conditions.append(cfg)

    print(f"\nRunning {len(conditions)} conditions (DroPE derived from RoPE models):")
    for c in conditions:
        print(f"  - {c.name}")
    print()

    all_results = []
    t0 = time.time()

    for cfg in conditions:
        # Build dataloaders from cached data with per-condition ordering
        loaders = create_dataloaders_from_cache(
            train_tokens, train_labels, val_tokens, val_labels,
            ordering=cfg.data.ordering,
            batch_size=cfg.train.batch_size,
            seed=cfg.data.seed,
        )
        result = run_single_experiment(cfg, dataloaders=loaders)
        all_results.append(result)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.1f}s")
    print(f"{'='*60}")

    # Save combined results
    output_path = Path(conditions[0].output_dir) / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: list[dict]):
    """Print a summary comparison table of all experiments."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    header = f"{'Condition':<35} {'Val Loss':>10} {'Within%':>10} {'Entropy':>10} {'Probe@L4':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        name = r["name"]
        val_loss = r["train"]["final_val_loss"]

        # Average within-ratio across all layers and heads
        within_ratios = r["analysis"]["within_ratio"]
        avg_within = sum(sum(layer) / len(layer) for layer in within_ratios) / len(within_ratios)

        # Average entropy
        entropies = r["analysis"]["entropy"]
        avg_entropy = sum(sum(layer) / len(layer) for layer in entropies) / len(entropies)

        # Probe accuracy at layer 4 (or last available)
        probe_idx = min(4, len(r["probes"]) - 1)
        probe_acc = r["probes"][probe_idx]["val_acc"]

        print(f"{name:<35} {val_loss:>10.4f} {avg_within:>9.1%} {avg_entropy:>10.2f} {probe_acc:>9.1%}")

        # DroPE variant
        if "drope" in r:
            drope_name = name.replace("rope", "drope")
            drope_val = r["drope"]["train"]["val_loss"]
            drope_within = r["drope"]["analysis"]["within_ratio"]
            avg_drope_within = sum(sum(l) / len(l) for l in drope_within) / len(drope_within)
            drope_entropy = r["drope"]["analysis"]["entropy"]
            avg_drope_entropy = sum(sum(l) / len(l) for l in drope_entropy) / len(drope_entropy)
            drope_probe = r["drope"]["probes"][min(4, len(r["drope"]["probes"]) - 1)]["val_acc"]
            print(
                f"  {drope_name:<33} {drope_val:>10.4f} {avg_drope_within:>9.1%} "
                f"{avg_drope_entropy:>10.2f} {drope_probe:>9.1%}"
            )

    print(f"\nKey: Within% = within-cluster attention ratio (higher = more clustering)")
    print(f"     Probe@L4 = linear probe accuracy for topic at layer 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention clustering experiments")
    parser.add_argument("--full", action="store_true", help="Run full experimental matrix")
    parser.add_argument("--debug", action="store_true", help="Quick debug run (small models)")
    parser.add_argument("--name", type=str, help="Run single named condition")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    if args.full or args.debug:
        run_full_matrix(debug=args.debug)
    elif args.name:
        cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
        run_single_experiment(cfg)
    else:
        print("Usage: python run_experiment.py --debug (for quick test)")
        print("       python run_experiment.py --full (for full matrix)")
