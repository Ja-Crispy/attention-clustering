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
import math
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
    analysis = analyze_model(model, val_loader, device, num_topics=config.data.num_topics, num_batches=5)

    print(f"\nRunning linear probes for {config.name}...")
    probes = run_probing_experiment(model, val_loader, device, num_batches=10)

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
            drope_model, val_loader, device, num_topics=config.data.num_topics, num_batches=5
        )
        drope_probes = run_probing_experiment(drope_model, val_loader, device, num_batches=10)

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


def run_sanity_check() -> list[dict]:
    """Run a quick sanity check with easy synthetic data.

    Uses high topic concentration (100x) so topics are trivially distinguishable.
    If methodology is sound, we should see:
    - Val loss well below random (ln(512) = 6.24)
    - Clear probe accuracy differences between topic_ordered and shuffled after DroPE
    - Higher within-cluster attention ratios for topic_ordered

    Config: vocab=512, 3 topics, concentration=100, 256 seq_len
    Model: 4 layers, 256 hidden, 4 heads (~3.8M params)
    Training: 5K steps, batch_size=64
    """
    base_model = ModelConfig(
        hidden_size=256,
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=64,
        intermediate_size=688,
        vocab_size=512,
        max_seq_len=256,
    )
    base_data = DataConfig(
        num_topics=3,
        vocab_size=512,
        seq_len=256,
        num_train_sequences=20_000,
        num_val_sequences=2_000,
        min_segments=2,
        max_segments=4,
        min_segment_len=32,
        topic_concentration=100.0,
    )
    base_train = TrainConfig(
        num_steps=5_000,
        batch_size=64,
        warmup_steps=250,
        log_every=50,
        eval_every=250,
        save_every=1000,
    )

    # Generate shared data
    print("=" * 60)
    print("SANITY CHECK: Easy synthetic data (high topic concentration)")
    print("=" * 60)
    print(f"  Vocab: {base_data.vocab_size}, Topics: {base_data.num_topics}, "
          f"Concentration: {base_data.topic_concentration}")
    print(f"  Model: {base_model.num_layers}L/{base_model.hidden_size}d/"
          f"{base_model.num_q_heads}h (~{base_model.num_params_approx:,} params)")
    print(f"  Training: {base_train.num_steps} steps, batch {base_train.batch_size}")
    print(f"  Random baseline: ln({base_data.vocab_size}) = {math.log(base_data.vocab_size):.2f} nats")
    print()

    train_tokens, train_labels, val_tokens, val_labels, generator = generate_raw_data(base_data)
    print(f"  Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    # MHA only: shuffled vs topic_ordered (2 conditions + DroPE = 4 total)
    conditions = []
    for ordering in ["shuffled", "topic_ordered"]:
        name = f"sanity_mha_{ordering}_rope"
        cfg = ExperimentConfig(
            name=name,
            model=ModelConfig(**{**vars(base_model), "use_rope": True}),
            data=DataConfig(**{**vars(base_data), "ordering": ordering}),
            train=base_train,
            output_dir="results_sanity",
        )
        conditions.append(cfg)

    print(f"\nRunning {len(conditions)} conditions (+ DroPE for each):")
    for c in conditions:
        print(f"  - {c.name}")
    print()

    all_results = []
    t0 = time.time()

    for cfg in conditions:
        loaders = create_dataloaders_from_cache(
            train_tokens, train_labels, val_tokens, val_labels,
            ordering=cfg.data.ordering,
            batch_size=cfg.train.batch_size,
            seed=cfg.data.seed,
        )
        result = run_single_experiment(cfg, dataloaders=loaders)
        all_results.append(result)

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"SANITY CHECK COMPLETE in {total_time:.1f}s")
    print(f"{'=' * 60}")

    # Save combined
    output_path = Path("results_sanity") / "sanity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(all_results)

    # Print pass/fail verdict
    random_loss = math.log(base_data.vocab_size)
    print(f"\n{'=' * 60}")
    print("SANITY CHECK VERDICT")
    print(f"{'=' * 60}")
    for r in all_results:
        gap = random_loss - r["train"]["final_val_loss"]
        pct = gap / random_loss * 100
        status = "PASS" if pct > 20 else "FAIL"
        print(f"  {r['name']}: val_loss={r['train']['final_val_loss']:.3f} "
              f"(gap={gap:.2f} nats, {pct:.0f}% of random) [{status}]")
        if "drope" in r:
            drope_name = r["name"].replace("rope", "drope")
            drope_loss = r["drope"]["train"]["val_loss"]
            drope_gap = random_loss - drope_loss
            drope_pct = drope_gap / random_loss * 100
            drope_status = "PASS" if drope_pct > 10 else "FAIL"
            print(f"  {drope_name}: val_loss={drope_loss:.3f} "
                  f"(gap={drope_gap:.2f} nats, {drope_pct:.0f}% of random) [{drope_status}]")

    # Check probe differential
    if len(all_results) == 2:
        shuffled_drope_probes = all_results[0].get("drope", {}).get("probes", [])
        ordered_drope_probes = all_results[1].get("drope", {}).get("probes", [])
        if shuffled_drope_probes and ordered_drope_probes:
            print(f"\n  DroPE Probe Accuracy (topic_ordered - shuffled):")
            for i, (s, o) in enumerate(zip(shuffled_drope_probes, ordered_drope_probes)):
                diff = o["val_acc"] - s["val_acc"]
                layer_name = s.get("layer", f"layer_{i}")
                marker = " <<<" if abs(diff) > 0.05 else ""
                print(f"    {layer_name}: {diff:+.1%}{marker}")

    return all_results


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


def run_v2(debug: bool = False) -> list[dict]:
    """Run v2 experiment with Wikipedia data (~98M param model).

    Data must be prepared first: python -m data.wikipedia
    Or this will auto-prepare if cache doesn't exist.
    """
    from data.wikipedia import load_wiki_data, WikiPipelineConfig, WikipediaPipeline

    wiki_cfg = WikiPipelineConfig()

    # Load or prepare data
    try:
        train_tokens, train_labels, val_tokens, val_labels = load_wiki_data(wiki_cfg.cache_dir)
        print(f"Loaded wiki data: train={train_tokens.shape}, val={val_tokens.shape}")
    except FileNotFoundError:
        print("Wiki data not found. Running preparation pipeline...")
        pipeline = WikipediaPipeline(wiki_cfg)
        train_tokens, train_labels, val_tokens, val_labels = pipeline.run()

    # Model config: ~98M params
    if debug:
        base_model = ModelConfig(
            hidden_size=256, num_layers=4, num_q_heads=4, num_kv_heads=4,
            head_dim=64, intermediate_size=688,
            vocab_size=wiki_cfg.vocab_size, max_seq_len=wiki_cfg.seq_len,
        )
        base_train = TrainConfig(
            num_steps=200, batch_size=16, warmup_steps=20,
            log_every=20, eval_every=100, save_every=100,
        )
    else:
        base_model = ModelConfig(
            hidden_size=768, num_layers=12, num_q_heads=12, num_kv_heads=12,
            head_dim=64, intermediate_size=2048,
            vocab_size=wiki_cfg.vocab_size, max_seq_len=wiki_cfg.seq_len,
        )
        base_train = TrainConfig(
            num_steps=30_000, batch_size=64, warmup_steps=1000,
            log_every=100, eval_every=1000, save_every=5000,
        )

    base_data = DataConfig(
        num_topics=wiki_cfg.num_topics,
        vocab_size=wiki_cfg.vocab_size,
        seq_len=wiki_cfg.seq_len,
        num_train_sequences=wiki_cfg.num_train_sequences,
        num_val_sequences=wiki_cfg.num_val_sequences,
    )

    random_baseline = math.log(wiki_cfg.vocab_size)

    print(f"\n{'='*60}")
    print("V2 EXPERIMENT: Wikipedia Data")
    print(f"{'='*60}")
    print(f"  Model: {base_model.num_layers}L/{base_model.hidden_size}d/"
          f"{base_model.num_q_heads}h (~{base_model.num_params_approx:,} params)")
    print(f"  Data: {wiki_cfg.num_topics} topics, {wiki_cfg.vocab_size} vocab")
    print(f"  Training: {base_train.num_steps} steps, batch {base_train.batch_size}")
    print(f"  Random baseline: ln({wiki_cfg.vocab_size}) = {random_baseline:.2f} nats")

    # 4 conditions: MHA/GQA x shuffled/topic_ordered (each gets DroPE too)
    conditions = []
    for attn_name, num_kv in [("mha", base_model.num_q_heads), ("gqa", base_model.num_q_heads // 2)]:
        for ordering in ["shuffled", "topic_ordered"]:
            name = f"v2_{attn_name}_{ordering}_rope"
            cfg = ExperimentConfig(
                name=name,
                model=ModelConfig(
                    **{**vars(base_model), "num_kv_heads": num_kv, "use_rope": True}
                ),
                data=DataConfig(**{**vars(base_data), "ordering": ordering}),
                train=base_train,
                output_dir="results_v2",
            )
            conditions.append(cfg)

    print(f"\n  Running {len(conditions)} conditions (+ DroPE for each):")
    for c in conditions:
        print(f"    - {c.name}")
    print()

    all_results = []
    t0 = time.time()

    for cfg in conditions:
        # Resume: skip conditions that already have full results
        result_file = Path(cfg.output_dir) / cfg.name / "experiment_result.json"
        if result_file.exists():
            print(f"\n{'='*60}")
            print(f"SKIPPING {cfg.name} (results already exist)")
            print(f"{'='*60}")
            with open(result_file) as f:
                all_results.append(json.load(f))
            continue

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
    print(f"V2 EXPERIMENTS COMPLETE in {total_time:.1f}s")
    print(f"{'='*60}")

    output_path = Path("results_v2") / "all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(all_results)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention clustering experiments")
    parser.add_argument("--full", action="store_true", help="Run full experimental matrix")
    parser.add_argument("--debug", action="store_true", help="Quick debug run (small models)")
    parser.add_argument("--sanity", action="store_true", help="Sanity check with easy synthetic data")
    parser.add_argument("--v2", action="store_true", help="Run v2 experiment with Wikipedia data")
    parser.add_argument("--name", type=str, help="Run single named condition")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    if args.sanity:
        run_sanity_check()
    elif args.v2:
        run_v2(debug=args.debug)
    elif args.full or args.debug:
        run_full_matrix(debug=args.debug)
    elif args.name:
        cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
        run_single_experiment(cfg)
    else:
        print("Usage: python run_experiment.py --sanity  (sanity check)")
        print("       python run_experiment.py --v2      (v2 Wikipedia experiment)")
        print("       python run_experiment.py --debug   (quick test)")
        print("       python run_experiment.py --full    (full v1 matrix)")
