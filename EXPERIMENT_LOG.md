# Attention Clustering Experiment Log

## Hypothesis
Training data ordering (topic-clustered vs shuffled) installs durable computational
structure in attention heads. Specifically, topic-ordered training biases attention
toward semantic clustering that persists even after positional encoding removal (DroPE).

## Architecture: Mini-Qwen3
- **Key feature**: QK-Norm (RMSNorm on Q,K before RoPE) — forces cosine similarity
- Hidden: 512, Layers: 8, Heads: 8Q, Head dim: 64
- SwiGLU MLP (intermediate 1376), RMSNorm, RoPE (theta=1M)
- Tied embeddings, ~27M params (MHA) / ~24M params (GQA)
- GQA variant: 8Q/4KV (2:1 ratio, Qwen3-style)

## Experimental Matrix (2x2x2)
| Attention | Data Ordering | Position Enc | Name |
|-----------|---------------|--------------|------|
| MHA | shuffled | RoPE | mha_shuffled_rope |
| MHA | topic_ordered | RoPE | mha_topic_ordered_rope |
| MHA | shuffled | DroPE | mha_shuffled_drope |
| MHA | topic_ordered | DroPE | mha_topic_ordered_drope |
| GQA | shuffled | RoPE | gqa_shuffled_rope |
| GQA | topic_ordered | RoPE | gqa_topic_ordered_rope |
| GQA | shuffled | DroPE | gqa_shuffled_drope |
| GQA | topic_ordered | DroPE | gqa_topic_ordered_drope |

## DroPE Protocol
- Phase 1: Train with RoPE (full training budget)
- Phase 2: Remove RoPE, recalibrate with ~1% of training steps on same-length data
- Evaluate attention patterns after both phases

## Synthetic Data: Micro-Languages
- 5 topics, each a Markov chain over shared vocab (4096 tokens)
- Topics distinguished by bigram transitions, NOT individual tokens
- Sequences: 512 tokens, 2-5 topic segments each
- Ground-truth topic labels per token for evaluation

## Mech Interp Toolkit
- Attention entropy per head (within-topic vs cross-topic)
- Linear probes for topic identity at each layer
- SVD on QK/OV circuits
- TensorLens for holistic model analysis
- Cluster attention ratio (within-topic / cross-topic attention mass)

## Optimizer Stack
- Muon: 2D hidden weights (built-in muP scaling)
- AdamW: embeddings, head, norms, biases
- Cosine LR with warmup

## Build Progress
- [x] Config dataclasses (config.py)
- [x] Project structure
- [x] Model components — RoPE, RMSNorm, SwiGLU (model/components.py)
- [x] Transformer — Attention w/ QK-Norm + GQA, Block, full model (model/transformer.py)
- [x] Synthetic data generator — Markov chains, topic ordering (data/synthetic.py)
- [x] Training loop — Muon+AdamW, cosine LR, DroPE recalibration (train.py)
- [x] Analysis tools — attention entropy, cluster ratio, probes, SVD (analysis/)
- [x] Experiment runner — full 2x2x2 matrix with summary (run_experiment.py)
- [x] Smoke test — all components verified end-to-end
- [x] Muon optimizer installed and integrated
- [ ] Visualization scripts (attention heatmaps, probe curves, comparison plots)
- [ ] First real training run (debug mode)

## Smoke Test Results (2026-02-06)
- Model forward pass: MHA and GQA both work, DroPE toggle works
- Data generator: 5 topics, Markov chains, topic ordering verified
- Training loop: loss decreasing, Muon+AdamW optimizer working
- Analysis: attention extraction, entropy, cluster ratios, QK/OV circuits, linear probes — all functional
- Muon param split: ~97% params via Muon (hidden 2D), ~3% via AdamW (embed/norms)

## Decisions & Notes
- Using pure PyTorch (not HF/nanoGPT) for maximum control
- Mini-Qwen3 over mini-Llama because QK-Norm makes attention patterns
  purely directional (cosine similarity), cleaner for mech interp
- GQA + MHA variants for comparison — GQA forces head pairs to share
  KV representations, may affect clustering dynamics
- Vocab 4096 is small (synthetic) — tied embeddings are important here
- DroPE is NOT a separate training run — it's a post-hoc modification
  of RoPE-trained models (drop RoPE + short recalibration)
