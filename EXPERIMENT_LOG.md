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

## Run 1: Original Synthetic Data (2026-02-06)

### Config
- Data: 5 topics, 4096 vocab, concentration=5.0, 100K train / 10K val sequences
- Model: 8 layers, 512 hidden, 8 heads, ~27M params (MHA) / ~24M params (GQA)
- Training: 10K steps, Muon (0.02) + AdamW (3e-4), batch 64, seq_len 512
- 4 conditions (MHA/GQA x shuffled/topic_ordered), each with DroPE recalibration
- Total runtime: ~15 hours on L4 GPU

### Results Summary
| Condition | Val Loss | Avg Within% | Avg Entropy | Probe@L4 |
|-----------|----------|-------------|-------------|----------|
| mha_shuffled_rope | 7.9702 | 59.3% | 3.66 | 93.0% |
| mha_shuffled_drope | 8.0973 | 46.4% | 4.61 | 73.8% |
| mha_topic_ordered_rope | 7.9545 | 59.5% | 3.67 | 93.6% |
| mha_topic_ordered_drope | 8.0629 | 47.3% | 4.56 | 77.6% |
| gqa_shuffled_rope | 7.9965 | 59.4% | 3.53 | 93.0% |
| gqa_shuffled_drope | 8.1075 | 47.1% | 4.50 | 73.2% |
| gqa_topic_ordered_rope | 7.9776 | 59.2% | 3.52 | 93.3% |
| gqa_topic_ordered_drope | 8.0843 | 47.2% | 4.47 | 76.5% |

### Key Findings
1. **Models barely learned**: Val loss ~8.0 vs ln(4096)=8.32 random baseline.
   Only covered ~30% of learnable gap. Loss still decreasing at step 10K.
2. **Data was the bottleneck**: Markov chain transition entropy = 7.28 nats.
   Learnable gap only 1.04 nats. Concentration=5.0 too weak.
3. **Head specialization is real but positional**: Layer 1 shows bimodal
   within-cluster ratios (0.92 vs 0.30) with RoPE. All collapse to ~0.47 after DroPE.
4. **Probe signal survives DroPE (positive finding!)**: Topic-ordered DroPE retains
   +3-4% better probe accuracy at ALL layers vs shuffled DroPE.
   Consistent across MHA and GQA. This supports the hypothesis.
5. **Within-cluster ratio**: No meaningful difference after DroPE (~47% for both).
6. **Muon was active**: LR logs confirm peak at 0.02 (Muon), not 0.0003 (AdamW fallback).

### Diagnosis
Root cause: synthetic data too noisy. With concentration=5.0, topics are barely
distinguishable. The theoretical best loss (7.28 nats, perplexity 1454) means the model
can only learn very subtle statistical biases. Despite this, the probe signal after DroPE
is encouraging — topic-ordered training does install *something* durable.

### Bugs Fixed During Run 1
- sklearn `lbfgs` convergence: bumped `max_iter` to 2000, added `StandardScaler`
- Data regeneration: was generating 100K sequences 12x (once per condition). Fixed to
  generate once and share via `create_dataloaders_from_cache()`
- tqdm visibility over SSH: changed `leave=False` to `leave=True`

---

## Sanity Check (pre-Run 2)

### Purpose
Validate methodology detects ordering effects when signal is strong (easy data).
If this fails, the pipeline has bugs. If it passes, proceed to real data.

### Config
- Data: 3 topics, 512 vocab, concentration=100.0, 20K train / 2K val, seq_len=256
- Model: 4 layers, 256 hidden, 4 heads, ~3.8M params
- Training: 5K steps, MHA only (2 conditions + DroPE)
- Random baseline: ln(512) = 6.24 nats
- Expected learnable gap: ~1.7 nats (much better than Run 1's 1.04)
- Command: `python run_experiment.py --sanity`

---

## v2 Plan: Real Data Experiment

### Changes from Run 1
- **Data**: Wikipedia articles + sentence-transformer embedding clustering (8 topics)
  - Download ~500K articles, embed with `all-MiniLM-L6-v2`, K-means into 8 clusters
  - Train BPE tokenizer (16K vocab) on the corpus
  - Rich semantic structure, real language, learnable gap >>2 nats
- **Model**: Mini-Qwen3 v2 (~98-100M params)
  - 768 hidden, 12 layers, 12 heads, head_dim=64, SwiGLU (intermediate ~2048)
  - Same QK-Norm architecture, just scaled up
- **Training**: 30K steps on A100 80GB (Prime Intellect, $0.51/hr spot)
- **Budget**: ~$3-5 total
- **Why not SYNTH by Pleias**: Q&A exercise format, not plain text. Not ideal for
  causal LM pretraining comparison.

---

## Decisions & Notes
- Using pure PyTorch (not HF/nanoGPT) for maximum control
- Mini-Qwen3 over mini-Llama because QK-Norm makes attention patterns
  purely directional (cosine similarity), cleaner for mech interp
- GQA + MHA variants for comparison — GQA forces head pairs to share
  KV representations, may affect clustering dynamics
- Vocab 4096 is small (synthetic) — tied embeddings are important here
- DroPE is NOT a separate training run — it's a post-hoc modification
  of RoPE-trained models (drop RoPE + short recalibration)
