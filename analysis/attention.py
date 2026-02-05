"""Attention pattern analysis for mechanistic interpretability.

Tools for extracting and analyzing attention patterns from trained models,
with focus on semantic clustering metrics.
"""

import torch
import numpy as np
from typing import Optional

from config import ModelConfig
from model.transformer import Transformer


@torch.no_grad()
def extract_attention_patterns(
    model: Transformer,
    input_ids: torch.Tensor,
    device: Optional[torch.device] = None,
) -> list[torch.Tensor]:
    """Extract attention weight matrices from all layers.

    Args:
        model: Trained transformer model
        input_ids: (batch, seq_len) token ids
        device: Device to run on

    Returns:
        List of (batch, num_heads, seq_len, seq_len) tensors, one per layer
    """
    if device:
        input_ids = input_ids.to(device)
    model.eval()
    result = model(input_ids, return_attention=True)
    return result["attention_weights"]


@torch.no_grad()
def extract_hidden_states(
    model: Transformer,
    input_ids: torch.Tensor,
    device: Optional[torch.device] = None,
) -> list[torch.Tensor]:
    """Extract hidden states from all layers (for linear probing).

    Returns:
        List of (batch, seq_len, hidden_size) tensors.
        Index 0 = after embedding, index i = after layer i.
    """
    if device:
        input_ids = input_ids.to(device)
    model.eval()
    result = model(input_ids, return_hidden_states=True)
    return result["hidden_states"]


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention distributions per head.

    Higher entropy = more diffuse attention.
    Lower entropy = more concentrated/peaked attention.

    Args:
        attn_weights: (batch, num_heads, seq_len, seq_len)

    Returns:
        (batch, num_heads) tensor of entropy values
    """
    # Clamp for numerical stability
    attn = attn_weights.clamp(min=1e-10)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (batch, heads, seq_len)
    return entropy.mean(dim=-1)  # average over query positions


def compute_cluster_attention_ratio(
    attn_weights: torch.Tensor,
    topic_labels: torch.Tensor,
    num_topics: int = 5,
) -> dict[str, torch.Tensor]:
    """Compute within-cluster vs cross-cluster attention mass.

    This is the core metric: does attention preferentially flow between
    tokens of the same topic?

    Args:
        attn_weights: (batch, num_heads, seq_len, seq_len) - NOTE: these are
            for the input sequence (seq_len-1 if using next-token prediction).
        topic_labels: (batch, seq_len) topic id per token
        num_topics: number of distinct topics

    Returns:
        dict with:
            - within_ratio: (batch, num_heads) fraction of attention within same topic
            - cross_ratio: (batch, num_heads) fraction of attention across topics
            - per_topic_within: (batch, num_heads, num_topics) per-topic breakdown
    """
    B, H, T_q, T_k = attn_weights.shape

    # Align labels with attention dimensions (trim if needed)
    labels_q = topic_labels[:, :T_q]  # (B, T_q)
    labels_k = topic_labels[:, :T_k]  # (B, T_k)

    # Create same-topic mask: (B, T_q, T_k)
    same_topic = (labels_q.unsqueeze(-1) == labels_k.unsqueeze(-2)).float()
    same_topic = same_topic.unsqueeze(1)  # (B, 1, T_q, T_k)

    # Mask attention by causal mask (lower triangular) to only count valid positions
    causal = torch.tril(torch.ones(T_q, T_k, device=attn_weights.device))
    causal = causal.unsqueeze(0).unsqueeze(0)

    # Within-topic attention mass
    within_mass = (attn_weights * same_topic * causal).sum(dim=-1).mean(dim=-1)  # (B, H)
    total_mass = (attn_weights * causal).sum(dim=-1).mean(dim=-1)  # (B, H)

    within_ratio = within_mass / total_mass.clamp(min=1e-10)
    cross_ratio = 1.0 - within_ratio

    # Per-topic breakdown
    per_topic_within = torch.zeros(B, H, num_topics, device=attn_weights.device)
    for t in range(num_topics):
        topic_mask_q = (labels_q == t).float().unsqueeze(1).unsqueeze(-1)  # (B,1,T_q,1)
        topic_mask_k = (labels_k == t).float().unsqueeze(1).unsqueeze(-2)  # (B,1,1,T_k)
        topic_same = topic_mask_q * topic_mask_k  # within this topic

        topic_within = (attn_weights * topic_same * causal).sum(dim=(-1, -2))
        topic_total = (attn_weights * topic_mask_q.expand_as(attn_weights) * causal).sum(dim=(-1, -2))
        per_topic_within[:, :, t] = topic_within / topic_total.clamp(min=1e-10)

    return {
        "within_ratio": within_ratio,
        "cross_ratio": cross_ratio,
        "per_topic_within": per_topic_within,
    }


def compute_head_specialization(
    attn_weights: torch.Tensor,
    topic_labels: torch.Tensor,
    num_topics: int = 5,
) -> torch.Tensor:
    """Measure how much each head specializes for specific topics.

    Uses the variance of per-topic within-cluster ratios across topics.
    High variance = head treats topics differently (specialization).
    Low variance = head treats all topics similarly (general).

    Args:
        attn_weights: (batch, num_heads, seq_len, seq_len)
        topic_labels: (batch, seq_len)

    Returns:
        (num_heads,) specialization scores (higher = more specialized)
    """
    ratios = compute_cluster_attention_ratio(attn_weights, topic_labels, num_topics)
    per_topic = ratios["per_topic_within"]  # (B, H, num_topics)
    # Variance across topics, averaged across batch
    return per_topic.var(dim=-1).mean(dim=0)  # (H,)


def extract_qk_ov_weights(model: Transformer) -> dict[str, list[torch.Tensor]]:
    """Extract QK and OV circuit weight matrices for SVD analysis.

    For each layer, computes:
        W_QK = W_Q^T @ W_K  (what tokens attend to)
        W_OV = W_V @ W_O    (what information is moved)

    Returns:
        dict with 'qk' and 'ov' keys, each a list of tensors per layer.
    """
    qk_matrices = []
    ov_matrices = []

    for layer in model.layers:
        attn = layer.attn

        W_q = attn.q_proj.weight.data  # (num_q_heads * head_dim, hidden)
        W_k = attn.k_proj.weight.data  # (num_kv_heads * head_dim, hidden)
        W_v = attn.v_proj.weight.data  # (num_kv_heads * head_dim, hidden)
        W_o = attn.o_proj.weight.data  # (hidden, num_q_heads * head_dim)

        hd = attn.head_dim

        # W_QK per KV head
        W_q_heads = W_q.view(attn.num_q_heads, hd, -1)  # (n_q, hd, hidden)
        W_k_heads = W_k.view(attn.num_kv_heads, hd, -1)  # (n_kv, hd, hidden)

        qk_per_kv = []
        for kv_idx in range(attn.num_kv_heads):
            q_idx = kv_idx * attn.num_kv_groups
            wqk = W_q_heads[q_idx] @ W_k_heads[kv_idx].T  # (hd, hd)
            qk_per_kv.append(wqk)
        qk_matrices.append(torch.stack(qk_per_kv))

        # W_OV per Q head
        W_v_expanded = W_v.view(attn.num_kv_heads, hd, -1)
        W_o_heads = W_o.view(-1, attn.num_q_heads, hd).permute(1, 2, 0)  # (n_q, hd, hidden)

        ov_per_head = []
        for q_idx in range(attn.num_q_heads):
            kv_idx = q_idx // attn.num_kv_groups
            wov = W_v_expanded[kv_idx].T @ W_o_heads[q_idx]  # (hidden, hidden)
            ov_per_head.append(wov)
        ov_matrices.append(torch.stack(ov_per_head))

    return {"qk": qk_matrices, "ov": ov_matrices}


def analyze_model(
    model: Transformer,
    dataloader,
    device: torch.device,
    num_batches: int = 10,
    num_topics: int = 5,
) -> dict:
    """Run full mech interp analysis on a trained model.

    Returns a dict with all analysis results.
    """
    model.eval()
    all_entropy = []
    all_within_ratio = []
    all_cross_ratio = []
    all_specialization = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attn_weights = extract_attention_patterns(model, input_ids, device)

        for layer_idx, aw in enumerate(attn_weights):
            entropy = compute_attention_entropy(aw)
            cluster = compute_cluster_attention_ratio(aw, labels, num_topics)
            spec = compute_head_specialization(aw, labels, num_topics)

            if i == 0:
                all_entropy.append([])
                all_within_ratio.append([])
                all_cross_ratio.append([])
                all_specialization.append([])

            all_entropy[layer_idx].append(entropy.cpu())
            all_within_ratio[layer_idx].append(cluster["within_ratio"].cpu())
            all_cross_ratio[layer_idx].append(cluster["cross_ratio"].cpu())
            all_specialization[layer_idx].append(spec.cpu())

    # Aggregate
    results = {
        "entropy": [torch.cat(e).mean(0).numpy() for e in all_entropy],
        "within_ratio": [torch.cat(w).mean(0).numpy() for w in all_within_ratio],
        "cross_ratio": [torch.cat(c).mean(0).numpy() for c in all_cross_ratio],
        "specialization": [torch.stack(s).mean(0).numpy() for s in all_specialization],
        "qk_ov": extract_qk_ov_weights(model),
    }

    return results
