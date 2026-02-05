"""Linear probing for topic identity at each layer.

Tests whether topic identity is linearly decodable from residual stream activations.
Comparing where this happens across training regimes reveals whether topic-ordered
training builds topic representations at earlier layers.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model.transformer import Transformer
from .attention import extract_hidden_states


def collect_probe_data(
    model: Transformer,
    dataloader,
    device: torch.device,
    num_batches: int = 20,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Collect hidden states and topic labels for probe training.

    Returns:
        hidden_per_layer: list of (N, hidden_size) arrays, one per layer
        all_labels: (N,) array of topic ids
    """
    model.eval()
    hidden_per_layer = None
    all_labels = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"]  # (B, seq_len)

        hidden_states = extract_hidden_states(model, input_ids, device)

        # Flatten batch and sequence dims, subsample to keep memory bounded
        B, T = labels.shape
        step = 8  # take every 8th token
        indices = list(range(0, T, step))

        flat_labels = labels[:, indices].reshape(-1).numpy()
        all_labels.append(flat_labels)

        if hidden_per_layer is None:
            hidden_per_layer = [[] for _ in range(len(hidden_states))]

        for layer_idx, hs in enumerate(hidden_states):
            flat_hidden = hs[:, indices].reshape(-1, hs.shape[-1]).cpu().float().numpy()
            hidden_per_layer[layer_idx].append(flat_hidden)

    all_labels = np.concatenate(all_labels)
    hidden_per_layer = [np.concatenate(h) for h in hidden_per_layer]

    return hidden_per_layer, all_labels


def train_linear_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    max_iter: int = 500,
    seed: int = 42,
) -> dict:
    """Train a logistic regression probe on hidden states to predict topic.

    Args:
        hidden_states: (N, hidden_size)
        labels: (N,) topic ids
        train_fraction: fraction of data for training

    Returns:
        dict with train_acc, val_acc
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    perm = rng.permutation(N)
    split = int(N * train_fraction)

    X_train = hidden_states[perm[:split]]
    y_train = labels[perm[:split]]
    X_val = hidden_states[perm[split:]]
    y_val = labels[perm[split:]]

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))

    return {"train_acc": train_acc, "val_acc": val_acc}


def run_probing_experiment(
    model: Transformer,
    dataloader,
    device: torch.device,
    num_batches: int = 20,
) -> list[dict]:
    """Run linear probing at every layer.

    Returns:
        List of probe results per layer (index 0 = embedding layer).
    """
    hidden_per_layer, labels = collect_probe_data(model, dataloader, device, num_batches)

    results = []
    for layer_idx, hidden in enumerate(hidden_per_layer):
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        probe_result = train_linear_probe(hidden, labels)
        probe_result["layer"] = layer_name
        probe_result["layer_idx"] = layer_idx
        results.append(probe_result)
        print(
            f"  Probe {layer_name}: "
            f"train_acc={probe_result['train_acc']:.3f} "
            f"val_acc={probe_result['val_acc']:.3f}"
        )

    return results
