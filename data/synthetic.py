"""Synthetic micro-language data generator.

Generates sequences from N topic-specific Markov chains over a shared vocabulary.
Topics are distinguishable from bigram transition patterns, NOT individual tokens
(token unigram distributions overlap substantially between topics).

Each training sequence contains multiple topic segments with ground-truth labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path

from config import DataConfig


class MicroLanguageGenerator:
    """Generates synthetic multi-topic sequences for attention clustering experiments.

    Each topic is a first-order Markov chain over a shared vocabulary. Topics differ
    in their transition matrices — individual tokens are ambiguous, but short sequences
    (3-5 tokens) are diagnostic of the generating topic.

    Vocab layout:
        0 = PAD
        1 = BOS
        2 = EOS
        3..vocab_size-1 = regular tokens
    """

    SPECIAL_TOKENS = 3  # PAD=0, BOS=1, EOS=2

    def __init__(self, config: DataConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.num_topics = config.num_topics
        self.vocab_size = config.vocab_size
        self.regular_vocab = config.vocab_size - self.SPECIAL_TOKENS
        self.tokens_per_topic = self.regular_vocab // config.num_topics

        self.transitions = self._build_transition_matrices()
        self.unigrams = self._build_unigram_distributions()
        # Precompute CDFs for fast sampling (avoids repeated rng.choice overhead)
        self.transition_cdfs = np.cumsum(self.transitions, axis=-1)
        self.unigram_cdfs = np.cumsum(self.unigrams, axis=-1)

    def _build_transition_matrices(self) -> np.ndarray:
        """Build per-topic bigram transition matrices.

        Each topic has:
        - A base uniform-ish distribution over all regular tokens
        - Boosted transitions within its "preferred" token range
        - Unique "signature" bigram patterns

        Returns: (num_topics, vocab_size, vocab_size) array
        """
        V = self.vocab_size
        S = self.SPECIAL_TOKENS
        n_topics = self.num_topics
        concentration = self.config.topic_concentration

        transitions = np.zeros((n_topics, V, V))

        for t in range(n_topics):
            # Base: Dirichlet-drawn rows over regular tokens only
            base = self.rng.dirichlet(np.ones(self.regular_vocab) * 0.5, size=V)

            # Boost intra-topic transitions
            topic_start = S + t * self.tokens_per_topic
            topic_end = topic_start + self.tokens_per_topic
            base[:, topic_start - S : topic_end - S] *= concentration

            # Re-normalize each row
            row_sums = base.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            base /= row_sums

            # Map back to full vocab space (special tokens have ~0 probability)
            trans = np.zeros((V, V))
            trans[:, S:] = base
            # Tiny probability mass on special tokens to avoid numerical issues
            trans[:, :S] = 1e-8
            # Re-normalize
            trans /= trans.sum(axis=1, keepdims=True)

            # Add signature trigram patterns: for certain (a,b) pairs in this topic,
            # strongly prefer specific next tokens
            n_signatures = min(50, self.tokens_per_topic // 2)
            sig_tokens = self.rng.choice(
                range(topic_start, topic_end), size=n_signatures * 2, replace=False
            )
            for i in range(0, len(sig_tokens) - 1, 2):
                a, b = sig_tokens[i], sig_tokens[i + 1]
                # From token a, strongly prefer token b
                trans[a, S:] *= 0.3  # dampen other transitions
                trans[a, b] *= 10.0  # boost signature
                trans[a] /= trans[a].sum()

            transitions[t] = trans

        return transitions

    def _build_unigram_distributions(self) -> np.ndarray:
        """Build per-topic unigram distributions (marginals of transition matrices)."""
        # Use stationary distribution approximation: average row
        unigrams = np.zeros((self.num_topics, self.vocab_size))
        for t in range(self.num_topics):
            unigrams[t] = self.transitions[t].mean(axis=0)
            unigrams[t] /= unigrams[t].sum()
        return unigrams

    def generate_sequence(
        self,
        seq_len: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single sequence with multiple topic segments.

        Returns:
            tokens: (seq_len,) int array of token ids
            labels: (seq_len,) int array of topic ids per token
        """
        rng = rng or self.rng
        seq_len = seq_len or self.config.seq_len

        # Decide number of segments and their topics
        num_segments = rng.integers(self.config.min_segments, self.config.max_segments + 1)
        topics = rng.choice(self.num_topics, size=num_segments, replace=True)

        # Create segment boundaries (ensure minimum segment length)
        min_seg = self.config.min_segment_len
        remaining = seq_len - num_segments * min_seg
        if remaining < 0:
            # Not enough room for all segments at min length — reduce segments
            num_segments = seq_len // min_seg
            topics = topics[:num_segments]
            remaining = seq_len - num_segments * min_seg

        # Distribute remaining tokens across segments
        extra = rng.multinomial(remaining, np.ones(num_segments) / num_segments)
        seg_lengths = np.full(num_segments, min_seg) + extra

        # Adjust last segment to exactly fill seq_len
        seg_lengths[-1] = seq_len - seg_lengths[:-1].sum()

        tokens = np.zeros(seq_len, dtype=np.int64)
        labels = np.zeros(seq_len, dtype=np.int64)

        pos = 0
        for seg_idx in range(num_segments):
            topic = topics[seg_idx]
            seg_len = int(seg_lengths[seg_idx])

            # First token of segment: sample from unigram via CDF
            if pos == 0:
                prev = int(np.searchsorted(self.unigram_cdfs[topic], rng.random()))
            else:
                prev = tokens[pos - 1]

            # Vectorized: draw all uniform random numbers at once, sample via CDF
            end = min(pos + seg_len, seq_len)
            uniforms = rng.random(end - pos)
            for i in range(end - pos):
                next_token = int(np.searchsorted(self.transition_cdfs[topic, prev], uniforms[i]))
                tokens[pos + i] = next_token
                labels[pos + i] = topic
                prev = next_token

            pos += seg_len

        return tokens, labels

    def generate_dataset(
        self,
        num_sequences: int,
        seq_len: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a full dataset of sequences.

        Returns:
            tokens: (num_sequences, seq_len) int array
            labels: (num_sequences, seq_len) int array of topic ids
        """
        seq_len = seq_len or self.config.seq_len
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        all_tokens = np.zeros((num_sequences, seq_len), dtype=np.int64)
        all_labels = np.zeros((num_sequences, seq_len), dtype=np.int64)

        from tqdm import trange
        for i in trange(num_sequences, desc="Generating sequences"):
            seq_rng = np.random.default_rng(rng.integers(0, 2**31))
            all_tokens[i], all_labels[i] = self.generate_sequence(seq_len, seq_rng)

        return all_tokens, all_labels

    def dominant_topic(self, labels: np.ndarray) -> np.ndarray:
        """Get the dominant (most frequent) topic per sequence.

        Args:
            labels: (num_sequences, seq_len)
        Returns:
            (num_sequences,) array of dominant topic ids
        """
        return np.array([np.bincount(row, minlength=self.num_topics).argmax() for row in labels])


class MicroLanguageDataset(Dataset):
    """PyTorch Dataset wrapping generated micro-language data.

    Supports shuffled and topic-ordered modes.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        ordering: str = "shuffled",
        seed: int = 42,
    ):
        """
        Args:
            tokens: (N, seq_len) token ids
            labels: (N, seq_len) topic labels
            ordering: "shuffled" or "topic_ordered"
            seed: random seed for shuffling
        """
        self.tokens = tokens
        self.labels = labels
        self.ordering = ordering

        if ordering == "topic_ordered":
            # Sort by dominant topic, then by secondary patterns within topic
            num_classes = int(labels.max()) + 1
            dominant = np.array(
                [np.bincount(row, minlength=num_classes).argmax() for row in labels]
            )
            sort_idx = np.argsort(dominant, kind="stable")
            self.tokens = self.tokens[sort_idx]
            self.labels = self.labels[sort_idx]
        elif ordering == "shuffled":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(self.tokens))
            self.tokens = self.tokens[perm]
            self.labels = self.labels[perm]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.tokens[idx].copy()).long(),
            "labels": torch.from_numpy(self.labels[idx].copy()).long(),
        }


def generate_raw_data(
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MicroLanguageGenerator]:
    """Generate raw train/val arrays (no ordering applied yet).

    Returns:
        train_tokens, train_labels, val_tokens, val_labels, generator
    """
    generator = MicroLanguageGenerator(config)

    print(f"Generating {config.num_train_sequences} training sequences...")
    train_tokens, train_labels = generator.generate_dataset(
        config.num_train_sequences, seed=config.seed
    )
    print(f"Generating {config.num_val_sequences} validation sequences...")
    val_tokens, val_labels = generator.generate_dataset(
        config.num_val_sequences, seed=config.seed + 1
    )

    return train_tokens, train_labels, val_tokens, val_labels, generator


def create_dataloaders_from_cache(
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    ordering: str = "shuffled",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders from pre-generated arrays (avoids re-generating data).

    Returns:
        train_loader, val_loader
    """
    train_dataset = MicroLanguageDataset(
        train_tokens.copy(), train_labels.copy(), ordering=ordering, seed=seed
    )
    val_dataset = MicroLanguageDataset(
        val_tokens.copy(), val_labels.copy(), ordering="shuffled", seed=seed + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(ordering == "shuffled"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_dataloaders(
    config: DataConfig,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, MicroLanguageGenerator]:
    """Create train and validation dataloaders (generates data fresh).

    Returns:
        train_loader, val_loader, generator
    """
    train_tokens, train_labels, val_tokens, val_labels, generator = generate_raw_data(config)

    train_loader, val_loader = create_dataloaders_from_cache(
        train_tokens, train_labels, val_tokens, val_labels,
        ordering=config.ordering, batch_size=batch_size,
        num_workers=num_workers, seed=config.seed,
    )

    return train_loader, val_loader, generator
