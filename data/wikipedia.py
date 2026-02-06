"""Wikipedia data pipeline for attention clustering v2.

Pipeline steps (each cached to disk):
1. Download & extract paragraphs from Wikipedia
2. Embed paragraphs with sentence-transformers
3. K-means cluster embeddings into topics
4. Train BPE tokenizer on corpus
5. Tokenize all paragraphs
6. Build multi-topic training sequences

Usage:
    python -m data.wikipedia                        # Run full pipeline
    python -m data.wikipedia --stats                # Print cache stats
    python -m data.wikipedia --num-articles 50000   # Custom article count

Requirements:
    pip install datasets sentence-transformers tokenizers scikit-learn
"""

import json
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm, trange


@dataclass
class WikiPipelineConfig:
    num_articles: int = 100_000
    num_topics: int = 8
    vocab_size: int = 16384
    seq_len: int = 512
    num_train_sequences: int = 500_000
    num_val_sequences: int = 10_000
    min_segments: int = 2
    max_segments: int = 4
    min_segment_len: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"
    min_paragraph_words: int = 50
    max_paragraph_words: int = 500
    cache_dir: str = "data/wiki_cache"
    seed: int = 42


def _check_requirements():
    """Check that all required packages are installed."""
    missing = []
    for pkg, name in [
        ("datasets", "datasets"),
        ("sentence_transformers", "sentence-transformers"),
        ("tokenizers", "tokenizers"),
        ("sklearn", "scikit-learn"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"Missing packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


class WikipediaPipeline:
    """Downloads, embeds, clusters, and tokenizes Wikipedia articles.

    Each step caches its output to disk. If cache exists, the step is skipped.
    """

    def __init__(self, config: WikiPipelineConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download Wikipedia and extract paragraphs
    # ------------------------------------------------------------------
    def step1_download_and_extract(self) -> list[str]:
        cache_file = self.cache_dir / "paragraphs.pkl"
        if cache_file.exists():
            print(f"Step 1: Loading cached paragraphs from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)  # noqa: S301 — trusted local data

        _check_requirements()
        from datasets import load_dataset

        num = self.config.num_articles
        print(f"Step 1: Downloading {num:,} Wikipedia articles...")

        # Try newer dataset first, fall back to older
        try:
            ds = load_dataset(
                "wikimedia/wikipedia", "20231101.en",
                split="train", streaming=True, trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                "wikipedia", "20220301.en",
                split="train", streaming=True, trust_remote_code=True,
            )

        paragraphs = []
        min_w = self.config.min_paragraph_words
        max_w = self.config.max_paragraph_words

        for i, article in enumerate(tqdm(ds, total=num, desc="Downloading")):
            if i >= num:
                break
            text = article["text"]
            for para in text.split("\n\n"):
                para = para.strip()
                # Skip section headers, templates, table rows
                if para.startswith("=") or para.startswith("{") or para.startswith("|"):
                    continue
                words = para.split()
                if min_w <= len(words) <= max_w:
                    paragraphs.append(para)

        print(f"  Extracted {len(paragraphs):,} paragraphs from {num:,} articles")

        with open(cache_file, "wb") as f:
            pickle.dump(paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return paragraphs

    # ------------------------------------------------------------------
    # Step 2: Embed paragraphs with sentence-transformers
    # ------------------------------------------------------------------
    def step2_embed(self, paragraphs: list[str]) -> np.ndarray:
        cache_file = self.cache_dir / "embeddings.npy"
        if cache_file.exists():
            print(f"Step 2: Loading cached embeddings from {cache_file}")
            return np.load(cache_file)

        _check_requirements()
        from sentence_transformers import SentenceTransformer

        print(f"Step 2: Embedding {len(paragraphs):,} paragraphs "
              f"with {self.config.embedding_model}...")

        model = SentenceTransformer(self.config.embedding_model)
        embeddings = model.encode(
            paragraphs, show_progress_bar=True, batch_size=256,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        print(f"  Embeddings shape: {embeddings.shape}")

        np.save(cache_file, embeddings)
        return embeddings

    # ------------------------------------------------------------------
    # Step 3: K-means clustering
    # ------------------------------------------------------------------
    def step3_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        cache_file = self.cache_dir / "cluster_labels.npy"
        if cache_file.exists():
            print(f"Step 3: Loading cached cluster labels from {cache_file}")
            return np.load(cache_file)

        from sklearn.cluster import MiniBatchKMeans

        k = self.config.num_topics
        print(f"Step 3: Clustering {len(embeddings):,} embeddings into {k} topics...")

        km = MiniBatchKMeans(
            n_clusters=k, random_state=self.config.seed,
            batch_size=4096, n_init=3, verbose=1,
        )
        labels = km.fit_predict(embeddings)

        print("  Cluster sizes:")
        for i in range(k):
            count = (labels == i).sum()
            print(f"    Topic {i}: {count:,} paragraphs ({count / len(labels) * 100:.1f}%)")

        np.save(cache_file, labels.astype(np.int32))
        np.save(self.cache_dir / "cluster_centers.npy", km.cluster_centers_)
        return labels

    # ------------------------------------------------------------------
    # Step 4: Train BPE tokenizer
    # ------------------------------------------------------------------
    def step4_train_tokenizer(self, paragraphs: list[str]):
        cache_file = self.cache_dir / "tokenizer.json"
        if cache_file.exists():
            print(f"Step 4: Loading cached tokenizer from {cache_file}")
            from tokenizers import Tokenizer
            return Tokenizer.from_file(str(cache_file))

        _check_requirements()
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.normalizers import NFKC
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        print(f"Step 4: Training BPE tokenizer (vocab_size={self.config.vocab_size})...")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train_from_iterator(paragraphs, trainer=trainer)
        tokenizer.save(str(cache_file))
        print(f"  Tokenizer vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer

    # ------------------------------------------------------------------
    # Step 5: Tokenize all paragraphs
    # ------------------------------------------------------------------
    def step5_tokenize(self, paragraphs: list[str], tokenizer) -> tuple[list[np.ndarray], np.ndarray]:
        """Tokenize paragraphs, filter short ones, return (tokenized, keep_mask).

        Returns:
            tokenized: list of token arrays (only paragraphs with >= 20 tokens)
            keep_mask: boolean mask over original paragraph indices
        """
        cache_file = self.cache_dir / "tokenized.pkl"
        mask_file = self.cache_dir / "tokenize_keep_mask.npy"
        if cache_file.exists() and mask_file.exists():
            print(f"Step 5: Loading cached tokenized paragraphs from {cache_file}")
            with open(cache_file, "rb") as f:
                tokenized = pickle.load(f)  # noqa: S301 — trusted local data
            keep_mask = np.load(mask_file)
            return tokenized, keep_mask

        print(f"Step 5: Tokenizing {len(paragraphs):,} paragraphs...")
        tokenized = []
        keep_mask = np.zeros(len(paragraphs), dtype=bool)
        batch_size = 10_000

        for i in tqdm(range(0, len(paragraphs), batch_size), desc="Tokenizing"):
            batch = paragraphs[i : i + batch_size]
            encoded = tokenizer.encode_batch(batch)
            for j, enc in enumerate(encoded):
                ids = np.array(enc.ids, dtype=np.int32)
                if len(ids) >= 20:
                    tokenized.append(ids)
                    keep_mask[i + j] = True

        print(f"  Kept {len(tokenized):,} paragraphs (>= 20 tokens)")

        lengths = [len(t) for t in tokenized]
        print(f"  Token lengths: mean={np.mean(lengths):.0f}, "
              f"median={np.median(lengths):.0f}, "
              f"min={np.min(lengths)}, max={np.max(lengths)}")

        with open(cache_file, "wb") as f:
            pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(mask_file, keep_mask)
        return tokenized, keep_mask

    # ------------------------------------------------------------------
    # Step 6: Build multi-topic sequences
    # ------------------------------------------------------------------
    def step6_build_sequences(
        self,
        tokenized: list[np.ndarray],
        cluster_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_file = self.cache_dir / "train_tokens.npy"
        if train_file.exists():
            print("Step 6: Loading cached sequences...")
            return (
                np.load(self.cache_dir / "train_tokens.npy"),
                np.load(self.cache_dir / "train_labels.npy"),
                np.load(self.cache_dir / "val_tokens.npy"),
                np.load(self.cache_dir / "val_labels.npy"),
            )

        print("Step 6: Building multi-topic sequences...")

        # Build concatenated token pools per topic (for fast random slicing)
        topic_pools = {}
        for t in range(self.config.num_topics):
            indices = np.where(cluster_labels == t)[0]
            paras = [tokenized[i] for i in indices if i < len(tokenized)]
            if not paras:
                raise ValueError(f"Topic {t} has no tokenized paragraphs!")
            pool = np.concatenate(paras)
            topic_pools[t] = pool
            print(f"  Topic {t}: {len(paras):,} paragraphs, {len(pool):,} tokens")

        rng = np.random.default_rng(self.config.seed)
        total = self.config.num_train_sequences + self.config.num_val_sequences

        all_tokens = np.zeros((total, self.config.seq_len), dtype=np.int32)
        all_labels = np.zeros((total, self.config.seq_len), dtype=np.int32)

        for i in trange(total, desc="Building sequences"):
            t, l = self._build_one_sequence(topic_pools, rng)
            all_tokens[i] = t
            all_labels[i] = l

        n_train = self.config.num_train_sequences
        train_tokens = all_tokens[:n_train]
        train_labels = all_labels[:n_train]
        val_tokens = all_tokens[n_train:]
        val_labels = all_labels[n_train:]

        np.save(self.cache_dir / "train_tokens.npy", train_tokens)
        np.save(self.cache_dir / "train_labels.npy", train_labels)
        np.save(self.cache_dir / "val_tokens.npy", val_tokens)
        np.save(self.cache_dir / "val_labels.npy", val_labels)

        print(f"  Train: {train_tokens.shape}, Val: {val_tokens.shape}")
        return train_tokens, train_labels, val_tokens, val_labels

    def _build_one_sequence(
        self,
        topic_pools: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build one multi-topic sequence from concatenated topic pools."""
        seq_len = self.config.seq_len
        min_seg = self.config.min_segment_len
        n_segs = rng.integers(self.config.min_segments, self.config.max_segments + 1)

        # Ensure segments fit
        max_possible = seq_len // min_seg
        n_segs = min(n_segs, max(1, max_possible))
        topics = rng.choice(self.config.num_topics, size=n_segs, replace=True)

        # Distribute tokens across segments
        base = np.full(n_segs, min_seg)
        remaining = seq_len - base.sum()
        extra = rng.multinomial(remaining, np.ones(n_segs) / n_segs)
        seg_lens = base + extra
        seg_lens[-1] = seq_len - seg_lens[:-1].sum()

        tokens = np.zeros(seq_len, dtype=np.int32)
        labels = np.zeros(seq_len, dtype=np.int32)
        pos = 0

        for seg_idx in range(n_segs):
            topic = topics[seg_idx]
            seg_len = int(seg_lens[seg_idx])
            pool = topic_pools[topic]

            # Slice a contiguous chunk from the topic pool
            max_start = max(0, len(pool) - seg_len)
            start = rng.integers(0, max_start + 1)
            take = min(seg_len, len(pool) - start)

            tokens[pos : pos + take] = pool[start : start + take]
            labels[pos : pos + take] = topic

            # If pool shorter than segment (very unlikely), fill remainder
            filled = take
            while filled < seg_len:
                s = rng.integers(0, max(1, len(pool)))
                t = min(seg_len - filled, len(pool) - s)
                tokens[pos + filled : pos + filled + t] = pool[s : s + t]
                labels[pos + filled : pos + filled + t] = topic
                filled += t

            pos += seg_len

        return tokens, labels

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the full pipeline. Skips cached steps."""
        print("=" * 60)
        print("Wikipedia Data Pipeline")
        print("=" * 60)
        print(f"  Articles: {self.config.num_articles:,}")
        print(f"  Topics: {self.config.num_topics}")
        print(f"  Vocab: {self.config.vocab_size:,}")
        print(f"  Sequences: {self.config.num_train_sequences:,} train + "
              f"{self.config.num_val_sequences:,} val")
        print(f"  Cache: {self.cache_dir}")
        print()

        paragraphs = self.step1_download_and_extract()
        embeddings = self.step2_embed(paragraphs)
        cluster_labels_all = self.step3_cluster(embeddings)
        tokenizer = self.step4_train_tokenizer(paragraphs)
        tokenized, keep_mask = self.step5_tokenize(paragraphs, tokenizer)

        # Align cluster labels with filtered paragraphs
        cluster_labels = cluster_labels_all[keep_mask]
        assert len(cluster_labels) == len(tokenized), (
            f"Mismatch: {len(cluster_labels)} labels vs {len(tokenized)} tokenized"
        )

        train_tokens, train_labels, val_tokens, val_labels = self.step6_build_sequences(
            tokenized, cluster_labels
        )

        # Save pipeline config
        with open(self.cache_dir / "pipeline_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"\nPipeline complete!")
        print(f"  Train: {train_tokens.shape}")
        print(f"  Val: {val_tokens.shape}")
        return train_tokens, train_labels, val_tokens, val_labels


def load_wiki_data(
    cache_dir: str = "data/wiki_cache",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-built Wikipedia data arrays from cache.

    Returns:
        train_tokens, train_labels, val_tokens, val_labels
    """
    cache = Path(cache_dir)
    if not (cache / "train_tokens.npy").exists():
        raise FileNotFoundError(
            f"Wiki data not found in {cache_dir}. "
            "Run `python -m data.wikipedia` first to prepare data."
        )

    return (
        np.load(cache / "train_tokens.npy"),
        np.load(cache / "train_labels.npy"),
        np.load(cache / "val_tokens.npy"),
        np.load(cache / "val_labels.npy"),
    )


def print_stats(cache_dir: str = "data/wiki_cache"):
    """Print statistics about cached Wikipedia data."""
    cache = Path(cache_dir)
    if not cache.exists():
        print(f"No cache found at {cache_dir}")
        return

    print(f"Cache directory: {cache}")
    print()

    cfg_file = cache / "pipeline_config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        print("Pipeline config:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()

    for name in ["train_tokens", "train_labels", "val_tokens", "val_labels"]:
        f = cache / f"{name}.npy"
        if f.exists():
            arr = np.load(f, mmap_mode="r")
            size_mb = f.stat().st_size / 1e6
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, size={size_mb:.1f}MB")

    labels_file = cache / "cluster_labels.npy"
    if labels_file.exists():
        labels = np.load(labels_file)
        n_topics = labels.max() + 1
        print(f"\n  Clusters: {n_topics} topics, {len(labels):,} paragraphs")
        for i in range(n_topics):
            count = (labels == i).sum()
            print(f"    Topic {i}: {count:,} ({count / len(labels) * 100:.1f}%)")

    total_size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"\n  Total cache size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia data pipeline")
    parser.add_argument("--stats", action="store_true", help="Print cache stats")
    parser.add_argument("--num-articles", type=int, default=100_000)
    parser.add_argument("--num-topics", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--num-train", type=int, default=500_000)
    parser.add_argument("--num-val", type=int, default=10_000)
    parser.add_argument("--cache-dir", default="data/wiki_cache")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.cache_dir)
    else:
        config = WikiPipelineConfig(
            num_articles=args.num_articles,
            num_topics=args.num_topics,
            vocab_size=args.vocab_size,
            num_train_sequences=args.num_train,
            num_val_sequences=args.num_val,
            cache_dir=args.cache_dir,
        )
        pipeline = WikipediaPipeline(config)
        pipeline.run()
