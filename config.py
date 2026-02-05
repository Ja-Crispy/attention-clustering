"""Experiment configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 8
    num_q_heads: int = 8
    num_kv_heads: int = 8  # 8 = MHA, 4 = GQA (2:1 ratio)
    head_dim: int = 64
    intermediate_size: int = 1376
    vocab_size: int = 4096
    max_seq_len: int = 512
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    use_rope: bool = True  # toggle for DroPE
    dropout: float = 0.0

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_q_heads

    @property
    def num_kv_groups(self) -> int:
        assert self.num_q_heads % self.num_kv_heads == 0
        return self.num_q_heads // self.num_kv_heads

    @property
    def num_params_approx(self) -> int:
        """Rough parameter count estimate."""
        d = self.hidden_size
        qkv = d * (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        o = self.num_q_heads * self.head_dim * d
        mlp = 3 * d * self.intermediate_size  # gate + up + down
        per_layer = qkv + o + mlp
        embed = d * self.vocab_size
        total = self.num_layers * per_layer + embed
        if not self.tie_embeddings:
            total += embed
        return total


@dataclass
class DataConfig:
    num_topics: int = 5
    vocab_size: int = 4096
    seq_len: int = 512
    num_train_sequences: int = 100_000
    num_val_sequences: int = 10_000
    min_segments: int = 2
    max_segments: int = 5
    min_segment_len: int = 64
    topic_concentration: float = 5.0  # how much topics prefer "own" tokens
    ordering: str = "shuffled"  # "shuffled" or "topic_ordered"
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_steps: int = 10_000
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    seed: int = 42
    save_every: int = 1000
    log_every: int = 50
    eval_every: int = 500
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_wandb: bool = False
    compile: bool = False  # torch.compile


@dataclass
class ExperimentConfig:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results"

    def variant(self, **kwargs) -> "ExperimentConfig":
        """Create a copy with overrides. Supports dotted keys like 'model.use_rope'."""
        import copy
        cfg = copy.deepcopy(self)
        for key, value in kwargs.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return cfg


# --- Preset configurations ---

def mha_config() -> ModelConfig:
    """Standard multi-head attention (no GQA)."""
    return ModelConfig(num_q_heads=8, num_kv_heads=8)


def gqa_config() -> ModelConfig:
    """Grouped-query attention (Qwen3-style 2:1 ratio)."""
    return ModelConfig(num_q_heads=8, num_kv_heads=4)


def experiment_matrix() -> list[ExperimentConfig]:
    """Generate the full 2x2x2 experimental matrix.

    Axes:
        - Attention: MHA vs GQA
        - Data ordering: shuffled vs topic_ordered
        - Positional encoding: RoPE vs DroPE (no RoPE)
    """
    configs = []
    for attn_name, num_kv in [("mha", 8), ("gqa", 4)]:
        for ordering in ["shuffled", "topic_ordered"]:
            for rope_name, use_rope in [("rope", True), ("drope", False)]:
                name = f"{attn_name}_{ordering}_{rope_name}"
                cfg = ExperimentConfig(
                    name=name,
                    model=ModelConfig(num_kv_heads=num_kv, use_rope=use_rope),
                    data=DataConfig(ordering=ordering),
                )
                configs.append(cfg)
    return configs
