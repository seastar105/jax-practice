import dataclasses
import json
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound="JsonSerializable")


class JsonSerializable:
    def to_json(self, path: str, **dump_kwargs):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, **dump_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls: Type[T], path: str) -> T:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)


@dataclasses.dataclass
class TransformerConfig(JsonSerializable):
    vocab_size: int = 50304
    num_layers: int = 12
    dim: int = 768
    dim_ff: int = 3072
    num_heads: int = 12
    context_length: int = 1024
    ff_class: str = "vanilla"  # "vanilla" or "glu"
    ff_activation: str = "gelu"
    ff_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_bias: bool = False
    norm_class: str = "layernorm"
    tie_embedding: bool = True
    use_remat: bool = False


@dataclasses.dataclass
class TrainConfig(JsonSerializable):
    log_interval: int = 100
    val_interval: int = 500
    per_device_batch_size: int = 32
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    learning_rate_scheduler: str = "linear_warmup_cosine_decay"
    total_steps: int = 10000
    warmup_steps: int = 400
    decay_steps: int = 400
    wandb_project_name: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclasses.dataclass
class Config(JsonSerializable):
    transformer_config: TransformerConfig = dataclasses.field(default_factory=TransformerConfig)
    train_config: TrainConfig = dataclasses.field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            transformer_config=TransformerConfig.from_dict(data["transformer_config"]),
            train_config=TrainConfig.from_dict(data["train_config"]),
        )
