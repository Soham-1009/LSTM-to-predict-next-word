"""Model and training configuration definitions matching architecture.md specification."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """LSTM Language Model Hyperparameters."""

    vocab_size: int = 30000
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.30
    context_length: int = 35
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: Path | str) -> "ModelConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for data processing and training."""

    batch_size: int = 64
    global_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 10
    early_stopping_patience: int = 3
    seed: int = 42
    val_split_ratio: float = 0.10
    test_split_ratio: float = 0.10
    checkpoint_dir: str = "backend/model_artifacts/checkpoints"

