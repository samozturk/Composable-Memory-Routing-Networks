# file: cmrn/config.py
from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class CMRNConfig:
    """Configuration for the CMRN Model."""
    # Model architecture
    vocab_size: int = 50000
    d_model: int = 768
    num_experts: int = 16
    expert_types: List[str] = field(default_factory=lambda: ['general', 'symbolic', 'pattern', 'generative'])

    # Memory settings
    memory_size: int = 1_000_000
    memory_dim: int = 768
    memory_update_rate: float = 0.01

    # Routing configuration
    routing_strategy: Literal['learned', 'hash', 'random'] = 'learned'
    routing_k: int = 2
    routing_temperature: float = 1.0

    # Temporal settings
    temporal_layers: int = 3
    max_episode_length: int = 10000

    # Training settings
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = True

@dataclass
class TrainingConfig:
    """Configuration for the CMRNTrainer."""
    learning_rate: float = 5e-5
    batch_size: int = 32
    max_steps: int = 100_000
    memory_update_frequency: int = 100
    routing_temperature_schedule: str = 'cosine' # Placeholder for schedule type