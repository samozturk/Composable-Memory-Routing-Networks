# file: cmrn/model.py
import torch
import torch.nn as nn
from types import SimpleNamespace

from .config import CMRNConfig
from .experts import ExpertModule
from .routing import DynamicRouter
from .memory import PersistentMemory
from .temporal import TemporalAbstraction

class CMRN(nn.Module):
    """Composable Memory Routing Network."""
    def __init__(self, config: CMRNConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.router = DynamicRouter(config.d_model, config.num_experts)
        
        self.experts = nn.ModuleList([
            ExpertModule(config.d_model, expert_type)
            for expert_type in config.expert_types
            for _ in range(config.num_experts // len(config.expert_types))
        ])
        
        self.memory_system = PersistentMemory(config.memory_dim, config.memory_size)
        self.temporal_module = TemporalAbstraction(config.d_model, config.temporal_layers)
        
        # Final layer to project back to vocabulary size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids):
        # 1. Embedding
        x = self.token_embeddings(input_ids) # (B, L, D)

        # 2. Temporal Processing
        x, _ = self.temporal_module(x)

        # 3. Dynamic Routing
        routing_weights = self.router(x, k=self.config.routing_k, training=self.training)
        # routing_weights shape: (B, L, num_experts)
        
        # 4. Expert Processing
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        # expert_outputs shape: (B, L, num_experts, D)
        
        # 5. Weighted sum of expert outputs
        # routing_weights needs to be (B, L, num_experts, 1) to broadcast
        output_hidden_states = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=-2)
        # final output shape: (B, L, D)
        
        # 6. Language model head
        logits = self.lm_head(output_hidden_states)
        
        # Return a simple namespace for clear access to outputs
        return SimpleNamespace(
            logits=logits,
            routing_weights=routing_weights,
            memory_stats=f"{self.memory_system.index.ntotal}/{self.config.memory_size}"
        )

    # Add placeholder methods for features mentioned in the README
    def generate(self, input_ids, max_length=50, **kwargs):
        """Simple greedy decoding for text generation."""
        for _ in range(max_length):
            outputs = self(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

    def reset_memory(self):
        self.memory_system.__init__(self.config.memory_dim, self.config.memory_size)
        print("Memory has been reset.")