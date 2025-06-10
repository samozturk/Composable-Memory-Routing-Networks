# file: cmrn/routing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_routing(logits, k=2, training=True):
    """Top-k sparse routing with straight-through gradients."""
    if k >= logits.shape[-1]: # If k is all experts, just use softmax
        return torch.softmax(logits, dim=-1)
        
    if training:
        # Gumbel-Softmax for differentiable sampling during training
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        noisy_logits = logits + gumbel_noise
        top_k_indices = torch.topk(noisy_logits, k=k, dim=-1).indices
    else:
        # Deterministic top-k selection for inference
        top_k_indices = torch.topk(logits, k=k, dim=-1).indices

    # Create a sparse mask from top-k indices
    mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)

    # Apply mask and re-normalize (straight-through estimator part)
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    routing_weights = torch.softmax(masked_logits, dim=-1)

    return routing_weights

class DynamicRouter(nn.Module):
    """A simplified router that produces expert weights."""
    def __init__(self, d_model, num_experts):
        super().__init__()
        # This network determines which experts to route to based on token embeddings.
        self.routing_network = nn.Linear(d_model, num_experts)

    def forward(self, x, k=2, training=True):
        # x shape: (Batch, Sequence, Dim)
        routing_logits = self.routing_network(x) # (B, L, num_experts)
        routing_weights = sparse_routing(routing_logits, k, training)
        return routing_weights