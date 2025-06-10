# file: cmrn/memory.py
import torch
import torch.nn as nn

# FAISS is a required dependency for efficient search.
# This code assumes it is installed.
try:
    import faiss
except ImportError:
    print("Warning: faiss-gpu is not installed. PersistentMemory will not work.")
    faiss = None

class PersistentMemory(nn.Module):
    """Differentiable external memory with FAISS indexing."""
    def __init__(self, memory_dim, max_size=1000000):
        super().__init__()
        if faiss is None:
            raise ImportError("faiss-gpu package is required for PersistentMemory.")

        self.memory_dim = memory_dim
        self.max_size = max_size
        self.register_buffer('memory_bank', torch.randn(max_size, memory_dim))
        self.register_buffer('usage_tracker', torch.zeros(max_size))

        # FAISS index for efficient similarity search
        self.index = faiss.IndexFlatIP(memory_dim)
        # Add the initial random vectors to the index
        self.index.add(self.memory_bank.cpu().numpy())

    def query(self, query_vector, k=10):
        """Retrieve top-k similar memories."""
        scores, indices = self.index.search(query_vector.detach().cpu().numpy(), k)
        
        # Move indices to the correct device to gather from memory_bank
        device = self.memory_bank.device
        indices_tensor = torch.from_numpy(indices).to(device)
        
        retrieved_memories = self.memory_bank[indices_tensor]
        return retrieved_memories, torch.from_numpy(scores).to(device)

    def update(self, keys, values):
        """Update memory with new key-value pairs."""
        num_updates = keys.shape[0]
        # Find least recently used slots
        _, update_indices = torch.topk(self.usage_tracker, k=num_updates, largest=False)

        # Update memory bank and usage tracker
        self.memory_bank[update_indices] = values
        self.usage_tracker[update_indices] = torch.max(self.usage_tracker) + 1.0 # Increment usage
        
        # Update FAISS index (remove old, add new)
        self.index.remove_ids(update_indices.cpu().numpy())
        self.index.add(values.detach().cpu().numpy())


class MemoryAttention(nn.Module):
    """Replaces quadratic attention with memory-based retrieval."""
    def __init__(self, d_model, memory_heads=8):
        super().__init__()
        self.d_model = d_model
        self.memory_heads = memory_heads
        assert d_model % memory_heads == 0, "d_model must be divisible by memory_heads"
        self.head_dim = d_model // memory_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x, memory_bank: PersistentMemory):
        B, L, D = x.shape
        
        # Project queries
        queries = self.query_proj(x).view(B, L, self.memory_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3) # (B, H, L, head_dim)
        
        # Batch memory retrieval by reshaping
        head_queries = queries.reshape(-1, self.head_dim)
        
        retrieved_memories, scores = memory_bank.query(head_queries, k=32)
        # retrieved_memories shape: (B*H*L, k, head_dim)
        # scores shape: (B*H*L, k)
        
        # Apply softmax to scores and compute weighted sum
        weights = torch.softmax(scores, dim=-1).unsqueeze(1) # (B*H*L, 1, k)
        retrieved_memories = retrieved_memories.to(x.device) # Ensure device match
        
        # attended shape: (B*H*L, 1, head_dim) -> (B*H*L, head_dim)
        attended = torch.bmm(weights, retrieved_memories).squeeze(1)
        
        # Reshape back to (B, H, L, head_dim)
        attended = attended.view(B, self.memory_heads, L, self.head_dim)
        
        # Concatenate heads and project
        output = attended.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.output_proj(output)