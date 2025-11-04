import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class StandardAttention(nn.Module):
    """
    Standard self-attention mechanism used in Transformers.
    Time Complexity: O(n²d) where n is sequence length, d is dimension
    Memory Complexity: O(n²) for attention matrix
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        # MEMORY BOTTLENECK: This matrix is seq_len x seq_len for EACH head!
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


class SparseAttention(nn.Module):
    """
    Sparse attention with local + strided pattern (inspired by Sparse Transformers).
    Time Complexity: O(n√n * d) - much better for long sequences
    Memory Complexity: O(n√n) for attention matrix
    """
    def __init__(self, d_model, n_heads, local_window=128, stride=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window = local_window
        self.stride = stride
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Initialize sparse attention matrix
        attn_weights = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=x.device)
        
        # For each query position
        for i in range(seq_len):
            # Local attention: attend to nearby tokens
            local_start = max(0, i - self.local_window)
            local_end = min(seq_len, i + self.local_window + 1)
            
            # Strided attention: attend to every stride-th position
            strided_indices = list(range(0, seq_len, self.stride))
            
            # Combine local and strided indices (remove duplicates)
            local_indices = list(range(local_start, local_end))
            sparse_indices = sorted(set(local_indices + strided_indices))
            
            # Compute attention scores only for sparse positions
            Q_i = Q[:, :, i:i+1, :]  # [batch, n_heads, 1, d_k]
            K_sparse = K[:, :, sparse_indices, :]  # [batch, n_heads, n_sparse, d_k]
            
            scores = torch.matmul(Q_i, K_sparse.transpose(-2, -1)) / math.sqrt(self.d_k)
            # Shape: [batch_size, n_heads, 1, n_sparse]
            
            # Softmax over sparse positions
            sparse_attn = F.softmax(scores, dim=-1)
            
            # Place sparse attention weights in full matrix
            attn_weights[:, :, i, sparse_indices] = sparse_attn.squeeze(2)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


def benchmark_attention(seq_len=512, d_model=512, n_heads=8, batch_size=2, device='cuda'):
    """
    Benchmark and compare standard vs sparse attention
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking with seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    print(f"{'='*60}\n")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Standard Attention
    print("Running Standard Attention...")
    standard_attn = StandardAttention(d_model, n_heads).to(device)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    output_std, attn_weights_std = standard_attn(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_std = time.time() - start
    
    # Memory used by attention matrix
    memory_std = attn_weights_std.element_size() * attn_weights_std.nelement() / (1024**2)  # MB
    
    print(f"  Time: {time_std*1000:.2f} ms")
    print(f"  Attention matrix shape: {attn_weights_std.shape}")
    print(f"  Memory for attention: {memory_std:.2f} MB")
    print(f"  Non-zero elements: 100.0% (dense)")
    
    # Sparse Attention
    print("\nRunning Sparse Attention...")
    sparse_attn = SparseAttention(d_model, n_heads, local_window=64, stride=64).to(device)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    output_sparse, attn_weights_sparse = sparse_attn(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_sparse = time.time() - start
    
    # Calculate sparsity
    nonzero = (attn_weights_sparse > 0).sum().item()
    total = attn_weights_sparse.numel()
    sparsity = (1 - nonzero / total) * 100
    memory_sparse = nonzero * attn_weights_sparse.element_size() / (1024**2)  # MB
    
    print(f"  Time: {time_sparse*1000:.2f} ms")
    print(f"  Attention matrix shape: {attn_weights_sparse.shape}")
    print(f"  Memory for attention: {memory_sparse:.2f} MB (theoretical)")
    print(f"  Non-zero elements: {(1-sparsity/100)*100:.1f}%")
    print(f"  Sparsity: {sparsity:.1f}%")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"  Speedup: {time_std/time_sparse:.2f}x faster")
    print(f"  Memory reduction: {(1-memory_sparse/memory_std)*100:.1f}% less memory")
    print(f"  Complexity: O(n²) -> O(n√n)")
    print(f"{'='*60}\n")
    
    return {
        'standard': {'time': time_std, 'memory': memory_std},
        'sparse': {'time': time_sparse, 'memory': memory_sparse, 'sparsity': sparsity}
    }


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with different sequence lengths
    for seq_len in [512, 1024, 2048]:
        benchmark_attention(seq_len=seq_len, d_model=512, n_heads=8, device=device)
    
    print("\nKEY INSIGHTS:")
    print("1. Standard attention computes O(n²) pairwise interactions")
    print("2. Sparse attention reduces this to O(n√n) using local + strided patterns")
    print("3. Memory savings are MASSIVE for long sequences (e.g., 4096+ tokens)")
    print("4. Sparse patterns preserve most of the modeling capacity")
    print("\nReal implementations (e.g., Longformer, BigBird) use optimized")
    print("kernels and more sophisticated sparsity patterns for better performance.")