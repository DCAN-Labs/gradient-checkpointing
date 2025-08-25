"""
Benchmarking suite for gradient checkpointing.

Compares memory usage and training time across different checkpointing strategies:
- Standard backpropagation (O(n) memory, O(n) compute)
- Full checkpointing (O(1) memory, O(n²) compute)
- Selective checkpointing (O(√n) memory, O(n√n) compute)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from gradient_checkpointing import checkpoint, CheckpointedSequential, SelectiveCheckpoint


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    strategy: str
    peak_memory_mb: float
    avg_memory_mb: float
    total_time_seconds: float
    forward_time_seconds: float
    backward_time_seconds: float
    iterations: int
    model_params: int
    batch_size: int
    sequence_length: int


class MemoryMonitor:
    """Monitor memory usage during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset memory statistics."""
        self.memory_samples = []
        self.peak_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def sample(self):
        """Sample current memory usage."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, torch.cuda.max_memory_allocated() / 1024 / 1024)
        else:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        
        self.memory_samples.append(current_memory)
        return current_memory
    
    def get_stats(self) -> Tuple[float, float]:
        """Get memory statistics (peak, average)."""
        if not self.memory_samples:
            return 0.0, 0.0
        return self.peak_memory, np.mean(self.memory_samples)


class TestModel(nn.Module):
    """Test model for benchmarking."""
    
    def __init__(self, num_layers: int = 12, hidden_size: int = 768, 
                 sequence_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Build transformer-like layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, dropout) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return self.output_projection(x)


class TransformerLayer(nn.Module):
    """Single transformer layer for testing."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CheckpointedTestModel(TestModel):
    """Test model with full gradient checkpointing."""
    
    def forward(self, x):
        """Forward pass with checkpointing for each layer."""
        for layer in self.layers:
            x = checkpoint(layer, x)
        return self.output_projection(x)


class SelectiveCheckpointedTestModel(TestModel):
    """Test model with selective gradient checkpointing."""
    
    def __init__(self, *args, checkpoint_every_n: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_every_n = checkpoint_every_n
    
    def forward(self, x):
        """Forward pass with selective checkpointing."""
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every_n == 0:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return self.output_projection(x)


def run_benchmark(
    model: nn.Module,
    strategy_name: str,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    iterations: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BenchmarkResult:
    """
    Run a benchmark for a specific checkpointing strategy.
    
    Args:
        model: The model to benchmark
        strategy_name: Name of the checkpointing strategy
        batch_size: Batch size for training
        sequence_length: Sequence length for input
        hidden_size: Hidden dimension size
        iterations: Number of training iterations
        device: Device to run on
    
    Returns:
        BenchmarkResult with timing and memory statistics
    """
    model = model.to(device)
    model.train()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Memory monitor
    monitor = MemoryMonitor()
    
    # Timing
    forward_times = []
    backward_times = []
    total_start = time.time()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nRunning {strategy_name} benchmark...")
    print(f"Model parameters: {num_params:,}")
    print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
    
    for i in range(iterations):
        # Clear gradients and cache
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Generate random input
        x = torch.randn(sequence_length, batch_size, hidden_size, device=device)
        target = torch.randn(sequence_length, batch_size, hidden_size, device=device)
        
        # Forward pass
        forward_start = time.time()
        output = model(x)
        loss = criterion(output, target)
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        
        # Sample memory after forward
        monitor.sample()
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)
        
        # Sample memory after backward
        monitor.sample()
        
        # Optimizer step
        optimizer.step()
        
        if (i + 1) % max(1, iterations // 10) == 0:
            print(f"  Iteration {i+1}/{iterations}: "
                  f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
    
    total_time = time.time() - total_start
    peak_mem, avg_mem = monitor.get_stats()
    
    result = BenchmarkResult(
        strategy=strategy_name,
        peak_memory_mb=peak_mem,
        avg_memory_mb=avg_mem,
        total_time_seconds=total_time,
        forward_time_seconds=np.mean(forward_times),
        backward_time_seconds=np.mean(backward_times),
        iterations=iterations,
        model_params=num_params,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    print(f"  Peak memory: {peak_mem:.2f} MB")
    print(f"  Average memory: {avg_mem:.2f} MB")
    print(f"  Total time: {total_time:.2f}s")
    
    return result


def compare_strategies(
    num_layers: int = 12,
    hidden_size: int = 768,
    batch_size: int = 4,
    sequence_length: int = 512,
    iterations: int = 10
) -> Dict[str, BenchmarkResult]:
    """
    Compare different checkpointing strategies.
    
    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        batch_size: Batch size for training
        sequence_length: Sequence length
        iterations: Number of training iterations
    
    Returns:
        Dictionary mapping strategy names to results
    """
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Gradient Checkpointing Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Device: {device}")
    print(f"{'='*60}")
    
    # Strategy 1: Standard backpropagation
    model = TestModel(num_layers, hidden_size, sequence_length)
    results['standard'] = run_benchmark(
        model, "Standard Backprop", batch_size, sequence_length, 
        hidden_size, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Strategy 2: Full checkpointing
    model = CheckpointedTestModel(num_layers, hidden_size, sequence_length)
    results['full_checkpoint'] = run_benchmark(
        model, "Full Checkpointing", batch_size, sequence_length,
        hidden_size, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Strategy 3: Selective checkpointing (every 2 layers)
    model = SelectiveCheckpointedTestModel(
        num_layers, hidden_size, sequence_length, checkpoint_every_n=2
    )
    results['selective_2'] = run_benchmark(
        model, "Selective (every 2)", batch_size, sequence_length,
        hidden_size, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Strategy 4: Selective checkpointing (every 3 layers)
    model = SelectiveCheckpointedTestModel(
        num_layers, hidden_size, sequence_length, checkpoint_every_n=3
    )
    results['selective_3'] = run_benchmark(
        model, "Selective (every 3)", batch_size, sequence_length,
        hidden_size, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return results


def print_comparison_table(results: Dict[str, BenchmarkResult]):
    """Print a comparison table of results."""
    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")
    print(f"{'Strategy':<20} {'Peak Mem (MB)':<15} {'Avg Mem (MB)':<15} "
          f"{'Forward (s)':<12} {'Backward (s)':<12} {'Total (s)':<10}")
    print(f"{'-'*80}")
    
    # Get baseline (standard) for comparison
    baseline = results.get('standard')
    
    for key, result in results.items():
        mem_reduction = ""
        time_overhead = ""
        
        if baseline and key != 'standard':
            mem_reduction = f"(-{(1 - result.peak_memory_mb/baseline.peak_memory_mb)*100:.1f}%)"
            time_overhead = f"(+{(result.total_time_seconds/baseline.total_time_seconds - 1)*100:.1f}%)"
        
        print(f"{result.strategy:<20} "
              f"{result.peak_memory_mb:<15.2f} "
              f"{result.avg_memory_mb:<15.2f} "
              f"{result.forward_time_seconds:<12.4f} "
              f"{result.backward_time_seconds:<12.4f} "
              f"{result.total_time_seconds:<10.2f}")
        
        if mem_reduction or time_overhead:
            print(f"{'':>20} {mem_reduction:<15} {'':>15} {'':>12} {'':>12} {time_overhead}")
    
    print(f"{'='*80}")
    
    # Print trade-off analysis
    print(f"\nTrade-off Analysis:")
    print(f"{'-'*40}")
    
    if baseline:
        for key, result in results.items():
            if key != 'standard':
                mem_saving = (1 - result.peak_memory_mb/baseline.peak_memory_mb) * 100
                compute_overhead = (result.total_time_seconds/baseline.total_time_seconds - 1) * 100
                efficiency = mem_saving / max(compute_overhead, 0.1)  # Avoid division by zero
                
                print(f"{result.strategy}:")
                print(f"  Memory savings: {mem_saving:.1f}%")
                print(f"  Compute overhead: {compute_overhead:.1f}%")
                print(f"  Efficiency ratio: {efficiency:.2f}")


if __name__ == "__main__":
    # Run benchmarks with different configurations
    
    # Small model configuration
    print("\n" + "="*80)
    print("SMALL MODEL CONFIGURATION")
    print("="*80)
    results_small = compare_strategies(
        num_layers=6,
        hidden_size=512,
        batch_size=8,
        sequence_length=256,
        iterations=10
    )
    print_comparison_table(results_small)
    
    # Medium model configuration
    print("\n" + "="*80)
    print("MEDIUM MODEL CONFIGURATION")
    print("="*80)
    results_medium = compare_strategies(
        num_layers=12,
        hidden_size=768,
        batch_size=4,
        sequence_length=512,
        iterations=10
    )
    print_comparison_table(results_medium)
    
    # Large model configuration (if enough memory)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
        print("\n" + "="*80)
        print("LARGE MODEL CONFIGURATION")
        print("="*80)
        results_large = compare_strategies(
            num_layers=24,
            hidden_size=1024,
            batch_size=2,
            sequence_length=1024,
            iterations=5
        )
        print_comparison_table(results_large)