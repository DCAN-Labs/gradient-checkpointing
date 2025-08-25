"""
Memory profiling and visualization tools for gradient checkpointing.

This module provides tools to profile memory usage, visualize memory timelines,
and analyze the trade-offs between memory and compute.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import gc
import psutil
from contextlib import contextmanager
from collections import defaultdict


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    active_mb: float
    phase: str  # 'forward', 'backward', 'optimizer'
    layer_name: str = ""
    iteration: int = 0


@dataclass 
class ProfileResult:
    """Complete profiling result for a model."""
    model_name: str
    strategy: str
    snapshots: List[MemorySnapshot]
    peak_memory_mb: float
    average_memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    num_parameters: int
    batch_size: int
    input_shape: Tuple[int, ...]


class MemoryProfiler:
    """Advanced memory profiler for PyTorch models."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize memory profiler.
        
        Args:
            device: Device to profile ('cuda' or 'cpu')
        """
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self.hooks = []
        self.start_time = None
        
    def reset(self):
        """Reset profiler state."""
        self.snapshots = []
        self.remove_hooks()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def snapshot(self, phase: str, layer_name: str = "", iteration: int = 0):
        """
        Take a memory snapshot.
        
        Args:
            phase: Current phase ('forward', 'backward', etc.)
            layer_name: Name of the current layer
            iteration: Current iteration number
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        timestamp = time.time() - self.start_time
        
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            active = allocated  # For CUDA, active = allocated
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            allocated = mem_info.rss / 1024 / 1024
            reserved = mem_info.vms / 1024 / 1024
            active = allocated
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            allocated_mb=allocated,
            reserved_mb=reserved,
            active_mb=active,
            phase=phase,
            layer_name=layer_name,
            iteration=iteration
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def register_hooks(self, model: nn.Module):
        """
        Register forward and backward hooks on all layers.
        
        Args:
            model: Model to profile
        """
        def make_forward_hook(name):
            def hook(module, input, output):
                self.snapshot(phase='forward', layer_name=name)
            return hook
        
        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.snapshot(phase='backward', layer_name=name)
            return hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                forward_hook = module.register_forward_hook(make_forward_hook(name))
                backward_hook = module.register_full_backward_hook(make_backward_hook(name))
                self.hooks.extend([forward_hook, backward_hook])
    
    @contextmanager
    def profile(self, model: nn.Module, auto_register: bool = True):
        """
        Context manager for profiling.
        
        Args:
            model: Model to profile
            auto_register: Whether to automatically register hooks
        """
        self.reset()
        
        if auto_register:
            self.register_hooks(model)
        
        self.start_time = time.time()
        
        try:
            yield self
        finally:
            self.remove_hooks()
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        if not self.snapshots:
            return 0.0
        return max(s.allocated_mb for s in self.snapshots)
    
    def get_average_memory(self) -> float:
        """Get average memory usage."""
        if not self.snapshots:
            return 0.0
        return np.mean([s.allocated_mb for s in self.snapshots])
    
    def get_memory_by_phase(self) -> Dict[str, List[float]]:
        """Get memory usage grouped by phase."""
        by_phase = defaultdict(list)
        for snapshot in self.snapshots:
            by_phase[snapshot.phase].append(snapshot.allocated_mb)
        return dict(by_phase)
    
    def get_memory_by_layer(self) -> Dict[str, List[float]]:
        """Get memory usage grouped by layer."""
        by_layer = defaultdict(list)
        for snapshot in self.snapshots:
            if snapshot.layer_name:
                by_layer[snapshot.layer_name].append(snapshot.allocated_mb)
        return dict(by_layer)


class MemoryVisualizer:
    """Visualization tools for memory profiling results."""
    
    @staticmethod
    def plot_memory_timeline(
        profiles: List[ProfileResult],
        title: str = "Memory Usage Timeline",
        save_path: Optional[str] = None
    ):
        """
        Plot memory usage timeline for multiple strategies.
        
        Args:
            profiles: List of profiling results
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(profiles))
        
        for i, profile in enumerate(profiles):
            timestamps = [s.timestamp * 1000 for s in profile.snapshots]  # Convert to ms
            memory = [s.allocated_mb for s in profile.snapshots]
            
            plt.plot(timestamps, memory, label=profile.strategy, 
                    color=colors[i], linewidth=2, alpha=0.8)
            
            # Mark peak memory
            peak_idx = np.argmax(memory)
            plt.scatter(timestamps[peak_idx], memory[peak_idx], 
                       color=colors[i], s=100, zorder=5)
            plt.annotate(f'{memory[peak_idx]:.1f} MB',
                        xy=(timestamps[peak_idx], memory[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, color=colors[i])
        
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Memory (MB)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_memory_vs_compute_tradeoff(
        profiles: List[ProfileResult],
        save_path: Optional[str] = None
    ):
        """
        Plot memory vs compute trade-off scatter plot.
        
        Args:
            profiles: List of profiling results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        memory_values = [p.peak_memory_mb for p in profiles]
        compute_values = [p.total_time_ms for p in profiles]
        strategies = [p.strategy for p in profiles]
        
        # Normalize values for better visualization
        baseline_memory = memory_values[0]  # Assume first is baseline
        baseline_compute = compute_values[0]
        
        memory_savings = [(1 - m/baseline_memory) * 100 for m in memory_values]
        compute_overhead = [(c/baseline_compute - 1) * 100 for c in compute_values]
        
        colors = sns.color_palette("viridis", len(profiles))
        
        for i, (mem_save, comp_over, strategy) in enumerate(
            zip(memory_savings, compute_overhead, strategies)
        ):
            plt.scatter(comp_over, mem_save, s=200, c=[colors[i]], 
                       edgecolors='black', linewidth=2, alpha=0.7)
            plt.annotate(strategy, (comp_over, mem_save),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)
        
        # Add ideal trade-off line
        x_range = plt.xlim()
        y_range = plt.ylim()
        plt.plot([0, max(compute_overhead)], [0, max(memory_savings)],
                'k--', alpha=0.3, label='Linear trade-off')
        
        plt.xlabel('Compute Overhead (%)', fontsize=12)
        plt.ylabel('Memory Savings (%)', fontsize=12)
        plt.title('Memory-Compute Trade-off Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add quadrant labels
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_layer_memory_heatmap(
        profiler: MemoryProfiler,
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of memory usage by layer and phase.
        
        Args:
            profiler: Memory profiler with snapshots
            save_path: Optional path to save the plot
        """
        # Organize data by layer and phase
        data_dict = defaultdict(lambda: defaultdict(list))
        
        for snapshot in profiler.snapshots:
            if snapshot.layer_name:
                data_dict[snapshot.layer_name][snapshot.phase].append(snapshot.allocated_mb)
        
        # Create matrix for heatmap
        layers = sorted(data_dict.keys())
        phases = ['forward', 'backward']
        
        matrix = []
        for layer in layers:
            row = []
            for phase in phases:
                values = data_dict[layer][phase]
                row.append(np.mean(values) if values else 0)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Plot heatmap
        plt.figure(figsize=(8, max(6, len(layers) * 0.3)))
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=phases, yticklabels=layers,
                   cbar_kws={'label': 'Memory (MB)'})
        
        plt.title('Memory Usage by Layer and Phase', fontsize=14, fontweight='bold')
        plt.xlabel('Phase', fontsize=12)
        plt.ylabel('Layer', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_comparison_bars(
        profiles: List[ProfileResult],
        metrics: List[str] = ['peak_memory_mb', 'total_time_ms'],
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing different strategies.
        
        Args:
            profiles: List of profiling results
            metrics: Metrics to compare
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        strategies = [p.strategy for p in profiles]
        colors = sns.color_palette("husl", len(profiles))
        
        for ax, metric in zip(axes, metrics):
            values = [getattr(p, metric) for p in profiles]
            
            bars = ax.bar(strategies, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
            
            # Format axis
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(strategies) > 3:
                ax.set_xticklabels(strategies, rotation=45, ha='right')
        
        plt.suptitle('Strategy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def profile_checkpointing_strategies(
    model_factory: callable,
    input_shape: Tuple[int, ...],
    strategies: Dict[str, Dict[str, Any]],
    batch_size: int = 4,
    iterations: int = 5
) -> List[ProfileResult]:
    """
    Profile multiple checkpointing strategies.
    
    Args:
        model_factory: Function to create model
        input_shape: Shape of input tensor
        strategies: Dictionary of strategy configurations
        batch_size: Batch size for profiling
        iterations: Number of iterations to profile
    
    Returns:
        List of profiling results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    for strategy_name, config in strategies.items():
        print(f"\nProfiling {strategy_name}...")
        
        # Create model with strategy
        model = model_factory(**config)
        model = model.to(device)
        model.train()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Setup profiler
        profiler = MemoryProfiler(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Profile
        with profiler.profile(model):
            forward_times = []
            backward_times = []
            total_start = time.time()
            
            for i in range(iterations):
                # Clear cache
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Generate input
                x = torch.randn(batch_size, *input_shape, device=device, requires_grad=True)
                target = torch.randn_like(x)
                
                # Forward pass
                profiler.snapshot('forward', iteration=i)
                forward_start = time.time()
                output = model(x)
                loss = criterion(output, target)
                forward_time = (time.time() - forward_start) * 1000  # ms
                forward_times.append(forward_time)
                
                # Backward pass
                profiler.snapshot('backward', iteration=i)
                backward_start = time.time()
                loss.backward()
                backward_time = (time.time() - backward_start) * 1000  # ms
                backward_times.append(backward_time)
                
                # Optimizer step
                profiler.snapshot('optimizer', iteration=i)
                optimizer.step()
                optimizer.zero_grad()
            
            total_time = (time.time() - total_start) * 1000  # ms
        
        # Create result
        result = ProfileResult(
            model_name=model.__class__.__name__,
            strategy=strategy_name,
            snapshots=profiler.snapshots,
            peak_memory_mb=profiler.get_peak_memory(),
            average_memory_mb=profiler.get_average_memory(),
            forward_time_ms=np.mean(forward_times),
            backward_time_ms=np.mean(backward_times),
            total_time_ms=total_time,
            num_parameters=num_params,
            batch_size=batch_size,
            input_shape=input_shape
        )
        
        results.append(result)
        
        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  Peak memory: {result.peak_memory_mb:.2f} MB")
        print(f"  Average memory: {result.average_memory_mb:.2f} MB")
        print(f"  Forward time: {result.forward_time_ms:.2f} ms")
        print(f"  Backward time: {result.backward_time_ms:.2f} ms")
    
    return results


def demonstrate_profiling():
    """Demonstrate profiling and visualization capabilities."""
    print("=" * 80)
    print("Memory Profiling and Visualization Demo")
    print("=" * 80)
    
    # Define a simple test model
    class TestModel(nn.Module):
        def __init__(self, num_layers=6, use_checkpoint=False, checkpoint_freq=2):
            super().__init__()
            self.use_checkpoint = use_checkpoint
            self.checkpoint_freq = checkpoint_freq
            
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.LayerNorm(512)
                ) for _ in range(num_layers)
            ])
            
            self.output = nn.Linear(512, 512)
        
        def forward(self, x):
            from gradient_checkpointing import checkpoint
            
            for i, layer in enumerate(self.layers):
                if self.use_checkpoint and i % self.checkpoint_freq == 0 and self.training:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
            
            return self.output(x)
    
    # Define strategies
    strategies = {
        'No Checkpointing': {
            'use_checkpoint': False
        },
        'Full Checkpointing': {
            'use_checkpoint': True,
            'checkpoint_freq': 1
        },
        'Selective (every 2)': {
            'use_checkpoint': True,
            'checkpoint_freq': 2
        },
        'Selective (every 3)': {
            'use_checkpoint': True,
            'checkpoint_freq': 3
        }
    }
    
    # Profile strategies
    print("\nProfiling different strategies...")
    results = profile_checkpointing_strategies(
        model_factory=TestModel,
        input_shape=(512,),
        strategies=strategies,
        batch_size=8,
        iterations=5
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    visualizer = MemoryVisualizer()
    
    # Plot memory timeline
    visualizer.plot_memory_timeline(
        results,
        title="Memory Usage Over Time",
        save_path="memory_timeline.png"
    )
    
    # Plot trade-off analysis
    visualizer.plot_memory_vs_compute_tradeoff(
        results,
        save_path="memory_compute_tradeoff.png"
    )
    
    # Plot comparison bars
    visualizer.plot_comparison_bars(
        results,
        metrics=['peak_memory_mb', 'total_time_ms'],
        save_path="strategy_comparison.png"
    )
    
    print("\nVisualization complete! Plots saved as:")
    print("  - memory_timeline.png")
    print("  - memory_compute_tradeoff.png")
    print("  - strategy_comparison.png")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Peak Mem (MB)':<15} {'Avg Mem (MB)':<15} "
          f"{'Forward (ms)':<15} {'Backward (ms)':<15}")
    print("-" * 80)
    
    baseline = results[0]
    for result in results:
        mem_savings = (1 - result.peak_memory_mb/baseline.peak_memory_mb) * 100
        compute_overhead = (result.total_time_ms/baseline.total_time_ms - 1) * 100
        
        print(f"{result.strategy:<20} {result.peak_memory_mb:<15.2f} "
              f"{result.average_memory_mb:<15.2f} {result.forward_time_ms:<15.2f} "
              f"{result.backward_time_ms:<15.2f}")
        
        if result.strategy != baseline.strategy:
            print(f"{'':>20} {f'(-{mem_savings:.1f}%)':<15} "
                  f"{'':>15} {f'(+{compute_overhead:.1f}%)':<15}")


if __name__ == "__main__":
    demonstrate_profiling()