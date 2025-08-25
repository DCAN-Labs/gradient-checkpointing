"""
Optimal checkpoint selection using dynamic programming.

This module implements algorithms to find the optimal checkpoint locations
given a memory budget, minimizing recomputation while staying within memory constraints.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LayerProfile:
    """Profile information for a single layer."""
    layer_idx: int
    forward_compute_cost: float  # Time in milliseconds
    backward_compute_cost: float  # Time in milliseconds
    activation_memory: float  # Memory in MB
    parameter_memory: float  # Memory in MB
    name: str = ""


@dataclass
class CheckpointingPlan:
    """Optimal checkpointing plan."""
    checkpoint_layers: List[int]  # Indices of layers to checkpoint
    total_memory: float  # Total memory usage in MB
    total_compute: float  # Total compute time in ms
    memory_savings: float  # Memory saved compared to no checkpointing
    compute_overhead: float  # Additional compute time compared to no checkpointing


class OptimalCheckpointer:
    """
    Find optimal checkpoint locations using dynamic programming.
    
    The algorithm minimizes recomputation cost while staying within memory budget.
    """
    
    def __init__(self, layer_profiles: List[LayerProfile]):
        """
        Initialize optimal checkpointer.
        
        Args:
            layer_profiles: Profile information for each layer
        """
        self.layer_profiles = layer_profiles
        self.num_layers = len(layer_profiles)
        
        # Precompute cumulative costs
        self._precompute_costs()
    
    def _precompute_costs(self):
        """Precompute cumulative memory and compute costs."""
        self.cumulative_memory = np.zeros(self.num_layers + 1)
        self.cumulative_forward_compute = np.zeros(self.num_layers + 1)
        self.cumulative_backward_compute = np.zeros(self.num_layers + 1)
        
        for i, profile in enumerate(self.layer_profiles):
            self.cumulative_memory[i + 1] = (
                self.cumulative_memory[i] + profile.activation_memory
            )
            self.cumulative_forward_compute[i + 1] = (
                self.cumulative_forward_compute[i] + profile.forward_compute_cost
            )
            self.cumulative_backward_compute[i + 1] = (
                self.cumulative_backward_compute[i] + profile.backward_compute_cost
            )
        
        # Total costs without checkpointing
        self.total_memory_no_checkpoint = self.cumulative_memory[-1]
        self.total_compute_no_checkpoint = (
            self.cumulative_forward_compute[-1] + self.cumulative_backward_compute[-1]
        )
    
    def find_optimal_checkpoints(
        self, 
        memory_budget: float,
        max_checkpoints: Optional[int] = None
    ) -> CheckpointingPlan:
        """
        Find optimal checkpoint locations using dynamic programming.
        
        Args:
            memory_budget: Maximum memory budget in MB
            max_checkpoints: Maximum number of checkpoints allowed
        
        Returns:
            Optimal checkpointing plan
        """
        if max_checkpoints is None:
            max_checkpoints = self.num_layers
        
        # DP state: dp[i][k] = (min_compute, checkpoints)
        # i = layer index, k = number of checkpoints used
        dp = {}
        
        def solve(layer_idx: int, checkpoints_left: int, memory_used: float) -> Tuple[float, List[int]]:
            """
            Recursive DP with memoization.
            
            Returns:
                (total_compute_cost, checkpoint_locations)
            """
            # Base case: processed all layers
            if layer_idx >= self.num_layers:
                return 0.0, []
            
            # Check if we've seen this state
            state = (layer_idx, checkpoints_left, int(memory_used))
            if state in dp:
                return dp[state]
            
            best_cost = float('inf')
            best_checkpoints = []
            
            # Option 1: Don't checkpoint at this layer
            if memory_used + self.layer_profiles[layer_idx].activation_memory <= memory_budget:
                next_memory = memory_used + self.layer_profiles[layer_idx].activation_memory
                cost, checkpoints = solve(layer_idx + 1, checkpoints_left, next_memory)
                total_cost = cost + self.layer_profiles[layer_idx].forward_compute_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_checkpoints = checkpoints
            
            # Option 2: Checkpoint at this layer (if we have checkpoints left)
            if checkpoints_left > 0:
                # Checkpointing means we don't store activations but need to recompute
                recompute_cost = self.layer_profiles[layer_idx].forward_compute_cost
                cost, checkpoints = solve(layer_idx + 1, checkpoints_left - 1, memory_used)
                total_cost = cost + self.layer_profiles[layer_idx].forward_compute_cost + recompute_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_checkpoints = [layer_idx] + checkpoints
            
            dp[state] = (best_cost, best_checkpoints)
            return best_cost, best_checkpoints
        
        # Find optimal solution
        total_compute, checkpoint_layers = solve(0, max_checkpoints, 0)
        
        # Calculate actual memory usage with checkpointing
        total_memory = sum(
            profile.parameter_memory for profile in self.layer_profiles
        )
        
        # Add activation memory for non-checkpointed layers
        for i, profile in enumerate(self.layer_profiles):
            if i not in checkpoint_layers:
                total_memory += profile.activation_memory
        
        # Create plan
        plan = CheckpointingPlan(
            checkpoint_layers=checkpoint_layers,
            total_memory=total_memory,
            total_compute=total_compute,
            memory_savings=self.total_memory_no_checkpoint - total_memory,
            compute_overhead=total_compute - self.total_compute_no_checkpoint
        )
        
        return plan
    
    def find_optimal_k_checkpoints(self, k: int) -> CheckpointingPlan:
        """
        Find optimal locations for exactly k checkpoints.
        
        This uses a different DP formulation optimized for fixed k.
        
        Args:
            k: Number of checkpoints to place
        
        Returns:
            Optimal checkpointing plan
        """
        if k >= self.num_layers:
            # Checkpoint everything
            return CheckpointingPlan(
                checkpoint_layers=list(range(self.num_layers)),
                total_memory=sum(p.parameter_memory for p in self.layer_profiles),
                total_compute=self.total_compute_no_checkpoint + self.cumulative_forward_compute[-1],
                memory_savings=self.total_memory_no_checkpoint,
                compute_overhead=self.cumulative_forward_compute[-1]
            )
        
        # DP: dp[i][j] = minimum memory to process layers 0..i with j checkpoints
        dp = np.full((self.num_layers + 1, k + 1), float('inf'))
        parent = {}
        
        # Base case
        dp[0][0] = 0
        
        for i in range(1, self.num_layers + 1):
            for j in range(min(i, k) + 1):
                # Option 1: Don't checkpoint layer i-1
                if j <= i - 1:
                    mem_cost = dp[i-1][j] + self.layer_profiles[i-1].activation_memory
                    if mem_cost < dp[i][j]:
                        dp[i][j] = mem_cost
                        parent[(i, j)] = (i-1, j, False)
                
                # Option 2: Checkpoint layer i-1
                if j > 0 and j - 1 <= i - 1:
                    mem_cost = dp[i-1][j-1]  # No activation memory needed
                    if mem_cost < dp[i][j]:
                        dp[i][j] = mem_cost
                        parent[(i, j)] = (i-1, j-1, True)
        
        # Reconstruct solution
        checkpoint_layers = []
        i, j = self.num_layers, k
        
        while i > 0 and (i, j) in parent:
            prev_i, prev_j, is_checkpoint = parent[(i, j)]
            if is_checkpoint:
                checkpoint_layers.append(prev_i)
            i, j = prev_i, prev_j
        
        checkpoint_layers.reverse()
        
        # Calculate costs
        total_memory = dp[self.num_layers][k] + sum(
            p.parameter_memory for p in self.layer_profiles
        )
        
        # Compute overhead from recomputation
        compute_overhead = sum(
            self.layer_profiles[idx].forward_compute_cost 
            for idx in checkpoint_layers
        )
        
        return CheckpointingPlan(
            checkpoint_layers=checkpoint_layers,
            total_memory=total_memory,
            total_compute=self.total_compute_no_checkpoint + compute_overhead,
            memory_savings=self.total_memory_no_checkpoint - total_memory,
            compute_overhead=compute_overhead
        )


class SegmentedCheckpointing:
    """
    Segment-based checkpointing strategy.
    
    Divides the model into segments and checkpoints between segments.
    This is simpler than full DP but still effective.
    """
    
    @staticmethod
    def compute_optimal_segments(
        num_layers: int,
        memory_budget_ratio: float = 0.5
    ) -> int:
        """
        Compute optimal number of segments based on memory budget.
        
        The optimal number of segments k minimizes:
        - Memory: O(n/k) where n is number of layers
        - Compute: O(n + n/k) due to recomputation
        
        Args:
            num_layers: Total number of layers
            memory_budget_ratio: Fraction of full memory to use (0-1)
        
        Returns:
            Optimal number of segments
        """
        # Theoretical optimal: k = sqrt(n * (1 - memory_budget_ratio))
        optimal_k = int(np.sqrt(num_layers * (1 - memory_budget_ratio)))
        return max(1, min(num_layers, optimal_k))
    
    @staticmethod
    def get_segment_boundaries(num_layers: int, num_segments: int) -> List[int]:
        """
        Get layer indices that mark segment boundaries.
        
        Args:
            num_layers: Total number of layers
            num_segments: Number of segments
        
        Returns:
            List of layer indices to checkpoint
        """
        if num_segments <= 1:
            return []
        
        segment_size = num_layers / num_segments
        boundaries = []
        
        for i in range(1, num_segments):
            boundary = int(i * segment_size)
            if boundary < num_layers:
                boundaries.append(boundary)
        
        return boundaries


def profile_model(model: nn.Module, input_shape: Tuple[int, ...]) -> List[LayerProfile]:
    """
    Profile a model to get compute and memory costs per layer.
    
    Args:
        model: The model to profile
        input_shape: Shape of input tensor
    
    Returns:
        List of layer profiles
    """
    profiles = []
    device = next(model.parameters()).device
    
    # Create dummy input
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    
    # Hook to capture activations
    activation_sizes = {}
    handles = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_sizes[name] = output.numel() * output.element_size() / 1024 / 1024  # MB
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)
    
    # Forward pass to collect activation sizes
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Create profiles
    for i, (name, module) in enumerate(model.named_modules()):
        if len(list(module.children())) == 0:  # Leaf modules only
            # Calculate parameter memory
            param_memory = sum(
                p.numel() * p.element_size() for p in module.parameters()
            ) / 1024 / 1024  # MB
            
            # Get activation memory
            activation_memory = activation_sizes.get(name, 0.0)
            
            # Estimate compute cost (simplified)
            forward_cost = activation_memory * 0.1  # Rough estimate
            backward_cost = forward_cost * 2  # Backward typically 2x forward
            
            profile = LayerProfile(
                layer_idx=i,
                forward_compute_cost=forward_cost,
                backward_compute_cost=backward_cost,
                activation_memory=activation_memory,
                parameter_memory=param_memory,
                name=name
            )
            profiles.append(profile)
    
    return profiles


def demonstrate_optimal_checkpointing():
    """Demonstrate optimal checkpointing on a sample model."""
    print("=" * 80)
    print("Optimal Checkpointing Demonstration")
    print("=" * 80)
    
    # Create sample layer profiles
    num_layers = 12
    profiles = []
    
    for i in range(num_layers):
        # Simulate varying layer costs
        if i % 3 == 0:  # Every 3rd layer is more expensive
            activation_memory = 100.0  # MB
            forward_cost = 50.0  # ms
        else:
            activation_memory = 50.0  # MB
            forward_cost = 20.0  # ms
        
        profile = LayerProfile(
            layer_idx=i,
            forward_compute_cost=forward_cost,
            backward_compute_cost=forward_cost * 2,
            activation_memory=activation_memory,
            parameter_memory=10.0,
            name=f"layer_{i}"
        )
        profiles.append(profile)
    
    # Create optimizer
    optimizer = OptimalCheckpointer(profiles)
    
    print(f"\nModel Configuration:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total activation memory: {optimizer.total_memory_no_checkpoint:.1f} MB")
    print(f"  Total compute time: {optimizer.total_compute_no_checkpoint:.1f} ms")
    
    # Test different memory budgets
    print(f"\n{'-'*80}")
    print("Testing different memory budgets:")
    print(f"{'-'*80}")
    
    for budget_ratio in [0.3, 0.5, 0.7]:
        memory_budget = optimizer.total_memory_no_checkpoint * budget_ratio
        plan = optimizer.find_optimal_checkpoints(memory_budget)
        
        print(f"\nMemory budget: {budget_ratio*100:.0f}% ({memory_budget:.1f} MB)")
        print(f"  Checkpoint layers: {plan.checkpoint_layers}")
        print(f"  Total memory: {plan.total_memory:.1f} MB")
        print(f"  Memory savings: {plan.memory_savings:.1f} MB ({plan.memory_savings/optimizer.total_memory_no_checkpoint*100:.1f}%)")
        print(f"  Compute overhead: {plan.compute_overhead:.1f} ms ({plan.compute_overhead/optimizer.total_compute_no_checkpoint*100:.1f}%)")
    
    # Test fixed number of checkpoints
    print(f"\n{'-'*80}")
    print("Testing fixed number of checkpoints:")
    print(f"{'-'*80}")
    
    for k in [2, 4, 6]:
        plan = optimizer.find_optimal_k_checkpoints(k)
        
        print(f"\nWith {k} checkpoints:")
        print(f"  Checkpoint layers: {plan.checkpoint_layers}")
        print(f"  Total memory: {plan.total_memory:.1f} MB")
        print(f"  Memory savings: {plan.memory_savings:.1f} MB ({plan.memory_savings/optimizer.total_memory_no_checkpoint*100:.1f}%)")
        print(f"  Compute overhead: {plan.compute_overhead:.1f} ms ({plan.compute_overhead/optimizer.total_compute_no_checkpoint*100:.1f}%)")
    
    # Test segmented approach
    print(f"\n{'-'*80}")
    print("Segmented checkpointing approach:")
    print(f"{'-'*80}")
    
    for memory_ratio in [0.3, 0.5, 0.7]:
        num_segments = SegmentedCheckpointing.compute_optimal_segments(num_layers, memory_ratio)
        boundaries = SegmentedCheckpointing.get_segment_boundaries(num_layers, num_segments)
        
        print(f"\nMemory budget ratio: {memory_ratio}")
        print(f"  Optimal segments: {num_segments}")
        print(f"  Checkpoint at layers: {boundaries}")


if __name__ == "__main__":
    demonstrate_optimal_checkpointing()