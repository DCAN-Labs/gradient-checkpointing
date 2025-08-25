"""
Optimal checkpoint selection for 3D medical imaging models using dynamic programming.

This module implements algorithms to find optimal checkpoint locations in 3D U-Net,
V-Net, and other medical imaging architectures, minimizing recomputation while 
staying within GPU memory constraints when processing large 3D MRI/CT volumes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MedicalLayerProfile:
    """Profile information for a single layer in a medical imaging model."""
    layer_idx: int
    forward_compute_cost: float  # Time in milliseconds
    backward_compute_cost: float  # Time in milliseconds
    activation_memory: float  # Memory in MB (critical for 3D volumes)
    parameter_memory: float  # Memory in MB
    volume_dimensions: Optional[Tuple[int, int, int]] = None  # (D, H, W) for 3D
    layer_type: str = ""  # e.g., "conv3d", "pooling", "upsampling"
    name: str = ""


@dataclass
class MedicalCheckpointingPlan:
    """Optimal checkpointing plan for medical imaging models."""
    checkpoint_layers: List[int]  # Indices of layers to checkpoint
    total_memory: float  # Total memory usage in MB
    total_compute: float  # Total compute time in ms
    memory_savings: float  # Memory saved compared to no checkpointing
    compute_overhead: float  # Additional compute time
    estimated_batch_size: int = 1  # Maximum batch size with this plan


class OptimalMedicalCheckpointer:
    """
    Find optimal checkpoint locations for 3D medical imaging models.
    
    Optimized for architectures processing large 3D volumes like:
    - 3D U-Net for brain MRI segmentation
    - V-Net for volumetric medical image segmentation
    - nnU-Net for various medical imaging tasks
    """
    
    def __init__(self, layer_profiles: List[MedicalLayerProfile]):
        """
        Initialize optimal checkpointer for medical imaging.
        
        Args:
            layer_profiles: Profile information for each layer
        """
        self.layer_profiles = layer_profiles
        self.num_layers = len(layer_profiles)
        
        # Precompute cumulative costs
        self._precompute_costs()
        
        # Identify critical layers (e.g., bottleneck in U-Net)
        self._identify_critical_layers()
    
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
    
    def _identify_critical_layers(self):
        """Identify critical layers for medical imaging models."""
        self.critical_layers = []
        
        for i, profile in enumerate(self.layer_profiles):
            # Critical layers are those with high memory usage (e.g., encoder outputs)
            if profile.activation_memory > np.mean([p.activation_memory for p in self.layer_profiles]) * 1.5:
                self.critical_layers.append(i)
            
            # Also mark transition layers (e.g., pooling, upsampling)
            if profile.layer_type in ["pooling", "upsampling", "maxpool3d", "upsample3d"]:
                if i not in self.critical_layers:
                    self.critical_layers.append(i)
    
    def find_optimal_checkpoints_for_volume(
        self, 
        volume_size: Tuple[int, int, int],
        gpu_memory_gb: float,
        batch_size: int = 1
    ) -> MedicalCheckpointingPlan:
        """
        Find optimal checkpoints for a specific 3D volume size.
        
        Args:
            volume_size: (D, H, W) dimensions of the 3D volume
            gpu_memory_gb: Available GPU memory in GB
            batch_size: Desired batch size
        
        Returns:
            Optimal checkpointing plan
        """
        # Convert GB to MB
        memory_budget = gpu_memory_gb * 1024
        
        # Reserve memory for model parameters and optimizer states
        parameter_memory = sum(p.parameter_memory for p in self.layer_profiles)
        optimizer_memory = parameter_memory * 2  # Rough estimate for Adam
        available_memory = memory_budget - parameter_memory - optimizer_memory
        
        # Adjust for batch size
        per_sample_budget = available_memory / batch_size
        
        return self.find_optimal_checkpoints(per_sample_budget, prioritize_critical=True)
    
    def find_optimal_checkpoints(
        self, 
        memory_budget: float,
        max_checkpoints: Optional[int] = None,
        prioritize_critical: bool = True
    ) -> MedicalCheckpointingPlan:
        """
        Find optimal checkpoint locations using dynamic programming.
        
        Args:
            memory_budget: Maximum memory budget in MB
            max_checkpoints: Maximum number of checkpoints allowed
            prioritize_critical: Whether to prioritize critical layers
        
        Returns:
            Optimal checkpointing plan for medical imaging
        """
        if max_checkpoints is None:
            max_checkpoints = self.num_layers
        
        # DP state: dp[i][k] = (min_compute, checkpoints)
        dp = {}
        
        def solve(layer_idx: int, checkpoints_left: int, memory_used: float) -> Tuple[float, List[int]]:
            """
            Recursive DP with memoization, optimized for medical imaging.
            """
            if layer_idx >= self.num_layers:
                return 0.0, []
            
            state = (layer_idx, checkpoints_left, int(memory_used))
            if state in dp:
                return dp[state]
            
            best_cost = float('inf')
            best_checkpoints = []
            
            profile = self.layer_profiles[layer_idx]
            
            # Option 1: Don't checkpoint at this layer
            if memory_used + profile.activation_memory <= memory_budget:
                next_memory = memory_used + profile.activation_memory
                cost, checkpoints = solve(layer_idx + 1, checkpoints_left, next_memory)
                total_cost = cost + profile.forward_compute_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_checkpoints = checkpoints
            
            # Option 2: Checkpoint at this layer
            if checkpoints_left > 0:
                # Bonus for checkpointing critical layers
                bonus = 0
                if prioritize_critical and layer_idx in self.critical_layers:
                    bonus = -profile.forward_compute_cost * 0.1  # 10% bonus
                
                recompute_cost = profile.forward_compute_cost
                cost, checkpoints = solve(layer_idx + 1, checkpoints_left - 1, memory_used)
                total_cost = cost + profile.forward_compute_cost + recompute_cost + bonus
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_checkpoints = [layer_idx] + checkpoints
            
            dp[state] = (best_cost, best_checkpoints)
            return best_cost, best_checkpoints
        
        # Find optimal solution
        total_compute, checkpoint_layers = solve(0, max_checkpoints, 0)
        
        # Calculate actual memory usage
        total_memory = sum(profile.parameter_memory for profile in self.layer_profiles)
        
        for i, profile in enumerate(self.layer_profiles):
            if i not in checkpoint_layers:
                total_memory += profile.activation_memory
        
        # Estimate maximum batch size
        estimated_batch_size = max(1, int(memory_budget / total_memory))
        
        plan = MedicalCheckpointingPlan(
            checkpoint_layers=checkpoint_layers,
            total_memory=total_memory,
            total_compute=total_compute,
            memory_savings=self.total_memory_no_checkpoint - total_memory,
            compute_overhead=total_compute - self.total_compute_no_checkpoint,
            estimated_batch_size=estimated_batch_size
        )
        
        return plan
    
    def find_unet_optimal_checkpoints(self) -> MedicalCheckpointingPlan:
        """
        Find optimal checkpoints specifically for U-Net architecture.
        
        U-Net has encoder-decoder structure with skip connections.
        Optimal strategy often involves checkpointing at:
        - Encoder-decoder boundary (bottleneck)
        - After each downsampling
        - Before each upsampling
        """
        checkpoint_layers = []
        
        # Find encoder-decoder boundary (typically middle of the network)
        bottleneck_idx = self.num_layers // 2
        checkpoint_layers.append(bottleneck_idx)
        
        # Add checkpoints at pooling/upsampling layers
        for i, profile in enumerate(self.layer_profiles):
            if profile.layer_type in ["pooling", "upsampling", "maxpool3d", "upsample3d"]:
                if i not in checkpoint_layers:
                    checkpoint_layers.append(i)
        
        checkpoint_layers.sort()
        
        # Calculate costs
        total_memory = sum(p.parameter_memory for p in self.layer_profiles)
        compute_overhead = 0
        
        for i, profile in enumerate(self.layer_profiles):
            if i not in checkpoint_layers:
                total_memory += profile.activation_memory
            else:
                compute_overhead += profile.forward_compute_cost
        
        return MedicalCheckpointingPlan(
            checkpoint_layers=checkpoint_layers,
            total_memory=total_memory,
            total_compute=self.total_compute_no_checkpoint + compute_overhead,
            memory_savings=self.total_memory_no_checkpoint - total_memory,
            compute_overhead=compute_overhead,
            estimated_batch_size=1
        )


class VolumetricSegmentedCheckpointing:
    """
    Segment-based checkpointing strategy for 3D volumetric medical data.
    
    Optimized for processing large 3D MRI/CT volumes with limited GPU memory.
    """
    
    @staticmethod
    def compute_optimal_segments_for_volume(
        volume_size: Tuple[int, int, int],
        num_layers: int,
        gpu_memory_gb: float
    ) -> int:
        """
        Compute optimal number of segments for 3D medical volumes.
        
        Args:
            volume_size: (D, H, W) dimensions of 3D volume
            num_layers: Total number of layers in the model
            gpu_memory_gb: Available GPU memory in GB
        
        Returns:
            Optimal number of segments
        """
        # Calculate volume memory footprint
        voxels = np.prod(volume_size)
        volume_memory_gb = (voxels * 4 * 32) / (1024**3)  # Assuming 32 channels, float32
        
        # Memory pressure ratio
        memory_pressure = volume_memory_gb / gpu_memory_gb
        
        if memory_pressure > 0.7:
            # High memory pressure - aggressive checkpointing
            optimal_k = int(np.sqrt(num_layers * 2))
        elif memory_pressure > 0.4:
            # Moderate memory pressure
            optimal_k = int(np.sqrt(num_layers))
        else:
            # Low memory pressure
            optimal_k = int(np.sqrt(num_layers * 0.5))
        
        return max(1, min(num_layers, optimal_k))
    
    @staticmethod
    def get_medical_architecture_segments(
        architecture: str,
        num_layers: int
    ) -> List[int]:
        """
        Get checkpoint locations based on medical imaging architecture.
        
        Args:
            architecture: Architecture name ("unet", "vnet", "nnunet")
            num_layers: Total number of layers
        
        Returns:
            List of layer indices to checkpoint
        """
        if architecture.lower() == "unet":
            # U-Net: checkpoint at each resolution change
            # Typically 4-5 resolution levels
            num_levels = 5
            segment_size = num_layers // (num_levels * 2)  # encoder + decoder
            return [i * segment_size for i in range(1, num_levels * 2)]
        
        elif architecture.lower() == "vnet":
            # V-Net: similar to U-Net but with residual connections
            # Checkpoint after each residual block
            num_blocks = num_layers // 4  # Assuming 4 layers per block
            return [i * 4 for i in range(1, num_blocks)]
        
        elif architecture.lower() == "nnunet":
            # nnU-Net: adaptive architecture
            # Use aggressive checkpointing due to large volumes
            return list(range(2, num_layers, 3))
        
        else:
            # Default: uniform distribution
            num_segments = int(np.sqrt(num_layers))
            segment_size = num_layers // num_segments
            return [i * segment_size for i in range(1, num_segments)]


def profile_medical_model(
    model: nn.Module, 
    volume_shape: Tuple[int, int, int, int]
) -> List[MedicalLayerProfile]:
    """
    Profile a 3D medical imaging model for memory and compute costs.
    
    Args:
        model: The 3D medical imaging model (U-Net, V-Net, etc.)
        volume_shape: Shape of input volume (B, C, D, H, W)
    
    Returns:
        List of layer profiles
    """
    profiles = []
    device = next(model.parameters()).device if torch.cuda.is_available() else 'cpu'
    
    # Create dummy 3D volume
    x = torch.randn(*volume_shape, device=device, requires_grad=True)
    
    # Hook to capture activations
    activation_sizes = {}
    handles = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Calculate memory for 3D tensors
                memory_mb = output.numel() * output.element_size() / (1024 * 1024)
                activation_sizes[name] = memory_mb
                
                # Store volume dimensions if 5D tensor (batch, channel, depth, height, width)
                if output.dim() == 5:
                    activation_sizes[f"{name}_dims"] = output.shape[2:]
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
            ) / (1024 * 1024)  # MB
            
            # Get activation memory
            activation_memory = activation_sizes.get(name, 0.0)
            
            # Get volume dimensions if available
            volume_dims = activation_sizes.get(f"{name}_dims", None)
            
            # Determine layer type
            layer_type = module.__class__.__name__.lower()
            
            # Estimate compute cost (more accurate for 3D operations)
            if "conv3d" in layer_type:
                forward_cost = activation_memory * 0.5  # 3D convolutions are expensive
            elif "pool" in layer_type or "upsamp" in layer_type:
                forward_cost = activation_memory * 0.1  # Pooling/upsampling is cheaper
            else:
                forward_cost = activation_memory * 0.2
            
            backward_cost = forward_cost * 2.5  # Backward pass for 3D is expensive
            
            profile = MedicalLayerProfile(
                layer_idx=i,
                forward_compute_cost=forward_cost,
                backward_compute_cost=backward_cost,
                activation_memory=activation_memory,
                parameter_memory=param_memory,
                volume_dimensions=volume_dims,
                layer_type=layer_type,
                name=name
            )
            profiles.append(profile)
    
    return profiles


def demonstrate_medical_checkpointing():
    """Demonstrate optimal checkpointing for 3D medical imaging models."""
    print("=" * 80)
    print("Optimal Checkpointing for 3D Medical Imaging")
    print("=" * 80)
    
    # Simulate a 3D U-Net architecture for brain MRI segmentation
    num_layers = 23  # Typical U-Net depth
    profiles = []
    
    # Encoder path (downsampling)
    for i in range(0, 11):
        if i % 3 == 0:  # Pooling layers
            activation_memory = 50.0 * (2 ** (i // 3))  # Memory increases with depth
            forward_cost = 10.0
            layer_type = "maxpool3d"
        else:  # Conv layers
            activation_memory = 100.0 * (2 ** (i // 3))
            forward_cost = 50.0 * (2 ** (i // 3))
            layer_type = "conv3d"
        
        profile = MedicalLayerProfile(
            layer_idx=i,
            forward_compute_cost=forward_cost,
            backward_compute_cost=forward_cost * 2.5,
            activation_memory=activation_memory,
            parameter_memory=20.0,
            layer_type=layer_type,
            name=f"encoder_{i}"
        )
        profiles.append(profile)
    
    # Bottleneck
    profile = MedicalLayerProfile(
        layer_idx=11,
        forward_compute_cost=100.0,
        backward_compute_cost=250.0,
        activation_memory=200.0,
        parameter_memory=40.0,
        layer_type="conv3d",
        name="bottleneck"
    )
    profiles.append(profile)
    
    # Decoder path (upsampling)
    for i in range(12, 23):
        if (i - 12) % 3 == 0:  # Upsampling layers
            activation_memory = 50.0 * (2 ** ((22 - i) // 3))
            forward_cost = 15.0
            layer_type = "upsample3d"
        else:  # Conv layers
            activation_memory = 100.0 * (2 ** ((22 - i) // 3))
            forward_cost = 50.0 * (2 ** ((22 - i) // 3))
            layer_type = "conv3d"
        
        profile = MedicalLayerProfile(
            layer_idx=i,
            forward_compute_cost=forward_cost,
            backward_compute_cost=forward_cost * 2.5,
            activation_memory=activation_memory,
            parameter_memory=20.0,
            layer_type=layer_type,
            name=f"decoder_{i}"
        )
        profiles.append(profile)
    
    # Create optimizer
    optimizer = OptimalMedicalCheckpointer(profiles)
    
    print(f"\n3D U-Net Model Configuration:")
    print(f"  Architecture: U-Net for brain MRI segmentation")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total activation memory: {optimizer.total_memory_no_checkpoint:.1f} MB")
    print(f"  Total compute time: {optimizer.total_compute_no_checkpoint:.1f} ms")
    
    # Test for different GPU configurations
    print(f"\n{'-'*80}")
    print("Optimization for different GPU memory configurations:")
    print(f"{'-'*80}")
    
    gpu_configs = [
        (8, "RTX 3070 (8GB)"),
        (16, "V100 (16GB)"),
        (24, "RTX 3090 (24GB)"),
        (40, "A100 (40GB)")
    ]
    
    for gpu_memory, gpu_name in gpu_configs:
        print(f"\n{gpu_name}:")
        
        # Test different volume sizes
        volume_sizes = [
            ((128, 128, 128), "Standard brain MRI"),
            ((256, 256, 128), "High-res brain MRI"),
            ((512, 512, 64), "Whole-body CT slice")
        ]
        
        for volume_size, description in volume_sizes:
            plan = optimizer.find_optimal_checkpoints_for_volume(
                volume_size, gpu_memory, batch_size=1
            )
            
            print(f"\n  {description} {volume_size}:")
            print(f"    Checkpoint layers: {plan.checkpoint_layers[:5]}..." if len(plan.checkpoint_layers) > 5 else f"    Checkpoint layers: {plan.checkpoint_layers}")
            print(f"    Memory usage: {plan.total_memory:.1f} MB")
            print(f"    Memory savings: {plan.memory_savings:.1f} MB ({plan.memory_savings/optimizer.total_memory_no_checkpoint*100:.1f}%)")
            print(f"    Compute overhead: {plan.compute_overhead:.1f} ms ({plan.compute_overhead/optimizer.total_compute_no_checkpoint*100:.1f}%)")
            print(f"    Estimated max batch size: {plan.estimated_batch_size}")
    
    # Test U-Net specific optimization
    print(f"\n{'-'*80}")
    print("U-Net specific optimization:")
    print(f"{'-'*80}")
    
    unet_plan = optimizer.find_unet_optimal_checkpoints()
    print(f"\nU-Net optimized checkpointing:")
    print(f"  Checkpoint layers: {unet_plan.checkpoint_layers}")
    print(f"  Total memory: {unet_plan.total_memory:.1f} MB")
    print(f"  Memory savings: {unet_plan.memory_savings:.1f} MB")
    print(f"  Compute overhead: {unet_plan.compute_overhead:.1f} ms")
    
    # Test architecture-specific segmentation
    print(f"\n{'-'*80}")
    print("Architecture-specific segmented checkpointing:")
    print(f"{'-'*80}")
    
    for arch in ["unet", "vnet", "nnunet"]:
        segments = VolumetricSegmentedCheckpointing.get_medical_architecture_segments(
            arch, num_layers
        )
        print(f"\n{arch.upper()} checkpoints: {segments}")


if __name__ == "__main__":
    demonstrate_medical_checkpointing()