"""
Gradient checkpointing implementation for 3D medical imaging with PyTorch.

This module demonstrates memory-efficient training for 3D MRI segmentation and
classification models by selectively storing activations during forward pass
and recomputing them during backward pass.
"""

import torch
import torch.nn as nn
from typing import Callable, Any, Tuple, List
import numpy as np
import weakref


class CheckpointFunction(torch.autograd.Function):
    """Custom autograd function for gradient checkpointing in medical imaging models."""
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        """
        Forward pass: run the function but don't save intermediate activations.
        Critical for 3D medical imaging where activation tensors can be very large.
        
        Args:
            ctx: Context object for storing information for backward pass
            run_function: The function to checkpoint (e.g., 3D convolution blocks)
            preserve_rng_state: Whether to preserve RNG state for deterministic recomputation
            *args: Arguments to pass to run_function (typically 3D volume tensors)
        """
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        
        # Save non-tensor inputs
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        
        # Save RNG state if needed (important for dropout in medical models)
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.cuda_rng_state = torch.cuda.get_rng_state()
        
        # Save tensors for backward (only input tensors, not intermediate activations)
        ctx.save_for_backward(*tensor_inputs)
        
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass: recompute forward pass to get intermediate activations,
        then compute gradients. Essential for fitting large 3D volumes in GPU memory.
        """
        # Retrieve saved tensors
        tensor_inputs = ctx.saved_tensors
        
        # Reconstruct full arguments list
        inputs = list(ctx.inputs)
        for i, tensor_idx in enumerate(ctx.tensor_indices):
            inputs[tensor_idx] = tensor_inputs[i]
        
        # Restore RNG state if needed
        if ctx.preserve_rng_state:
            rng_cpu_state = torch.get_rng_state()
            torch.set_rng_state(ctx.cpu_rng_state)
            if torch.cuda.is_available():
                rng_cuda_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
        
        # Recompute forward pass with gradient tracking enabled
        with torch.enable_grad():
            # Detach inputs and make them require grad
            detached_inputs = []
            for inp in inputs:
                if torch.is_tensor(inp):
                    inp = inp.detach()
                    inp.requires_grad = True
                detached_inputs.append(inp)
            
            outputs = ctx.run_function(*detached_inputs)
        
        # Compute gradients
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        # Filter out None gradients
        filtered_grad_outputs = []
        for grad in grad_outputs:
            if grad is not None:
                filtered_grad_outputs.append(grad)
        
        torch.autograd.backward(outputs, filtered_grad_outputs)
        
        # Restore RNG state
        if ctx.preserve_rng_state:
            torch.set_rng_state(rng_cpu_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(rng_cuda_state)
        
        # Collect gradients for inputs
        grads = []
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                grads.append(inp.grad)
            else:
                grads.append(None)
        
        return None, None, *grads


def checkpoint(function: Callable[..., Any], *args, preserve_rng_state: bool = True) -> Any:
    """
    Checkpoint a function to save memory during 3D medical image processing.
    
    Particularly useful for 3D U-Net, V-Net, and other volumetric architectures
    where intermediate feature maps can consume significant GPU memory.
    
    Args:
        function: The function to checkpoint (e.g., encoder/decoder blocks)
        *args: Arguments to pass to the function (3D volume tensors)
        preserve_rng_state: Whether to preserve RNG state for deterministic behavior
    
    Returns:
        Output of the function
    """
    return CheckpointFunction.apply(function, preserve_rng_state, *args)


class CheckpointedMedicalSequential(nn.Module):
    """
    Sequential container with gradient checkpointing for 3D medical imaging models.
    Optimized for architectures like 3D U-Net, V-Net, and nnU-Net.
    """
    
    def __init__(self, *modules, checkpoint_segments: int = 1):
        """
        Initialize checkpointed sequential container for medical imaging.
        
        Args:
            *modules: Modules to run sequentially (e.g., 3D conv blocks)
            checkpoint_segments: Number of segments to checkpoint 
                                (1 = full checkpointing, best for large 3D volumes)
        """
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.checkpoint_segments = checkpoint_segments
    
    def forward(self, x):
        """Forward pass with checkpointing for 3D medical volumes."""
        if not self.training or self.checkpoint_segments == 0:
            # No checkpointing during evaluation or if disabled
            for module in self.modules_list:
                x = module(x)
            return x
        
        # Divide modules into segments for checkpointing
        modules = list(self.modules_list)
        segment_size = len(modules) // self.checkpoint_segments
        
        for i in range(0, len(modules), segment_size):
            segment = modules[i:i + segment_size]
            if len(segment) == 0:
                break
            
            # Create a function that runs this segment
            def run_segment(x, segment=segment):
                for module in segment:
                    x = module(x)
                return x
            
            # Checkpoint this segment
            x = checkpoint(run_segment, x)
        
        return x


class SelectiveCheckpointMedical:
    """
    Selective checkpointing for 3D medical imaging models.
    
    Allows fine-grained control over which layers to checkpoint,
    essential for balancing memory usage and computation in 3D CNNs.
    """
    
    def __init__(self, model: nn.Module, checkpoint_layers: List[int] = None):
        """
        Initialize selective checkpointing for medical imaging models.
        
        Args:
            model: The 3D medical imaging model (U-Net, V-Net, etc.)
            checkpoint_layers: List of layer indices to checkpoint
                             (typically deeper layers with larger feature maps)
        """
        self.model = model
        self.checkpoint_layers = checkpoint_layers or []
        self._original_forwards = {}
        self._setup_checkpointing()
    
    def _setup_checkpointing(self):
        """Wrap specified layers with checkpointing."""
        layers = list(self.model.modules())
        
        for idx in self.checkpoint_layers:
            if idx < len(layers):
                layer = layers[idx]
                # Store original forward
                self._original_forwards[idx] = layer.forward
                
                # Create checkpointed forward
                def make_checkpointed_forward(original_forward):
                    def checkpointed_forward(self, *args, **kwargs):
                        if self.training:
                            # Use checkpoint during training
                            return checkpoint(lambda *a: original_forward(*a, **kwargs), *args)
                        else:
                            # No checkpointing during evaluation
                            return original_forward(*args, **kwargs)
                    return checkpointed_forward
                
                # Replace forward method
                import types
                layer.forward = types.MethodType(
                    make_checkpointed_forward(self._original_forwards[idx]), 
                    layer
                )
    
    def restore(self):
        """Restore original forward methods."""
        layers = list(self.model.modules())
        for idx, original_forward in self._original_forwards.items():
            if idx < len(layers):
                layers[idx].forward = original_forward


def memory_efficient_medical_training(
    model: nn.Module,
    data_loader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 1,
    checkpoint_segments: int = 0,
    mixed_precision: bool = True
):
    """
    Memory-efficient training loop for 3D medical imaging models.
    
    Designed for training on large 3D MRI/CT volumes with limited GPU memory.
    Combines gradient accumulation, checkpointing, and mixed precision.
    
    Args:
        model: 3D medical imaging model (U-Net, V-Net, etc.)
        data_loader: DataLoader for 3D medical volumes
        loss_fn: Loss function (Dice, Cross-Entropy, etc.)
        optimizer: Optimizer (Adam, SGD, etc.)
        accumulation_steps: Number of batches to accumulate gradients
        checkpoint_segments: Number of segments to checkpoint (0 = no checkpointing)
        mixed_precision: Whether to use automatic mixed precision (AMP)
    """
    model.train()
    
    # Setup mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    for batch_idx, (volumes, targets) in enumerate(data_loader):
        # volumes shape: [B, C, D, H, W] for 3D medical images
        
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            # Forward pass with optional checkpointing
            if checkpoint_segments > 0:
                # Wrap model forward with checkpointing
                outputs = checkpoint(model, volumes)
            else:
                outputs = model(volumes)
            
            # Compute loss (e.g., Dice loss for segmentation)
            loss = loss_fn(outputs, targets)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
        
        # Backward pass
        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
    
    # Final update if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        if mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()


class VolumetricCheckpointStrategy:
    """
    Advanced checkpointing strategy for 3D volumetric medical data.
    Optimizes memory usage based on volume dimensions and model architecture.
    """
    
    def __init__(self, volume_size: Tuple[int, int, int], model_depth: int):
        """
        Initialize volumetric checkpointing strategy.
        
        Args:
            volume_size: (D, H, W) dimensions of input 3D volume
            model_depth: Number of layers in the model
        """
        self.volume_size = volume_size
        self.model_depth = model_depth
        self.memory_per_voxel = 4  # bytes for float32
        
    def calculate_memory_usage(self, batch_size: int, channels: int) -> float:
        """
        Calculate approximate memory usage for 3D volumes.
        
        Returns:
            Memory usage in GB
        """
        voxels = batch_size * channels * np.prod(self.volume_size)
        return (voxels * self.memory_per_voxel) / (1024**3)
    
    def recommend_checkpoint_layers(self, available_memory_gb: float) -> List[int]:
        """
        Recommend which layers to checkpoint based on available GPU memory.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            List of layer indices to checkpoint
        """
        # Simple heuristic: checkpoint deeper layers which typically have more channels
        checkpoint_layers = []
        
        # Checkpoint every nth layer based on memory constraints
        if available_memory_gb < 8:
            # Aggressive checkpointing for low memory
            checkpoint_interval = 2
        elif available_memory_gb < 16:
            # Moderate checkpointing
            checkpoint_interval = 3
        else:
            # Light checkpointing for high memory
            checkpoint_interval = 4
            
        checkpoint_layers = list(range(0, self.model_depth, checkpoint_interval))
        return checkpoint_layers