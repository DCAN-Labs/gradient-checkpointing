"""
Basic gradient checkpointing implementation in PyTorch.

This module demonstrates how to selectively not store activations during forward pass
and recompute them during backward pass to save memory.
"""

import torch
import torch.nn as nn
from typing import Callable, Any, Tuple, List
import weakref


class CheckpointFunction(torch.autograd.Function):
    """Custom autograd function for gradient checkpointing."""
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        """
        Forward pass: run the function but don't save intermediate activations.
        
        Args:
            ctx: Context object for storing information for backward pass
            run_function: The function to checkpoint
            preserve_rng_state: Whether to preserve RNG state for deterministic recomputation
            *args: Arguments to pass to run_function
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
        
        # Save RNG state if needed
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
        then compute gradients.
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
    Checkpoint a function to save memory during training.
    
    Args:
        function: The function to checkpoint
        *args: Arguments to pass to the function
        preserve_rng_state: Whether to preserve RNG state for deterministic behavior
    
    Returns:
        Output of the function
    """
    return CheckpointFunction.apply(function, preserve_rng_state, *args)


class CheckpointedSequential(nn.Module):
    """Sequential container with gradient checkpointing support."""
    
    def __init__(self, *modules, checkpoint_segments: int = 1):
        """
        Initialize checkpointed sequential container.
        
        Args:
            *modules: Modules to run sequentially
            checkpoint_segments: Number of segments to checkpoint (1 = full checkpointing)
        """
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.checkpoint_segments = checkpoint_segments
    
    def forward(self, x):
        """Forward pass with checkpointing."""
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


class SelectiveCheckpoint:
    """
    Selective checkpointing with configurable checkpoint locations.
    
    This allows fine-grained control over which layers to checkpoint.
    """
    
    def __init__(self, model: nn.Module, checkpoint_layers: List[int] = None):
        """
        Initialize selective checkpointing.
        
        Args:
            model: The model to apply checkpointing to
            checkpoint_layers: List of layer indices to checkpoint
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


def memory_efficient_gradient_accumulation(
    model: nn.Module,
    data_loader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 1,
    checkpoint_segments: int = 0
):
    """
    Training loop with gradient accumulation and optional checkpointing.
    
    Args:
        model: The model to train
        data_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        accumulation_steps: Number of batches to accumulate gradients over
        checkpoint_segments: Number of segments to checkpoint (0 = no checkpointing)
    """
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Forward pass with optional checkpointing
        if checkpoint_segments > 0:
            # Wrap model forward with checkpointing
            outputs = checkpoint(model, inputs)
        else:
            outputs = model(inputs)
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Final update if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()