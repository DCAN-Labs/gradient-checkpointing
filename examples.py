"""
Usage examples for gradient checkpointing library.

This module provides comprehensive examples of how to use gradient checkpointing
in various scenarios, from basic usage to advanced optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import our gradient checkpointing modules
from gradient_checkpointing import (
    checkpoint, 
    CheckpointedSequential,
    SelectiveCheckpoint,
    memory_efficient_gradient_accumulation
)
from benchmark import compare_strategies, print_comparison_table
from optimal_checkpointing import (
    OptimalCheckpointer,
    LayerProfile,
    SegmentedCheckpointing
)
from architecture_specific import (
    ResNetCheckpointing,
    TransformerCheckpointing,
    UNetCheckpointing,
    MixedPrecisionCheckpointing
)
from profiling_visualization import (
    MemoryProfiler,
    MemoryVisualizer,
    profile_checkpointing_strategies
)


def example_1_basic_checkpointing():
    """Example 1: Basic gradient checkpointing usage."""
    print("\n" + "="*60)
    print("Example 1: Basic Gradient Checkpointing")
    print("="*60)
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=784, hidden_size=256, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Build layers
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(nn.Linear(input_size, hidden_size))
                else:
                    self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.LayerNorm(hidden_size))
            
            self.output = nn.Linear(hidden_size, 10)
        
        def forward(self, x, use_checkpoint=False):
            for layer in self.layers:
                if use_checkpoint and self.training:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
            return self.output(x)
    
    # Create model and data
    model = SimpleModel()
    batch_size = 32
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))
    
    # Training without checkpointing
    print("\nTraining without checkpointing:")
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Forward and backward
    output = model(x, use_checkpoint=False)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    
    # Training with checkpointing
    print("\nTraining with checkpointing:")
    model.zero_grad()
    output = model(x, use_checkpoint=True)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    
    print("\nKey takeaway: Both methods produce the same gradients,")
    print("but checkpointing trades compute for memory.")


def example_2_checkpointed_sequential():
    """Example 2: Using CheckpointedSequential container."""
    print("\n" + "="*60)
    print("Example 2: CheckpointedSequential Container")
    print("="*60)
    
    # Create a model using CheckpointedSequential
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Wrap with checkpointing
    checkpointed_model = CheckpointedSequential(*model, checkpoint_segments=2)
    
    print("Model structure:")
    print(f"  Total layers: {len(model)}")
    print(f"  Checkpoint segments: 2")
    print(f"  Layers per segment: ~{len(model)//2}")
    
    # Test forward pass
    x = torch.randn(16, 512)
    checkpointed_model.train()
    output = checkpointed_model(x)
    print(f"\nOutput shape: {output.shape}")
    
    # Compare memory usage
    if torch.cuda.is_available():
        device = 'cuda'
        model = model.to(device)
        checkpointed_model = checkpointed_model.to(device)
        x = x.to(device)
        
        # Without checkpointing
        torch.cuda.reset_peak_memory_stats()
        output = model(x)
        loss = output.mean()
        loss.backward()
        mem_without = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # With checkpointing
        torch.cuda.reset_peak_memory_stats()
        output = checkpointed_model(x)
        loss = output.mean()
        loss.backward()
        mem_with = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"\nMemory comparison:")
        print(f"  Without checkpointing: {mem_without:.2f} MB")
        print(f"  With checkpointing: {mem_with:.2f} MB")
        print(f"  Memory saved: {(1 - mem_with/mem_without)*100:.1f}%")


def example_3_selective_checkpointing():
    """Example 3: Selective layer checkpointing."""
    print("\n" + "="*60)
    print("Example 3: Selective Layer Checkpointing")
    print("="*60)
    
    # Create a deeper model
    class DeepModel(nn.Module):
        def __init__(self, num_layers=12):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.LayerNorm(256)
                ) for _ in range(num_layers)
            ])
            self.output = nn.Linear(256, 10)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    model = DeepModel(num_layers=12)
    
    # Apply selective checkpointing to specific layers
    # Checkpoint every 3rd layer (indices: 2, 5, 8, 11)
    checkpoint_layers = [2, 5, 8, 11]
    selective_cp = SelectiveCheckpoint(model, checkpoint_layers)
    
    print(f"Model configuration:")
    print(f"  Total layers: 12")
    print(f"  Checkpointed layers: {checkpoint_layers}")
    print(f"  Memory-compute trade-off: ~{len(checkpoint_layers)/12*100:.0f}% recomputation")
    
    # Test training
    model.train()
    x = torch.randn(8, 256)
    y = torch.randint(0, 10, (8,))
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining loss: {loss.item():.4f}")
    
    # Restore original model
    selective_cp.restore()
    print("Checkpointing removed - model restored to original state")


def example_4_optimal_checkpoint_selection():
    """Example 4: Finding optimal checkpoint locations."""
    print("\n" + "="*60)
    print("Example 4: Optimal Checkpoint Selection")
    print("="*60)
    
    # Create layer profiles for a hypothetical model
    num_layers = 16
    profiles = []
    
    print("Creating layer profiles for optimization...")
    for i in range(num_layers):
        # Simulate varying costs (transformer-like pattern)
        if i % 4 == 0:  # Attention layers
            activation_memory = 150.0  # MB
            forward_cost = 80.0  # ms
        else:  # FFN layers
            activation_memory = 100.0  # MB
            forward_cost = 40.0  # ms
        
        profile = LayerProfile(
            layer_idx=i,
            forward_compute_cost=forward_cost,
            backward_compute_cost=forward_cost * 2,
            activation_memory=activation_memory,
            parameter_memory=20.0,
            name=f"layer_{i}"
        )
        profiles.append(profile)
    
    # Create optimizer
    optimizer = OptimalCheckpointer(profiles)
    
    print(f"\nModel statistics:")
    print(f"  Total layers: {num_layers}")
    print(f"  Total activation memory: {optimizer.total_memory_no_checkpoint:.1f} MB")
    print(f"  Total compute time: {optimizer.total_compute_no_checkpoint:.1f} ms")
    
    # Find optimal checkpoints for different memory budgets
    print("\nOptimal checkpointing strategies:")
    print("-" * 50)
    
    for budget_ratio in [0.3, 0.5, 0.7]:
        memory_budget = optimizer.total_memory_no_checkpoint * budget_ratio
        plan = optimizer.find_optimal_checkpoints(memory_budget)
        
        print(f"\nMemory budget: {budget_ratio*100:.0f}% ({memory_budget:.0f} MB)")
        print(f"  Checkpoint layers: {plan.checkpoint_layers}")
        print(f"  Memory usage: {plan.total_memory:.1f} MB")
        print(f"  Memory saved: {plan.memory_savings:.1f} MB")
        print(f"  Compute overhead: +{plan.compute_overhead/optimizer.total_compute_no_checkpoint*100:.1f}%")
        
        # Efficiency metric
        efficiency = plan.memory_savings / max(plan.compute_overhead, 1)
        print(f"  Efficiency score: {efficiency:.2f}")
    
    # Compare with segmented approach
    print("\n" + "-"*50)
    print("Segmented checkpointing comparison:")
    
    for memory_ratio in [0.3, 0.5, 0.7]:
        num_segments = SegmentedCheckpointing.compute_optimal_segments(
            num_layers, memory_ratio
        )
        boundaries = SegmentedCheckpointing.get_segment_boundaries(
            num_layers, num_segments
        )
        
        print(f"\nMemory ratio: {memory_ratio}")
        print(f"  Segments: {num_segments}")
        print(f"  Boundaries: {boundaries}")


def example_5_architecture_specific():
    """Example 5: Architecture-specific strategies."""
    print("\n" + "="*60)
    print("Example 5: Architecture-Specific Strategies")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example 5a: ResNet with skip connections
    print("\n--- ResNet Checkpointing ---")
    resnet = ResNetCheckpointing.create_checkpointed_resnet(
        block_type="basic",
        layers=[2, 2, 2, 2],  # ResNet-18 configuration
        checkpoint_stages=[1, 2]  # Checkpoint stages 1 and 2
    )
    print("Created ResNet-18 with checkpointing at stages 1 and 2")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224, device=device)
    resnet = resnet.to(device)
    resnet.train()
    output = resnet(x)
    print(f"Output shape: {output.shape}")
    
    # Example 5b: Transformer with attention
    print("\n--- Transformer Checkpointing ---")
    transformer = TransformerCheckpointing.create_checkpointed_transformer(
        num_layers=6,
        d_model=512,
        nhead=8,
        checkpoint_every_n=2
    )
    print("Created 6-layer Transformer with checkpointing every 2 layers")
    
    # Test forward pass
    seq_len, batch_size = 100, 4
    x = torch.randn(seq_len, batch_size, 512, device=device)
    transformer = transformer.to(device)
    transformer.train()
    output = transformer(x)
    print(f"Output shape: {output.shape}")
    
    # Example 5c: U-Net with encoder-decoder
    print("\n--- U-Net Checkpointing ---")
    unet = UNetCheckpointing.create_checkpointed_unet(
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256],
        checkpoint_encoder=True,
        checkpoint_decoder=False
    )
    print("Created U-Net with encoder checkpointing only")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256, device=device)
    unet = unet.to(device)
    unet.train()
    output = unet(x)
    print(f"Output shape: {output.shape}")
    
    print("\nKey insight: Different architectures benefit from different")
    print("checkpointing strategies based on their connectivity patterns.")


def example_6_memory_profiling():
    """Example 6: Memory profiling and visualization."""
    print("\n" + "="*60)
    print("Example 6: Memory Profiling and Analysis")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a test model
    class ProfileTestModel(nn.Module):
        def __init__(self, use_checkpoint=False):
            super().__init__()
            self.use_checkpoint = use_checkpoint
            self.layer1 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.LayerNorm(1024)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.LayerNorm(1024)
            )
            self.layer3 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.LayerNorm(512)
            )
            self.output = nn.Linear(512, 10)
        
        def forward(self, x):
            if self.use_checkpoint and self.training:
                x = checkpoint(self.layer1, x)
                x = checkpoint(self.layer2, x)
                x = checkpoint(self.layer3, x)
            else:
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
            return self.output(x)
    
    # Profile without checkpointing
    print("\nProfiling without checkpointing...")
    model_no_cp = ProfileTestModel(use_checkpoint=False).to(device)
    profiler = MemoryProfiler(device)
    
    with profiler.profile(model_no_cp):
        x = torch.randn(16, 512, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        
        optimizer = optim.Adam(model_no_cp.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        profiler.snapshot('start')
        output = model_no_cp(x)
        profiler.snapshot('after_forward')
        loss = criterion(output, y)
        loss.backward()
        profiler.snapshot('after_backward')
        optimizer.step()
        profiler.snapshot('after_optimizer')
    
    peak_no_cp = profiler.get_peak_memory()
    avg_no_cp = profiler.get_average_memory()
    
    print(f"  Peak memory: {peak_no_cp:.2f} MB")
    print(f"  Average memory: {avg_no_cp:.2f} MB")
    
    # Profile with checkpointing
    print("\nProfiling with checkpointing...")
    model_cp = ProfileTestModel(use_checkpoint=True).to(device)
    profiler_cp = MemoryProfiler(device)
    
    with profiler_cp.profile(model_cp):
        x = torch.randn(16, 512, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        
        optimizer = optim.Adam(model_cp.parameters())
        
        # Training step
        profiler_cp.snapshot('start')
        output = model_cp(x)
        profiler_cp.snapshot('after_forward')
        loss = criterion(output, y)
        loss.backward()
        profiler_cp.snapshot('after_backward')
        optimizer.step()
        profiler_cp.snapshot('after_optimizer')
    
    peak_cp = profiler_cp.get_peak_memory()
    avg_cp = profiler_cp.get_average_memory()
    
    print(f"  Peak memory: {peak_cp:.2f} MB")
    print(f"  Average memory: {avg_cp:.2f} MB")
    
    # Compare results
    print("\nMemory savings with checkpointing:")
    print(f"  Peak memory reduced by: {(1 - peak_cp/peak_no_cp)*100:.1f}%")
    print(f"  Average memory reduced by: {(1 - avg_cp/avg_no_cp)*100:.1f}%")
    
    # Memory by phase analysis
    print("\nMemory usage by phase (with checkpointing):")
    by_phase = profiler_cp.get_memory_by_phase()
    for phase, memories in by_phase.items():
        if memories:
            print(f"  {phase}: {np.mean(memories):.2f} MB (avg)")


def example_7_gradient_accumulation():
    """Example 7: Combining checkpointing with gradient accumulation."""
    print("\n" + "="*60)
    print("Example 7: Gradient Accumulation with Checkpointing")
    print("="*60)
    
    # Create model and data
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Create dummy dataset
    dataset = TensorDataset(
        torch.randn(100, 784),
        torch.randint(0, 10, (100,))
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Training with gradient accumulation and checkpointing...")
    print("  Effective batch size: 40 (4 accumulation steps × 10)")
    print("  Checkpoint segments: 2")
    
    # Train with gradient accumulation and checkpointing
    memory_efficient_gradient_accumulation(
        model=model,
        data_loader=dataloader,
        loss_fn=criterion,
        optimizer=optimizer,
        accumulation_steps=4,
        checkpoint_segments=2
    )
    
    print("\nTraining complete!")
    print("This approach allows training with larger effective batch sizes")
    print("while keeping memory usage low.")


def example_8_mixed_precision():
    """Example 8: Combining checkpointing with mixed precision."""
    print("\n" + "="*60)
    print("Example 8: Mixed Precision + Checkpointing")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping mixed precision example.")
        return
    
    device = 'cuda'
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)
    
    # Training data
    x = torch.randn(32, 1024, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    print("Training with mixed precision and checkpointing...")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Training step with mixed precision and checkpointing
    with torch.cuda.amp.autocast():
        # Use checkpointing for middle layers
        def forward_with_checkpoint(x):
            x = model[0](x)  # First layer
            x = model[1](x)
            x = checkpoint(model[2], x)  # Checkpoint middle layer
            x = checkpoint(model[3], x)
            x = model[4](x)
            x = model[5](x)
            x = model[6](x)  # Output layer
            return x
        
        output = forward_with_checkpoint(x)
        loss = criterion(output, y)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"\nResults:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    print("\nMixed precision reduces memory for activations (FP16)")
    print("while checkpointing reduces stored activations.")
    print("Combined, they provide maximum memory efficiency!")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*80)
    print("GRADIENT CHECKPOINTING: Comprehensive Examples")
    print("="*80)
    
    examples = [
        ("Basic Checkpointing", example_1_basic_checkpointing),
        ("Checkpointed Sequential", example_2_checkpointed_sequential),
        ("Selective Checkpointing", example_3_selective_checkpointing),
        ("Optimal Checkpoint Selection", example_4_optimal_checkpoint_selection),
        ("Architecture-Specific Strategies", example_5_architecture_specific),
        ("Memory Profiling", example_6_memory_profiling),
        ("Gradient Accumulation", example_7_gradient_accumulation),
        ("Mixed Precision", example_8_mixed_precision)
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Running: {name}")
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
    print("\nSummary of techniques demonstrated:")
    print("1. Basic checkpoint function usage")
    print("2. Sequential container with automatic checkpointing")
    print("3. Selective layer-wise checkpointing")
    print("4. Dynamic programming for optimal checkpoint placement")
    print("5. Architecture-specific strategies (ResNet, Transformer, U-Net)")
    print("6. Memory profiling and analysis")
    print("7. Gradient accumulation for large batch training")
    print("8. Mixed precision training combination")
    print("\nUse these techniques to train larger models with limited GPU memory!")


if __name__ == "__main__":
    run_all_examples()