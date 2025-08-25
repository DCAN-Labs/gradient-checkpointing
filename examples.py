"""
Usage examples for gradient checkpointing in 3D brain MRI analysis.

This module provides comprehensive examples of using gradient checkpointing
for 3D brain MRI segmentation, tumor detection, and parcellation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import our gradient checkpointing modules
from gradient_checkpointing import (
    checkpoint, 
    CheckpointedMedicalSequential,
    SelectiveCheckpointMedical,
    memory_efficient_medical_training
)
from benchmark import compare_strategies, print_comparison_table
from optimal_checkpointing import (
    OptimalMedicalCheckpointer,
    MedicalLayerProfile,
    VolumetricSegmentedCheckpointing
)
from architecture_specific import (
    UNet3DCheckpointing,
    BrainMRICheckpointing,
    MixedPrecisionBrainCheckpointing
)
from profiling_visualization import (
    MemoryProfiler,
    MemoryVisualizer,
    profile_checkpointing_strategies
)


def example_1_basic_3d_unet_checkpointing():
    """Example 1: Basic gradient checkpointing for 3D U-Net."""
    print("\n" + "="*60)
    print("Example 1: 3D U-Net with Gradient Checkpointing")
    print("="*60)
    
    # Define a simple 3D U-Net block
    class UNet3DBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            return x
    
    class Simple3DUNet(nn.Module):
        def __init__(self, in_channels=1, num_classes=4):  # 4 classes for brain segmentation
            super().__init__()
            # Encoder
            self.enc1 = UNet3DBlock(in_channels, 32)
            self.pool1 = nn.MaxPool3d(2)
            self.enc2 = UNet3DBlock(32, 64)
            self.pool2 = nn.MaxPool3d(2)
            
            # Bottleneck
            self.bottleneck = UNet3DBlock(64, 128)
            
            # Decoder
            self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
            self.dec1 = UNet3DBlock(128, 64)  # 128 due to skip connection
            self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
            self.dec2 = UNet3DBlock(64, 32)  # 64 due to skip connection
            
            # Output
            self.output = nn.Conv3d(32, num_classes, kernel_size=1)
        
        def forward(self, x, use_checkpoint=False):
            # Encoder with skip connections
            if use_checkpoint and self.training:
                enc1 = checkpoint(self.enc1, x)
                x = checkpoint(self.pool1, enc1)
                enc2 = checkpoint(self.enc2, x)
                x = checkpoint(self.pool2, enc2)
                
                # Bottleneck
                x = checkpoint(self.bottleneck, x)
                
                # Decoder with skip connections
                x = checkpoint(self.up1, x)
                x = torch.cat([x, enc2], dim=1)
                x = checkpoint(self.dec1, x)
                x = checkpoint(self.up2, x)
                x = torch.cat([x, enc1], dim=1)
                x = checkpoint(self.dec2, x)
            else:
                enc1 = self.enc1(x)
                x = self.pool1(enc1)
                enc2 = self.enc2(x)
                x = self.pool2(enc2)
                
                x = self.bottleneck(x)
                
                x = self.up1(x)
                x = torch.cat([x, enc2], dim=1)
                x = self.dec1(x)
                x = self.up2(x)
                x = torch.cat([x, enc1], dim=1)
                x = self.dec2(x)
            
            return self.output(x)
    
    # Create model and sample 3D MRI data
    model = Simple3DUNet()
    batch_size = 2  # Small batch due to 3D volume size
    volume_size = (1, 128, 128, 128)  # (channels, depth, height, width)
    x = torch.randn(batch_size, *volume_size)
    y = torch.randint(0, 4, (batch_size, 128, 128, 128))  # Segmentation labels
    
    print(f"\n3D MRI Volume Shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Volume dimensions: {volume_size[1:]} voxels")
    print(f"  Memory per volume: ~{x.element_size() * x.nelement() / 1024**2:.1f} MB")
    
    # Training without checkpointing
    print("\nTraining without checkpointing:")
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
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
    
    print("\nKey benefit: Checkpointing enables training on larger 3D volumes")
    print("that wouldn't fit in GPU memory otherwise.")


def example_2_brain_sequential_checkpointing():
    """Example 2: Using CheckpointedMedicalSequential for brain parcellation."""
    print("\n" + "="*60)
    print("Example 2: Brain Parcellation with CheckpointedMedicalSequential")
    print("="*60)
    
    # Create brain parcellation residual blocks
    class BrainParcellationBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm3d(channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm3d(channels)
        
        def forward(self, x):
            residual = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x + residual)
            return x
    
    # Create brain parcellation encoder using CheckpointedMedicalSequential
    brain_encoder = CheckpointedMedicalSequential(
        nn.Conv3d(1, 16, 3, padding=1),
        nn.BatchNorm3d(16),
        nn.ReLU(inplace=True),
        BrainParcellationBlock(16),
        nn.Conv3d(16, 32, 2, stride=2),  # Downsampling
        BrainParcellationBlock(32),
        BrainParcellationBlock(32),
        nn.Conv3d(32, 64, 2, stride=2),  # Downsampling
        BrainParcellationBlock(64),
        BrainParcellationBlock(64),
        BrainParcellationBlock(64),
        checkpoint_segments=3  # Checkpoint every 3-4 layers
    )
    
    print("Brain Parcellation Encoder Structure:")
    print(f"  Total blocks: 11")
    print(f"  Checkpoint segments: 3")
    print(f"  Memory-intensive layers: Residual blocks at each resolution")
    
    # Test with 3D brain MRI volume
    brain_volume = torch.randn(1, 1, 128, 128, 128)  # Standard brain MRI
    print(f"\nBrain MRI Volume: {brain_volume.shape}")
    
    brain_encoder.train()
    output = brain_encoder(brain_volume)
    print(f"Encoder output shape: {output.shape}")
    
    # Estimate memory savings
    total_params = sum(p.numel() for p in brain_encoder.parameters())
    print(f"\nModel parameters: {total_params / 1e6:.2f}M")
    print("Estimated memory savings: ~40% with 3 checkpoint segments")


def example_3_selective_layer_checkpointing():
    """Example 3: Selective checkpointing for brain tumor detection."""
    print("\n" + "="*60)
    print("Example 3: Selective Layer Checkpointing for Brain Tumor Detection")
    print("="*60)
    
    class BrainTumorBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.lrelu1 = nn.LeakyReLU(0.01, inplace=True)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.norm2 = nn.InstanceNorm3d(out_channels)
            self.lrelu2 = nn.LeakyReLU(0.01, inplace=True)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.lrelu1(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.lrelu2(x)
            return x
    
    # Build brain tumor detection architecture
    model = nn.Sequential(
        # Stage 1 - highest resolution
        BrainTumorBlock(1, 32),
        nn.MaxPool3d(2),
        # Stage 2
        BrainTumorBlock(32, 64),
        nn.MaxPool3d(2),
        # Stage 3
        BrainTumorBlock(64, 128),
        nn.MaxPool3d(2),
        # Stage 4 - bottleneck (highest memory usage)
        BrainTumorBlock(128, 256),
        nn.MaxPool3d(2),
        # Stage 5 - deepest
        BrainTumorBlock(256, 320)
    )
    
    # Identify memory-intensive layers (deeper layers with more channels)
    checkpoint_layers = [6, 8, 10]  # Checkpoint stages 3, 4, 5
    
    print("Brain Tumor Detection Architecture:")
    print(f"  Total stages: 5")
    print(f"  Channels: 32 → 64 → 128 → 256 → 320")
    print(f"  Selective checkpointing at layers: {checkpoint_layers}")
    print("  (Targeting high-channel layers to maximize memory savings)")
    
    # Apply selective checkpointing
    selective_checkpoint = SelectiveCheckpointMedical(model, checkpoint_layers)
    
    # Test with brain MRI volume (T1 post-contrast for tumor detection)
    brain_mri_volume = torch.randn(1, 1, 155, 240, 240)  # BraTS standard size
    print(f"\nBrain MRI Volume (T1+Gd): {brain_mri_volume.shape}")
    
    model.train()
    output = model(brain_mri_volume)
    print(f"Output shape: {output.shape}")
    
    print("\nMemory optimization strategy:")
    print("  - Checkpoint deeper layers (256, 320 channels)")
    print("  - Keep shallow layers in memory (faster)")
    print("  - Estimated memory reduction: ~50% with minimal speed impact")


def example_4_optimal_checkpointing_strategy():
    """Example 4: Finding optimal checkpointing for specific hardware."""
    print("\n" + "="*60)
    print("Example 4: Optimal Checkpointing for GPU Memory Constraints")
    print("="*60)
    
    # Simulate layer profiles for a 3D segmentation model
    num_layers = 20
    profiles = []
    
    # Create realistic profile for 3D medical model
    channels = [1, 32, 32, 64, 64, 128, 128, 256, 256, 512,  # Encoder
                512, 256, 256, 128, 128, 64, 64, 32, 32, 4]   # Decoder
    
    for i in range(num_layers):
        # Memory scales with channels and spatial dimensions
        spatial_reduction = 2 ** min(i // 4, 3)  # Reduce spatial dims in encoder
        if i >= 10:  # Decoder - spatial dims increase
            spatial_reduction = 2 ** max(0, (19 - i) // 4)
        
        activation_memory = (channels[i] * 128 * 128 * 128) / (spatial_reduction ** 3) * 4 / (1024**2)
        
        profile = MedicalLayerProfile(
            layer_idx=i,
            forward_compute_cost=activation_memory * 0.1,
            backward_compute_cost=activation_memory * 0.25,
            activation_memory=activation_memory,
            parameter_memory=channels[i] * 0.5,
            layer_type="conv3d" if i % 2 == 0 else "norm",
            name=f"layer_{i}"
        )
        profiles.append(profile)
    
    # Create optimizer
    optimizer = OptimalMedicalCheckpointer(profiles)
    
    # Test for different GPU configurations
    gpu_configs = [
        (8, "RTX 2080 (8GB) - Radiology workstation"),
        (16, "V100 (16GB) - Research cluster"),
        (24, "RTX 3090 (24GB) - Deep learning workstation"),
        (40, "A100 (40GB) - HPC cluster")
    ]
    
    print("Finding optimal checkpointing for different GPUs:\n")
    
    for gpu_memory, description in gpu_configs:
        # Brain MRI volume: 256x256x256
        volume_size = (256, 256, 256)
        plan = optimizer.find_optimal_checkpoints_for_volume(
            volume_size, gpu_memory, batch_size=1
        )
        
        print(f"{description}:")
        print(f"  Checkpointed layers: {len(plan.checkpoint_layers)}/{num_layers}")
        print(f"  Memory usage: {plan.total_memory:.1f} MB")
        print(f"  Memory savings: {plan.memory_savings/optimizer.total_memory_no_checkpoint*100:.1f}%")
        print(f"  Compute overhead: {plan.compute_overhead/optimizer.total_compute_no_checkpoint*100:.1f}%")
        print(f"  Max batch size for 256³ volume: {plan.estimated_batch_size}")
        print()


def example_5_mixed_precision_checkpointing():
    """Example 5: Combining mixed precision with checkpointing."""
    print("\n" + "="*60)
    print("Example 5: Mixed Precision + Checkpointing for 3D Medical Imaging")
    print("="*60)
    
    class BrainSegmentationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 32, 3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 4, 1)  # 4 classes: background, GM, WM, CSF
            )
        
        def forward(self, x, use_checkpoint=False):
            if use_checkpoint and self.training:
                x = checkpoint(self.encoder, x)
                x = checkpoint(self.decoder, x)
            else:
                x = self.encoder(x)
                x = self.decoder(x)
            return x
    
    model = BrainSegmentationModel()
    
    # Create sample brain MRI data
    batch_size = 2
    brain_mri = torch.randn(batch_size, 1, 128, 128, 128)
    labels = torch.randint(0, 4, (batch_size, 128, 128, 128))
    
    print("Brain Segmentation Model:")
    print(f"  Input: T1-weighted MRI {brain_mri.shape}")
    print(f"  Output: 4-class segmentation (BG, GM, WM, CSF)")
    
    # Setup training
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("\nTraining configurations:")
    print("1. Standard FP32 training")
    print("2. FP32 with checkpointing")
    print("3. Mixed precision (FP16) training")
    print("4. Mixed precision with checkpointing (maximum efficiency)")
    
    # Configuration 4: Mixed precision + checkpointing
    model.train()
    
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        output = model(brain_mri, use_checkpoint=True)
        loss = criterion(output, labels)
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    print(f"\nLoss: {loss.item():.4f}")
    print("\nMemory savings with mixed precision + checkpointing:")
    print("  - FP16 reduces activation memory by ~50%")
    print("  - Checkpointing reduces stored activations by ~60%")
    print("  - Combined: ~80% memory reduction")
    print("  - Enables 4x larger batch size or 2x larger volumes")


def example_6_multi_gpu_checkpointing():
    """Example 6: Distributed training with checkpointing for large brain MRI cohorts."""
    print("\n" + "="*60)
    print("Example 6: Multi-GPU Training for Large Brain MRI Cohorts")
    print("="*60)
    
    print("Scenario: Training on large-scale brain MRI dataset")
    print("  - Dataset: 10,000 T1-weighted brain MRI scans")
    print("  - Task: Whole-brain parcellation (100+ regions)")
    print("  - Hardware: 4x V100 GPUs (16GB each)")
    
    print("\nDistributed training strategy:")
    print("1. Data parallelism across 4 GPUs")
    print("2. Gradient checkpointing on each GPU")
    print("3. Gradient accumulation for effective batch size")
    
    # Pseudo-code for distributed setup
    print("\nPseudo-code for distributed training:")
    print("""
    # Initialize distributed training
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    
    # Create model with checkpointing
    model = BrainParcellationNet()
    model = apply_checkpointing(model, checkpoint_ratio=0.5)
    model = model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Distributed data loading for brain MRI
    dataset = BrainMRIDataset('path/to/brain_mri_data')
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)
    
    # Training loop with gradient accumulation
    accumulation_steps = 4
    for epoch in range(num_epochs):
        for i, (mri_volume, parcellation) in enumerate(dataloader):
            outputs = model(mri_volume)
            loss = criterion(outputs, parcellation) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    """)
    
    print("\nEffective training configuration:")
    print("  - Local batch size: 2 volumes per GPU")
    print("  - Gradient accumulation: 4 steps")
    print("  - Effective batch size: 2 * 4 * 4 = 32 volumes")
    print("  - Memory per GPU: ~14GB (fits in 16GB V100)")
    print("  - Training speed: ~100 volumes/minute")


def run_all_examples():
    """Run all brain MRI imaging examples."""
    print("\n" + "="*80)
    print("Gradient Checkpointing Examples for 3D Brain MRI Analysis")
    print("="*80)
    
    examples = [
        ("3D U-Net Brain MRI Segmentation", example_1_basic_3d_unet_checkpointing),
        ("Brain Parcellation Sequential Checkpointing", example_2_brain_sequential_checkpointing),
        ("Brain Tumor Detection Selective Checkpointing", example_3_selective_layer_checkpointing),
        ("Optimal GPU Memory Management", example_4_optimal_checkpointing_strategy),
        ("Mixed Precision Brain Segmentation", example_5_mixed_precision_checkpointing),
        ("Multi-GPU Brain MRI Cohort Training", example_6_multi_gpu_checkpointing)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            print("Continuing with next example...")
    
    print("\n" + "="*80)
    print("Brain MRI checkpointing examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    run_all_examples()