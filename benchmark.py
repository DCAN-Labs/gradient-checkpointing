"""
Benchmarking suite for gradient checkpointing in 3D medical imaging.

Compares memory usage and training time across different checkpointing strategies
for 3D medical imaging models (U-Net, V-Net, nnU-Net):
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
from gradient_checkpointing import checkpoint, CheckpointedMedicalSequential, SelectiveCheckpointMedical


@dataclass
class MedicalBenchmarkResult:
    """Results from a medical imaging benchmark run."""
    strategy: str
    architecture: str  # U-Net, V-Net, nnU-Net
    peak_memory_mb: float
    avg_memory_mb: float
    total_time_seconds: float
    forward_time_seconds: float
    backward_time_seconds: float
    iterations: int
    model_params: int
    batch_size: int
    volume_shape: Tuple[int, ...]  # (D, H, W) for 3D volumes
    voxels_per_second: float  # Processing speed


class MemoryMonitor:
    """Monitor memory usage during medical model training."""
    
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


class UNet3DTest(nn.Module):
    """3D U-Net for medical imaging benchmarking."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 4, 
                 base_features: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        features = base_features
        in_ch = in_channels
        
        for _ in range(depth):
            self.encoders.append(self._conv_block(in_ch, features))
            self.pools.append(nn.MaxPool3d(2))
            in_ch = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = self._conv_block(in_ch, features)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose3d(features, features//2, 2, stride=2))
            self.decoders.append(self._conv_block(features, features//2))
            features //= 2
        
        # Output
        self.output = nn.Conv3d(features, num_classes, 1)
    
    def _conv_block(self, in_channels, out_channels):
        """Create a 3D convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through U-Net."""
        # Encoder
        encoder_outputs = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for decoder, upconv, skip in zip(self.decoders, self.upconvs, 
                                         reversed(encoder_outputs)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.output(x)


class VNet3DTest(nn.Module):
    """3D V-Net for medical imaging benchmarking."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        
        # Initial convolution
        self.input_tr = nn.Conv3d(in_channels, 16, 5, padding=2)
        
        # Encoder blocks with residual connections
        self.down_tr32 = self._res_block(16, 32, stride=2)
        self.down_tr64 = self._res_block(32, 64, stride=2)
        self.down_tr128 = self._res_block(64, 128, stride=2)
        self.down_tr256 = self._res_block(128, 256, stride=2)
        
        # Decoder blocks
        self.up_tr256 = self._up_block(256, 256)
        self.up_tr128 = self._up_block(256, 128)
        self.up_tr64 = self._up_block(128, 64)
        self.up_tr32 = self._up_block(64, 32)
        
        # Output
        self.output = nn.Conv3d(32, num_classes, 1)
    
    def _res_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with optional downsampling."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _up_block(self, in_channels, out_channels):
        """Create an upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through V-Net."""
        x = self.input_tr(x)
        
        # Encoder
        down1 = self.down_tr32(x)
        down2 = self.down_tr64(down1)
        down3 = self.down_tr128(down2)
        down4 = self.down_tr256(down3)
        
        # Decoder with skip connections
        up4 = self.up_tr256(down4)
        up3 = self.up_tr128(up4 + down3)
        up2 = self.up_tr64(up3 + down2)
        up1 = self.up_tr32(up2 + down1)
        
        return self.output(up1)


class CheckpointedUNet3D(UNet3DTest):
    """3D U-Net with full gradient checkpointing."""
    
    def forward(self, x):
        """Forward pass with checkpointing for each block."""
        # Encoder
        encoder_outputs = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = checkpoint(encoder, x)
            encoder_outputs.append(x)
            x = checkpoint(pool, x)
        
        # Bottleneck
        x = checkpoint(self.bottleneck, x)
        
        # Decoder with skip connections
        for decoder, upconv, skip in zip(self.decoders, self.upconvs, 
                                         reversed(encoder_outputs)):
            x = checkpoint(upconv, x)
            x = torch.cat([x, skip], dim=1)
            x = checkpoint(decoder, x)
        
        return self.output(x)


class SelectiveCheckpointedUNet3D(UNet3DTest):
    """3D U-Net with selective gradient checkpointing."""
    
    def __init__(self, *args, checkpoint_encoder=True, checkpoint_decoder=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_decoder = checkpoint_decoder
    
    def forward(self, x):
        """Forward pass with selective checkpointing."""
        # Encoder (checkpoint if memory-intensive)
        encoder_outputs = []
        for encoder, pool in zip(self.encoders, self.pools):
            if self.checkpoint_encoder:
                x = checkpoint(encoder, x)
            else:
                x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        
        # Bottleneck (always checkpoint - highest memory usage)
        x = checkpoint(self.bottleneck, x)
        
        # Decoder
        for decoder, upconv, skip in zip(self.decoders, self.upconvs, 
                                         reversed(encoder_outputs)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            if self.checkpoint_decoder:
                x = checkpoint(decoder, x)
            else:
                x = decoder(x)
        
        return self.output(x)


def run_medical_benchmark(
    model: nn.Module,
    strategy_name: str,
    architecture_name: str,
    batch_size: int,
    volume_shape: Tuple[int, int, int],
    iterations: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> MedicalBenchmarkResult:
    """
    Run a benchmark for medical imaging models.
    
    Args:
        model: The 3D medical model to benchmark
        strategy_name: Name of the checkpointing strategy
        architecture_name: Architecture type (U-Net, V-Net, etc.)
        batch_size: Batch size for training
        volume_shape: Shape of 3D volume (D, H, W)
        iterations: Number of training iterations
        device: Device to run on
    
    Returns:
        MedicalBenchmarkResult with timing and memory statistics
    """
    model = model.to(device)
    model.train()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Memory monitor
    monitor = MemoryMonitor()
    
    # Timing
    forward_times = []
    backward_times = []
    total_start = time.time()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    total_voxels = batch_size * np.prod(volume_shape)
    
    print(f"\nRunning {strategy_name} benchmark for {architecture_name}...")
    print(f"Model parameters: {num_params:,}")
    print(f"Batch size: {batch_size}, Volume shape: {volume_shape}")
    print(f"Total voxels per batch: {total_voxels:,}")
    
    for i in range(iterations):
        # Clear gradients and cache
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Generate random 3D medical volume
        x = torch.randn(batch_size, 1, *volume_shape, device=device)
        target = torch.randint(0, 4, (batch_size, *volume_shape), device=device)
        
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
        
        if (i + 1) % max(1, iterations // 5) == 0:
            voxels_per_sec = total_voxels / (forward_time + backward_time)
            print(f"  Iteration {i+1}/{iterations}: "
                  f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s, "
                  f"Speed: {voxels_per_sec/1e6:.2f}M voxels/s")
    
    total_time = time.time() - total_start
    peak_mem, avg_mem = monitor.get_stats()
    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)
    voxels_per_second = total_voxels / (avg_forward + avg_backward)
    
    result = MedicalBenchmarkResult(
        strategy=strategy_name,
        architecture=architecture_name,
        peak_memory_mb=peak_mem,
        avg_memory_mb=avg_mem,
        total_time_seconds=total_time,
        forward_time_seconds=avg_forward,
        backward_time_seconds=avg_backward,
        iterations=iterations,
        model_params=num_params,
        batch_size=batch_size,
        volume_shape=volume_shape,
        voxels_per_second=voxels_per_second
    )
    
    print(f"  Peak memory: {peak_mem:.2f} MB")
    print(f"  Average memory: {avg_mem:.2f} MB")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Processing speed: {voxels_per_second/1e6:.2f}M voxels/s")
    
    return result


def compare_medical_strategies(
    architecture: str = "unet",
    volume_shape: Tuple[int, int, int] = (64, 128, 128),
    batch_size: int = 2,
    iterations: int = 10
) -> Dict[str, MedicalBenchmarkResult]:
    """
    Compare different checkpointing strategies for medical imaging.
    
    Args:
        architecture: Model architecture ("unet" or "vnet")
        volume_shape: 3D volume dimensions (D, H, W)
        batch_size: Batch size for training
        iterations: Number of training iterations
    
    Returns:
        Dictionary mapping strategy names to results
    """
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"3D Medical Imaging Gradient Checkpointing Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Architecture: {architecture.upper()}")
    print(f"  Volume shape: {volume_shape} voxels")
    print(f"  Batch size: {batch_size}")
    print(f"  Total voxels: {batch_size * np.prod(volume_shape):,}")
    print(f"  Device: {device}")
    print(f"{'='*80}")
    
    # Choose model based on architecture
    if architecture.lower() == "unet":
        ModelClass = UNet3DTest
        CheckpointedModelClass = CheckpointedUNet3D
        SelectiveModelClass = SelectiveCheckpointedUNet3D
        arch_name = "3D U-Net"
    else:  # vnet
        ModelClass = VNet3DTest
        CheckpointedModelClass = VNet3DTest  # Use same for now
        SelectiveModelClass = VNet3DTest
        arch_name = "3D V-Net"
    
    # Strategy 1: Standard backpropagation
    model = ModelClass()
    results['standard'] = run_medical_benchmark(
        model, "Standard Backprop", arch_name, batch_size, 
        volume_shape, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Strategy 2: Full checkpointing
    model = CheckpointedModelClass()
    results['full_checkpoint'] = run_medical_benchmark(
        model, "Full Checkpointing", arch_name, batch_size,
        volume_shape, iterations, device
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Strategy 3: Selective checkpointing (encoder only)
    if architecture.lower() == "unet":
        model = SelectiveModelClass(checkpoint_encoder=True, checkpoint_decoder=False)
        results['selective_encoder'] = run_medical_benchmark(
            model, "Selective (Encoder)", arch_name, batch_size,
            volume_shape, iterations, device
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # Strategy 4: Selective checkpointing (bottleneck only)
    if architecture.lower() == "unet":
        model = SelectiveModelClass(checkpoint_encoder=False, checkpoint_decoder=False)
        results['selective_bottleneck'] = run_medical_benchmark(
            model, "Bottleneck Only", arch_name, batch_size,
            volume_shape, iterations, device
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    return results


def print_medical_comparison(results: Dict[str, MedicalBenchmarkResult]):
    """Print a comparison table of medical imaging benchmark results."""
    print(f"\n{'='*100}")
    print(f"Medical Imaging Results Summary")
    print(f"{'='*100}")
    print(f"{'Strategy':<25} {'Peak Mem (MB)':<15} {'Avg Mem (MB)':<15} "
          f"{'Forward (s)':<12} {'Backward (s)':<12} {'MVoxels/s':<12}")
    print(f"{'-'*100}")
    
    # Get baseline (standard) for comparison
    baseline = results.get('standard')
    
    for key, result in results.items():
        mem_reduction = ""
        speed_factor = ""
        
        if baseline and key != 'standard':
            mem_reduction = f"(-{(1 - result.peak_memory_mb/baseline.peak_memory_mb)*100:.1f}%)"
            speed_factor = f"({result.voxels_per_second/baseline.voxels_per_second:.2f}x)"
        
        print(f"{result.strategy:<25} "
              f"{result.peak_memory_mb:<15.2f} "
              f"{result.avg_memory_mb:<15.2f} "
              f"{result.forward_time_seconds:<12.4f} "
              f"{result.backward_time_seconds:<12.4f} "
              f"{result.voxels_per_second/1e6:<12.2f}")
        
        if mem_reduction or speed_factor:
            print(f"{'':>25} {mem_reduction:<15} {'':>15} {'':>12} {'':>12} {speed_factor}")
    
    print(f"{'='*100}")
    
    # Print medical-specific analysis
    print(f"\nMedical Imaging Trade-off Analysis:")
    print(f"{'-'*50}")
    
    if baseline:
        for key, result in results.items():
            if key != 'standard':
                mem_saving = (1 - result.peak_memory_mb/baseline.peak_memory_mb) * 100
                speed_ratio = result.voxels_per_second / baseline.voxels_per_second
                
                print(f"{result.strategy}:")
                print(f"  Memory savings: {mem_saving:.1f}%")
                print(f"  Processing speed: {speed_ratio:.2f}x baseline")
                print(f"  Suitable for: ", end="")
                
                if mem_saving > 50:
                    print("Large 3D volumes (256³+), whole-brain MRI")
                elif mem_saving > 30:
                    print("Standard clinical volumes (128³-256³)")
                else:
                    print("Small ROI analysis, real-time processing")


def benchmark_clinical_scenarios():
    """Benchmark typical clinical imaging scenarios."""
    print("\n" + "="*80)
    print("CLINICAL IMAGING SCENARIO BENCHMARKS")
    print("="*80)
    
    scenarios = [
        ("Brain MRI T1", (128, 128, 128), 2),  # Standard brain MRI
        ("High-res Brain", (256, 256, 128), 1),  # High-resolution brain
        ("Cardiac CT", (64, 256, 256), 2),  # Cardiac imaging
        ("Abdominal CT", (40, 512, 512), 1),  # Large abdominal scan
    ]
    
    all_results = {}
    
    for scenario_name, volume_shape, batch_size in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"Volume: {volume_shape}, Batch: {batch_size}")
        print(f"Total voxels: {batch_size * np.prod(volume_shape):,}")
        print(f"{'='*80}")
        
        try:
            results = compare_medical_strategies(
                architecture="unet",
                volume_shape=volume_shape,
                batch_size=batch_size,
                iterations=5
            )
            all_results[scenario_name] = results
            print_medical_comparison(results)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠️  Out of memory for {scenario_name} - volume too large")
                print(f"  💡 Gradient checkpointing would be essential here!")
            else:
                raise e
    
    return all_results


if __name__ == "__main__":
    # Run medical imaging benchmarks
    
    # Small volume test
    print("\n" + "="*80)
    print("SMALL VOLUME (ROI Analysis)")
    print("="*80)
    results_small = compare_medical_strategies(
        architecture="unet",
        volume_shape=(32, 64, 64),
        batch_size=4,
        iterations=10
    )
    print_medical_comparison(results_small)
    
    # Standard clinical volume
    print("\n" + "="*80)
    print("STANDARD CLINICAL VOLUME")
    print("="*80)
    results_standard = compare_medical_strategies(
        architecture="unet",
        volume_shape=(64, 128, 128),
        batch_size=2,
        iterations=10
    )
    print_medical_comparison(results_standard)
    
    # Run clinical scenario benchmarks
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
        benchmark_clinical_scenarios()
    else:
        print("\n⚠️  Skipping large volume benchmarks - insufficient GPU memory")
        print("💡 This is where gradient checkpointing becomes essential!")