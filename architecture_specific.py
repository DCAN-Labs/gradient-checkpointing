"""
Architecture-specific gradient checkpointing strategies.

Different architectures (ResNet, Transformer, U-Net) require different
checkpointing strategies due to their unique connectivity patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Any
from gradient_checkpointing import checkpoint


class ResNetCheckpointing:
    """
    Checkpointing strategy for ResNet architectures.
    
    ResNets have skip connections that need special handling to avoid
    storing intermediate activations twice.
    """
    
    class BasicBlock(nn.Module):
        """ResNet Basic Block with checkpointing support."""
        
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x, use_checkpoint: bool = False):
            """Forward pass with optional checkpointing."""
            identity = x
            
            if use_checkpoint and self.training:
                # Checkpoint the main branch computation
                def main_branch(x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    return out
                
                out = checkpoint(main_branch, x)
            else:
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
            
            # Skip connection doesn't need checkpointing
            out += self.shortcut(identity)
            out = F.relu(out)
            return out
    
    class BottleneckBlock(nn.Module):
        """ResNet Bottleneck Block with checkpointing support."""
        
        expansion = 4
        
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels * self.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
        
        def forward(self, x, use_checkpoint: bool = False):
            """Forward pass with optional checkpointing."""
            identity = x
            
            if use_checkpoint and self.training:
                # Checkpoint the bottleneck computation
                def bottleneck(x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = F.relu(self.bn2(self.conv2(out)))
                    out = self.bn3(self.conv3(out))
                    return out
                
                out = checkpoint(bottleneck, x)
            else:
                out = F.relu(self.bn1(self.conv1(x)))
                out = F.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
            
            out += self.shortcut(identity)
            out = F.relu(out)
            return out
    
    @staticmethod
    def create_checkpointed_resnet(
        block_type: str = "basic",
        layers: List[int] = [2, 2, 2, 2],
        checkpoint_stages: List[int] = [2, 3]
    ) -> nn.Module:
        """
        Create a ResNet with stage-level checkpointing.
        
        Args:
            block_type: "basic" or "bottleneck"
            layers: Number of blocks in each stage
            checkpoint_stages: Which stages to checkpoint (0-indexed)
        
        Returns:
            ResNet model with checkpointing
        """
        class CheckpointedResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 64
                
                # Initial convolution
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                
                # Create stages
                if block_type == "basic":
                    block_class = ResNetCheckpointing.BasicBlock
                else:
                    block_class = ResNetCheckpointing.BottleneckBlock
                
                self.stage1 = self._make_stage(block_class, 64, layers[0], 1)
                self.stage2 = self._make_stage(block_class, 128, layers[1], 2)
                self.stage3 = self._make_stage(block_class, 256, layers[2], 2)
                self.stage4 = self._make_stage(block_class, 512, layers[3], 2)
                
                # Final layers
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                final_channels = 512 * (block_class.expansion if hasattr(block_class, 'expansion') else 1)
                self.fc = nn.Linear(final_channels, 1000)
            
            def _make_stage(self, block_class, out_channels, num_blocks, stride):
                layers = []
                layers.append(block_class(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * (block_class.expansion if hasattr(block_class, 'expansion') else 1)
                
                for _ in range(1, num_blocks):
                    layers.append(block_class(self.in_channels, out_channels, 1))
                
                return nn.Sequential(*layers)
            
            def forward(self, x):
                # Initial layers
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                
                # Process stages with selective checkpointing
                for i, stage in enumerate([self.stage1, self.stage2, self.stage3, self.stage4]):
                    if i in checkpoint_stages and self.training:
                        x = checkpoint(stage, x)
                    else:
                        x = stage(x)
                
                # Final layers
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return CheckpointedResNet()


class TransformerCheckpointing:
    """
    Checkpointing strategy for Transformer architectures.
    
    Transformers benefit from layer-wise checkpointing due to uniform structure.
    """
    
    class TransformerLayer(nn.Module):
        """Single transformer layer with checkpointing support."""
        
        def __init__(self, d_model: int = 512, nhead: int = 8, 
                     dim_feedforward: int = 2048, dropout: float = 0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
        
        def forward(self, x, mask: Optional[torch.Tensor] = None):
            """Forward pass with residual connections."""
            # Self-attention with residual
            attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
            x = self.norm1(x + attn_out)
            
            # Feed-forward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            
            return x
    
    @staticmethod
    def create_checkpointed_transformer(
        num_layers: int = 12,
        d_model: int = 512,
        nhead: int = 8,
        checkpoint_every_n: int = 2
    ) -> nn.Module:
        """
        Create a Transformer with periodic checkpointing.
        
        Args:
            num_layers: Number of transformer layers
            d_model: Model dimension
            nhead: Number of attention heads
            checkpoint_every_n: Checkpoint every N layers
        
        Returns:
            Transformer model with checkpointing
        """
        class CheckpointedTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    TransformerCheckpointing.TransformerLayer(d_model, nhead)
                    for _ in range(num_layers)
                ])
                self.checkpoint_every_n = checkpoint_every_n
            
            def forward(self, x, mask: Optional[torch.Tensor] = None):
                for i, layer in enumerate(self.layers):
                    if i % self.checkpoint_every_n == 0 and self.training:
                        x = checkpoint(layer, x, mask)
                    else:
                        x = layer(x, mask)
                return x
        
        return CheckpointedTransformer()
    
    @staticmethod
    def create_gradient_checkpointed_bert(
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        checkpoint_activations: bool = True
    ) -> nn.Module:
        """
        Create a BERT-like model with gradient checkpointing.
        
        Args:
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            checkpoint_activations: Whether to use checkpointing
        
        Returns:
            BERT-like model
        """
        class BertLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, num_heads)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm2 = nn.LayerNorm(hidden_size)
            
            def forward(self, hidden_states, attention_mask=None):
                # Attention block
                attn_output, _ = self.attention(
                    hidden_states, hidden_states, hidden_states,
                    attn_mask=attention_mask
                )
                hidden_states = self.norm1(hidden_states + attn_output)
                
                # MLP block
                mlp_output = self.mlp(hidden_states)
                hidden_states = self.norm2(hidden_states + mlp_output)
                
                return hidden_states
        
        class CheckpointedBert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(30522, hidden_size)  # Vocab size
                self.layers = nn.ModuleList([BertLayer() for _ in range(num_layers)])
                self.pooler = nn.Linear(hidden_size, hidden_size)
                self.checkpoint_activations = checkpoint_activations
            
            def forward(self, input_ids, attention_mask=None):
                hidden_states = self.embeddings(input_ids)
                
                for layer in self.layers:
                    if self.checkpoint_activations and self.training:
                        hidden_states = checkpoint(layer, hidden_states, attention_mask)
                    else:
                        hidden_states = layer(hidden_states, attention_mask)
                
                pooled_output = self.pooler(hidden_states.mean(dim=1))
                return hidden_states, pooled_output
        
        return CheckpointedBert()


class UNetCheckpointing:
    """
    Checkpointing strategy for U-Net architectures.
    
    U-Nets have encoder-decoder structure with skip connections that require
    careful handling to avoid redundant storage.
    """
    
    class UNetBlock(nn.Module):
        """Basic U-Net block."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x
    
    @staticmethod
    def create_checkpointed_unet(
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        checkpoint_encoder: bool = True,
        checkpoint_decoder: bool = False
    ) -> nn.Module:
        """
        Create a U-Net with selective checkpointing.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Channel sizes for each level
            checkpoint_encoder: Whether to checkpoint encoder
            checkpoint_decoder: Whether to checkpoint decoder
        
        Returns:
            U-Net model with checkpointing
        """
        class CheckpointedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_blocks = nn.ModuleList()
                self.decoder_blocks = nn.ModuleList()
                self.pool = nn.MaxPool2d(2, 2)
                self.upconv = nn.ModuleList()
                
                # Encoder
                in_ch = in_channels
                for f in features:
                    self.encoder_blocks.append(UNetCheckpointing.UNetBlock(in_ch, f))
                    in_ch = f
                
                # Bottleneck
                self.bottleneck = UNetCheckpointing.UNetBlock(features[-1], features[-1] * 2)
                
                # Decoder
                for i in range(len(features) - 1, -1, -1):
                    self.upconv.append(
                        nn.ConvTranspose2d(
                            features[i] * 2 if i == len(features) - 1 else features[i + 1],
                            features[i], 2, 2
                        )
                    )
                    self.decoder_blocks.append(
                        UNetCheckpointing.UNetBlock(features[i] * 2, features[i])
                    )
                
                # Output
                self.output = nn.Conv2d(features[0], out_channels, 1)
            
            def forward(self, x):
                # Encoder path with skip connections
                skip_connections = []
                
                for i, block in enumerate(self.encoder_blocks):
                    if checkpoint_encoder and self.training:
                        x = checkpoint(block, x)
                    else:
                        x = block(x)
                    skip_connections.append(x)
                    x = self.pool(x)
                
                # Bottleneck
                x = self.bottleneck(x)
                
                # Decoder path
                for i, (upconv, block) in enumerate(zip(self.upconv, self.decoder_blocks)):
                    x = upconv(x)
                    # Concatenate skip connection
                    skip = skip_connections[-(i + 1)]
                    x = torch.cat([x, skip], dim=1)
                    
                    if checkpoint_decoder and self.training:
                        x = checkpoint(block, x)
                    else:
                        x = block(x)
                
                return self.output(x)
        
        return CheckpointedUNet()


class MixedPrecisionCheckpointing:
    """
    Combine gradient checkpointing with mixed precision training.
    
    This can further reduce memory usage and speed up training.
    """
    
    @staticmethod
    def checkpoint_with_mixed_precision(
        func: Callable,
        *args,
        use_amp: bool = True,
        **kwargs
    ) -> Any:
        """
        Checkpoint a function with automatic mixed precision.
        
        Args:
            func: Function to checkpoint
            *args: Arguments to the function
            use_amp: Whether to use automatic mixed precision
            **kwargs: Keyword arguments to the function
        
        Returns:
            Output of the function
        """
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return checkpoint(func, *args, **kwargs)
        else:
            return checkpoint(func, *args, **kwargs)
    
    @staticmethod
    def train_with_mixed_precision_checkpointing(
        model: nn.Module,
        dataloader,
        num_epochs: int = 1,
        checkpoint_freq: int = 2
    ):
        """
        Training loop with mixed precision and checkpointing.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            num_epochs: Number of epochs
            checkpoint_freq: Checkpoint every N layers
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')


def demonstrate_architecture_specific():
    """Demonstrate architecture-specific checkpointing."""
    print("=" * 80)
    print("Architecture-Specific Checkpointing Demonstration")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test ResNet checkpointing
    print("\n1. ResNet Checkpointing")
    print("-" * 40)
    resnet = ResNetCheckpointing.create_checkpointed_resnet(
        block_type="basic",
        layers=[2, 2, 2, 2],
        checkpoint_stages=[1, 2]  # Checkpoint stages 1 and 2
    )
    print(f"Created ResNet with checkpointing at stages 1 and 2")
    
    # Test with dummy input
    x = torch.randn(2, 3, 224, 224, device=device)
    resnet = resnet.to(device)
    resnet.train()
    output = resnet(x)
    print(f"Output shape: {output.shape}")
    
    # Test Transformer checkpointing
    print("\n2. Transformer Checkpointing")
    print("-" * 40)
    transformer = TransformerCheckpointing.create_checkpointed_transformer(
        num_layers=6,
        d_model=512,
        checkpoint_every_n=2
    )
    print(f"Created Transformer with checkpointing every 2 layers")
    
    # Test with dummy input
    seq_len, batch_size, d_model = 100, 2, 512
    x = torch.randn(seq_len, batch_size, d_model, device=device)
    transformer = transformer.to(device)
    transformer.train()
    output = transformer(x)
    print(f"Output shape: {output.shape}")
    
    # Test U-Net checkpointing
    print("\n3. U-Net Checkpointing")
    print("-" * 40)
    unet = UNetCheckpointing.create_checkpointed_unet(
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256],
        checkpoint_encoder=True,
        checkpoint_decoder=False
    )
    print(f"Created U-Net with encoder checkpointing")
    
    # Test with dummy input
    x = torch.randn(2, 3, 256, 256, device=device)
    unet = unet.to(device)
    unet.train()
    output = unet(x)
    print(f"Output shape: {output.shape}")
    
    # Memory comparison
    print("\n4. Memory Usage Comparison")
    print("-" * 40)
    
    def measure_memory_usage(model, input_tensor, use_checkpoint=False):
        """Measure peak memory usage during forward-backward pass."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Forward pass
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # Create dummy loss
        loss = output.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            peak_memory = 0
        
        return peak_memory
    
    if torch.cuda.is_available():
        # Compare memory usage for BERT-like model
        print("\nBERT-like model memory comparison:")
        
        # Without checkpointing
        bert_no_cp = TransformerCheckpointing.create_gradient_checkpointed_bert(
            hidden_size=768,
            num_layers=6,
            checkpoint_activations=False
        ).to(device)
        
        input_ids = torch.randint(0, 30522, (2, 128), device=device)
        mem_no_cp = measure_memory_usage(bert_no_cp, input_ids)
        print(f"  Without checkpointing: {mem_no_cp:.2f} MB")
        
        # With checkpointing
        bert_cp = TransformerCheckpointing.create_gradient_checkpointed_bert(
            hidden_size=768,
            num_layers=6,
            checkpoint_activations=True
        ).to(device)
        
        mem_cp = measure_memory_usage(bert_cp, input_ids)
        print(f"  With checkpointing: {mem_cp:.2f} MB")
        print(f"  Memory saved: {(1 - mem_cp/mem_no_cp)*100:.1f}%")


if __name__ == "__main__":
    demonstrate_architecture_specific()