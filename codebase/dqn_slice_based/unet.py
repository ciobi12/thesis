import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvBlock(nn.Module):
    """Simple convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class SimpleEncoderDecoderPolicy(nn.Module):
    """
    Lightweight Encoder-Decoder Policy Network for 64x64 slice reconstruction.
    Takes multi-channel input: current slice + previous predictions.
    Outputs probability map directly (not Q-values).
    
    Architecture:
    Input (n_channels, 64, 64) -> Encoder -> Bottleneck (128, 8, 8) -> Decoder -> Output (1, 64, 64)
    """
    
    def __init__(self, n_channels=4):
        """
        Args:
            n_channels: Number of input channels (1 current + history_len previous)
        """
        super(SimpleEncoderDecoderPolicy, self).__init__()
        self.n_channels = n_channels
        
        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.enc1 = SimpleConvBlock(n_channels, 32)      # 64x64
        self.pool1 = nn.MaxPool2d(2, 2)                  # -> 32x32
        
        self.enc2 = SimpleConvBlock(32, 64)              # 32x32
        self.pool2 = nn.MaxPool2d(2, 2)                  # -> 16x16
        
        self.enc3 = SimpleConvBlock(64, 128)             # 16x16
        self.pool3 = nn.MaxPool2d(2, 2)                  # -> 8x8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            SimpleConvBlock(128, 128),
            SimpleConvBlock(128, 128)
        )
        
        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec1 = SimpleConvBlock(192, 64)  # 64 + 128 from skip connection
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # 16x16 -> 32x32
        self.dec2 = SimpleConvBlock(96, 32)   # 32 + 64 from skip connection
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)   # 32x32 -> 64x64
        self.dec3 = SimpleConvBlock(48, 16)   # 16 + 32 from skip connection
        
        # Output layer: predict probabilities for each pixel
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output probabilities in [0, 1]
        )

    def forward(self, x):
        # x shape: (B, n_channels, 64, 64)
        # Encoder with skip connections
        e1 = self.enc1(x)        # (B, 32, 64, 64)
        p1 = self.pool1(e1)      # (B, 32, 32, 32)
        
        e2 = self.enc2(p1)       # (B, 64, 32, 32)
        p2 = self.pool2(e2)      # (B, 64, 16, 16)
        
        e3 = self.enc3(p2)       # (B, 128, 16, 16)
        p3 = self.pool3(e3)      # (B, 128, 8, 8)
        
        # Bottleneck
        b = self.bottleneck(p3)  # (B, 128, 8, 8)
        
        # Decoder with skip connections
        u1 = self.up1(b)         # (B, 64, 16, 16)
        d1 = torch.cat([u1, e3], dim=1)  # (B, 128, 16, 16)
        d1 = self.dec1(d1)       # (B, 64, 16, 16)
        
        u2 = self.up2(d1)        # (B, 32, 32, 32)
        d2 = torch.cat([u2, e2], dim=1)  # (B, 64, 32, 32)
        d2 = self.dec2(d2)       # (B, 32, 32, 32)
        
        u3 = self.up3(d2)        # (B, 16, 64, 64)
        d3 = torch.cat([u3, e1], dim=1)  # (B, 48, 64, 64)
        d3 = self.dec3(d3)       # (B, 16, 64, 64)
        
        # Probability output
        probs = self.output(d3)  # (B, 1, 64, 64)
        
        return probs.squeeze(1)  # (B, 64, 64)


class TinyEncoderDecoderPolicy(nn.Module):
    """
    Even lighter version with fewer parameters.
    Good for faster training and lower memory.
    
    Architecture:
    Input (n_channels, 64, 64) -> Encoder -> Bottleneck (64, 16, 16) -> Decoder -> Output (1, 64, 64)
    """
    
    def __init__(self, n_channels=4):
        super(TinyEncoderDecoderPolicy, self).__init__()
        self.n_channels = n_channels
        
        # Encoder: 64x64 -> 32x32 -> 16x16
        self.enc1 = SimpleConvBlock(n_channels, 16)      # 64x64
        self.pool1 = nn.MaxPool2d(2, 2)                  # -> 32x32
        
        self.enc2 = SimpleConvBlock(16, 32)              # 32x32
        self.pool2 = nn.MaxPool2d(2, 2)                  # -> 16x16
        
        # Bottleneck
        self.bottleneck = SimpleConvBlock(32, 64)        # 16x16
        
        # Decoder: 16x16 -> 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # 16x16 -> 32x32
        self.dec1 = SimpleConvBlock(64, 32)   # 32 + 32 from skip
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)   # 32x32 -> 64x64
        self.dec2 = SimpleConvBlock(32, 16)   # 16 + 16 from skip
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # (B, 16, 64, 64)
        p1 = self.pool1(e1)      # (B, 16, 32, 32)
        
        e2 = self.enc2(p1)       # (B, 32, 32, 32)
        p2 = self.pool2(e2)      # (B, 32, 16, 16)
        
        # Bottleneck
        b = self.bottleneck(p2)  # (B, 64, 16, 16)
        
        # Decoder
        u1 = self.up1(b)         # (B, 32, 32, 32)
        d1 = torch.cat([u1, e2], dim=1)  # (B, 64, 32, 32)
        d1 = self.dec1(d1)       # (B, 32, 32, 32)
        
        u2 = self.up2(d1)        # (B, 16, 64, 64)
        d2 = torch.cat([u2, e1], dim=1)  # (B, 32, 64, 64)
        d2 = self.dec2(d2)       # (B, 16, 64, 64)
        
        # Probability output
        probs = self.output(d2)  # (B, 1, 64, 64)
        
        return probs.squeeze(1)  # (B, 64, 64)


# Test the models
if __name__ == "__main__":
    # Test with multi-channel input (1 current + 3 history)
    n_channels = 4
    
    # Test SimpleEncoderDecoderPolicy
    model1 = SimpleEncoderDecoderPolicy(n_channels=n_channels)
    x = torch.randn(4, n_channels, 64, 64)
    out1 = model1(x)
    print("SimpleEncoderDecoderPolicy:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out1.shape}")
    print(f"  Output range: [{out1.min():.3f}, {out1.max():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print()
    
    # Test TinyEncoderDecoderPolicy
    model2 = TinyEncoderDecoderPolicy(n_channels=n_channels)
    out2 = model2(x)
    print("TinyEncoderDecoderPolicy:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out2.shape}")
    print(f"  Output range: [{out2.min():.3f}, {out2.max():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")