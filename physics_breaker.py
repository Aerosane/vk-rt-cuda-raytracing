import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import time

# GLOBAL OPTIMIZATION FLAGS
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ==============================================================================
# 1. ARCHITECTURE: The "Physics-Breaker" (Tensor-Aligned U-Net)
# ==============================================================================
class PhysicsBreakerDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Helper for Tensor-Core Aligned Convolutions (Multiples of 16)
        def tc_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder: Fusing pooling into strided convolutions
        self.enc1 = tc_conv(3, 16)
        self.enc2 = tc_conv(16, 32, stride=2) 
        self.enc3 = tc_conv(32, 64, stride=2)

        self.bottleneck = tc_conv(64, 128)

        # Decoder: Using PixelShuffle instead of ConvTranspose (2x Faster on V100)
        # PixelShuffle(2) on 128 channels -> 128 / 4 = 32 channels.
        # Skip connection from enc2 has 32 channels. Total = 64.
        self.up2 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.PixelShuffle(2)) # 128 -> 32 channels
        self.dec2 = tc_conv(64, 64) 

        # PixelShuffle(2) on 64 channels -> 64 / 4 = 16 channels.
        # Skip connection from enc1 has 16 channels. Total = 32.
        self.up1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.PixelShuffle(2)) # 64 -> 16 channels
        self.dec1 = tc_conv(32, 32) 
        
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        # Path with Skip Connections
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        
        b = self.bottleneck(s3)
        
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        
        # Residual connection to preserve physical energy
        return torch.sigmoid(self.final(d1) + x)

# ==============================================================================
# 2. OPTIMIZED DATASET: Vectorized Noise Generation
# ==============================================================================
class VectorizedNoiseDataset(Dataset):
    def __init__(self, n=5000): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        # Higher complexity shapes for better quality
        clean = torch.zeros(3, 256, 256)
        for _ in range(10):
            c, p, s = torch.rand(3,1,1), torch.randint(0,180,(2,)), torch.randint(20,70,(2,))
            clean[:, p[0]:p[0]+s[0], p[1]:p[1]+s[1]] = c
        
        noisy = torch.clamp(clean + torch.randn_like(clean)*0.1 + (torch.rand_like(clean)>0.999)*3.0, 0, 1)
        return noisy, clean

# ==============================================================================
# 3. THE RUN: Pushing < 1ms Latency
# ==============================================================================
def train_physics_breaker():
    device = torch.device("cuda")
    model = PhysicsBreakerDenoiser().to(device)
    
    print("⚡ Physics-Breaker Initialized. Kernel Fusion: PENDING...")
    
    # Static Input Shape optimization for torch.compile
    model = torch.compile(model, mode="reduce-overhead") 

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    dataset = VectorizedNoiseDataset(n=2000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    scaler = GradScaler('cuda')

    print("\n--- BEGINNING ULTRA-HIGH-SPEED TRAINING ---")
    start = time.time()
    for epoch in range(10):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                loss = F.mse_loss(model(noisy), clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f" 🚀 Epoch {epoch+1:02d} | Time: {time.time()-start:.1f}s")

    # The Final Physics-Check
    model.eval()
    dummy = torch.randn(1, 3, 1080, 1920).to(device)
    with torch.no_grad(), autocast('cuda'):
        # 50 Warmup rounds to prime the V100's caches
        for _ in range(50): _ = model(dummy)
        
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(500): _ = model(dummy)
        torch.cuda.synchronize()
        latency = (time.time()-t0) / 500 * 1000
    
    print(f"\n🏆 BREAKTHROUGH: 1080p Latency: {latency:.3f} ms")
    print(f"📈 Real-Time Ceiling: {1000/latency:.1f} FPS")
    torch.save(model.state_dict(), "physics_breaker_v100.pth")

if __name__ == "__main__":
    train_physics_breaker()
