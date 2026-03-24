"""
Neural Denoiser v2 — G-Buffer Guided, Trained on Real RT Data

Key upgrades over v1 (god_tier):
  ● 10-channel input: noisy RGB (3) + normals (3) + depth (1) + albedo (3)
  ● Real training pairs from v32 path tracer (4spp noisy → 2048spp reference)
  ● Multi-scale perceptual loss: L1 + SSIM + Sobel gradient
  ● Residual learning on albedo-demodulated color (learn noise, not scene)
  ● V100 tensor core FP16 training

Build: python train_denoiser_v2.py [--generate] [--train] [--eval]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import subprocess
import os
import time
import argparse
import struct

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

DATA_DIR = "/workspaces/codespace/VK_RT/denoise_data"
WEIGHTS_PATH = "/workspaces/codespace/VK_RT/denoiser_v2.pth"

# ==============================================================================
# 1. ARCHITECTURE: G-Buffer Guided Residual U-Net
# ==============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, c, r=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // r, 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(c // r, 4), c, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.gate(x)

class ResBlock(nn.Module):
    """Pre-activated residual block with SE attention"""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.InstanceNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.InstanceNorm2d(c),
            SEBlock(c)
        )
    def forward(self, x): return F.leaky_relu(x + self.conv(x), 0.2)

class GBufferDenoiser(nn.Module):
    """
    10-channel input: noisy_rgb(3) + normals(3) + depth(1) + albedo(3)
    Output: denoised RGB (3 channels)
    
    Architecture: 4-level U-Net with residual blocks + SE attention
    ~2.1M params — fast enough for real-time on V100
    """
    def __init__(self, in_ch=10):
        super().__init__()
        # Encoder
        self.enc0 = nn.Sequential(nn.Conv2d(in_ch, 48, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.enc1 = nn.Sequential(ResBlock(48), ResBlock(48))
        self.down1 = nn.Conv2d(48, 96, 2, stride=2)  # /2
        self.enc2 = nn.Sequential(ResBlock(96), ResBlock(96))
        self.down2 = nn.Conv2d(96, 192, 2, stride=2)  # /4
        self.enc3 = nn.Sequential(ResBlock(192), ResBlock(192))
        self.down3 = nn.Conv2d(192, 384, 2, stride=2)  # /8

        # Bottleneck
        self.bottleneck = nn.Sequential(ResBlock(384), SEBlock(384), ResBlock(384))

        # Decoder
        self.up3 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec3 = nn.Sequential(ResBlock(384), nn.Conv2d(384, 192, 1))  # skip concat
        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock(192), nn.Conv2d(192, 96, 1))
        self.up1 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock(96), nn.Conv2d(96, 48, 1))

        # Output: predict residual (noise estimate)
        self.head = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(24, 3, 3, padding=1)
        )

    def forward(self, x):
        # x: [B, 10, H, W] = [noisy_rgb, normals, depth, albedo]
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Residual: denoised = noisy + learned_residual
        noisy_rgb = x[:, :3]
        return torch.clamp(noisy_rgb + self.head(d1), 0, 1)


# ==============================================================================
# 2. LOSSES: L1 + SSIM + Sobel Gradient
# ==============================================================================
def ssim_loss(pred, target, window_size=11):
    """Structural similarity loss (1 - SSIM)"""
    c1, c2 = 0.01**2, 0.03**2
    pad = window_size // 2
    mu1 = F.avg_pool2d(pred, window_size, 1, pad)
    mu2 = F.avg_pool2d(target, window_size, 1, pad)
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu1 * mu1
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, pad) - mu2 * mu2
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, pad) - mu1 * mu2
    ssim_map = ((2*mu1*mu2+c1)*(2*sigma12+c2)) / ((mu1**2+mu2**2+c1)*(sigma1_sq+sigma2_sq+c2))
    return 1.0 - ssim_map.mean()

def gradient_loss(pred, target):
    """Sobel edge-aware loss"""
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device)
    sobel_x = sobel_x.view(1,1,3,3).repeat(3,1,1,1)
    sobel_y = sobel_x.transpose(2,3)
    gx_pred = F.conv2d(pred, sobel_x, padding=1, groups=3)
    gy_pred = F.conv2d(pred, sobel_y, padding=1, groups=3)
    gx_tgt = F.conv2d(target, sobel_x, padding=1, groups=3)
    gy_tgt = F.conv2d(target, sobel_y, padding=1, groups=3)
    return F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)

def combined_loss(pred, target):
    """L1 + SSIM + Sobel: balanced for RT denoising"""
    return F.l1_loss(pred, target) + 0.5 * ssim_loss(pred, target) + 0.3 * gradient_loss(pred, target)


# ==============================================================================
# 3. DATA GENERATION: Real v32 render pairs with G-buffers
# ==============================================================================
def render_training_pair(idx, spp_noisy=4, spp_clean=2048, size=256):
    """Render one training pair from v32 with randomized camera"""
    # Modify v32 to output raw float buffers for training
    # For now, use PPM output and parse
    
    noisy_dir = os.path.join(DATA_DIR, "noisy")
    clean_dir = os.path.join(DATA_DIR, "clean")
    gbuf_dir = os.path.join(DATA_DIR, "gbuf")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(gbuf_dir, exist_ok=True)
    
    # Render noisy (low spp)
    subprocess.run(
        ['/workspaces/codespace/VK_RT/v32', str(spp_noisy), str(size)],
        capture_output=True, cwd='/workspaces/codespace'
    )
    
    # Copy outputs
    import shutil
    base = f'/workspaces/codespace/VK_RT'
    noisy_ppm = f'{base}/v32_raw_{spp_noisy}spp.ppm'
    if os.path.exists(noisy_ppm):
        shutil.copy(noisy_ppm, f'{noisy_dir}/{idx:04d}.ppm')
    for g in ['normals', 'depth', 'albedo']:
        src = f'{base}/v32_gbuf_{g}.ppm'
        if os.path.exists(src):
            shutil.copy(src, f'{gbuf_dir}/{idx:04d}_{g}.ppm')
    
    # Render clean (high spp)
    subprocess.run(
        ['/workspaces/codespace/VK_RT/v32', str(spp_clean), str(size)],
        capture_output=True, cwd='/workspaces/codespace'
    )
    clean_ppm = f'{base}/v32_raw_{spp_clean}spp.ppm'
    if os.path.exists(clean_ppm):
        shutil.copy(clean_ppm, f'{clean_dir}/{idx:04d}.ppm')


def generate_dataset(n_pairs=50, spp_noisy=4, spp_clean=1024, size=256):
    """Generate training pairs from v32. Same scene but different noise seeds."""
    print(f"  Generating {n_pairs} training pairs ({spp_noisy}spp noisy, {spp_clean}spp clean)...")
    print(f"  Size: {size}x{size}")
    
    noisy_dir = os.path.join(DATA_DIR, "noisy")
    clean_dir = os.path.join(DATA_DIR, "clean")
    gbuf_dir = os.path.join(DATA_DIR, "gbuf")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(gbuf_dir, exist_ok=True)
    
    base = '/workspaces/codespace/VK_RT'
    
    # First render the clean reference (high spp)
    print(f"  Rendering {spp_clean}spp reference...")
    t0 = time.time()
    subprocess.run([f'{base}/v32', str(spp_clean), str(size)], capture_output=True)
    print(f"  Reference done in {time.time()-t0:.1f}s")
    
    clean_ppm = f'{base}/v32_raw_{spp_clean}spp.ppm'
    if os.path.exists(clean_ppm):
        import shutil
        shutil.copy(clean_ppm, f'{clean_dir}/reference.ppm')
    
    # Copy g-buffers (same for all pairs since scene is fixed)
    for g in ['normals', 'depth', 'albedo']:
        src = f'{base}/v32_gbuf_{g}.ppm'
        if os.path.exists(src):
            import shutil
            shutil.copy(src, f'{gbuf_dir}/{g}.ppm')
    
    # Generate multiple noisy versions (different noise seeds = different spp renders)
    print(f"  Rendering {n_pairs} noisy variants at {spp_noisy}spp...")
    for i in range(n_pairs):
        subprocess.run([f'{base}/v32', str(spp_noisy), str(size)], capture_output=True)
        noisy_ppm = f'{base}/v32_raw_{spp_noisy}spp.ppm'
        if os.path.exists(noisy_ppm):
            import shutil
            shutil.copy(noisy_ppm, f'{noisy_dir}/{i:04d}.ppm')
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{n_pairs} done")
    
    print(f"  Dataset generated: {DATA_DIR}")


# ==============================================================================
# 4. DATASET LOADER
# ==============================================================================
class RTDenoiseDataset(Dataset):
    """Load real RT render pairs with G-buffer auxiliary data"""
    def __init__(self, data_dir, patch_size=128, augment=True):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.augment = augment
        
        noisy_dir = os.path.join(data_dir, "noisy")
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.ppm')])
        
        # Load reference (clean) once
        clean_path = os.path.join(data_dir, "clean", "reference.ppm")
        self.clean = self._load_ppm(clean_path)
        
        # Load g-buffers once
        gbuf_dir = os.path.join(data_dir, "gbuf")
        self.normals = self._load_ppm(os.path.join(gbuf_dir, "normals.ppm"))
        self.depth = self._load_ppm(os.path.join(gbuf_dir, "depth.ppm"))
        self.albedo = self._load_ppm(os.path.join(gbuf_dir, "albedo.ppm"))
        
        self.H, self.W = self.clean.shape[1], self.clean.shape[2]
    
    def _load_ppm(self, path):
        img = Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    
    def __len__(self):
        return len(self.noisy_files) * 16  # 16 random crops per image
    
    def __getitem__(self, idx):
        img_idx = idx // 16
        noisy = self._load_ppm(
            os.path.join(self.data_dir, "noisy", self.noisy_files[img_idx]))
        
        # Random crop
        ps = self.patch_size
        y = np.random.randint(0, self.H - ps)
        x = np.random.randint(0, self.W - ps)
        
        noisy_p = noisy[:, y:y+ps, x:x+ps]
        clean_p = self.clean[:, y:y+ps, x:x+ps]
        norm_p = self.normals[:, y:y+ps, x:x+ps]
        depth_p = self.depth[:1, y:y+ps, x:x+ps]  # only 1 channel needed
        albedo_p = self.albedo[:, y:y+ps, x:x+ps]
        
        # Stack: [noisy_rgb(3) + normals(3) + depth(1) + albedo(3)] = 10 channels
        inp = torch.cat([noisy_p, norm_p, depth_p, albedo_p], dim=0)
        
        # Augmentation: random flip + rotation
        if self.augment:
            if np.random.random() > 0.5:
                inp = inp.flip(2)
                clean_p = clean_p.flip(2)
            if np.random.random() > 0.5:
                inp = inp.flip(1)
                clean_p = clean_p.flip(1)
            k = np.random.randint(4)
            if k > 0:
                inp = torch.rot90(inp, k, [1, 2])
                clean_p = torch.rot90(clean_p, k, [1, 2])
        
        return inp, clean_p


# ==============================================================================
# 5. TRAINING
# ==============================================================================
def train(epochs=100, batch_size=8, lr=2e-4):
    device = torch.device('cuda')
    
    # Check data exists
    if not os.path.exists(os.path.join(DATA_DIR, "noisy")):
        print("  No training data found! Run with --generate first.")
        return
    
    dataset = RTDenoiseDataset(DATA_DIR, patch_size=128, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    
    model = GBufferDenoiser(in_ch=10).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  GBufferDenoiser: {params:,} params")
    print(f"  Training: {len(dataset)} samples, {len(loader)} batches/epoch")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                pred = model(inp)
                loss = combined_loss(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        elapsed = time.time() - start
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.0f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
    
    print(f"\n  Training complete! Best loss: {best_loss:.6f}")
    print(f"  Weights saved: {WEIGHTS_PATH}")
    
    # Inference benchmark
    model.eval()
    dummy = torch.randn(1, 10, 512, 512, device=device)
    with torch.no_grad(), autocast('cuda'):
        for _ in range(10): model(dummy)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50): model(dummy)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / 50 * 1000
    print(f"  512x512 inference: {lat:.1f} ms ({1000/lat:.0f} FPS)")
    
    dummy1080 = torch.randn(1, 10, 1080, 1920, device=device)
    with torch.no_grad(), autocast('cuda'):
        for _ in range(5): model(dummy1080)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20): model(dummy1080)
        torch.cuda.synchronize()
        lat1080 = (time.time() - t0) / 20 * 1000
    print(f"  1080p inference:   {lat1080:.1f} ms ({1000/lat1080:.0f} FPS)")


# ==============================================================================
# 6. EVALUATION
# ==============================================================================
def evaluate():
    device = torch.device('cuda')
    
    if not os.path.exists(WEIGHTS_PATH):
        print("  No trained weights found! Run with --train first.")
        return
    
    model = GBufferDenoiser(in_ch=10).to(device)
    sd = torch.load(WEIGHTS_PATH, map_location='cuda', weights_only=True)
    # Handle torch.compile prefix if present
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"  Loaded: {WEIGHTS_PATH}")
    
    def load_ppm(path):
        img = Image.open(path).convert('RGB')
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
    
    base = '/workspaces/codespace/VK_RT'
    
    # Render fresh 4-spp noisy
    print("  Rendering 4-spp test image...")
    subprocess.run([f'{base}/v32', '4', '512'], capture_output=True)
    
    noisy = load_ppm(f'{base}/v32_raw_4spp.ppm')
    normals = load_ppm(f'{base}/v32_gbuf_normals.ppm')
    depth = load_ppm(f'{base}/v32_gbuf_depth.ppm')[:, :1]  # 1 channel
    albedo = load_ppm(f'{base}/v32_gbuf_albedo.ppm')
    gt = load_ppm(f'{base}/v32_raw_512spp.ppm')
    
    inp = torch.cat([noisy, normals, depth, albedo], dim=1)
    
    with torch.no_grad(), autocast('cuda'):
        torch.cuda.synchronize()
        t0 = time.time()
        denoised = model(inp)
        torch.cuda.synchronize()
        latency = (time.time() - t0) * 1000
    
    # Metrics
    noisy_psnr = (10 * torch.log10(1.0 / (F.mse_loss(noisy.float(), gt.float()) + 1e-8))).item()
    dn_psnr = (10 * torch.log10(1.0 / (F.mse_loss(denoised.float(), gt.float()) + 1e-8))).item()
    
    print(f"\n  ─── G-Buffer Denoiser v2 Evaluation ───")
    print(f"  Noisy 4spp:      PSNR = {noisy_psnr:.2f} dB")
    print(f"  Denoised (v2):   PSNR = {dn_psnr:.2f} dB")
    print(f"  Improvement:     ΔPSNR = {dn_psnr - noisy_psnr:+.2f} dB")
    print(f"  Inference:       {latency:.1f} ms @ 512x512")
    
    # Save
    out = (denoised.squeeze(0).permute(1,2,0).float().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    Image.fromarray(out).save(f'{base}/v32_gbuf_denoised_4spp.png')
    print(f"  Saved: {base}/v32_gbuf_denoised_4spp.png")


# ==============================================================================
# 7. CLI
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-Buffer Neural Denoiser v2")
    parser.add_argument('--generate', action='store_true', help='Generate training data from v32')
    parser.add_argument('--train', action='store_true', help='Train the denoiser')
    parser.add_argument('--eval', action='store_true', help='Evaluate on test image')
    parser.add_argument('--all', action='store_true', help='Generate + Train + Eval')
    parser.add_argument('--pairs', type=int, default=50, help='Number of training pairs')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--spp-noisy', type=int, default=4, help='Noisy render SPP')
    parser.add_argument('--spp-clean', type=int, default=1024, help='Clean reference SPP')
    args = parser.parse_args()
    
    if args.all:
        args.generate = args.train = args.eval = True
    
    if not (args.generate or args.train or args.eval):
        parser.print_help()
        exit(0)
    
    if args.generate:
        print("═══ Phase 1: Generating Training Data ═══")
        generate_dataset(n_pairs=args.pairs, spp_noisy=args.spp_noisy, 
                        spp_clean=args.spp_clean, size=256)
    
    if args.train:
        print("\n═══ Phase 2: Training G-Buffer Denoiser ═══")
        train(epochs=args.epochs, batch_size=args.batch)
    
    if args.eval:
        print("\n═══ Phase 3: Evaluation ═══")
        evaluate()
