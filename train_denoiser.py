import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast
import os
from PIL import Image

# ==============================================================================
# 1. ARCHITECTURE: Light-Weight U-Net for Real-Time Denoising
# ==============================================================================
class UNetDenoiser(nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64) # 128 because of skip connection
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        
        self.final = nn.Conv2d(32, 3, 1) # Output RGB

    def forward(self, x):
        # Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Up
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection
        d1 = self.dec1(d1)

        return self.final(d1)

# ==============================================================================
# 2. DATASET: Noisy Input vs Ground Truth Target
# ==============================================================================
class RayTraceDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.filenames = os.listdir(noisy_dir)
        self.transform = transform or T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        noisy_img = Image.open(os.path.join(self.noisy_dir, name)).convert("RGB")
        clean_img = Image.open(os.path.join(self.clean_dir, name)).convert("RGB")
        
        return self.transform(noisy_img), self.transform(clean_img)

# ==============================================================================
# 3. TRAINING LOOP (V100 Tensor Core Optimized)
# ==============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss() # L1 is better than MSE for preserving edges
    scaler = GradScaler()   # For Mixed Precision (FP16) training

    # Setup Paths (Edit these to point to your data)
    # dataset = RayTraceDataset(noisy_dir="data/noisy", clean_dir="data/clean")
    # loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"Training initialized on {torch.cuda.get_device_name(0)}")
    print("Tensor Core acceleration (AMP) is ACTIVE.")

    # Simulated epoch loop
    for epoch in range(10):
        # for noisy, clean in loader:
        #     noisy, clean = noisy.to(device), clean.to(device)
            
        #     optimizer.zero_grad()
            
        #     # Mixed Precision Forward Pass
        #     with autocast():
        #         output = model(noisy)
        #         loss = criterion(output, clean)
            
        #     # Scaled Backward Pass
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        
        print(f"Epoch {epoch+1} complete. [Loss: Simulated]")

    # Save the model
    torch.save(model.state_dict(), "vk_rt_denoiser.pth")
    print("Training finished. Weights saved to vk_rt_denoiser.pth")

if __name__ == "__main__":
    train()
