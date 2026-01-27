import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path

class BranchingStructureDataset(Dataset):
    """Dataset loader for CT-like branching structure images (images/ and segm/ folders)"""
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name.replace("_ct", "_mask")

        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0

        # Add channel dimension if grayscale
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        else:
            image = np.transpose(image, (2, 0, 1))

        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]

        return torch.from_numpy(image), torch.from_numpy(mask)

class RetinalStructureDataset(Dataset):
    """Dataset loader for CT-like retinal structure images (images/ and segm/ folders)"""
    def __init__(self, drive_root, stare_root, split="train", transform=None):
        self.drive_images_dir = Path(drive_root) / split / "images"
        self.drive_masks_dir = Path(drive_root) / split / "segm"
        self.stare_images_dir = Path(stare_root) / split / "images"
        self.stare_masks_dir = Path(stare_root) / split / "segm"
        self.transform = transform

        self.drive_image_files = sorted(list(self.drive_images_dir.glob("*.tif")))
        self.stare_image_files = sorted(list(self.stare_images_dir.glob("*.ppm")))

        # Combine both datasets
        self.image_files = self.drive_image_files + self.stare_image_files
        
        self.drive_masks_files = sorted(list(self.drive_masks_dir.glob("*.gif")))
        self.stare_masks_files = sorted(list(self.stare_masks_dir.glob("*.ppm")))

        self.mask_files = self.drive_masks_files + self.stare_masks_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Open and convert to grayscale, resize to 384x384
        image = Image.open(img_path).convert('L').resize((384, 384), Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((384, 384), Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        # Add channel dimension (always grayscale)
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return torch.from_numpy(image), torch.from_numpy(mask)

class UNet(nn.Module):
    """Simple U-Net for binary segmentation"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for better handling of class imbalance"""
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    """BCE + Dice loss for robust training"""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    total_loss = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            # IoU
            intersection = (preds * masks).sum(dim=(2, 3))
            union = (preds + masks).clamp(0, 1).sum(dim=(2, 3))
            iou = (intersection / (union + 1e-6)).mean()
            # Dice
            dice = (2 * intersection) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6)
            # Loss
            loss = combined_loss(outputs, masks)
            total_iou += iou.item()
            total_dice += dice.mean().item()
            total_loss += loss.item()
    return total_loss / len(dataloader), total_iou / len(dataloader), total_dice / len(dataloader)

def main():
    # Configuration
    data_root = "../../../data/"
    batch_size = 8
    num_epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Data loading (new folder structure)
    # train_images = os.path.join(data_root, "train/images")
    # train_masks = os.path.join(data_root, "train/segm")
    # val_images = os.path.join(data_root, "val/images")
    # val_masks = os.path.join(data_root, "val/segm")

    # train_dataset = BranchingStructureDataset(train_images, train_masks)
    # val_dataset = BranchingStructureDataset(val_images, val_masks)

    train_dataset = RetinalStructureDataset(os.path.join(data_root, "DRIVE"), os.path.join(data_root, "STARE"), split="train")
    val_dataset = RetinalStructureDataset(os.path.join(data_root, "DRIVE"), os.path.join(data_root, "STARE"), split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)

    # Training loop
    best_iou = 0
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou, val_dice = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        scheduler.step(val_iou)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_unet.pth")
            print(f"Saved best model with IoU: {best_iou:.4f}")
    # Save metrics plot
    plot_training_metrics(train_losses, val_losses, val_ious, val_dices, "unet_training_metrics.png")
    print("Saved training metrics plot to unet_training_metrics.png")

def plot_training_metrics(train_losses, val_losses, val_ious, val_dices, save_path):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, val_losses, label='Val Loss', color='orange')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, val_ious, label='Val IoU', color='green')
    axs[1, 0].set_title('Validation IoU')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('IoU')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, val_dices, label='Val Dice', color='red')
    axs[1, 1].set_title('Validation Dice')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Dice')
    axs[1, 1].legend()

    plt.tight_layout()
    for ax in axs.flat:
        ax.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
    