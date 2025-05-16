import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import pickle

from tqdm import tqdm

import torch
import numpy as np

import csv


class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames)

    def __len__(self):
        return len(self.filenames)

    def pad_to_shape(self, img_np, target_shape):
        pad_height = target_shape[0] - img_np.shape[0]
        pad_width = target_shape[1] - img_np.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(img_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        study_id, slice_tag = filename.replace(".jpg", "").split("_slice_")
        slice_num = int(slice_tag)

        stack = []
        shapes = []
        for offset in [-1, 0, 1]:
            n = slice_num + offset
            neighbor_file = f"{study_id}_slice_{n:03d}.jpg"
            neighbor_path = os.path.join(self.image_dir, neighbor_file)
            if os.path.exists(neighbor_path):
                img = Image.open(neighbor_path).convert("L")
            else:
                img = Image.open(os.path.join(self.image_dir, filename)).convert("L")

            img_np = np.array(img, dtype=np.float32) / 255.0
            stack.append(img_np)
            shapes.append(img_np.shape)

        max_shape = np.max(shapes, axis=0)
        stack = [self.pad_to_shape(s, max_shape) for s in stack]
        image = np.stack(stack, axis=0)

        mask_path = os.path.join(self.mask_dir, study_id, filename.replace(".jpg", ".npy"))
        mask = np.load(mask_path).astype(np.int64)
        mask = self.pad_to_shape(mask, max_shape)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

# Batch padding to SMP-required divisible-by-32 sizes
def pad_batch_to_32(batch):
    images, masks = zip(*batch)
    h = max(img.shape[1] for img in images)
    w = max(img.shape[2] for img in images)
    def ceil32(x): return ((x + 31) // 32) * 32
    H, W = ceil32(h), ceil32(w)

    padded_imgs, padded_masks = [], []
    for img, mask in zip(images, masks):
        pad = (0, W - img.shape[2], 0, H - img.shape[1])
        padded_imgs.append(F.pad(img, pad))
        padded_masks.append(F.pad(mask, pad))

    return torch.stack(padded_imgs), torch.stack(padded_masks)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

image_dir = "/data_vault/hexai/OAICartilage/image_manual_crops"
mask_dir = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"
all_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Split: 80% train, 10% val, 10% test
train_f, temp_f = train_test_split(all_files, test_size=0.2, random_state=42)
val_f, test_f = train_test_split(temp_f, test_size=0.5, random_state=42)

print(f"Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

train_ds = KneeSegmentation25D(image_dir, mask_dir, train_f)
val_ds = KneeSegmentation25D(image_dir, mask_dir, val_f)
test_ds = KneeSegmentation25D(image_dir, mask_dir, test_f)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_batch_to_32, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=pad_batch_to_32, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=pad_batch_to_32, num_workers=4)

# Save split to a pickle file
split_save_path = "/home/feg48/2.5D_seg/knee_split.pkl"
with open(split_save_path, "wb") as f:
    pickle.dump({
        "train": train_f,
        "val": val_f,
        "test": test_f
    }, f)

print(f" Saved split info to: {split_save_path}")


def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)

import csv

def train_model(epochs=30, save_path="/home/feg48/2.5D_seg/best_model.pth", log_csv_path="/home/feg48/2.5D_seg/training_log.csv"):
    best_iou = 0.0
    metrics_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_ious = []

        print(f"\n Epoch {epoch+1} started...")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1)
            for p, t in zip(pred_labels, y):
                train_ious.append(iou_score(p.cpu(), t.cpu()))

        avg_train_loss = total_loss / len(train_loader)
        mean_train_iou = np.mean(train_ious)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train IoU: {mean_train_iou:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                for p, t in zip(pred, y):
                    val_ious.append(iou_score(p.cpu(), t.cpu()))

        avg_val_loss = val_loss / len(val_loader)
        mean_val_iou = np.mean(val_ious)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val IoU: {mean_val_iou:.4f}")

        # Save best model
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save(model.state_dict(), save_path)
            print(f" ✅ Saved best model at Epoch {epoch+1} with IoU={mean_val_iou:.4f}")

        # Append metrics for CSV logging
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_iou": mean_train_iou,
            "val_loss": avg_val_loss,
            "val_iou": mean_val_iou
        })

    print(f"\nTraining finished. Best IoU: {best_iou:.4f}")

    # Save metrics to CSV
    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_iou", "val_loss", "val_iou"])
        writer.writeheader()
        for row in metrics_log:
            writer.writerow(row)
    print(f"Training metrics saved to: {log_csv_path}")

train_model(epochs=30)


# Test

# Dice Score Function
def dice_score(pred, target, num_classes=5):
    dices = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dices.append(dice.item())
    return np.mean(dices)

# IoU Score Function
def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)

# Load Best Model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Run Test 
def test_model(test_loader):
    test_ious = []
    test_dices = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            for p, t in zip(pred, y):
                test_ious.append(iou_score(p.cpu(), t.cpu()))
                test_dices.append(dice_score(p.cpu(), t.cpu()))

    mean_iou = np.mean(test_ious)
    mean_dice = np.mean(test_dices)
    print(f"Test IoU:   {mean_iou:.4f}")
    print(f"Test Dice:  {mean_dice:.4f}")

test_model(test_loader)


