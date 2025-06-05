from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.v2 as T

class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, is_test_dataset, target_shape=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames)
        self.target_shape = target_shape
        self.is_test_dataset = is_test_dataset

        # Gaussian noise transforms
        self.noise_1 = T.GaussianNoise(mean=0.0, sigma=0.03, clip=True)  # 0.05
        self.noise_2 = T.GaussianNoise(mean=0.0, sigma=0.10, clip=True)   #0.1

    def resize_image(self, img_np):
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_resized = img_pil.resize(self.target_shape, Image.BILINEAR)
        return np.array(img_resized, dtype=np.float32) / 255.0

    def resize_mask(self, mask_np):
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))
        mask_resized = mask_pil.resize(self.target_shape, Image.NEAREST)
        return np.array(mask_resized, dtype=np.int64)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        study_id, slice_tag = filename.replace(".jpg", "").split("_slice_")
        slice_num = int(slice_tag)

        stack = []
        for offset in [-1, 0, 1]:
            n = slice_num + offset
            neighbor_file = f"{study_id}_slice_{n:03d}.jpg"
            neighbor_path = os.path.join(self.image_dir, neighbor_file)
            if os.path.exists(neighbor_path):
                img = Image.open(neighbor_path).convert("L")
            else:
                img = Image.open(os.path.join(self.image_dir, filename)).convert("L")

            img_np = np.array(img, dtype=np.float32) / 255.0
            img_resized = self.resize_image(img_np)
            stack.append(img_resized)

        image = np.stack(stack, axis=0)  # shape: (3, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # print(image.shape) 

        # Add Gaussian noise here 
        # using noise with sigma=0.05 then sigma=0.1
        if not self.is_test_dataset:
            image_tensor = self.noise_1(image_tensor)
            image_tensor = self.noise_2(image_tensor).squeeze(0)

        mask_path = os.path.join(self.mask_dir, study_id, filename.replace(".jpg", ".npy"))
        mask = np.load(mask_path).astype(np.int64)
        mask_resized = self.resize_mask(mask)

        return image_tensor, torch.tensor(mask_resized, dtype=torch.long)

import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5
).to(device)

# Use DiceLoss for multi-class segmentation
loss_fn = DiceLoss(mode='multiclass', classes=None, from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from sklearn.model_selection import train_test_split
import pickle

image_dir = "/data_vault/hexai/OAICartilage/image_manual_crops"
mask_dir = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"
all_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Split: 80% train, 10% val, 10% test
train_f, temp_f = train_test_split(all_files, test_size=0.2, random_state=42)
val_f, test_f = train_test_split(temp_f, test_size=0.5, random_state=42)

print(f"Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

train_ds = KneeSegmentation25D(image_dir, mask_dir, train_f, False)
val_ds = KneeSegmentation25D(image_dir, mask_dir, val_f, False)
test_ds = KneeSegmentation25D(image_dir, mask_dir, test_f, True)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, drop_last=True)


# Save split to a pickle file
split_save_path = "/data_vault/hexai/OAICartilage/knee_split_256_noise_notebook.pkl"
with open(split_save_path, "wb") as f:
    pickle.dump({
        "train": train_f,
        "val": val_f,
        "test": test_f
    }, f)

print(f"✅ Saved split info to: {split_save_path}")

from tqdm import tqdm
import csv

def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)

model_save_path = "/home/feg48/2.5D_seg/best_model_256_noise_notebook.pth" 
# [Modified] Added validation loss calculation, per-epoch metrics logging, and CSV export by Fengyi
def train_model(epochs=15, save_path="/home/feg48/2.5D_seg/best_model_256_noise_notebook.pth", log_csv_path="/home/feg48/2.5D_seg/256_noise_training_log.csv"): 
    best_iou = 0.0
    metrics_log = []  # Added To store training/validation metrics for logging by Fengyi

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # train_ious = []    # Added Collect per-image IoU scores for training by Fengyi

        print(f"\n Epoch {epoch+1} started...")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            # loss = loss_fn(preds, y)

            # Compute Dice loss
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Added Compute training IoU for each image in the batch by Fengyi
            # train_ious.append(0.9)
            # pred_labels = torch.argmax(preds, dim=1)
            # for p, t in zip(pred_labels, y):
            #     train_ious.append(iou_score(p.cpu(), t.cpu()))

        avg_train_loss = total_loss / len(train_loader)
        # mean_train_iou = np.mean(train_ious)
        # print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train IoU: {mean_train_iou:.4f}")
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        # Added Validation loss and IoU computation by Fengyi
        model.eval()
        val_loss = 0
        val_ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                # Compute Dice loss
                loss = loss_fn(out, y)
                # loss = loss_fn(out, y)    # Added by Fengyi
                val_loss += loss.item()   # Added by Fengyi
                pred = torch.argmax(out, dim=1)
                # print("Prediction shape:", preds.shape)
                # break  # test on just one batch for now
                for p, t in zip(pred, y):
                    val_ious.append(iou_score(p.cpu(), t.cpu()))

        avg_val_loss = val_loss / len(val_loader)   # Added by Fengyi
        mean_val_iou = np.mean(val_ious)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val IoU: {mean_val_iou:.4f}")

        # Save best model
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save(model.state_dict(), save_path)
            print(f" ✅ Saved best model at Epoch {epoch+1} with IoU={mean_val_iou:.4f}")

        # Append metrics for CSV logging, Added by Fengyi
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            # "train_iou": mean_train_iou,
            "val_loss": avg_val_loss,
            "val_iou": mean_val_iou
        })

    print(f"\nTraining finished. Best IoU: {best_iou:.4f}")

    # Save metrics to CSV, Added by Fengyi
    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_iou"])
        writer.writeheader()
        for row in metrics_log:
            writer.writerow(row)
    print(f"Training metrics saved to: {log_csv_path}")

train_model(epochs=30, save_path=model_save_path)


# Test
import torch
import numpy as np

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
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

save_dir = "/home/feg48/2.5D_seg/256_probs"

# Run Test 
def test_model_and_save_probs(test_loader, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    test_ious = []
    test_dices = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            out = model(x)  # raw logits
            prob = F.softmax(out, dim=1)  # softmax probabilities [B, C, H, W]

            pred = torch.argmax(prob, dim=1)

            # Save per-sample softmax probabilities (for fusion)
            for i in range(x.size(0)):
                sample_prob = prob[i].cpu().numpy()  # shape [C, H, W]
                save_path = os.path.join(save_dir, f"prob_{idx * test_loader.batch_size + i:04d}.npy")
                np.save(save_path, sample_prob)

                p = pred[i].cpu()
                t = y[i].cpu()
                test_ious.append(iou_score(p, t))
                test_dices.append(dice_score(p, t))

    mean_iou = np.mean(test_ious)
    mean_dice = np.mean(test_dices)
    print(f"✅ Test IoU:   {mean_iou:.4f}")
    print(f"✅ Test Dice:  {mean_dice:.4f}")
    print(f"✅ Saved softmax probs to: {save_dir}")

test_model_and_save_probs(test_loader, model, save_dir)
