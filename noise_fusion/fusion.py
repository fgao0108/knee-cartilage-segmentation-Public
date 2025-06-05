import torch 
import os
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import pickle

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# define model
def load_unet_model(weight_path, device, encoder="resnet34", in_channels=3, num_classes=5):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()

# Paths to saved model weights
model_save_path_256 = "/home/feg48/2.5D_seg/best_model_256_noise_notebook.pth"
model_save_path_512 = "/home/feg48/2.5D_seg/best_model_512_noise_notebook.pth"

#load models
model_256 = load_unet_model(model_save_path_256, device)
model_512 = load_unet_model(model_save_path_512, device)

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

# Load and upsample + align probs
def load_and_align_probs(prob_path_256, prob_path_512, target_size=(512, 512)):
    prob_256 = np.load(prob_path_256)  # [C, 256, 256]
    prob_512 = np.load(prob_path_512)  # [C, 512, 512]

    prob_256_tensor = torch.from_numpy(prob_256).unsqueeze(0)  # [1, C, H, W]
    prob_512_tensor = torch.from_numpy(prob_512).unsqueeze(0)  # [1, C, H, W]

    prob_256_resized = F.interpolate(prob_256_tensor, size=target_size, mode='bilinear', align_corners=False)

    return prob_256_resized.squeeze(0), prob_512_tensor.squeeze(0)  # both [C, 512, 512]

# Fuse and evaluate on test set
def fuse_probs_and_evaluate(test_loader, save_dir_256, save_dir_512, num_classes=5):
    ious = []
    dices = []

    for idx, (_, y) in enumerate(test_loader):
        batch_size = y.size(0)

        for i in range(batch_size):
            prob_path_256 = os.path.join(save_dir_256, f"prob_{idx * batch_size + i:04d}.npy")
            prob_path_512 = os.path.join(save_dir_512, f"prob_{idx * batch_size + i:04d}.npy")

            prob_256_resized, prob_512 = load_and_align_probs(prob_path_256, prob_path_512)

            fused_prob = (prob_256_resized + prob_512) / 2  # [C, 512, 512]
            pred = torch.argmax(fused_prob, dim=0)  # [512, 512]

            target = y[i].cpu()

            ious.append(iou_score(pred, target, num_classes))
            dices.append(dice_score(pred, target, num_classes))

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    print(f"✅ Fused Test IoU:  {mean_iou:.4f}")
    print(f"✅ Fused Test Dice: {mean_dice:.4f}")

    return mean_iou, mean_dice

# DataLoader
class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, is_test_dataset, target_shape=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames)
        self.target_shape = target_shape
        self.is_test_dataset = is_test_dataset

        # Gaussian noise transforms
        self.noise_1 = T.GaussianNoise(mean=0.0, sigma=0.03, clip=True)
        self.noise_2 = T.GaussianNoise(mean=0.0, sigma=0.10, clip=True)

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

#Load saved split and recreate test_loader_512
split_save_path_512 = "/data_vault/hexai/OAICartilage/knee_split_512_noise_notebook.pkl"

with open(split_save_path_512, "rb") as f:
    split_info = pickle.load(f)

test_f = split_info["test"]

# Manually specify these based on your dataset organization and batch size
image_dir_512 = "/data_vault/hexai/OAICartilage/image_manual_crops"   
mask_dir_512 = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"  
batch_size = 8  

test_ds_512 = KneeSegmentation25D(image_dir_512, mask_dir_512, test_f, True, target_shape=(512, 512))
test_loader_512 = DataLoader(test_ds_512, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

print(f"✅ Loaded test_loader_512 with {len(test_ds_512)} samples.")


# Then call fusion and evaluation function
save_dir_256 = "/home/feg48/2.5D_seg/256_probs"
save_dir_512 = "/home/feg48/2.5D_seg/512_probs"

fuse_probs_and_evaluate(test_loader_512, save_dir_256, save_dir_512)
