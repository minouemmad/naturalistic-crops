import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BDDDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.mask_paths = [os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        if self.transform:
            image = self.transform(image)
        mask = transforms.Resize((720, 1280))(transforms.ToTensor()(mask))  # Resize and convert mask to tensor
        return image, mask, img_path

# Uses replace_stride_with_dilation=[False, True, True] to make feature maps denser (1/8 x 1/8)
class DilatedResNet50(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True], **kwargs)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_feat:
            return x
        return self.avgpool(x).flatten(1)

def load_model_weights(model, weight_path):
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def generate_heatmap(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor, return_feat=True)
    heatmap = features.mean(dim=1).squeeze()
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(720, 1280), mode='bilinear').squeeze(0).squeeze(0)
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    return heatmap

import torch.nn as nn

def sample_crops(img_tensor, crop_size=(112, 112), stride=56):
    """
    Sample overlapping crops from the input image tensor.
    """
    _, _, h, w = img_tensor.shape
    crops = []
    for y in range(0, h - crop_size[0] + 1, stride):
        for x in range(0, w - crop_size[1] + 1, stride):
            crop = img_tensor[:, :, y:y+crop_size[0], x:x+crop_size[1]]
            crops.append(crop)
    return crops

def calculate_average_feature_activation(crops, model):
    """
    Compute feature activations for each crop and return the average activation per crop.
    """
    activations = []
    with torch.no_grad():
        for crop in crops:
            features = model(crop, return_feat=True)
            avg_activation = features.mean().item()  # Compute average activation per crop
            activations.append(avg_activation)
    return activations


def create_activation_map(activations, img_shape, crop_size=(112, 112), stride=56):
    """
    Construct an activation map of the same shape as the input image using crop activations.
    """
    h, w = img_shape[1:]
    activation_map = np.zeros((h, w))
    count_map = np.zeros((h, w))  # To average overlapping regions

    idx = 0
    for y in range(0, h - crop_size[0] + 1, stride):
        for x in range(0, w - crop_size[1] + 1, stride):
            activation_map[y:y+crop_size[0], x:x+crop_size[1]] += activations[idx]
            count_map[y:y+crop_size[0], x:x+crop_size[1]] += 1
            idx += 1

    activation_map /= np.maximum(count_map, 1)  # Avoid division by zero
    return activation_map


def calculate_overlap_score(heatmap, mask, threshold=0.5):
    """
    Calculate overlap between thresholded heatmap and mask for rare classes.
    """
    heatmap_tensor = torch.tensor(heatmap).to(mask.device)
    thresholded_heatmap = (heatmap_tensor > threshold).float()
    overlap = (thresholded_heatmap * mask).sum() / (mask.sum() + 1e-6)
    return overlap.item()


# Crop-based feature extraction
def extract_crop_features(model, img_tensor, crop_size=224):
    height, width = img_tensor.shape[2:]
    crop_activations = []
    for y in range(0, height - crop_size + 1, crop_size // 2):
        for x in range(0, width - crop_size + 1, crop_size // 2):
            crop = img_tensor[:, :, y:y+crop_size, x:x+crop_size]
            with torch.no_grad():
                feature = model(crop.to(device), return_feat=True).mean().item()
            crop_activations.append(((y, x), feature))
    return crop_activations

output_folder = 'heatmap_semantic_visualizations_BDD100K_11/09/2004'
os.makedirs(output_folder, exist_ok=True)

def visualize_heatmap(image, heatmap, crop_activations, weight_name, filename, overlap_score):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Heatmap Visualization - Model: {weight_name}\nImage: {filename}', fontsize=14, y=1.05)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f'Heatmap Overlay\nOverlap Score: {overlap_score:.4f}', fontsize=12, color='blue')

    for (y, x), activation in crop_activations:
        plt.gca().add_patch(plt.Rectangle((x, y), 224, 224, fill=False, edgecolor='red', linewidth=1))
        plt.text(x, y, f'{activation:.2f}', color='white', fontsize=8, weight='bold')

    output_path = os.path.join(output_folder, f'heatmap_{weight_name}_{filename}')
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')

# Initialize dataset and model
bdd_data = BDDDataset(
    root_dir='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train',
    mask_dir='/scratch/ch3451/datasets/bdd100k-evals/labels/sem_seg/masks/train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
dataloader = DataLoader(bdd_data, batch_size=1, shuffle=False)
weight_paths = [
    "/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth",
    "/scratch/ch3451/share/for-pathways/resnet50-imagenet-supervised.pth"
]
model = DilatedResNet50().to(device)

for weight_path in weight_paths:
    model = load_model_weights(model, weight_path)
    weight_name = os.path.basename(weight_path).split('.')[0]
    
    for i, (img, mask, img_path) in enumerate(dataloader):
        if i == 50:
            break
        img_tensor = img.to(device)
        mask = mask.to(device)
        crops = sample_crops(img_tensor, crop_size=(112, 112))

        activations = calculate_average_feature_activation(crops, model)
        activation_map = create_activation_map(activations, img_tensor.shape, crop_size=(112, 112))

        overlap_score = calculate_overlap_score(activation_map, mask.squeeze())

        #heatmap = generate_heatmap(model, img_tensor)
        #overlap_score = calculate_overlap_score(heatmap, mask.squeeze())

        img_denormalized = denormalize(img.squeeze(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_pil = transforms.ToPILImage()(img_denormalized)
        crop_activations = extract_crop_features(model, img_tensor)
        
        #visualize_heatmap(img_pil, heatmap, crop_activations, weight_name, os.path.basename(img_path[0]), overlap_score)
        visualize_heatmap(img_pil, activation_map, i, weight_name, os.path.basename(img_path), overlap_score)
        print(f'Overlap score for {os.path.basename(img_path[0])} ({weight_name}): {overlap_score:.4f}')

