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

# Dataset Class with Image and Mask Loading
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
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        mask = transforms.Resize((720, 1280))(transforms.ToTensor()(mask))
        return image, mask, img_path

# Modified ResNet50 with Dilation
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
            return x  # Return high-res feature map
        return self.avgpool(x).flatten(1)

# Loading Model Weights (DINO or ImageNet)
def load_model_weights(model, weight_path):
    if weight_path == "imagenet":  # If the path is "imagenet", load default ResNet50
        model = models.resnet50(pretrained=True)  # Default ImageNet weights
    else:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# Generate Heatmap by Averaging Feature Channels
def generate_heatmap(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor, return_feat=True)
    heatmap = features.mean(dim=1).squeeze()
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(720, 1280), mode='bilinear').squeeze(0).squeeze(0)
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    return heatmap

# Overlap Score Calculation (Heatmap vs Mask)
def calculate_overlap_score(heatmap, mask, threshold=0.5):
    heatmap_tensor = torch.tensor(heatmap).to(mask.device)
    thresholded_heatmap = (heatmap_tensor > threshold).float()
    overlap = (thresholded_heatmap * mask).sum() / (mask.sum() + 1e-6)
    return overlap.item()


def calculate_overlap_scores(heatmap, mask, threshold=0.5, num_classes=19):
    """
    Calculate overlap scores for each class in the segmentation mask, comparing 
    the thresholded heatmap with each class mask.
    
    Parameters:
    - heatmap: 2D numpy array of the heatmap.
    - mask: 2D torch tensor of the segmentation mask with class labels.
    - threshold: Threshold to binarize the heatmap.
    - num_classes: Total number of classes (19 including 'void').

    Returns:
    - class_overlap_scores: List of overlap scores for each class.
    - average_overlap: Average overlap score across classes (excluding 'void').
    - median_overlap: Median overlap score across classes (excluding 'void').
    """
    # Convert heatmap to tensor and threshold it
    heatmap_tensor = torch.tensor(heatmap).to(mask.device)
    thresholded_heatmap = (heatmap_tensor > threshold).float()

    # Store overlap scores for each class
    class_overlap_scores = []

    for cls in range(num_classes - 1):  # Exclude 'void' (class 255)
        class_mask = (mask == cls).float()  # Binary mask for current class
        class_area = class_mask.sum()

        if class_area > 0:  # Only calculate overlap if class is present
            overlap = (thresholded_heatmap * class_mask).sum() / (class_area + 1e-6)
            class_overlap_scores.append(overlap.item())
        else:
            class_overlap_scores.append(0)  # If class not present, set overlap to 0

    # Calculate average and median overlap (excluding 'void')
    average_overlap = sum(class_overlap_scores) / len(class_overlap_scores)
    median_overlap = np.median(class_overlap_scores)

    return class_overlap_scores, average_overlap, median_overlap



# Crop Sampling with Average Activation Calculation
def sample_crops(img_tensor, crop_size=(112, 112), stride=56):
    _, _, h, w = img_tensor.shape
    crops = []
    for y in range(0, h - crop_size[0] + 1, stride):
        for x in range(0, w - crop_size[1] + 1, stride):
            crop = img_tensor[:, :, y:y+crop_size[0], x:x+crop_size[1]]
            crops.append(crop)
    return crops

def calculate_average_feature_activation(crops, model):
    activations = []
    with torch.no_grad():
        for crop in crops:
            features = model(crop.to(device), return_feat=True)
            avg_activation = features.mean().item()
            activations.append(avg_activation)
    return activations

# Visualization
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

    output_path = os.path.join('heatmap_semantic_visualizations_BDD100K_DilatedResNet', f'heatmap_{weight_name}_{filename}')
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')

# Main Execution
bdd_data = BDDDataset(
    root_dir='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train',
    mask_dir='/scratch/ch3451/datasets/bdd100k-evals/labels/sem_seg/masks/train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
dataloader = DataLoader(bdd_data, batch_size=1, shuffle=False)
import torchvision.models as models

# Updated weight_paths using default ResNet50-IN1K weights
weight_paths = [
    "/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth",  # DINO BDD weights
    "imagenet"  # Using default ResNet50 weights from torchvision (ImageNet-1K)
]

# Load DINO-BDD model weights
dino_model = DilatedResNet50().to(device)

imagenet_model = models.resnet50(pretrained=True).to(device)  # Load ImageNet-pretrained ResNet50


for weight_path in weight_paths:
    dino_model = load_model_weights(dino_model, weight_path)
    imagenet_model=load_model_weights(imagenet_model, weight_path)
    weight_name = os.path.basename(weight_path).split('.')[0]

    for i, (img, mask, img_path) in enumerate(dataloader):
        if i == 50:  # Limiting to 50 images for now
            break

        img_tensor = img.to(device)
        mask = mask.to(device)
        crops = sample_crops(img_tensor, crop_size=(112, 112))
        activations = calculate_average_feature_activation(crops, dino_model)

        # Generate heatmap using DINO-BDD model
        dino_heatmap = generate_heatmap(dino_model, img_tensor)
        dino_class_overlap, dino_avg_overlap, dino_median_overlap = calculate_overlap_scores(dino_heatmap, mask.squeeze())

        # Generate heatmap using supervised ImageNet model
        #imagenet_heatmap = generate_heatmap(imagenet_model, img_tensor)
        #imagenet_class_overlap, imagenet_avg_overlap, imagenet_median_overlap = calculate_overlap_scores(imagenet_heatmap, mask.squeeze())


        # Visualization
        img_denormalized = denormalize(img.squeeze(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_pil = transforms.ToPILImage()(img_denormalized)

        # Ensure img_path is a string by indexing the first element if it’s a tuple
        if isinstance(img_path, tuple):
            img_path = img_path[0]

        visualize_heatmap(img_pil, dino_heatmap, activations, weight_name, os.path.basename(img_path), dino_avg_overlap)
        
        # Display the results for the current image
        print(f"Image: {os.path.basename(img_path[0])}")
        print(f"DINO-BDD - Average overlap: {dino_avg_overlap:.4f}, Median overlap: {dino_median_overlap:.4f}")
        #print(f"ImageNet - Average overlap: {imagenet_avg_overlap:.4f}, Median overlap: {imagenet_median_overlap:.4f}")

