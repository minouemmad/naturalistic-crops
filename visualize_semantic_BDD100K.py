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

# Custom wrapper for ResNet50 to support return_feat argument
class ResNet50WithFeatureOutput(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)

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


# Update Loading Model Weights (DINO or ImageNet)
def load_model_weights(model, weight_path):
    if weight_path == "imagenet":  # If the path is "imagenet", load ImageNet weights explicitly
        weights = ResNet50_Weights.DEFAULT  # Use DEFAULT for the latest ImageNet weights
        model.load_state_dict(weights.get_state_dict())
    else:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

import torch
import torch.nn.functional as F
import numpy as np


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


"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io

# Define the small object classes based on your class list
SMALL_CLASSES = [5, 6, 7, 11, 12, 17, 18]  # 'pole', 'traffic light', 'traffic sign', 'person', 'rider', 'motorcycle', 'bicycle'

def calculate_overlap_score(heatmap, mask, threshold=0.5):
    # Convert heatmap to a binary map based on the threshold
    binary_heatmap = (heatmap > threshold)

    # Convert binary_heatmap to a PyTorch tensor if it's not already
    if isinstance(binary_heatmap, np.ndarray):
        binary_heatmap = torch.from_numpy(binary_heatmap)

    # Move binary_heatmap to the same device as mask
    binary_heatmap = binary_heatmap.to(mask.device).bool()

    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Calculate intersection and union
    intersection = (binary_heatmap & mask).sum().item()
    union = (binary_heatmap | mask).sum().item()

    # Handle division by zero
    if union == 0:
        return 0.0
    else:
        return intersection / union


def calculate_average_median_overlap(dino_heatmaps, imagenet_heatmaps, masks, threshold=0.5, small_classes=SMALL_CLASSES):
    dino_overlap_scores = []
    imagenet_overlap_scores = []
    per_image_scores = []

    for idx, (dino_heatmap, imagenet_heatmap, mask) in enumerate(zip(dino_heatmaps, imagenet_heatmaps, masks)):
        # Create a binary mask for small classes
        small_mask = torch.zeros_like(mask, dtype=torch.bool)  # Ensures small_mask is a boolean tensor
        for cls in small_classes:
            small_mask |= (mask == cls)  # Perform bitwise OR on boolean tensors 
        # Calculate IoU for DINO and ImageNet heatmaps
        dino_iou = calculate_overlap_score(dino_heatmap, small_mask, threshold)
        imagenet_iou = calculate_overlap_score(imagenet_heatmap, small_mask, threshold)

        dino_overlap_scores.append(dino_iou)
        imagenet_overlap_scores.append(imagenet_iou)
        per_image_scores.append({'dino': dino_iou, 'imagenet': imagenet_iou})

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} images")

    # Compute average and median scores
    dino_avg, dino_median = np.mean(dino_overlap_scores), np.median(dino_overlap_scores)
    imagenet_avg, imagenet_median = np.mean(imagenet_overlap_scores), np.median(imagenet_overlap_scores)

    print(f"\nDINO Overlap - Average IoU: {dino_avg:.4f}, Median IoU: {dino_median:.4f}")
    print(f"ImageNet Overlap - Average IoU: {imagenet_avg:.4f}, Median IoU: {imagenet_median:.4f}")

    return (dino_avg, dino_median), (imagenet_avg, imagenet_median), per_image_scores

def annotate_image_with_score(image_tensor, score):
    # Convert tensor to PIL Image
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    # Initialize drawing context
    draw = ImageDraw.Draw(image_pil)
    
    # Define font (fallback to default if Arial is not available)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Define text position and content
    text = f"IoU: {score:.2f}"
    text_position = (10, 10)  # Top-left corner

    # Draw text with a black outline for visibility
    draw.text(text_position, text, fill="white", font=font)

    return transforms.ToTensor()(image_pil)

def create_grid(images, heatmaps, scores, model_title, output_dir, nrow=4):
    annotated_images = []
    for img, heatmap, score in zip(images, heatmaps, scores):
        # Convert heatmap to RGB using a colormap
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.cpu().numpy()
        else:
            heatmap_np = heatmap  # Already a NumPy array

        heatmap_rgb = plt.get_cmap('jet')(heatmap_np)[:, :, :3]  # Drop alpha channel
        heatmap_tensor = transforms.ToTensor()(heatmap_rgb).to(img.device)

        # Overlay heatmap on the original image
        overlayed = (0.5 * img + 0.5 * heatmap_tensor).clamp(0, 1)

        # Annotate the overlayed image with the IoU score
        annotated = annotate_image_with_score(overlayed, score)
        annotated_images.append(annotated)

    # Create a grid from annotated images
    grid = make_grid(annotated_images, nrow=nrow, padding=10, normalize=False)

    # Convert grid to NumPy array for plotting
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Initialize plot
    plt.figure(figsize=(20, 10))
    plt.imshow(grid_np)
    plt.axis('off')

    # Optionally, add average IoU score as the title
    if isinstance(scores[0], dict):  # Check if scores contain dictionaries
        if model_title == "DINO-ResNet":
            avg_iou = np.mean([s['dino'] for s in scores])
        else:
            avg_iou = np.mean([s['imagenet'] for s in scores])
    else:
        avg_iou = np.mean(scores)  # Assume scores are floats if not dictionaries
    plt.title(f"{model_title} Heatmap Grid\nAverage IoU: {avg_iou:.4f}", fontsize=20)

    # Save the grid image
    filename = f"{model_title}_heatmap_grid.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Saved {model_title} heatmap grid to {os.path.join(output_dir, filename)}")

def compare_models_heatmaps(images, dino_heatmaps, imagenet_heatmaps, per_image_scores, titles, base_output_dir="heatmap_grids", nrow=4):
    output_dir = os.path.join(base_output_dir, f"comparison_{'_vs_'.join(titles)}")
    os.makedirs(output_dir, exist_ok=True)

    # Extract individual scores for DINO and ImageNet
    dino_scores = [score['dino'] for score in per_image_scores]
    imagenet_scores = [score['imagenet'] for score in per_image_scores]

    # Create and save grids for both models
    create_grid(images, dino_heatmaps, dino_scores, titles[0], output_dir, nrow)
    create_grid(images, imagenet_heatmaps, imagenet_scores, titles[1], output_dir, nrow)

# ----------------- Main Execution -----------------

# Assuming BDDDataset is properly defined elsewhere
bdd_data = BDDDataset(
    root_dir='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train',
    mask_dir='/scratch/ch3451/datasets/bdd100k-evals/labels/sem_seg/masks/train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
dataloader = DataLoader(bdd_data, batch_size=1, shuffle=False)

# Import and initialize your models
import torchvision.models as models

# Assuming DilatedResNet50 and ResNet50WithFeatureOutput are defined appropriately
dino_model = DilatedResNet50().to(device)  # Instantiate DINO model
imagenet_model = ResNet50WithFeatureOutput().to(device)  # Instantiate ImageNet model

# Lists to store heatmaps, images, and masks
dino_heatmaps, imagenet_heatmaps, original_images = [], [], []
masks = []

# Generate heatmaps and collect data
for i, (img, mask, img_path) in enumerate(dataloader):
    if i >= 52:  # Process only the first 52 images for this example
        break

    img_tensor = img.to(device)

    # Generate heatmaps
    dino_heatmap = generate_heatmap(dino_model, img_tensor)
    dino_heatmaps.append(dino_heatmap)
    imagenet_heatmap = generate_heatmap(imagenet_model, img_tensor)
    imagenet_heatmaps.append(imagenet_heatmap)

    # Collect masks
    masks.append(mask.to(device))

    # Denormalize the original image for visualization
    denormalize_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    denormalized_img = denormalize_transform(img_tensor.squeeze()).clamp(0, 1)
    original_images.append(denormalized_img)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} images")

# Calculate average, median, and per-image overlap scores
(dino_avg, dino_median), (imagenet_avg, imagenet_median), per_image_scores = calculate_average_median_overlap(
    dino_heatmaps, imagenet_heatmaps, masks, threshold=0.5, small_classes=SMALL_CLASSES
)

# Generate and save the comparison grids with annotated scores
compare_models_heatmaps(
    original_images,
    dino_heatmaps,
    imagenet_heatmaps,
    per_image_scores,
    titles=["DINO-ResNet", "ImageNet-ResNet"]
)
"""




def calculate_average_median_overlap(dino_heatmaps, imagenet_heatmaps, masks, threshold=0.5):
    dino_overlap_scores = []
    imagenet_overlap_scores = []

    for dino_heatmap, imagenet_heatmap, mask in zip(dino_heatmaps, imagenet_heatmaps, masks):
        dino_overlap = calculate_overlap_score(dino_heatmap, mask, threshold)
        imagenet_overlap = calculate_overlap_score(imagenet_heatmap, mask, threshold)
        
        dino_overlap_scores.append(dino_overlap)
        imagenet_overlap_scores.append(imagenet_overlap)
    
    dino_avg, dino_median = np.mean(dino_overlap_scores), np.median(dino_overlap_scores)
    imagenet_avg, imagenet_median = np.mean(imagenet_overlap_scores), np.median(imagenet_overlap_scores)
    
    print(f"DINO Overlap - Average: {dino_avg}, Median: {dino_median}")
    print(f"ImageNet Overlap - Average: {imagenet_avg}, Median: {imagenet_median}")
    
    return (dino_avg, dino_median), (imagenet_avg, imagenet_median)



from torchvision.utils import make_grid

def create_grid(images, heatmaps, model_title, output_dir, nrow=4):
    combined_images = []
    for img, heatmap in zip(images, heatmaps):
        heatmap_rgb = plt.get_cmap('jet')(heatmap)[..., :3]
        heatmap_tensor = transforms.ToTensor()(heatmap_rgb).to(img.device)

        overlayed = (0.5 * img + 0.5 * heatmap_tensor).clamp(0, 1)
        combined_images.append(overlayed)

    grid = make_grid(combined_images, nrow=nrow, padding=10, normalize=True)
    plt.figure(figsize=(20, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    filename = f"{model_title}_heatmap_grid.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Saved {model_title} heatmap grid to {os.path.join(output_dir, filename)}")

def compare_models_heatmaps(images, dino_heatmaps, imagenet_heatmaps, titles, base_output_dir="heatmap_grids", nrow=4):
    output_dir = os.path.join(base_output_dir, f"comparison_{'_vs_'.join(titles)}")
    os.makedirs(output_dir, exist_ok=True)

    create_grid(images, dino_heatmaps, titles[0], output_dir, nrow)
    create_grid(images, imagenet_heatmaps, titles[1], output_dir, nrow)


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
from torchvision.models import resnet50, ResNet50_Weights
# Updated weight_paths using default ResNet50-IN1K weights
weight_paths = [
    "/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth",  # DINO BDD weights
    "imagenet"  # Using default ResNet50 weights from torchvision (ImageNet-1K)
]

dino_model = DilatedResNet50().to(device)  # Instantiate DINO model
imagenet_model = ResNet50WithFeatureOutput().to(device)

dino_heatmaps, imagenet_heatmaps, original_images = [], [], []
masks = []

for i, (img, mask, img_path) in enumerate(dataloader):
    if i == 52:
        break

    img_tensor = img.to(device)

    # Generate heatmaps
    dino_heatmap = generate_heatmap(dino_model, img_tensor)
    dino_heatmaps.append(dino_heatmap)
    imagenet_heatmap = generate_heatmap(imagenet_model, img_tensor)
    imagenet_heatmaps.append(imagenet_heatmap)

    masks.append(mask.to(device))

    # Denormalize the original image
    original_images.append(denormalize(
        img_tensor.squeeze(),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))

# Calculate average and median overlap scores
calculate_average_median_overlap(dino_heatmaps, imagenet_heatmaps, masks)

# Generate and save the comparison grids

compare_models_heatmaps(original_images, dino_heatmaps, imagenet_heatmaps, ["DINO-ResNet", "ImageNet-ResNet"])
