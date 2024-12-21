import torch
import torch.nn.functional as F
from torchvision import transforms, models  # Added 'models' here
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torchvision.models import resnet50, ResNet50_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for images and masks
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
        return image, mask, os.path.basename(img_path)

# Modified ResNet50 with dilation
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

    @staticmethod
    def load_dino_weights(model, weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model

# Generate Heatmap by Averaging Feature Channels
def generate_heatmap(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor, return_feat=True)
    heatmap = features.mean(dim=1).squeeze()
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(720, 1280), mode='bilinear').squeeze(0).squeeze(0)
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap /= (heatmap.max() + 1e-6)  # Avoid division by zero, ensure normalized
    return heatmap

def calculate_overlap_score(heatmap, mask, threshold=0.5):
    heatmap_binary = (heatmap > threshold).astype(float)
    mask_binary = (mask > threshold).cpu().numpy()  # Threshold the mask to binary

    intersection = (heatmap_binary * mask_binary).sum()
    union = heatmap_binary.sum() + mask_binary.sum() - intersection
    overlap_score = intersection / (union + 1e-6)
    return overlap_score


# Denormalize image
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


# Adjustments to iterative refinement to handle empty heatmaps
def iterative_refinement(heatmap, threshold=0.5, max_iterations=10):
    refined_regions = []
    refined_heatmaps = []
    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float).to(device)

    for i in range(max_iterations):
        h_filter = (heatmap_tensor.max(1)[0] > threshold).int()
        w_filter = (heatmap_tensor.max(0)[0] > threshold).int()

        if torch.nonzero(h_filter).numel() == 0 or torch.nonzero(w_filter).numel() == 0:
            threshold *= 0.9  # Adjust threshold
            if threshold < 0.1:  # Minimal threshold fallback
                threshold = 0.1
            continue

        h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]]
        w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]]
        refined_regions.append((h_min.item() / heatmap_tensor.shape[0],
                                w_min.item() / heatmap_tensor.shape[1],
                                h_max.item() / heatmap_tensor.shape[0],
                                w_max.item() / heatmap_tensor.shape[1]))

        refined_region = heatmap_tensor[h_min:h_max, w_min:w_max].clone()
        refined_heatmaps.append(refined_region.cpu().numpy())
        heatmap_tensor[h_min:h_max, w_min:w_max] = 0  # Zero-out refined region

    # Fallback if no refinement is found
    if not refined_regions:
        refined_regions.append((0, 0, 1, 1))  # Full area as fallback
        refined_heatmaps.append(heatmap)
    return refined_regions, refined_heatmaps

# Visualize multiple refined heatmaps
def visualize_refined_heatmaps(image, original_heatmap, mask, refined_regions, refined_heatmaps, filename, output_dir, overlap_score):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6 + len(refined_regions) * 2))

    # Display original image and heatmap
    plt.subplot(len(refined_regions) + 2, 1, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(len(refined_regions) + 2, 1, 2)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(original_heatmap, cmap="jet", alpha=0.5)
    plt.title(f"Original Heatmap\nOverlap: {overlap_score:.4f}")
    plt.axis("off")

    # Display refined heatmaps for each iteration
    for idx, (region, refined_heatmap) in enumerate(zip(refined_regions, refined_heatmaps)):
        plt.subplot(len(refined_regions) + 2, 1, idx + 3)
        plt.imshow(refined_heatmap, cmap="jet")
        plt.title(f"Refined Region {idx + 1}: {region}")
        plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Main script with iterative refinement logic
if __name__ == "__main__":
    dataset = BDDDataset(
        root_dir='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train',
        mask_dir='/scratch/ch3451/datasets/bdd100k-evals/labels/sem_seg/masks/train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = DilatedResNet50().to(device)

    DilatedResNet50.load_dino_weights(
        model, "/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth"
    )

    output_dir = "heatmap_outputs"

    for img, mask, filename_tuple in dataloader:
        filename = filename_tuple[0]

        img = img.to(device)
        heatmap = generate_heatmap(model, img)

        # Initialize the heatmap for meaningful refinement
        if heatmap.max() < 1e-3:
            print(f"Skipping {filename}: Heatmap is too weak.")
            continue

        # Iterative refinement with adjusted logic
        refined_regions, refined_heatmaps = iterative_refinement(torch.tensor(heatmap), threshold=0.5, max_iterations=5)

        # Compute overlap score
        overlap_score = calculate_overlap_score(heatmap, mask, threshold=0.5)

        # Denormalize image
        denorm_image = denormalize(img.squeeze(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Visualize refined heatmaps
        visualize_refined_heatmaps(denorm_image, heatmap, mask, refined_regions, refined_heatmaps, filename, output_dir, overlap_score)


