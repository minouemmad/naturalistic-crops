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

def calculate_overlap_score(heatmap, mask, threshold=0.5):
    heatmap_tensor = torch.tensor(heatmap).to(mask.device)
    thresholded_heatmap = (heatmap_tensor > threshold).float()
    overlap = (thresholded_heatmap * mask).sum() / (mask.sum() + 1e-6)
    return overlap.item()

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

