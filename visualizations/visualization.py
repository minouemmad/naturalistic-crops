import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision import transforms

def visualize_heatmap(image, heatmap, weight_name, filename, overlap_score):
    # Move tensors to CPU if necessary and convert to NumPy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Adjust shape for visualization (H, W, C)
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()

    # Create directories if they do not exist
    output_dir = 'heatmap_semantic_visualizations_BDD100K_DilatedResNet'
    os.makedirs(output_dir, exist_ok=True)

    # Extract the subdirectories and ensure their existence
    relative_path = os.path.relpath(filename, start='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train')
    save_path_dir = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(save_path_dir, exist_ok=True)

    output_path = os.path.join(output_dir, relative_path.replace('/', '_'))  # Replace '/' with '_' to flatten the path
    output_path = os.path.splitext(output_path)[0] + f"_{weight_name}.jpg"  # Ensure a proper filename format

    # Generate the heatmap visualization
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

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f'Saved {output_path}')

