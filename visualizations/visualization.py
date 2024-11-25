import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms

def visualize_heatmap(image, heatmap, weight_name, filename, overlap_score):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Heatmap Visualization - Model: {weight_name}\nImage: {filename}', fontsize=14, y=1.05)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f'Overlap Score: {overlap_score:.4f}', color='blue')

    output_path = os.path.join('visualizations', f'heatmap_{weight_name}_{filename}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Saved {output_path}')

