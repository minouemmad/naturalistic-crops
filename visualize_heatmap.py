import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet18FeatureExtractor(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)

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

model = ResNet18FeatureExtractor()

# Load pretrained weights for ResNet-18
pretrained_weights = models.ResNet18_Weights.IMAGENET1K_V1
model.load_state_dict(pretrained_weights.get_state_dict(progress=True))

model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

imagenet_data = datasets.ImageFolder(
    root='/vast/work/public/ml-datasets/imagenet/train',
        transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)

dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=False)

# Denormalize function to reverse normalization for visualization
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# Generate heatmap from the feature maps
def generate_heatmap(model, img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = model(img_tensor, return_feat=True)  # Get feature map from the last layer
    heatmap = features.mean(dim=1).squeeze()  # Averaging across channels
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0).squeeze(0)
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()  # Normalize to [0, 1]
    return heatmap

# Visualize and overlay heatmap on the original image
def visualize_heatmap(image, heatmap, idx):
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.title('Feature Activation Heatmap')
    plt.savefig(f'heatmap_{idx}.png')
    plt.show()

# Loop through a few images and visualize the heatmap
for i, (img, label) in enumerate(dataloader):
    if i == 5:  # Visualize for 5 images
        break

    img_tensor = img.to(device)
    heatmap = generate_heatmap(model, img_tensor)

    # Denormalize the image for visualization
    img_denormalized = denormalize(img.squeeze(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert tensor image back to PIL for visualization
    img_pil = transforms.ToPILImage()(img_denormalized)

    visualize_heatmap(img_pil, heatmap, i)

