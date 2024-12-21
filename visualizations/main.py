import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import BDDDataset
from models import DilatedResNet50, ResNet50WithFeatureOutput, load_model_weights
from utils import generate_heatmap, calculate_overlap_score, denormalize
from visualization import visualize_heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loader setup
bdd_data = BDDDataset(
    root_dir='/scratch/ch3451/datasets/bdd100k-evals/images/10k/train',
    mask_dir='/scratch/ch3451/datasets/bdd100k-evals/labels/sem_seg/masks/train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
dataloader = DataLoader(bdd_data, batch_size=1, shuffle=False)

# Load models
dino_model = DilatedResNet50().to(device)
imagenet_model = ResNet50WithFeatureOutput().to(device)
load_model_weights(dino_model, "/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth")
load_model_weights(imagenet_model, "imagenet")

# Process data
for img, mask, img_path in dataloader:
    img_tensor = img.to(device)
    dino_heatmap = generate_heatmap(dino_model, img_tensor)
    imagenet_heatmap = generate_heatmap(imagenet_model, img_tensor)

    mask = mask.to(device)
    dino_overlap = calculate_overlap_score(dino_heatmap, mask)
    imagenet_overlap = calculate_overlap_score(imagenet_heatmap, mask)

    original_image = denormalize(img_tensor.squeeze(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    visualize_heatmap(original_image, dino_heatmap, "DINO", img_path[0], dino_overlap)
    visualize_heatmap(original_image, imagenet_heatmap, "ImageNet", img_path[0], imagenet_overlap)

