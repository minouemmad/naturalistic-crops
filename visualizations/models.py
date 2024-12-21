import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model.load_state_dict(weights.get_state_dict(progress=True))  # Ensure 'progress' is used properly
    else:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model



