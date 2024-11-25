import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

class DilatedResNet50(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True], **kwargs)

    def forward(self, x, return_feat=False):
        x = super().forward(x)
        if return_feat:
            return x
        return x.flatten(1)

class ResNet50WithFeatureOutput(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)

    def forward(self, x, return_feat=False):
        x = super().forward(x)
        if return_feat:
            return x
        return x.flatten(1)

def load_model_weights(model, weight_path, device):
    if weight_path == "imagenet":
        weights = ResNet50_Weights.DEFAULT
        model.load_state_dict(weights.get_state_dict())
    else:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

