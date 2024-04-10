import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

def get_model_transfer_learning(model_name="resnet18", n_classes=57):
    # Get the requested architecture
    if hasattr(models, model_name):
        # Load model with default weights
        if "resnet" in model_name:
            model_transfer = getattr(models, model_name)(weights='ResNet18_Weights.DEFAULT')
        elif "vgg" in model_name:
            model_transfer = getattr(models, model_name)(weights='VGG16_Weights.DEFAULT')
        else:
            raise ValueError(f"Unsupported model type for {model_name}")
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Modify the classifier layer based on model architecture
    if "resnet" in model_name:
        num_ftrs = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(num_ftrs, n_classes)
    elif "vgg" in model_name:
        num_ftrs = model_transfer.classifier[6].in_features
        model_transfer.classifier[6] = nn.Linear(num_ftrs, n_classes)

    return model_transfer


