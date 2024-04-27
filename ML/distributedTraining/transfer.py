import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=57):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(weights='ResNet18_Weights.DEFAULT')

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    # loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    for name, param in model_transfer.named_parameters():
        if param.requires_grad:
            param.requires_grad = False #freeze

    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs  = model_transfer.fc.in_features#

    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer

