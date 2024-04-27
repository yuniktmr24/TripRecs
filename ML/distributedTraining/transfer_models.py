import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=57):
    # Dynamically load the model with pretrained weights
    if hasattr(models, model_name):
        ModelClass = getattr(models, model_name)
        try:
            # Attempt to load default weights, if available
            model_transfer = ModelClass(weights=ModelClass.Weights.DEFAULT)
        except AttributeError:
            # If default weights aren't specified, fall back to pretrained=True
            model_transfer = ModelClass(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} is not available in torchvision.")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Classifier modification depends on the architecture
    if 'resnet' in model_name:
        num_ftrs = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(num_ftrs, n_classes)
    elif 'vgg' in model_name:
        num_ftrs = model_transfer.classifier[6].in_features
        model_transfer.classifier[6] = nn.Linear(num_ftrs, n_classes)
    elif 'densenet' in model_name:
        num_ftrs = model_transfer.classifier.in_features
        model_transfer.classifier = nn.Linear(num_ftrs, n_classes)
    elif 'googlenet' in model_name:
        num_ftrs = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(num_ftrs, n_classes)
    elif 'mobilenet' in model_name:
        num_ftrs = model_transfer.classifier[-1].in_features
        model_transfer.classifier[-1] = nn.Linear(num_ftrs, n_classes)
    elif 'alexnet' in model_name:
        num_ftrs = model_transfer.classifier[6].in_features
        model_transfer.classifier[6] = nn.Linear(num_ftrs, n_classes)
    else:
        raise NotImplementedError(f"Model type {model_name} not implemented for classifier modification.")

    return model_transfer
