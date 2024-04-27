import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
import os

from helpers import compute_mean_and_std, get_data_location
# import matplotlib.pyplot as plt


# File - data.py
def get_data_loaders(
        world_size, rank, batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1,
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    # if num_workers == -1:
    #     # Use all cores
    #     num_workers = multiprocessing.cpu_count()

    num_workers = 0


    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
    data_transforms = {
        "train": transforms.Compose(
            [transforms.Resize(256),
             transforms.RandomCrop(224),
             transforms.RandAugment(4, 4),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        ),
        "valid": transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        ),
        "test": transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        ),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["train"]
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train",
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["valid"]
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_sampler_dist = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    valid_sampler_dist = DistributedSampler(valid_data, num_replicas=world_size, rank=rank)


    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler_dist,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler_dist,
        num_workers=num_workers
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        # (add the test transform)
        transform=data_transforms["test"]
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    test_sampler_dist = DistributedSampler(valid_data, num_replicas=world_size, rank=rank)


    data_loaders["test"] = torch.utils.data.DataLoader(
        # (remember to add shuffle=False as well)
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=test_sampler,
        shuffle=False
    )

    return data_loaders


def print_class_names(data_loaders):
    # Get class names from the train data loader
    class_names = data_loaders['train'].dataset.classes
    print(class_names)





