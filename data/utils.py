#
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

#
from typing import Optional

# Reference : https://github.com/PistonY/torch-toolbox
from torchtoolbox.transform import Cutout

#
from utils import logger


def get_train_valid_dataset(
    dataset: ImageFolder,
    split_ratio: Optional[float] = 0.2,
    random_state: Optional[int] = 42,
):
    """ """

    # Split the train/test with each class should appear on both train/test dataset
    #
    indices = list(range(len(dataset)))  # indices of the dataset
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=split_ratio,
        stratify=dataset.targets,
        random_state=random_state,
    )
    train_indices.sort()
    valid_indices.sort()

    return train_indices, valid_indices

    # # Creating sub dataset from valid indices
    # # Do not shuffle valid dataset, let the image in order
    # valid_indices.sort()
    # valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    # train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # return train_dataset, valid_dataset


def get_dataset_stats(dataset: ImageFolder):
    """
    Return the statistics (mean, standard deviation) of dataset
    Args:
        dataset (torch.utils.data.Dataset): Dataset to calculate statistics from

    Returns:
        stats (dict): dict with mean, std of the dataset
    """

    #
    mean = torch.zeros(3)
    std = torch.zeros(3)

    #
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for inputs, _ in loader:
        for d in range(3):
            mean[d] += inputs[:, d, :, :].mean()
            std[d] += inputs[:, d, :, :].std()
    mean.div_(len(loader))
    std.div_(len(loader))

    # package into a dictionary format
    stats = {"mean": mean.numpy(), "std": std.numpy()}
    return stats


def create_train_valiad_loader(opts):
    # Dataset preparation
    # 1. Set dataset mean, standard deviation
    # 2.

    # 1. Data Augumentation
    # 2. Split 80/20 of training dataset as train/valid
    # 1. All of the model is declared in folder ./models
    input_resolution = getattr(opts, "model.input_resolution", 224)
    root_train = getattr(opts, "dataset.root_train", None)
    batch_size = getattr(opts, "dataset.batch_size", 32)
    num_workers = getattr(opts, "dataset.num_workers", 1)

    assert root_train is not None, "[ERROR] Attribute $root_train should not be None"
    dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_set = [transforms.RandAugment()]
    train_transform_set = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_resolution
            ),  # Resize the image into a fixed shape
            # transforms.Resize((input_resolution, input_resolution)),
            # Reorder transform randomly
            transforms.RandomOrder(transform_set),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
            transforms.RandomErasing(),
        ]
    )
    valid_transform_set = transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ]
    )

    # Split 80/20 percent of training dataset as train/valid
    train_dataset = ImageFolder(root_train, transform=train_transform_set)
    train_indices, valid_indices = get_train_valid_dataset(train_dataset)

    train_dataset = ImageFolder(root_train, transform=train_transform_set)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = ImageFolder(root_train, transform=valid_transform_set)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    logger.info(str(train_dataset.dataset.transform))
    logger.info(str(valid_dataset.dataset.transform))

    # 
    num_gpus = getattr(opts, "ddp.num_gpus", 0)
    if num_gpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # DataLoader preparation
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, valid_dataset, train_loader, valid_loader


def create_eval_loader(opts):
    """ """

    #
    input_resolution = getattr(opts, "model.input_resolution", 224)
    root_eval = getattr(opts, "dataset.root_eval", None)
    batch_size = getattr(opts, "dataset.batch_size_eval", 32)
    # TODO: Deal with thread problem when evaluating with num_workers > 1
    # num_workers = getattr(opts, "dataset.num_workers", 1)
    num_workers = 1
    assert (
        root_eval is not None
    ), "[ERROR] Attribute $datasetroot_eval should not be None"

    dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    eval_transform_set = transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std),
        ]
    )

    # Create evaluate dataset & dataloader
    eval_dataset = ImageFolder(root_eval, transform=eval_transform_set)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return eval_dataset, eval_loader
