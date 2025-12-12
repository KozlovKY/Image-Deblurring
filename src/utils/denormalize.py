import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize(imgs):
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(IMAGENET_STD).reshape(1, -1, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs
