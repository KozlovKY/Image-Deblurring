import torch


def denormalize(imgs, mean, std):
    mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).reshape(1, -1, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs
