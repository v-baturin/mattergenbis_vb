from functools import partial
import torch

def l2_distance_loss(x, t, target):
    # Example: x and target are tensors of the same shape
    return torch.nn.functional.mse_loss(x, target)

def l1_distance_loss(x, t, target):
    return torch.nn.functional.l1_loss(x, target)

def volume(x, t, target):
    """
    Example volume loss function : compute the absolute difference between the actual volume and the targeted one
    """
    assert x.shape == target.shape, "x and target must have the same shape"
    
    return torch.sum(torch.abs(x - target))

LOSS_REGISTRY = {
    "l2_distance": l2_distance_loss,
    "l1_distance": l1_distance_loss,
}