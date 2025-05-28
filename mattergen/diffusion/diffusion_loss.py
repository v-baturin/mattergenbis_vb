import torch
from functools import partial

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
    "volume": volume,
    # Add more loss functions as needed
}

def make_combined_loss(guidance_dict: dict) -> callable:
    """
    Returns a loss function that combines all guidance losses defined in guidance_dict.
    Each key in guidance_dict must be in LOSS_REGISTRY, and the value is the target.
    """
    partial_losses = []
    for loss_name, target in guidance_dict.items():
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Loss '{loss_name}' not found in LOSS_REGISTRY.",f"Available losses: {list(LOSS_REGISTRY.keys())}")
        base_loss = LOSS_REGISTRY[loss_name]
        partial_losses.append(partial(base_loss, target=target))
    def combined_loss(x, t):
        #TODO: Verify that a simple sum is appropriate for combining the losses
        return sum(loss(x, t) for loss in partial_losses)
    return combined_loss