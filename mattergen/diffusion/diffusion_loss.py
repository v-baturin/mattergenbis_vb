import torch
from functools import partial
from typing import Callable, Dict, Any
from mattergen.common.data.chemgraph import ChemGraph

def l2_distance_loss(x, t, target):
    # Example: x and target are tensors of the same shape
    return torch.nn.functional.mse_loss(x, target)

def l1_distance_loss(x, t, target):
    return torch.nn.functional.l1_loss(x, target)

def volume(x, t, target):
    """
    Batched volume loss: computes the absolute difference between each actual volume and the target.
    x.cell: [N, 3, 3]
    target: float
    Returns: [N] tensor of losses
    """
    assert isinstance(x, ChemGraph), "x must be a ChemGraph object"
    cell = x.cell  # shape: [B, 3, 3]
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")
    # a, b, c: [N, 3]
    a, b, c = cell[:, 0, :], cell[:, 1, :], cell[:, 2, :]
    # dot(a, cross(b, c)): [N]
    # cross(b, c): [N, 3]
    vol = torch.abs(torch.sum(a * torch.cross(b, c, dim=1), dim=1))
    # Ensure target is broadcastable
    target_tensor = torch.as_tensor(target, dtype=vol.dtype, device=vol.device)
    return torch.abs(vol - target_tensor)

def make_combined_loss(guidance_dict: dict) -> callable:
    """
    Returns a loss function that combines all guidance losses defined in guidance_dict.
    Each key in guidance_dict must be in LOSS_REGISTRY, and the value is the target.
    """
    partial_losses = []
    for loss_name, target in guidance_dict.items():
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{loss_name}' not found in LOSS_REGISTRY.",
                f"Available losses: {list(LOSS_REGISTRY.keys())}"
            )
        base_loss = LOSS_REGISTRY[loss_name]
        partial_losses.append(partial(base_loss, target=target))
    def combined_loss(x, t):
    #TODO: Verify that a simple sum is appropriate for combining the losses
        return sum(loss(x, t) for loss in partial_losses)
    return combined_loss

LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "l2_distance": l2_distance_loss,
    "l1_distance": l1_distance_loss,
    "volume": volume,
    # Add more loss functions as needed
}