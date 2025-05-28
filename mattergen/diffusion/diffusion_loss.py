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
    Example volume loss function : compute the absolute difference between the actual volume and the targeted one
    """
    assert isinstance(x, ChemGraph), "x must be a ChemGraph object"
    cell = x.cell  # shape: [1, 3, 3]
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")
    cell = cell.squeeze(0)  # shape: [3, 3]
    a, b, c = cell[0], cell[1], cell[2]  # lattice vectors
    vol = torch.abs(torch.dot(a, torch.cross(b, c)))
    return torch.abs(vol - target)

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
        # If x is a list or tuple, apply losses to each ChemGraph and sum
        if isinstance(x, (list, tuple)):
            return sum(
                sum(loss(xi, t) for loss in partial_losses)
                for xi in x
            )
        # Otherwise, assume x is a single ChemGraph
        return sum(loss(x, t) for loss in partial_losses)
    return combined_loss

LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "l2_distance": l2_distance_loss,
    "l1_distance": l1_distance_loss,
    "volume": volume,
    # Add more loss functions as needed
}