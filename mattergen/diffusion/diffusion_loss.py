import torch
from functools import partial
from typing import Callable, Dict, Any
from mattergen.common.data.chemgraph import ChemGraph
from pymatgen.core import Element

def l2_distance_loss(x, t, target):
    # Example: x and target are tensors of the same shape
    return torch.nn.functional.mse_loss(x, target)

def l1_distance_loss(x, t, target):
    return torch.nn.functional.l1_loss(x, target)

def volume(x, t):
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
    return torch.abs(torch.sum(a * torch.cross(b, c, dim=1), dim=1))
    

def volume_loss(x, t, target):
    """
    Batched volume loss: computes the absolute difference between each actual volume and the target.
    x.cell: [N, 3, 3]
    target: float
    Returns: [N] tensor of losses
    """
    vol = volume(x, t)
    # Ensure target is broadcastable
    target_tensor = torch.as_tensor(target, dtype=vol.dtype, device=vol.device)
    loss = torch.abs(vol - target_tensor)
    return 10**-5*loss

def compute_species_pair(
    cell: torch.Tensor,         # (B, 3, 3) or (3, 3)
    frac: torch.Tensor,         # (B, N, 3) or (N, 3)
    atomic_numbers: torch.Tensor, # (B, N) or (N,)
    type_A: int,
    type_B: int,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0
) -> torch.Tensor:
    """
    Supports batched or single structures.
    Returns: (B,) if batched, scalar if single.
    """
    # If not batched, add batch dimension
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
        frac = frac.unsqueeze(0)
        atomic_numbers = atomic_numbers.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B = cell.shape[0]
    results = []
    for b in range(B):
        res = _compute_species_pair_single(
            cell[b], frac[b], atomic_numbers[b], type_A, type_B, kernel, sigma, r_cut, alpha
        )
        results.append(res)
    out = torch.stack(results)
    if squeeze_out:
        out = out.squeeze(0)
    return out

def _compute_species_pair_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types: torch.Tensor,
    type_A: int,
    type_B: int,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0
) -> torch.Tensor:
    """
    Compute a differentiable species‐pair value f[A,B], where
        f[A,B] = (1 / |A_A|) * sum_{i in A_A} sum_{j in A_B} g(d_ij),
    under PBC.  g(d) is either:
      - Gaussian:      exp[−(d/sigma)^2]
      - Soft‐cutoff:   sigmoid(alpha * (r_cut − d))

    Args:
      cell      (3×3) Tensor: rows are lattice vectors.
      frac      (N×3) Tensor: fractional coords.
      atomic_numbers list of N ints: atomic numbers.
      type_A    int: atomic number of species A (the one we want to compute the environment of).
      type_B    int: atomic number of species B (the one we are looking in the environment of species A).
      kernel    "gaussian" or "sigmoid"
      sigma     width for Gaussian (Å)
      r_cut     cutoff for sigmoid (Å)
      alpha     sharpness for sigmoid

    Returns:
      f_AB (scalar Tensor), with gradients flowing to cell and positions.
    """
    device = frac.device
    mask_A = (types == type_A)
    mask_B = (types == type_B)
    idx_A = mask_A.nonzero(as_tuple=True)[0]
    idx_B = mask_B.nonzero(as_tuple=True)[0]

    if idx_A.numel() == 0 or idx_B.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=cell.dtype)

    # Prepare cutoffs if needed
    if r_cut is None:
        r_cut= INTER_ATOMIC_CUTOFF[type_A] + INTER_ATOMIC_CUTOFF[type_B] + 0.5  # (N, N)
 
    # PBC images
    shifts = torch.stack(torch.meshgrid(
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
        indexing='ij'
    ), dim=-1).reshape(-1, 3)  # (27, 3)

    # Get frac for A and B
    frac_A = frac[idx_A]  # (n_A, 3)
    frac_B = frac[idx_B]  # (n_B, 3)

    # Expand B atoms to all images
    frac_B_images = frac_B.unsqueeze(1) + shifts.unsqueeze(0)  # (n_B, 27, 3)
    frac_B_images = frac_B_images.reshape(-1, 3)  # (n_B*27, 3)

    # Compute all distances from each A to all B images
    d = frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0)  # (n_A, n_B*27, 3)
    dc = torch.matmul(d, cell)  # (n_A, n_B*27, 3)
    dist = dc.norm(dim=-1)      # (n_A, n_B*27)

    # Kernel
    if kernel == "gaussian":
        G = torch.exp(- (dist / sigma).pow(2))
    elif kernel == "sigmoid":
        G = torch.sigmoid(alpha * (r_cut - dist))
    else:
        raise ValueError("kernel must be 'gaussian' or 'sigmoid'")

    # Sum over all pairs
    n_A = mask_A.sum()
    if n_A < 1e-8:
        return torch.tensor(0.0, device=device, dtype=cell.dtype)
    f_AB = G.sum() / n_A - int(type_A == type_B) # Subtract self-interaction 
    return f_AB

def environment_loss(
    x: ChemGraph,
    t: Any,
    target: dict,
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0
) -> torch.Tensor:
    """
    Computes the environment loss for a given ChemGraph.
    Example of target: {'O-H': 1, 'O-C': 1, 'C-C': 2}
    Meaning that the environment of O should have 1 H and 1 C, and the environment of C should have 2 C.
    The function computes the environment loss for the specified species in the ChemGraph.
    The loss is computed as the absolute difference between the computed environment and the target value.

    Args:
        x (ChemGraph): The input ChemGraph.
        t (Any): Unused, but required for compatibility.
        target (dict): The species of interest and the target value for each environment.
        kernel (str): Kernel type, either 'gaussian' or 'sigmoid'.
        sigma (float): Width for Gaussian kernel.
        r_cut (float | None): Cutoff for sigmoid kernel.
        alpha (float): Sharpness for sigmoid kernel.

    Returns:
        torch.Tensor: The computed environment loss.
    """
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")
    
    cell = x.cell
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")
    
    frac = x.pos
    atomic_numbers = x.atomic_numbers

    # Prepare target pairs and values as lists
    species_pairs = list(target.keys())
    target_values = list(target.values())
    f_AB_list = []
    for species_pair in species_pairs:
        if '-' not in species_pair:
            raise ValueError(f"Invalid species pair format: {species_pair}. Expected format 'A-B'.")
        type_A, type_B = (Element(sym).Z for sym in species_pair.split('-'))
        f_AB_list.append(
            compute_species_pair(
                cell=cell,
                frac=frac,
                atomic_numbers=atomic_numbers,
                type_A=type_A,
                type_B=type_B,
                kernel=kernel,
                sigma=sigma,
                r_cut=r_cut,
                alpha=alpha
            )
        )
    f_AB = torch.stack(f_AB_list)
    
    # Ensure target is broadcastable
    target_tensor = torch.tensor(target_values, dtype=f_AB.dtype, device=f_AB.device, requires_grad=True)
    
    # Compute the loss
    loss = torch.abs(f_AB - target_tensor)
    
    return loss.sum()  # Sum over all pairs to get a single loss value

INTER_ATOMIC_CUTOFF = {1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.7, 22: 1.6, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.2, 33: 1.19, 34: 1.2, 35: 1.2, 36: 1.16, 37: 2.2, 38: 1.95, 39: 1.9, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.4, 55: 2.44, 56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.9, 70: 1.87, 71: 1.87, 72: 1.75, 73: 1.7, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 84: 1.4, 85: 1.5, 86: 1.5, 87: 2.6, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.0, 92: 1.96, 93: 1.9, 94: 1.87, 95: 1.8, 96: 1.69}


def new_loss(x, t, target):
    """
    Example of a new loss function.
    This is just a placeholder and should be replaced with an actual implementation.
    """
    # Assuming x and target are tensors of the same shape
    pass

def make_combined_loss(guidance_dict: dict) -> callable:
    """
    Returns a loss function that combines all guidance losses defined in guidance_dict.
    Each key in guidance_dict must be in LOSS_REGISTRY, and the value is the target.
    More flexibility can be allowed, the value can be a dict containing parameters for the loss function.
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
    "volume": volume_loss,
    "environment": environment_loss,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}