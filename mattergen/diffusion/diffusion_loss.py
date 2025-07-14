import torch
from functools import partial
from typing import Callable, Dict, Any
from mattergen.common.data.chemgraph import ChemGraph
from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import pandas as pd
from mattersim.forcefield import MatterSimCalculator
from ase import Atoms
from ase.data import chemical_symbols

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
    atomic_numbers: torch.Tensor, # (\Sum N_i)
    num_atoms: torch.Tensor, # Number of atoms in each material, (B,)
    type_A: int,
    type_B: int,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0,
    mode: str | None = None
) -> torch.Tensor:
    """
    Supports batched or single structures.
    Returns: (B,) if batched, scalar if single.
    """
    # If not batched, add batch dimension
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B = cell.shape[0]
    results = []
    count = 0
    for b in range(B):
        count_ = count+num_atoms[b]
        res = _compute_species_pair_single(
            cell[b], frac[count:count_], atomic_numbers[count:count_], type_A, type_B, kernel, sigma, r_cut, alpha, mode
        )
        results.append(res)
        count = count_
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
    alpha: float = 8.0,
    mode: str | None = None
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
      types     list of N ints: atomic numbers.
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
        return cell.sum()*frac.sum() * 0.0 # No A or B atoms, return 0
    
    # Prepare cutoffs if needed
    if r_cut is None:
        r_cut= INTER_ATOMIC_CUTOFF[type_A] + INTER_ATOMIC_CUTOFF[type_B] + 1.0  # (N, N)
 
    # PBC images
    global shifts
    if shifts is None:
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
        # to keep the graph differentiable, return 0
        return cell.sum()*frac.sum() * 0.0
    f_AB = G.sum() / n_A - int(type_A == type_B) # Subtract self-interaction 
    return f_AB

def environment_loss(
    x: ChemGraph,
    t: Any,
    target: dict,
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    alpha: float = 8.0
) -> torch.Tensor:
    """
    Computes the environment loss for a given ChemGraph.
    Example of target: {'O-H': 1, 'O-C': [1,2.0], 'C-C': 2}
    Meaning that the environment of O should have 1 H and 1 C but with a r_cut of 2.0 for C, and the environment of C should have 2 C.
    The non-specified distance will be using the default r_cut, which is the sum of the covalent radii of the two species plus 1.0.
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
    num_atoms = x.num_atoms

    # Extract mode if present
    mode = target.pop("mode", None)

    global species_pairs
    global target_values
    global r_cuts

    # Prepare target pairs and values as lists
    if species_pairs is None or target_values is None:
        species_pairs = list(target.keys())
        target_values = [v[0] if isinstance(v, list) else v for v in target.values()]
        r_cuts =  [v[1] if isinstance(v, list) else None for v in target.values()]
    f_AB_list = []

    for species_pair, r_cut in zip(species_pairs,r_cuts):
        if species_pair == "mode":
            continue  # skip 'mode' key, already handled
        if '-' not in species_pair:
            raise ValueError(f"Invalid species pair format: {species_pair}. Expected format 'A-B'.")
        type_A, type_B = (Element(sym).Z for sym in species_pair.split('-'))
        f_AB_list.append(
            compute_species_pair(
                cell=cell,
                frac=frac,
                atomic_numbers=atomic_numbers,
                num_atoms=num_atoms,
                type_A=type_A,
                type_B=type_B,
                kernel=kernel,
                sigma=sigma,
                r_cut=r_cut,
                alpha=alpha,
                mode=mode
            )
        )
    f_AB = torch.stack(f_AB_list) # shape: (num_pairs, batch_size)
    
    # Ensure target is broadcastable
    global target_tensor
    if target_tensor is None:
        target_tensor = torch.tensor(target_values, dtype=f_AB.dtype, device=f_AB.device).unsqueeze(1) # (num_pairs, 1)
        if target_tensor.shape[1] == 1 and len(f_AB.shape) > 1:
            target_tensor = target_tensor.expand(-1, f_AB.shape[1])  # (num_pairs, batch_size)

    # Compute the loss
    if mode == "l1" or mode == None or mode == "test":
        loss = torch.abs(f_AB - target_tensor)
    elif mode == "l2":
        loss = torch.nn.functional.mse_loss(f_AB, target_tensor, reduction='none')
    elif mode == "huber": 
        loss = torch.nn.functional.huber_loss(f_AB, target_tensor, reduction='mean', delta = 1.5) 
    # eps sensitif ?
    elif mode == "divergence":
        pass  # Placeholder for divergence loss, not implemented
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'l1', 'huber', and 'l2'.")
    
    return loss.sum(dim=0)  # Sum over all pairs to get a single loss value

INTER_ATOMIC_CUTOFF = {1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.7, 22: 1.6, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.2, 33: 1.19, 34: 1.2, 35: 1.2, 36: 1.16, 37: 2.2, 38: 1.95, 39: 1.9, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.4, 55: 2.44, 56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.9, 70: 1.87, 71: 1.87, 72: 1.75, 73: 1.7, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 84: 1.4, 85: 1.5, 86: 1.5, 87: 2.6, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.0, 92: 1.96, 93: 1.9, 94: 1.87, 95: 1.8, 96: 1.69}
PDIAG = None
calc = None
species_pairs = None
target_values = None
r_cuts = None
target_tensor = None
shifts = None

def composition(num, pos):
    """
    Computes the composition of a list of atoms.
    li is a list of int with 101 beeing an empty atom.
    Returns a list of strings with the chemical symbols of the atoms.
    Example: [1, 101, 8, 8, 101] -> ['H', 'O', 'O']
    """
    return num[num != 101], pos[num != 101]

def energy_hull(x, t, target=None):
    """
    Computes the energy above the hull for a given composition and energy.
    x is a chemgraph batch
    The function uses a precomputed phase diagram to determine the energy above the hull.
    """
    global calc
    if calc is None:
        calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=x.cell.device)
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")

    B = x.cell.shape[0]
    count = 0
    e_hull = []
    for b in range(B):
        count_ = count+x.num_atoms[b]
        comp, sc_pos = composition(x.atomic_numbers[count:count_], x.pos[count:count_])
        atom = Atoms(comp, scaled_positions = sc_pos, cell = x.cell[b], pbc=True )
        atom.calc = calc
        energy = atom.get_potential_energy()  # Get the energy of the atoms
        e_hull.append(_energy_hull((atom.get_chemical_formula(), energy)))
        count = count_
    return e_hull

def _energy_hull(x):
    """
    Computes the energy above the hull for a given composition and energy.
    x is a (Compo, Energy) tuple (str, float)
    CSV : Compo , Energy
    """
    dir = "/Data/auguste.de-lambilly/mattergenbis/phase_diagram/" # This should be the directory where the phase diagram is saved
    global PDIAG
    if PDIAG is None:
        # Load the CSV file only once
        csv = pd.read_csv(dir + "LiCoO.csv")
        li = [ PDEntry(composition=Composition(csv["Formula"][i]), energy=csv["Energy"][i]) for i in range(len(csv)) ]
        PDIAG = PhaseDiagram(li)
        del csv, li
    x_ = PDEntry(composition = Composition(x[0]), energy=x[1])  # Assuming x has composition and energy attributes
    above_hull = PDIAG.get_e_above_hull(x_)
    return above_hull

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
    "energy_hull": energy_hull,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}