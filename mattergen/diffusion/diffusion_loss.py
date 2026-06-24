import torch
from functools import partial
from typing import Callable, Dict, Any
from mattergen.common.data.chemgraph import ChemGraph
from pymatgen.core import Element, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import pandas as pd
# from mattersim.datasets.utils.convertor import ChemGraphBatchConvertor
# from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from ase import Atoms
from ase.data import chemical_symbols

PDIAG = None
calc = None
converter = None
species_pairs = None
target_values = None
r_cuts = None
target_tensor = None
shifts = None

def clear_globals():
    global PDIAG, calc, converter, species_pairs, target_values, r_cuts, target_tensor
    PDIAG = None
    calc = None
    converter = None
    species_pairs = None
    target_values = None
    r_cuts = None
    target_tensor = None



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
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
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
    return 10 ** -5 * loss

def volume_pa(x, t):
    """
    Batched computatuion of volume per atom.
    """
    return volume(x, t) / x.num_atoms

def volume_pa_loss(x, t, target):
    """
    Batched computatuion of volume per atom.
    """
    vol_pa = volume_pa(x,t)
    target_tensor = torch.as_tensor(target, dtype=vol_pa.dtype, device=vol_pa.device)
    loss = torch.abs(vol_pa - target_tensor)
    return loss


# def compute_species_pair(
#         cell: torch.Tensor,  # (B, 3, 3) or (3, 3)
#         frac: torch.Tensor,  # (B, N, 3) or (N, 3)
#         atomic_numbers: torch.Tensor,  # (\Sum N_i)
#         num_atoms: torch.Tensor,  # Number of atoms in each material, (B,)
#         type_A: int,
#         type_B: int,
#         kernel: str = "sigmoid",
#         sigma: float = 1.0,
#         r_cut: float | None = None,
#         alpha: float = 8.0,
#         mode: str | None = None
# ) -> torch.Tensor:
#     """
#     Supports batched or single structures.
#     Returns: (B,) if batched, scalar if single.
#     """
#     # If not batched, add batch dimension
#     if cell.ndim == 2:
#         cell = cell.unsqueeze(0)
#         squeeze_out = True
#     else:
#         squeeze_out = False
#
#     B = cell.shape[0]
#     results = []
#     count = 0
#     for b in range(B):
#         count_ = count + num_atoms[b]
#         res = _compute_species_pair_single(
#             cell[b], frac[count:count_], atomic_numbers[count:count_], type_A, type_B, kernel, sigma, r_cut, alpha, mode
#         )
#         results.append(res)
#         count = count_
#     out = torch.stack(results)
#     if squeeze_out:
#         out = out.squeeze(0)
#     return out


# def _compute_species_pair_single(
#         cell: torch.Tensor,
#         frac: torch.Tensor,
#         types: torch.Tensor,
#         type_A: int,
#         type_B: int,
#         kernel: str = "sigmoid",
#         sigma: float = 1.0,
#         r_cut: float | None = None,
#         alpha: float = 8.0,
#         mode: str | None = None
# ) -> torch.Tensor:
#     """
#     Compute a differentiable species‐pair value f[A,B], where
#         f[A,B] = (1 / |A_A|) * sum_{i in A_A} sum_{j in A_B} g(d_ij),
#     under PBC.  g(d) is either:
#       - Gaussian:      exp[−(d/sigma)^2]
#       - Soft‐cutoff:   sigmoid(alpha * (r_cut − d))
#
#     Args:
#       cell      (3×3) Tensor: rows are lattice vectors.
#       frac      (N×3) Tensor: fractional coords.
#       types     list of N ints: atomic numbers.
#       type_A    int: atomic number of species A (the one we want to compute the environment of).
#       type_B    int: atomic number of species B (the one we are looking in the environment of species A).
#       kernel    "gaussian" or "sigmoid"
#       sigma     width for Gaussian (Å)
#       r_cut     cutoff for sigmoid (Å)
#       alpha     sharpness for sigmoid
#
#     Returns:
#       f_AB (scalar Tensor), with gradients flowing to cell and positions.
#     """
#     device = frac.device
#     mask_A = (types == type_A)
#     mask_B = (types == type_B)
#     idx_A = mask_A.nonzero(as_tuple=True)[0]
#     idx_B = mask_B.nonzero(as_tuple=True)[0]
#
#     if idx_A.numel() == 0 or idx_B.numel() == 0:
#         return cell.sum() * frac.sum() * 0.0  # No A or B atoms, return 0
#
#     # Prepare cutoffs if needed
#     if r_cut is None:
#         r_cut = INTER_ATOMIC_CUTOFF[type_A] + INTER_ATOMIC_CUTOFF[type_B] + 0.5  # (N, N)
#
#     # PBC images
#     global shifts
#     if shifts is None:
#         shifts = torch.stack(torch.meshgrid(
#             torch.arange(-1, 2, device=device),
#             torch.arange(-1, 2, device=device),
#             torch.arange(-1, 2, device=device),
#             indexing='ij'
#         ), dim=-1).reshape(-1, 3)  # (27, 3)
#
#     # Get frac for A and B
#     frac_A = frac[idx_A]  # (n_A, 3)
#     frac_B = frac[idx_B]  # (n_B, 3)
#
#     # Expand B atoms to all images
#     frac_B_images = frac_B.unsqueeze(1) + shifts.unsqueeze(0)  # (n_B, 27, 3)
#     frac_B_images = frac_B_images.reshape(-1, 3)  # (n_B*27, 3)
#
#     # Compute all distances from each A to all B images
#     d = frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0)  # (n_A, n_B*27, 3)
#     dc = torch.matmul(d, cell)  # (n_A, n_B*27, 3)
#     dist = dc.norm(dim=-1)  # (n_A, n_B*27)
#
#     # Kernel
#     if kernel == "gaussian":
#         G = torch.exp(- (dist / sigma).pow(2))
#     elif kernel == "sigmoid":
#         G = torch.sigmoid(alpha * (r_cut - dist))
#     else:
#         raise ValueError("kernel must be 'gaussian' or 'sigmoid'")
#
#     # Sum over all pairs
#     n_A = mask_A.sum()
#     if n_A < 1e-8:
#         # to keep the graph differentiable, return 0
#         return cell.sum() * frac.sum() * 0.0
#     f_AB = G.sum() / n_A - int(type_A == type_B)  # Subtract self-interaction
#     return f_AB

# --- Shared per-A soft neighbor counts (PBC, 27 images) ---
def _as_atomic_number_tuple(
    type_B: int | list[int] | tuple[int, ...] | set[int],
) -> tuple[int, ...]:
    """Normalize one or more neighbor atomic numbers to a de-duplicated tuple."""
    if isinstance(type_B, torch.Tensor):
        values = type_B.detach().cpu().reshape(-1).tolist()
    elif isinstance(type_B, (list, tuple, set)):
        values = list(type_B)
    else:
        values = [type_B]
    return tuple(dict.fromkeys(int(v) for v in values))


def _parse_coordination_constraint(species_constraint: str) -> tuple[int, tuple[int, ...]]:
    """
    Parse coordination keys.

    Supported forms:
      A-B
      A-[B,C,D]
      A-B,C,D
    """
    if "-" not in species_constraint:
        raise ValueError(f"Invalid species pair format: {species_constraint}. Expected 'A-B'.")

    species_A, species_B = species_constraint.split("-", maxsplit=1)
    species_A = species_A.strip()
    species_B = species_B.strip()

    if not species_A or not species_B:
        raise ValueError(f"Invalid species pair format: {species_constraint}. Expected 'A-B'.")

    if species_B.startswith("[") and species_B.endswith("]"):
        species_B = species_B[1:-1]

    species_B_symbols = [symbol.strip() for symbol in species_B.split(",") if symbol.strip()]
    if not species_B_symbols:
        raise ValueError(
            f"Invalid neighbor species set in {species_constraint}. Expected at least one element."
        )

    type_A = Element(species_A).Z
    type_Bs = tuple(Element(symbol).Z for symbol in species_B_symbols)
    return type_A, _as_atomic_number_tuple(type_Bs)


def _default_coordination_r_cut(type_A: int, type_Bs: tuple[int, ...]) -> float:
    """Use the largest default pair cutoff for an A-neighbor-set constraint."""
    return max(
        INTER_ATOMIC_CUTOFF[type_A] + INTER_ATOMIC_CUTOFF[type_B] + 0.5 for type_B in type_Bs
    )


def _soft_neighbor_counts_per_A_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types,                     # accepts numpy or torch
    type_A: int,
    type_B: int | list[int] | tuple[int, ...] | set[int],
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0,
    **kwargs
) -> torch.Tensor:
    """
    Returns a differentiable vector C (n_A,) of soft B-neighbor counts for each A atom:
        C_i = sum_{j in A_B} g(d_ij), with 27 PBC images.
    `type_B` may be a single atomic number or a set/list/tuple of atomic numbers.
    Kernel: 'gaussian' (exp[-(d/sigma)^2]) or 'sigmoid' (sigmoid(alpha*(r_cut-d))).
    If A is included in B, subtract 1.0 per A (legacy parity).
    """
    # Normalize inputs to torch on the same device/dtype (keeps grad from frac if it has one)
    frac = torch.as_tensor(frac, dtype=getattr(frac, "dtype", torch.float32),
                           device=getattr(frac, "device", None))
    cell = torch.as_tensor(cell, dtype=frac.dtype, device=frac.device)
    types = torch.as_tensor(types, dtype=torch.int64, device=frac.device)
    type_Bs = _as_atomic_number_tuple(type_B)

    device = frac.device
    mask_A = (types == type_A)
    mask_B = torch.zeros_like(mask_A, dtype=torch.bool)
    for type_B_i in type_Bs:
        mask_B = mask_B | (types == type_B_i)
    idx_A = mask_A.nonzero(as_tuple=True)[0]
    idx_B = mask_B.nonzero(as_tuple=True)[0]

    if idx_A.numel() == 0 or idx_B.numel() == 0:
        # Return a scalar-zero preserving the graph
        return cell.sum()*frac.sum() * torch.zeros(1, device=device)

    if r_cut is None:
        # chemistry-informed default
        r_cut = _default_coordination_r_cut(type_A, type_Bs)

    # PBC 27 images
    global shifts
    if shifts is None or shifts.device != device:
        shifts = torch.stack(torch.meshgrid(
            torch.arange(-1, 2, device=device),
            torch.arange(-1, 2, device=device),
            torch.arange(-1, 2, device=device),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)  # (27,3)

    frac_A = frac[idx_A]  # (n_A,3)
    frac_B = frac[idx_B]  # (n_B,3)

    # Expand B over images
    frac_B_images = (frac_B.unsqueeze(1) + shifts.unsqueeze(0)).reshape(-1, 3)  # (n_B*27,3)

    # Distances
    d = frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0)  # (n_A, n_B*27, 3)
    dc = torch.matmul(d, cell)                             # (n_A, n_B*27, 3)
    dist = dc.norm(dim=-1)                                 # (n_A, n_B*27)

    # Kernel
    if kernel == "gaussian":
        G = torch.exp(- (dist / sigma).pow(2))
    elif kernel == "sigmoid":
        G = torch.sigmoid(alpha * (r_cut - dist))
    else:
        raise ValueError("kernel must be 'gaussian' or 'sigmoid'")

    counts = G.sum(dim=1)  # (n_A,)

    # Remove self-interaction per A if A is part of the neighbor set.
    if int(type_A) in type_Bs:
        counts = counts - 1.0

    return counts


# --- Batched share metric: fraction of A atoms with ~target B neighbors ---
def _compute_target_share_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types: torch.Tensor,
    type_A: int,
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    tau: float = 0.5,
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0,
) -> torch.Tensor:
    """
    share = (1/|A|) * sum_i exp(- ((C_i - target)/tau)^2 ), where C_i are soft counts.
    """
    C = _soft_neighbor_counts_per_A_single(
        cell, frac, types, type_A, type_B, kernel=kernel, sigma=sigma, r_cut=r_cut, alpha=alpha
    )
    # If empty sentinel, return as-is (0-like scalar preserving graph)
    if C.numel() == 1 and C.squeeze().abs().sum() == 0:
        return C.squeeze()
    H = torch.exp(-((C - float(target)) / float(tau)).pow(2))
    return H.mean()


def compute_target_share(
    cell: torch.Tensor,           # (B, 3, 3) or (3, 3)
    frac: torch.Tensor,           # (B, N, 3) or (N, 3)
    atomic_numbers: torch.Tensor, # (sumN_i,)
    num_atoms: torch.Tensor,      # (B,)
    type_A: int,
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    tau: float = 0.5,
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    r_cut: float | None = None,
    alpha: float = 8.0,
) -> torch.Tensor:
    """
    Batched share metric. Returns (B,) if batched, scalar if single.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0); squeeze_out = True
    else:
        squeeze_out = False

    B = cell.shape[0]
    results = []
    count = 0
    for b in range(B):
        count_ = count + num_atoms[b]
        res = _compute_target_share_single(
            cell[b], frac[count:count_], atomic_numbers[count:count_],
            type_A, type_B,
            target=target, tau=tau, kernel=kernel, sigma=sigma, r_cut=r_cut, alpha=alpha
        )
        results.append(res)
        count = count_
    out = torch.stack(results)
    return out.squeeze(0) if squeeze_out else out


# --- New loss: maximize fraction at exact target coordination (minimize 1 - share) ---
def target_coordination_loss(
    x: ChemGraph,
    t: Any,
    target: dict,
    kernel: str = "sigmoid",
    sigma: float = 1.0,
    alpha: float = 8.0,
    default_tau: float = 0.5,
) -> torch.Tensor:
    """
    Target-coordination guidance. For each A-B in `target` with integer neighbor target k, minimize 1 - share_A(k; B),
    where share_A(k; B) is computed with a Gaussian window of width `tau` in coordination space.

    `target` can be:
      {'A-B': k}
      {'A-B': [k, r_cut]}
      {'A-B': [k, r_cut, tau]}
      {'A-[B,C,D]': k}
    """
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")

    cell = x.cell
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")

    frac = x.pos
    atomic_numbers = x.atomic_numbers
    num_atoms = x.num_atoms

    shares = []
    for species_pair, val in target.items():
        if species_pair == "mode":
            continue

        if isinstance(val, (list, tuple)):
            tgt = float(val[0])
            rcut = (None if len(val) < 2 or val[1] is None else float(val[1]))
            tau = (default_tau if len(val) < 3 or val[2] is None else float(val[2]))
        else:
            tgt = float(val); rcut = None; tau = default_tau

        ZA, ZBs = _parse_coordination_constraint(species_pair)

        sh = compute_target_share(
            cell=cell, frac=frac, atomic_numbers=atomic_numbers, num_atoms=num_atoms,
            type_A=ZA, type_B=ZBs,
            target=tgt, tau=tau, kernel=kernel, sigma=sigma, r_cut=rcut, alpha=alpha
        )  # (B,)
        shares.append(sh)

    if len(shares) == 0:
        # No valid pairs: return zero-like (B,) preserving graph
        zeros = torch.zeros_like(num_atoms, dtype=cell.dtype, device=cell.device)
        return zeros * 0.0

    shares_tensor = torch.stack(shares, dim=0)   # (P, B)
    loss = 1.0 - shares_tensor                    # (P, B)
    return loss.sum(dim=0)                        # (B,)


def dominant_environment_loss(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for target_coordination_loss."""
    return target_coordination_loss(*args, **kwargs)


def compute_mean_coordination(
        cell: torch.Tensor,  # (B, 3, 3) or (3, 3)
        frac: torch.Tensor,  # (B, N, 3) or (N, 3)
        atomic_numbers: torch.Tensor,  # (\Sum N_i)
        num_atoms: torch.Tensor,  # (B,)
        type_A: int,
        type_B: int | list[int] | tuple[int, ...] | set[int],
        kernel: str = "sigmoid",
        sigma: float = 1.0,
        r_cut: float | None = None,
        alpha: float = 8.0,
) -> torch.Tensor:
    """
    Batched mean A–B soft coordination:
       mean_i sum_j g(d_ij), using `_soft_neighbor_counts_per_A_single`.
    `type_B` may also be a collection of neighbor atomic numbers; those species are
    counted together using one cutoff. If `r_cut` is omitted, the cutoff is the
    maximum default cutoff over all A-B pairs in the collection.
    Returns: (B,) if batched, scalar if single.
    """
    # Normalize to batched
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B = cell.shape[0]
    results = []
    count = 0
    for b in range(B):
        count_ = count + num_atoms[b]
        C = _soft_neighbor_counts_per_A_single(
            cell[b], frac[count:count_], atomic_numbers[count:count_],
            type_A=type_A, type_B=type_B, kernel=kernel, sigma=sigma, r_cut=r_cut, alpha=alpha
        )
        # If empty sentinel, just append C.squeeze()
        if C.numel() == 1 and C.squeeze().abs().sum() == 0:
            results.append(C.squeeze())
        else:
            results.append(C.mean())
        count = count_
    out = torch.stack(results)
    if squeeze_out:
        out = out.squeeze(0)
    return out


def mean_coordination_loss(
        x: ChemGraph,
        t: Any,
        target: dict,
        kernel: str = "sigmoid",
        sigma: float = 1.0,
        alpha: float = 8.0
) -> torch.Tensor:
    """
    Computes the mean pair- or group-coordination loss for a given ChemGraph.
    Example of target: {'O-H': 1, 'O-C': [1,2.0], 'C-C': 2, 'H-[Pd,Ni,Pt]': 3}
    Meaning that the environment of O should have 1 H and 1 C but with a r_cut
    of 2.0 for C, the environment of C should have 2 C, and H should have a
    total of 3 Pd/Ni/Pt neighbors.
    The non-specified distance will be using the default r_cut, which is the sum of the covalent radii of the two species plus 0.5.
    The function computes the mean coordination loss for the specified species in the ChemGraph.
    The loss is computed as the absolute difference between the computed environment and the target value.

    Args:
        x (ChemGraph): The input ChemGraph.
        t (Any): Unused, but required for compatibility.
        target (dict): The species of interest and the target value for each coordination constraint.
        kernel (str): Kernel type, either 'gaussian' or 'sigmoid'.
        sigma (float): Width for Gaussian kernel.
        r_cut (float | None): Cutoff for sigmoid kernel.
        alpha (float): Sharpness for sigmoid kernel.

    Returns:
        torch.Tensor: The computed mean coordination loss.
    """
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")

    cell = x.cell
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")

    frac = x.pos
    atomic_numbers = x.atomic_numbers
    num_atoms = x.num_atoms

    # Extract mode without mutating the user-provided target dictionary.
    mode = target.get("mode", None)
    constraints = [
        (species_pair, val) for species_pair, val in target.items() if species_pair != "mode"
    ]
    f_AB_list = []
    target_values = []

    for species_pair, val in constraints:
        target_values.append(val[0] if isinstance(val, (list, tuple)) else val)
        r_cut = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else None
        type_A, type_Bs = _parse_coordination_constraint(species_pair)
        f_AB_list.append(
            compute_mean_coordination(
                cell=cell,
                frac=frac,
                atomic_numbers=atomic_numbers,
                num_atoms=num_atoms,
                type_A=type_A,
                type_B=type_Bs,
                kernel=kernel,
                sigma=sigma,
                r_cut=r_cut,
                alpha=alpha
            )
        )

    if len(f_AB_list) == 0:
        zeros = torch.zeros_like(num_atoms, dtype=cell.dtype, device=cell.device)
        return zeros * 0.0

    f_AB = torch.stack(f_AB_list)  # shape: (num_pairs,) or (num_pairs, B)

    # Force 2D: (num_pairs, B) with B=1 when single-structure
    if f_AB.ndim == 1:
        f_AB = f_AB.unsqueeze(1)  # (num_pairs, 1)

    # Prepare targets to match (num_pairs, B)
    B = f_AB.shape[1]
    target_vec = torch.as_tensor(target_values, dtype=f_AB.dtype, device=f_AB.device).view(
        -1, 1
    )  # (num_pairs, 1)
    target_tensor = target_vec.expand(-1, B)  # (num_pairs, B)

    # Compute the loss
    if mode == "l1" or mode == None or mode == "test":
        loss = torch.abs(f_AB - target_tensor)
    elif mode == "l2":
        loss = torch.nn.functional.mse_loss(f_AB, target_tensor, reduction='none')
    elif mode == "huber":
        loss = torch.nn.functional.huber_loss(f_AB, target_tensor, reduction='none', delta=1.5)
        # eps sensitif ?
    elif mode == "divergence":
        pass  # Placeholder for divergence loss, not implemented
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'l1', 'huber', and 'l2'.")

    return loss.sum(dim=0)  # Sum over all pairs to get a single loss value


def environment_loss(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for mean_coordination_loss."""
    return mean_coordination_loss(*args, **kwargs)


def group_coordination_loss(*args, **kwargs) -> torch.Tensor:
    """Alias for mean_coordination_loss with grouped neighbor-set keys."""
    return mean_coordination_loss(*args, **kwargs)


def group_target_coordination_loss(*args, **kwargs) -> torch.Tensor:
    """Alias for target_coordination_loss with grouped neighbor-set keys."""
    return target_coordination_loss(*args, **kwargs)


INTER_ATOMIC_CUTOFF = {1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
                       11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03,
                       20: 1.76, 21: 1.7, 22: 1.6, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
                       30: 1.22, 31: 1.22, 32: 1.2, 33: 1.19, 34: 1.2, 35: 1.2, 36: 1.16, 37: 2.2, 38: 1.95, 39: 1.9,
                       40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44,
                       49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.4, 55: 2.44, 56: 2.15, 57: 2.07,
                       58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92,
                       67: 1.92, 68: 1.89, 69: 1.9, 70: 1.87, 71: 1.87, 72: 1.75, 73: 1.7, 74: 1.62, 75: 1.51, 76: 1.44,
                       77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 84: 1.4, 85: 1.5, 86: 1.5,
                       87: 2.6, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.0, 92: 1.96, 93: 1.9, 94: 1.87, 95: 1.8, 96: 1.69}



def composition(num, pos):
    """
    Computes the composition of a list of atoms.
    li is a list of int with 101 beeing an empty atom.
    Returns a list of strings with the chemical symbols of the atoms.
    Example: [1, 101, 8, 8, 101] -> ['H', 'O', 'O']
    """
    return num[num != 101], pos[num != 101]


# def energy(x, t, target=None):
#     """
#     Computes the energy above the hull for a given composition and energy.
#     x is a chemgraph batch
#     The function uses a precomputed phase diagram to determine the energy above the hull.
#     """
#     global calc
#     global converter
#     if calc is None:
#         checkpoint = torch.load("/Data/auguste.de-lambilly/mattersim_torch/pretrained_models/mattersim-v1.0.0-1M.pth",
#                                 map_location="cuda")
#         model = M3Gnet(**checkpoint["model_args"], device="cuda")  # Add arguments as needed for your configuration
#         model.load_state_dict(checkpoint["model"])  # Load the model state dict, ensure it's on cuda
#         model.eval()  # Set to evaluation mode for inference
#         model = model.to(x.pos.device)  # Move model to the same device as x
#     if converter is None:
#         converter = ChemGraphBatchConvertor(twobody_cutoff=5.0, threebody_cutoff=4.0, pbc=True)
#     if not isinstance(x, ChemGraph):
#         raise ValueError("x must be a ChemGraph object")
#
#     inputs = converter.convert(x)
#     energies = []
#     for input in inputs:
#         if input is None:
#             # If no atoms, append 0 to results
#             energies.append(torch.zeros(1, device=x.pos.device) * x.pos.sum() * x.cell.sum())
#         else:
#             temp = model(input)
#             if temp.isnan().any():
#                 # If NaN, append 0 to results
#                 energies.append(torch.zeros(1, device=x.pos.device) * x.pos.sum() * x.cell.sum())
#             else:
#                 energies.append(temp)  # Otherwise compute the energy estimate
#     energies = torch.stack(energies)  # Stack the energies into a tensor
#     return energies


def _energy_hull(x):
    """
    Computes the energy above the hull for a given composition and energy.
    x is a (Compo, Energy) tuple (str, float)
    CSV : Compo , Energy
    """
    dir = "/Data/auguste.de-lambilly/mattergenbis/phase_diagram/"  # This should be the directory where the phase diagram is saved
    global PDIAG
    if PDIAG is None:
        # Load the CSV file only once
        csv = pd.read_csv(dir + "LiCoO.csv")
        li = [PDEntry(composition=Composition(csv["Formula"][i]), energy=csv["Energy"][i]) for i in range(len(csv))]
        PDIAG = PhaseDiagram(li)
        del csv, li
    x_ = PDEntry(composition=Composition(x[0]), energy=x[1])  # Assuming x has composition and energy attributes
    above_hull = PDIAG.get_e_above_hull(x_)
    return above_hull


def new_loss(x, t, target):
    """
    Example of a new loss function.
    This is just a placeholder and should be replaced with an actual implementation.
    """
    # x : ChemGraph object
    # t : timestep
    # target : target value
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
        # TODO: Verify that a simple sum is appropriate for combining the losses
        return sum(loss(x, t) for loss in partial_losses)

    return combined_loss


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "volume": volume_loss,
    "volume_pa": volume_pa_loss,
    "mean_coordination": mean_coordination_loss,
    "target_coordination": target_coordination_loss,
    "group_coordination": group_coordination_loss,
    "group_target_coordination": group_target_coordination_loss,
    "environment": environment_loss,
    "dominant_environment": dominant_environment_loss,
    # "energy": energy,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}
