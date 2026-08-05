import math

import torch
from typing import Any

from mattergen.common.data.chemgraph import ChemGraph
from pymatgen.core import Element

shifts = None
DEFAULT_COORDINATION_ALPHA = 2.0
DEFAULT_COORDINATION_MARGIN = 0.05
DEFAULT_COORDINATION_TEMPERATURE = 0.10
DEFAULT_COORDINATION_MODE = "soft_count"
COORDINATION_CONFIG_KEYS = {
    "mode",
    "alpha",
    "coordination_mode",
    "margin",
    "temperature",
}

def _as_atomic_number_tuple(
    atomic_numbers: int | list[int] | tuple[int, ...] | set[int],
) -> tuple[int, ...]:
    """Normalize one or more atomic numbers to a de-duplicated tuple."""
    if isinstance(atomic_numbers, torch.Tensor):
        values = atomic_numbers.detach().cpu().reshape(-1).tolist()
    elif isinstance(atomic_numbers, (list, tuple, set)):
        values = list(atomic_numbers)
    else:
        values = [atomic_numbers]
    return tuple(dict.fromkeys(int(v) for v in values))


def _validate_one_sided_coordination_groups(
    type_As: tuple[int, ...], type_Bs: tuple[int, ...]
) -> None:
    if len(type_As) > 1 and len(type_Bs) > 1:
        raise ValueError(
            "Coordination groups are supported on only one side: use either "
            "'A-[B1,B2]' or '[A1,A2]-B'."
        )


def _parse_species_group(
    species: str,
    *,
    species_constraint: str,
    side: str,
    allow_unbracketed_group: bool = False,
) -> tuple[int, ...]:
    """Parse one side of a coordination constraint into atomic numbers."""
    species = species.strip()
    is_bracketed = species.startswith("[") and species.endswith("]")
    has_bracket = "[" in species or "]" in species

    if has_bracket and not is_bracketed:
        raise ValueError(f"Malformed {side} species group in {species_constraint}.")

    if is_bracketed:
        species = species[1:-1]
    elif "," in species and not allow_unbracketed_group:
        raise ValueError(
            f"Grouped {side} species in {species_constraint} must be enclosed in brackets."
        )

    symbols = [symbol.strip() for symbol in species.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError(
            f"Invalid {side} species group in {species_constraint}. Expected at least one element."
        )

    return _as_atomic_number_tuple(tuple(Element(symbol).Z for symbol in symbols))


def _parse_coordination_constraint(
    species_constraint: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Parse coordination keys.

    Supported forms:
      A-B
      A-[B,C,D]
      A-B,C,D
      [A,B,C]-D

    At least one side must contain exactly one species.
    """
    if "-" not in species_constraint:
        raise ValueError(f"Invalid species pair format: {species_constraint}. Expected 'A-B'.")

    species_A, species_B = species_constraint.split("-", maxsplit=1)
    species_A = species_A.strip()
    species_B = species_B.strip()

    if not species_A or not species_B:
        raise ValueError(f"Invalid species pair format: {species_constraint}. Expected 'A-B'.")

    type_As = _parse_species_group(
        species_A,
        species_constraint=species_constraint,
        side="central",
    )
    type_Bs = _parse_species_group(
        species_B,
        species_constraint=species_constraint,
        side="neighbor",
        allow_unbracketed_group=True,
    )
    _validate_one_sided_coordination_groups(type_As, type_Bs)
    return type_As, type_Bs


def _default_coordination_r_cut(
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
) -> float:
    """Use the largest default pair cutoff across the coordination constraint."""
    type_As = _as_atomic_number_tuple(type_A)
    type_Bs = _as_atomic_number_tuple(type_B)
    return max(
        INTER_ATOMIC_CUTOFF[type_A_i] + INTER_ATOMIC_CUTOFF[type_B_i] + 0.5
        for type_A_i in type_As
        for type_B_i in type_Bs
    )


def _coordination_r_cut_per_center(
    center_types: torch.Tensor,
    type_Bs: tuple[int, ...],
    *,
    dtype: torch.dtype,
    r_cut: float | None,
) -> torch.Tensor:
    if r_cut is not None:
        return torch.full(
            center_types.shape, float(r_cut), dtype=dtype, device=center_types.device
        )
    return torch.tensor(
        [
            _default_coordination_r_cut(int(type_A_i), type_Bs)
            for type_A_i in center_types.detach().cpu().tolist()
        ],
        dtype=dtype,
        device=center_types.device,
    )


def _normalize_coordination_mode(coordination_mode: str) -> str:
    aliases = {
        "soft_count": "soft_count",
        "sigmoid": "soft_count",
        "kth_neighbor": "kth_neighbor",
        "softplus": "kth_neighbor",
    }
    try:
        return aliases[coordination_mode.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "coordination_mode must be 'soft_count' or 'kth_neighbor'."
        ) from exc


def _validate_target_coordination(target: float) -> int:
    target_float = float(target)
    if not math.isfinite(target_float) or not target_float.is_integer():
        raise ValueError(
            "The k-th-neighbor coordination target must be a non-negative integer."
        )
    target_int = int(target_float)
    if target_int < 0:
        raise ValueError(
            "The k-th-neighbor coordination target must be a non-negative integer."
        )
    return target_int


def _coordination_margin_penalties_per_A_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types,
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    r_cut: float | None = None,
    margin: float = DEFAULT_COORDINATION_MARGIN,
    temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Return the k-th-neighbor softplus margin penalty for every central A atom.

    For target coordination k > 0, d_(k) is pulled below r_cut - margin and
    d_(k+1) is pushed above r_cut + margin. For k = 0, only the nearest
    neighbor is pushed outside. The distances include 27 periodic B images,
    with only the zero-shift self-image excluded.
    """
    frac = torch.as_tensor(
        frac,
        dtype=getattr(frac, "dtype", torch.float32),
        device=getattr(frac, "device", None),
    )
    cell = torch.as_tensor(cell, dtype=frac.dtype, device=frac.device)
    types = torch.as_tensor(types, dtype=torch.int64, device=frac.device)
    type_As = _as_atomic_number_tuple(type_A)
    type_Bs = _as_atomic_number_tuple(type_B)
    _validate_one_sided_coordination_groups(type_As, type_Bs)

    target_int = _validate_target_coordination(target)
    margin = float(margin)
    temperature = float(temperature)
    if margin < 0.0:
        raise ValueError("coordination margin must be non-negative.")
    if temperature <= 0.0:
        raise ValueError("coordination temperature must be positive.")
    mask_A = torch.zeros_like(types, dtype=torch.bool)
    for type_A_i in type_As:
        mask_A = mask_A | (types == type_A_i)
    mask_B = torch.zeros_like(mask_A, dtype=torch.bool)
    for type_B_i in type_Bs:
        mask_B = mask_B | (types == type_B_i)
    idx_A = mask_A.nonzero(as_tuple=True)[0]
    idx_B = mask_B.nonzero(as_tuple=True)[0]

    if idx_A.numel() == 0 or idx_B.numel() == 0:
        return cell.sum() * frac.sum() * torch.zeros(1, device=frac.device)

    r_cut_per_A = _coordination_r_cut_per_center(
        types[idx_A], type_Bs, dtype=frac.dtype, r_cut=r_cut
    )

    global shifts
    if shifts is None or shifts.device != frac.device:
        shifts = torch.stack(
            torch.meshgrid(
                torch.arange(-1, 2, device=frac.device),
                torch.arange(-1, 2, device=frac.device),
                torch.arange(-1, 2, device=frac.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)

    frac_A = frac[idx_A]
    frac_B = frac[idx_B]
    frac_B_images = (frac_B.unsqueeze(1) + shifts.unsqueeze(0)).reshape(-1, 3)
    cartesian_displacements = torch.matmul(
        frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0),
        cell,
    )
    distances = cartesian_displacements.norm(dim=-1)

    # Exclude only i=j in the original cell. Periodic images of the same atom
    # remain valid candidate neighbors.
    zero_shift = (shifts == 0).all(dim=1)
    self_mask = (
        (idx_A[:, None] == idx_B[None, :])[:, :, None]
        & zero_shift[None, None, :]
    ).reshape(idx_A.numel(), -1)
    distances = distances.masked_fill(self_mask, torch.inf)
    ordered_distances = torch.sort(distances, dim=1).values

    # Zero-based index `target_int` is d_(k+1), including the k=0 case.
    if target_int >= ordered_distances.shape[1]:
        raise ValueError(
            f"Not enough B-neighbor images to define d_({target_int + 1})."
        )
    d_k_plus_1 = ordered_distances[:, target_int]
    if not torch.isfinite(d_k_plus_1).all():
        raise ValueError(
            f"Not enough B-neighbor images to define d_({target_int + 1})."
        )

    push_extra_outside = temperature * torch.nn.functional.softplus(
        ((r_cut_per_A + margin) - d_k_plus_1) / temperature
    )
    if target_int == 0:
        return push_extra_outside

    d_k = ordered_distances[:, target_int - 1]
    pull_k_inside = temperature * torch.nn.functional.softplus(
        (d_k - (r_cut_per_A - margin)) / temperature
    )
    return pull_k_inside + push_extra_outside


def _soft_neighbor_counts_per_A_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types,                     # accepts numpy or torch
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
    r_cut: float | None = None,
    alpha: float = DEFAULT_COORDINATION_ALPHA,
) -> torch.Tensor:
    """
    Returns a differentiable vector C (n_A,) of soft B-neighbor counts for each A atom:
        C_i = sum_{j in A_B} sigmoid(alpha * (r_cut - d_ij)),
        with 27 PBC images.
    `type_A` and `type_B` may be atomic numbers or collections of atomic numbers.
    If a central atom's type is included in B, subtract its self-interaction.
    """
    # Normalize inputs to torch on the same device/dtype (keeps grad from frac if it has one)
    frac = torch.as_tensor(frac, dtype=getattr(frac, "dtype", torch.float32),
                           device=getattr(frac, "device", None))
    cell = torch.as_tensor(cell, dtype=frac.dtype, device=frac.device)
    types = torch.as_tensor(types, dtype=torch.int64, device=frac.device)
    type_As = _as_atomic_number_tuple(type_A)
    type_Bs = _as_atomic_number_tuple(type_B)
    _validate_one_sided_coordination_groups(type_As, type_Bs)

    device = frac.device
    mask_A = torch.zeros_like(types, dtype=torch.bool)
    for type_A_i in type_As:
        mask_A = mask_A | (types == type_A_i)
    mask_B = torch.zeros_like(mask_A, dtype=torch.bool)
    for type_B_i in type_Bs:
        mask_B = mask_B | (types == type_B_i)
    idx_A = mask_A.nonzero(as_tuple=True)[0]
    idx_B = mask_B.nonzero(as_tuple=True)[0]

    if idx_A.numel() == 0 or idx_B.numel() == 0:
        # Return a scalar-zero preserving the graph
        return cell.sum()*frac.sum() * torch.zeros(1, device=device)

    r_cut_per_A = _coordination_r_cut_per_center(
        types[idx_A], type_Bs, dtype=frac.dtype, r_cut=r_cut
    )

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

    G = torch.sigmoid(alpha * (r_cut_per_A[:, None] - dist))

    counts = G.sum(dim=1)  # (n_A,)

    # Remove self-interaction only for centers whose type is in the neighbor set.
    center_types = types[idx_A]
    self_interaction = torch.zeros_like(center_types, dtype=torch.bool)
    for type_B_i in type_Bs:
        self_interaction = self_interaction | (center_types == type_B_i)
    counts = counts - self_interaction.to(dtype=counts.dtype)

    return counts


# --- Batched share metric: fraction of A atoms with ~target B neighbors ---
def _compute_target_coordination_share_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types: torch.Tensor,
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    tau: float = 0.5,
    r_cut: float | None = None,
    alpha: float = DEFAULT_COORDINATION_ALPHA,
    coordination_mode: str = DEFAULT_COORDINATION_MODE,
    margin: float = DEFAULT_COORDINATION_MARGIN,
    temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Compute one structure's coordination objective.

    In ``soft_count`` mode this returns the original share,
    (1/|A|) sum_i exp(-((C_i-target)/tau)^2). In ``kth_neighbor`` mode it
    returns the mean k-th-neighbor softplus margin penalty instead. The
    enclosing existing loss function handles the two return conventions.
    """
    coordination_mode = _normalize_coordination_mode(coordination_mode)
    if coordination_mode == "kth_neighbor":
        penalties = _coordination_margin_penalties_per_A_single(
            cell,
            frac,
            types,
            type_A,
            type_B,
            target=target,
            r_cut=r_cut,
            margin=margin,
            temperature=temperature,
        )
        if penalties.numel() == 1 and penalties.squeeze().abs().sum() == 0:
            return penalties.squeeze()
        return penalties.mean()

    C = _soft_neighbor_counts_per_A_single(
        cell, frac, types, type_A, type_B, r_cut=r_cut, alpha=alpha
    )
    # If empty sentinel, return as-is (0-like scalar preserving graph)
    if C.numel() == 1 and C.squeeze().abs().sum() == 0:
        return C.squeeze()
    H = torch.exp(-((C - float(target)) / float(tau)).pow(2))
    return H.mean()


def compute_target_coordination_share(
    cell: torch.Tensor,           # (B, 3, 3) or (3, 3)
    frac: torch.Tensor,           # (B, N, 3) or (N, 3)
    atomic_numbers: torch.Tensor, # (sumN_i,)
    num_atoms: torch.Tensor,      # (B,)
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    tau: float = 0.5,
    r_cut: float | None = None,
    alpha: float = DEFAULT_COORDINATION_ALPHA,
    coordination_mode: str = DEFAULT_COORDINATION_MODE,
    margin: float = DEFAULT_COORDINATION_MARGIN,
    temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Batched coordination objective. Returns (B,) if batched, scalar if single.

    The default ``soft_count`` mode returns the original target share.
    ``kth_neighbor`` returns the direct softplus margin penalty selected by the
    same existing function rather than defining a new loss type.
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
        res = _compute_target_coordination_share_single(
            cell[b], frac[count:count_], atomic_numbers[count:count_],
            type_A, type_B,
            target=target,
            tau=tau,
            r_cut=r_cut,
            alpha=alpha,
            coordination_mode=coordination_mode,
            margin=margin,
            temperature=temperature,
        )
        results.append(res)
        count = count_
    out = torch.stack(results)
    return out.squeeze(0) if squeeze_out else out


def compute_target_share(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for compute_target_coordination_share."""
    return compute_target_coordination_share(*args, **kwargs)


# --- New loss: maximize fraction at exact target coordination (minimize 1 - share) ---
def target_coordination_share_loss(
    x: ChemGraph,
    t: Any,
    target: dict,
    alpha: float = DEFAULT_COORDINATION_ALPHA,
    default_tau: float = 0.5,
    default_margin: float = DEFAULT_COORDINATION_MARGIN,
    default_temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Target-coordination guidance. For each A-B in `target` with integer neighbor target k, minimize 1 - share_A(k; B),
    where share_A(k; B) is computed with a Gaussian window of width `tau` in coordination space.

    The default ``coordination_mode='soft_count'`` preserves the original
    ``1 - share`` behavior. With ``coordination_mode='kth_neighbor'``, the same
    registered loss instead returns the direct softplus boundary penalty.

    `target` can be:
      {'alpha': 3.0, 'A-B': k}
      {'A-B': k}
      {'A-B': [k, r_cut]}
      {'A-B': [k, r_cut, tau]}
      {'coordination_mode': 'kth_neighbor', 'margin': 0.05,
       'temperature': 0.10, 'A-B': [k, r_cut]}
      {'coordination_mode': 'kth_neighbor',
       'A-B': [k, r_cut, margin, temperature]}
      {'A-[B,C,D]': k}
      {'[A,B,C]-D': k}
    """
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")

    cell = x.cell
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")

    frac = x.pos
    atomic_numbers = x.atomic_numbers
    num_atoms = x.num_atoms
    alpha = float(target.get("alpha", alpha))
    coordination_mode = _normalize_coordination_mode(
        target.get("coordination_mode", DEFAULT_COORDINATION_MODE)
    )
    default_margin = float(target.get("margin", default_margin))
    default_temperature = float(target.get("temperature", default_temperature))

    objectives = []
    for species_pair, val in target.items():
        if species_pair in COORDINATION_CONFIG_KEYS:
            continue

        if isinstance(val, (list, tuple)):
            tgt = float(val[0])
            rcut = (None if len(val) < 2 or val[1] is None else float(val[1]))
            if coordination_mode == "kth_neighbor":
                margin = (
                    default_margin if len(val) < 3 or val[2] is None else float(val[2])
                )
                temperature = (
                    default_temperature
                    if len(val) < 4 or val[3] is None
                    else float(val[3])
                )
                tau = default_tau
            else:
                tau = (default_tau if len(val) < 3 or val[2] is None else float(val[2]))
                margin = default_margin
                temperature = default_temperature
        else:
            tgt = float(val)
            rcut = None
            tau = default_tau
            margin = default_margin
            temperature = default_temperature

        ZAs, ZBs = _parse_coordination_constraint(species_pair)

        objective = compute_target_coordination_share(
            cell=cell, frac=frac, atomic_numbers=atomic_numbers, num_atoms=num_atoms,
            type_A=ZAs, type_B=ZBs,
            target=tgt,
            tau=tau,
            r_cut=rcut,
            alpha=alpha,
            coordination_mode=coordination_mode,
            margin=margin,
            temperature=temperature,
        )  # (B,)
        objectives.append(objective)

    if len(objectives) == 0:
        # No valid pairs: return zero-like (B,) preserving graph
        zeros = torch.zeros_like(num_atoms, dtype=cell.dtype, device=cell.device)
        return zeros * 0.0

    objectives_tensor = torch.stack(objectives, dim=0)  # (P, B)
    if coordination_mode == "kth_neighbor":
        return objectives_tensor.sum(dim=0)
    loss = 1.0 - objectives_tensor
    return loss.sum(dim=0)


def target_coordination_loss(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for target_coordination_share_loss."""
    return target_coordination_share_loss(*args, **kwargs)


def dominant_environment_loss(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for target_coordination_share_loss."""
    return target_coordination_share_loss(*args, **kwargs)


def compute_mean_coordination(
        cell: torch.Tensor,  # (B, 3, 3) or (3, 3)
        frac: torch.Tensor,  # (B, N, 3) or (N, 3)
        atomic_numbers: torch.Tensor,  # (\Sum N_i)
        num_atoms: torch.Tensor,  # (B,)
        type_A: int | list[int] | tuple[int, ...] | set[int],
        type_B: int | list[int] | tuple[int, ...] | set[int],
        r_cut: float | None = None,
        alpha: float = DEFAULT_COORDINATION_ALPHA,
        coordination_mode: str = DEFAULT_COORDINATION_MODE,
        target: float | None = None,
        margin: float = DEFAULT_COORDINATION_MARGIN,
        temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Batched A-B coordination objective selected by ``coordination_mode``.

    ``soft_count`` (default) returns the original mean soft coordination,
    mean_i sum_j g(d_ij). ``kth_neighbor`` returns the mean direct softplus
    margin penalty and therefore requires an integer ``target``.

    Either `type_A` or `type_B` may be a collection of atomic numbers. Grouped
    centers are pooled before taking the mean. If `r_cut` is omitted, each
    center element uses its own default cutoff; for a grouped neighbor set this
    is the maximum default cutoff over that center's neighbor pairs.
    Returns: (B,) if batched, scalar if single.
    """
    # Normalize to batched
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    coordination_mode = _normalize_coordination_mode(coordination_mode)
    if coordination_mode == "kth_neighbor" and target is None:
        raise ValueError(
            "target is required when coordination_mode='kth_neighbor'."
        )

    B = cell.shape[0]
    results = []
    count = 0
    for b in range(B):
        count_ = count + num_atoms[b]
        if coordination_mode == "kth_neighbor":
            values = _coordination_margin_penalties_per_A_single(
                cell[b],
                frac[count:count_],
                atomic_numbers[count:count_],
                type_A=type_A,
                type_B=type_B,
                target=target,
                r_cut=r_cut,
                margin=margin,
                temperature=temperature,
            )
        else:
            values = _soft_neighbor_counts_per_A_single(
                cell[b],
                frac[count:count_],
                atomic_numbers[count:count_],
                type_A=type_A,
                type_B=type_B,
                r_cut=r_cut,
                alpha=alpha,
            )
        # If empty sentinel, just append values.squeeze()
        if values.numel() == 1 and values.squeeze().abs().sum() == 0:
            results.append(values.squeeze())
        else:
            results.append(values.mean())
        count = count_
    out = torch.stack(results)
    if squeeze_out:
        out = out.squeeze(0)
    return out


def mean_coordination_loss(
        x: ChemGraph,
        t: Any,
        target: dict,
        alpha: float = DEFAULT_COORDINATION_ALPHA,
        default_margin: float = DEFAULT_COORDINATION_MARGIN,
        default_temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    """
    Computes the pair- or group-coordination loss for a given ChemGraph.

    The default ``coordination_mode='soft_count'`` preserves the original
    l1/l2/Huber comparison between mean sigmoid coordination and its target.
    With ``coordination_mode='kth_neighbor'``, the same existing loss returns
    the direct softplus penalty on d_(k) and d_(k+1). In that mode ``mode`` is
    ignored, because the two softplus terms are already the loss.

    Example of target: {'alpha': 3.0, 'O-H': 1, 'O-C': [1,2.0], 'C-C': 2,
                        'H-[Pd,Ni,Pt]': 3, '[H,C]-O': 2}
    K-th-neighbor example:
        {'coordination_mode': 'kth_neighbor', 'margin': 0.05,
         'temperature': 0.10, 'Co-O': [5, 2.42]}
    A pair can override the global margin and temperature with
    ``[target, r_cut, margin, temperature]``.
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
        alpha (float): Sharpness of the sigmoid neighbor count.

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
    alpha = float(target.get("alpha", alpha))
    coordination_mode = _normalize_coordination_mode(
        target.get("coordination_mode", DEFAULT_COORDINATION_MODE)
    )
    default_margin = float(target.get("margin", default_margin))
    default_temperature = float(target.get("temperature", default_temperature))
    constraints = [
        (species_pair, val)
        for species_pair, val in target.items()
        if species_pair not in COORDINATION_CONFIG_KEYS
    ]
    f_AB_list = []
    target_values = []

    for species_pair, val in constraints:
        target_value = val[0] if isinstance(val, (list, tuple)) else val
        target_values.append(target_value)
        r_cut = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else None
        margin = (
            val[2]
            if isinstance(val, (list, tuple)) and len(val) > 2 and val[2] is not None
            else default_margin
        )
        temperature = (
            val[3]
            if isinstance(val, (list, tuple)) and len(val) > 3 and val[3] is not None
            else default_temperature
        )
        type_As, type_Bs = _parse_coordination_constraint(species_pair)
        f_AB_list.append(
            compute_mean_coordination(
                cell=cell,
                frac=frac,
                atomic_numbers=atomic_numbers,
                num_atoms=num_atoms,
                type_A=type_As,
                type_B=type_Bs,
                r_cut=r_cut,
                alpha=alpha,
                coordination_mode=coordination_mode,
                target=target_value,
                margin=margin,
                temperature=temperature,
            )
        )

    if len(f_AB_list) == 0:
        zeros = torch.zeros_like(num_atoms, dtype=cell.dtype, device=cell.device)
        return zeros * 0.0

    f_AB = torch.stack(f_AB_list)  # shape: (num_pairs,) or (num_pairs, B)

    # Force 2D: (num_pairs, B) with B=1 when single-structure
    if f_AB.ndim == 1:
        f_AB = f_AB.unsqueeze(1)  # (num_pairs, 1)

    if coordination_mode == "kth_neighbor":
        return f_AB.sum(dim=0)

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
    """Alias for mean_coordination_loss with one-sided species-group keys."""
    return mean_coordination_loss(*args, **kwargs)


def group_target_coordination_loss(*args, **kwargs) -> torch.Tensor:
    """Alias for target_coordination_loss with one-sided species-group keys."""
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
