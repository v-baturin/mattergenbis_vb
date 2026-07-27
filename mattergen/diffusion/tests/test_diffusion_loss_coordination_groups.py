import pytest
import torch
from pymatgen.core import Element

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.diffusion_loss import (
    DEFAULT_COORDINATION_ALPHA,
    DEFAULT_COORDINATION_MARGIN,
    DEFAULT_COORDINATION_MODE,
    DEFAULT_COORDINATION_TEMPERATURE,
    INTER_ATOMIC_CUTOFF,
    compute_mean_coordination,
    compute_target_share,
    mean_coordination_loss,
    target_coordination_loss,
)


def test_default_coordination_alpha_is_two() -> None:
    assert DEFAULT_COORDINATION_ALPHA == 2.0
    assert DEFAULT_COORDINATION_MODE == "soft_count"
    assert DEFAULT_COORDINATION_MARGIN == 0.05
    assert DEFAULT_COORDINATION_TEMPERATURE == 0.10


def _group_coordination_system() -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # H
            [0.1, 0.0, 0.0],  # Pd, 1.0 A away
            [0.2, 0.0, 0.0],  # Ni, 2.0 A away
            [0.4, 0.0, 0.0],  # Pt, 4.0 A away
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor(
        [Element("H").Z, Element("Pd").Z, Element("Ni").Z, Element("Pt").Z],
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def _grouped_center_system() -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # H
            [0.1, 0.0, 0.0],  # H
            [0.3, 0.0, 0.0],  # Pd
            [0.2, 0.0, 0.0],  # Ni
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor(
        [Element("H").Z, Element("H").Z, Element("Pd").Z, Element("Ni").Z],
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def _kth_neighbor_system(
    oxygen_distances: tuple[float, ...] = (1.0, 2.0, 4.0),
    *,
    requires_grad: bool = False,
) -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [[0.0, 0.0, 0.0]]
        + [[distance / 10.0, 0.0, 0.0] for distance in oxygen_distances],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    atomic_numbers = torch.tensor(
        [Element("Co").Z] + [Element("O").Z] * len(oxygen_distances),
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def test_kth_neighbor_mode_matches_two_softplus_terms() -> None:
    x = _kth_neighbor_system()
    margin = 0.1
    temperature = 0.2
    actual = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=Element("Co").Z,
        type_B=Element("O").Z,
        coordination_mode="kth_neighbor",
        target=2,
        r_cut=2.5,
        margin=margin,
        temperature=temperature,
    )
    d_k = torch.tensor(2.0)
    d_k_plus_1 = torch.tensor(4.0)
    expected = temperature * torch.nn.functional.softplus(
        (d_k - (2.5 - margin)) / temperature
    ) + temperature * torch.nn.functional.softplus(
        ((2.5 + margin) - d_k_plus_1) / temperature
    )

    torch.testing.assert_close(actual, expected.unsqueeze(0))


def test_existing_losses_switch_to_same_kth_neighbor_objective() -> None:
    x = _kth_neighbor_system()
    target = {
        "coordination_mode": "kth_neighbor",
        "margin": 0.1,
        "temperature": 0.2,
        "Co-O": [2, 2.5],
    }
    expected = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=Element("Co").Z,
        type_B=Element("O").Z,
        coordination_mode="kth_neighbor",
        target=2,
        r_cut=2.5,
        margin=0.1,
        temperature=0.2,
    )

    torch.testing.assert_close(
        mean_coordination_loss(x, t=None, target=target),
        expected,
    )
    torch.testing.assert_close(
        target_coordination_loss(x, t=None, target=target),
        expected,
    )


def test_kth_neighbor_gradients_pull_and_push_only_boundary_neighbors() -> None:
    undercoordinated = _kth_neighbor_system(
        oxygen_distances=(1.0, 3.0, 4.0),
        requires_grad=True,
    )
    under_loss = mean_coordination_loss(
        undercoordinated,
        t=None,
        target={
            "coordination_mode": "kth_neighbor",
            "margin": 0.1,
            "temperature": 0.1,
            "Co-O": [2, 2.0],
        },
    )
    under_loss.sum().backward()
    assert undercoordinated.pos.grad[2, 0] > 0.0

    overcoordinated = _kth_neighbor_system(
        oxygen_distances=(1.0, 1.5, 4.0),
        requires_grad=True,
    )
    over_loss = mean_coordination_loss(
        overcoordinated,
        t=None,
        target={
            "coordination_mode": "kth_neighbor",
            "margin": 0.1,
            "temperature": 0.1,
            "Co-O": [1, 2.0],
        },
    )
    over_loss.sum().backward()
    assert overcoordinated.pos.grad[2, 0] < 0.0
    torch.testing.assert_close(
        overcoordinated.pos.grad[3],
        torch.zeros_like(overcoordinated.pos.grad[3]),
    )


def test_kth_neighbor_mode_supports_zero_coordination_target() -> None:
    x = _kth_neighbor_system(requires_grad=True)
    loss = mean_coordination_loss(
        x,
        t=None,
        target={
            "coordination_mode": "kth_neighbor",
            "Co-O": [0, 2.0, 0.1, 0.1],
        },
    )
    loss.sum().backward()

    assert x.pos.grad[1, 0] < 0.0
    torch.testing.assert_close(x.pos.grad[2], torch.zeros_like(x.pos.grad[2]))


def test_kth_neighbor_mode_requires_integer_target() -> None:
    x = _kth_neighbor_system()

    with pytest.raises(ValueError, match="non-negative integer"):
        mean_coordination_loss(
            x,
            t=None,
            target={
                "coordination_mode": "kth_neighbor",
                "Co-O": [1.5, 2.0],
            },
        )


def test_group_mean_coordination_uses_max_pair_cutoff() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    max_r_cut = max(
        INTER_ATOMIC_CUTOFF[type_a] + INTER_ATOMIC_CUTOFF[type_b] + 0.5
        for type_b in type_bs
    )

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
    )
    summed_pairs = torch.stack(
        [
            compute_mean_coordination(
                cell=x.cell,
                frac=x.pos,
                atomic_numbers=x.atomic_numbers,
                num_atoms=x.num_atoms,
                type_A=type_a,
                type_B=type_b,
                r_cut=max_r_cut,
            )
            for type_b in type_bs
        ]
    ).sum(dim=0)

    torch.testing.assert_close(grouped, summed_pairs)


def test_group_mean_coordination_loss_accepts_grouped_key_without_mutating_target() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
    )
    target = {"mode": "l1", "H-[Pd, Ni, Pt]": [float(grouped.item()), None]}

    loss = mean_coordination_loss(x, t=None, target=target)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert target["mode"] == "l1"


def test_group_target_coordination_loss_accepts_grouped_key() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    expected_share = compute_target_share(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
        target=2.0,
        tau=0.5,
    )

    loss = target_coordination_loss(
        x,
        t=None,
        target={"H-[Pd, Ni, Pt]": [2.0, None, 0.5]},
    )

    torch.testing.assert_close(loss, 1.0 - expected_share)


def test_grouped_centers_use_max_pair_cutoff_and_pool_all_centers() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z
    max_r_cut = max(
        INTER_ATOMIC_CUTOFF[type_a] + INTER_ATOMIC_CUTOFF[type_b] + 0.5
        for type_a in type_as
    )

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
    )
    mean_h, mean_pd = [
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=type_a,
            type_B=type_b,
            r_cut=max_r_cut,
        )
        for type_a in type_as
    ]
    pooled = (2 * mean_h + mean_pd) / 3

    torch.testing.assert_close(grouped, pooled)


def test_grouped_center_loss_accepts_grouped_key() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z
    alpha = 3.0
    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
        r_cut=2.5,
        alpha=alpha,
    )

    target = {
        "mode": "l1",
        "alpha": alpha,
        "[H,Pd]-Ni": [float(grouped.item()), 2.5],
    }
    loss = mean_coordination_loss(
        x,
        t=None,
        target=target,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert target["alpha"] == alpha


def test_grouped_center_target_coordination_loss_accepts_grouped_key() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z
    alpha = 3.0
    expected_share = compute_target_share(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
        target=1.0,
        tau=0.5,
        r_cut=2.5,
        alpha=alpha,
    )

    loss = target_coordination_loss(
        x,
        t=None,
        target={"alpha": alpha, "[H,Pd]-Ni": [1.0, 2.5, 0.5]},
    )

    torch.testing.assert_close(loss, 1.0 - expected_share)


def test_grouped_centers_remove_self_interaction_only_for_overlapping_species() -> None:
    x = _grouped_center_system()
    type_h = Element("H").Z
    type_pd = Element("Pd").Z
    r_cut = 2.5

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=(type_h, type_pd),
        type_B=type_pd,
        r_cut=r_cut,
    )
    mean_h, mean_pd = [
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=type_a,
            type_B=type_pd,
            r_cut=r_cut,
        )
        for type_a in (type_h, type_pd)
    ]
    separate = (2 * mean_h + mean_pd) / 3

    torch.testing.assert_close(grouped, separate)


def test_coordination_groups_on_both_sides_are_rejected() -> None:
    x = _group_coordination_system()

    with pytest.raises(ValueError, match="supported on only one side"):
        mean_coordination_loss(
            x,
            t=None,
            target={"[H,Pd]-[Ni,Pt]": 2},
        )

    with pytest.raises(ValueError, match="supported on only one side"):
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=(Element("H").Z, Element("Pd").Z),
            type_B=(Element("Ni").Z, Element("Pt").Z),
        )
