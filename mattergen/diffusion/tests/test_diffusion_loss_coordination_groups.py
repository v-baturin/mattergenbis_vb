import torch
from pymatgen.core import Element

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.diffusion_loss import (
    INTER_ATOMIC_CUTOFF,
    compute_mean_coordination,
    compute_target_share,
    mean_coordination_loss,
    target_coordination_loss,
)


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
