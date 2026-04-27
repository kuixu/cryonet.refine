from __future__ import annotations

import math

import torch

from CryoNetRefine.data.parse.restraints import ResolvedUserRestraints


def _slack_delta(delta: torch.Tensor, slack: float) -> torch.Tensor:
    if slack <= 0:
        return delta
    abs_delta = delta.abs()
    return torch.sign(delta) * torch.clamp(abs_delta - slack, min=0.0)


def _bond_loss(
    coords: torch.Tensor,
    atom_idx1: int,
    atom_idx2: int,
    distance_ideal: float,
    weight: float,
    slack: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    vec = coords[atom_idx1] - coords[atom_idx2]
    distance = torch.sqrt(torch.sum(vec * vec) + eps)
    delta = distance - distance_ideal
    delta_slack = _slack_delta(delta, slack)
    return coords.new_tensor(weight) * (delta_slack * delta_slack)


def _angle_degrees(
    coords: torch.Tensor,
    atom_idx1: int,
    atom_idx2: int,
    atom_idx3: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    v1 = coords[atom_idx1] - coords[atom_idx2]
    v2 = coords[atom_idx3] - coords[atom_idx2]
    n1 = torch.sqrt(torch.sum(v1 * v1) + eps)
    n2 = torch.sqrt(torch.sum(v2 * v2) + eps)
    cos_theta = torch.sum(v1 * v2) / (n1 * n2)
    cos_theta = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos_theta))


def _angle_loss(
    coords: torch.Tensor,
    atom_idx1: int,
    atom_idx2: int,
    atom_idx3: int,
    angle_ideal_deg: float,
    weight: float,
    slack_deg: float,
) -> torch.Tensor:
    angle_model = _angle_degrees(coords, atom_idx1, atom_idx2, atom_idx3)
    delta = angle_model - angle_ideal_deg
    delta_slack = _slack_delta(delta, slack_deg)
    return coords.new_tensor(weight) * (delta_slack * delta_slack)


def compute_user_restraint_losses(
    coords: torch.Tensor,
    restraints: ResolvedUserRestraints | None,
) -> dict[str, torch.Tensor]:
    if coords.ndim == 3:
        if coords.shape[0] != 1:
            raise ValueError("User restraint loss currently expects batch size 1.")
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [N, 3], got {tuple(coords.shape)}.")
    zero = coords.new_zeros(())
    if restraints is None:
        return {"user_bond": zero, "user_angle": zero}

    bond_losses = []
    for bond in restraints.bonds:
        bond_losses.append(
            _bond_loss(
                coords=coords,
                atom_idx1=bond.atom_idx1,
                atom_idx2=bond.atom_idx2,
                distance_ideal=bond.distance_ideal,
                weight=bond.weight,
                slack=bond.slack,
            )
        )
    angle_losses = []
    for angle in restraints.angles:
        angle_losses.append(
            _angle_loss(
                coords=coords,
                atom_idx1=angle.atom_idx1,
                atom_idx2=angle.atom_idx2,
                atom_idx3=angle.atom_idx3,
                angle_ideal_deg=angle.angle_ideal_deg,
                weight=angle.weight,
                slack_deg=angle.slack_deg,
            )
        )

    user_bond = torch.stack(bond_losses).mean() if bond_losses else zero
    user_angle = torch.stack(angle_losses).mean() if angle_losses else zero
    if not torch.isfinite(user_bond):
        raise ValueError("Non-finite user bond restraint loss encountered.")
    if not torch.isfinite(user_angle):
        raise ValueError("Non-finite user angle restraint loss encountered.")
    return {"user_bond": user_bond, "user_angle": user_angle}


def summarize_user_restraints(restraints: ResolvedUserRestraints | None) -> str:
    if restraints is None:
        return "user restraints: disabled"
    return (
        f"user restraints: {len(restraints.bonds)} bonds, "
        f"{len(restraints.angles)} angles"
    )
