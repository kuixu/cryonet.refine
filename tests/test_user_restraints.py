from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from CryoNetRefine.data.parse.secondary_structure_restraints import (
    NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD,
    _add_plane_parallelity,
    _na_hbond_angle_params,
)
from CryoNetRefine.data.parse.restraints import (
    load_user_restraints,
    merge_user_restraints_specs,
    parse_user_restraints_dict,
    resolve_user_restraints,
)
from CryoNetRefine.data.types import AtomV2, BondV2, Chain, Coords, Ensemble, Interface, Residue, StructureV2
from CryoNetRefine.data.write.utils import _load_restraint_bonds
from CryoNetRefine.loss.loss import refine_loss
from CryoNetRefine.loss.user_restraints import compute_user_restraint_losses


def _make_structure() -> StructureV2:
    atoms = np.array(
        [
            ("ZN", (0.0, 0.0, 0.0), True, 20.0, 0.0),
            ("ND1", (2.0, 0.0, 0.0), True, 20.0, 0.0),
            ("NE2", (0.0, 2.0, 0.0), True, 20.0, 0.0),
        ],
        dtype=AtomV2,
    )
    residues = np.array(
        [
            ("ZN", 0, 1, 0, 1, 0, 0, False, True, "1", "", "ZN"),
            ("HIS", 0, 2, 1, 2, 1, 1, True, True, "2", "", "HIS"),
        ],
        dtype=Residue,
    )
    chains = np.array(
        [
            ("A", 0, 0, 0, 0, 0, 3, 0, 2, 0, "A"),
        ],
        dtype=Chain,
    )
    bonds = np.array([], dtype=BondV2)
    interfaces = np.array([], dtype=Interface)
    mask = np.array([True], dtype=bool)
    coords = np.array([((0.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),), ((0.0, 2.0, 0.0),)], dtype=Coords)
    ensemble = np.array([(0, 3)], dtype=Ensemble)
    return StructureV2(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
        pocket=None,
    )


def test_load_and_resolve_user_restraints_json(tmp_path):
    restraints_path = tmp_path / "restraints.json"
    restraints_path.write_text(
        json.dumps(
            {
                "bonds": [
                    {
                        "atom1": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                        "atom2": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                        "distance_ideal": 2.1,
                        "sigma": 0.1,
                    }
                ],
                "angles": [
                    {
                        "atom1": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                        "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                        "atom3": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "NE2"},
                        "angle_ideal_deg": 90.0,
                        "sigma": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = load_user_restraints(restraints_path)
    resolved = resolve_user_restraints(spec, _make_structure())
    assert resolved is not None
    assert len(resolved.bonds) == 1
    assert len(resolved.angles) == 1
    assert resolved.bonds[0].atom_idx1 == 0
    assert resolved.bonds[0].atom_idx2 == 1
    assert resolved.angles[0].atom_idx2 == 0


def test_resolve_user_restraints_duplicate_match_error(tmp_path):
    restraints_path = tmp_path / "restraints.json"
    restraints_path.write_text(
        json.dumps(
            {
                "bonds": [
                    {
                        "atom1": {"chain": "A", "atom_name": "ND1"},
                        "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                        "distance_ideal": 2.1,
                        "sigma": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    structure = _make_structure()
    structure = StructureV2(
        atoms=np.array(
            [
                ("ZN", (0.0, 0.0, 0.0), True, 20.0, 0.0),
                ("ND1", (2.0, 0.0, 0.0), True, 20.0, 0.0),
                ("ND1", (0.0, 2.0, 0.0), True, 20.0, 0.0),
            ],
            dtype=AtomV2,
        ),
        bonds=structure.bonds,
        residues=np.array(
            [
                ("ZN", 0, 1, 0, 1, 0, 0, False, True, "1", "", "ZN"),
                ("HIS", 0, 2, 1, 1, 1, 1, True, True, "2", "", "HIS"),
                ("HIS", 0, 3, 2, 1, 2, 2, True, True, "3", "", "HIS"),
            ],
            dtype=Residue,
        ),
        chains=np.array([("A", 0, 0, 0, 0, 0, 3, 0, 3, 0, "A")], dtype=Chain),
        interfaces=structure.interfaces,
        mask=structure.mask,
        coords=np.array([((0.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),), ((0.0, 2.0, 0.0),)], dtype=Coords),
        ensemble=np.array([(0, 3)], dtype=Ensemble),
        pocket=None,
    )
    spec = load_user_restraints(restraints_path)
    try:
        resolve_user_restraints(spec, structure)
    except ValueError as exc:
        assert "matched multiple atoms" in str(exc)
    else:
        raise AssertionError("Expected duplicate selector match to raise ValueError")


def test_compute_user_restraint_losses_zero_at_ideal():
    structure = _make_structure()
    spec_dict = {
        "bonds": [
            {
                "atom1": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                "atom2": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                "distance_ideal": 2.0,
                "sigma": 0.1,
            }
        ],
        "angles": [
            {
                "atom1": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                "atom3": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "NE2"},
                "angle_ideal_deg": 90.0,
                "sigma": 1.0,
            }
        ],
    }
    spec = parse_user_restraints_dict(spec_dict)
    resolved = resolve_user_restraints(spec, structure)
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )
    losses = compute_user_restraint_losses(coords, resolved)
    assert torch.allclose(losses["user_bond"], torch.tensor(0.0))
    assert torch.allclose(losses["user_angle"], torch.tensor(0.0), atol=1e-5)


def test_compute_user_plane_parallelity_loss_zero_at_ideal():
    structure = _make_structure()
    plane = [
        {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
        {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
        {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "NE2"},
    ]
    spec = parse_user_restraints_dict(
        {
            "plane_parallelities": [
                {
                    "plane1": plane,
                    "plane2": plane,
                    "angle_ideal_deg": 0.0,
                    "sigma": 1.0,
                }
            ]
        }
    )
    resolved = resolve_user_restraints(spec, structure)
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )
    losses = compute_user_restraint_losses(coords, resolved)
    assert resolved is not None
    assert len(resolved.plane_parallelities) == 1
    assert torch.allclose(losses["user_plane_parallelity"], torch.tensor(0.0), atol=1e-3)


def test_merge_user_restraints_specs_user_overrides_default_duplicates():
    default_spec = parse_user_restraints_dict(
        {
            "bonds": [
                {
                    "atom1": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                    "atom2": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                    "distance_ideal": 2.1,
                    "sigma": 0.1,
                }
            ],
            "angles": [
                {
                    "atom1": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                    "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                    "atom3": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "NE2"},
                    "angle_ideal_deg": 90.0,
                    "sigma": 5.0,
                }
            ],
        }
    )
    user_spec = parse_user_restraints_dict(
        {
            "bonds": [
                {
                    "atom1": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                    "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                    "distance_ideal": 2.3,
                    "sigma": 0.2,
                }
            ],
            "angles": [
                {
                    "atom1": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "NE2"},
                    "atom2": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                    "atom3": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                    "angle_ideal_deg": 105.0,
                    "sigma": 3.0,
                }
            ],
        }
    )

    merged = merge_user_restraints_specs(default_spec, user_spec)
    assert merged is not None
    assert len(merged.bonds) == 1
    assert len(merged.angles) == 1
    assert merged.bonds[0].distance_ideal == 2.3
    assert merged.bonds[0].sigma == 0.2
    assert merged.angles[0].angle_ideal_deg == 105.0
    assert merged.angles[0].sigma == 3.0


def test_refine_loss_user_restraints_reduce_violation():
    structure = _make_structure()
    spec = parse_user_restraints_dict(
        {
            "bonds": [
                {
                    "atom1": {"chain": "A", "resseq": "1", "resname": "ZN", "atom_name": "ZN"},
                    "atom2": {"chain": "A", "resseq": "2", "resname": "HIS", "atom_name": "ND1"},
                    "distance_ideal": 2.0,
                    "sigma": 1.0,
                }
            ]
        }
    )
    resolved = resolve_user_restraints(spec, structure)
    coords = torch.nn.Parameter(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.SGD([coords], lr=1e-2)
    args = SimpleNamespace(
        weight_dict={
            "den": 0.0,
            "geometric": 0.0,
            "user_bond": 1.0,
            "user_angle": 0.0,
        },
        use_global_clash=False,
    )
    start_distance = torch.linalg.norm(coords.detach()[0, 0] - coords.detach()[0, 1]).item()
    for _ in range(100):
        optimizer.zero_grad()
        _, total_loss, _, _ = refine_loss(
            crop_idx=0,
            predicted_coords=coords,
            target_density=None,
            feats={"molecule_type": "PROTEIN"},
            args=args,
            final_global_refined_coords=coords,
            user_restraints=resolved,
        )
        total_loss.backward()
        optimizer.step()
    end_distance = torch.linalg.norm(coords.detach()[0, 0] - coords.detach()[0, 1]).item()
    assert abs(end_distance - 2.0) < abs(start_distance - 2.0)


def test_secondary_structure_restraint_bonds_not_exported_as_struct_conn(tmp_path):
    constraints_dir = tmp_path / "constraints"
    data_dir = tmp_path / "templates"
    constraints_dir.mkdir()
    data_dir.mkdir()
    constraints_path = constraints_dir / "model.json"
    constraints_path.write_text(
        json.dumps(
            {
                "bonds": [
                    {
                        "atom1": {"auth_asym_id": "A", "auth_seq_id": "1", "auth_comp_id": "ALA", "atom_name": "O"},
                        "atom2": {"auth_asym_id": "A", "auth_seq_id": "5", "auth_comp_id": "ALA", "atom_name": "N"},
                        "distance_ideal": 2.9,
                        "sigma": 0.05,
                        "restraint_source": "secondary_structure",
                        "secondary_structure_type": "protein_helix",
                    },
                    {
                        "atom1": {"auth_asym_id": "A", "auth_seq_id": "1", "auth_comp_id": "ZN", "atom_name": "ZN"},
                        "atom2": {"auth_asym_id": "A", "auth_seq_id": "2", "auth_comp_id": "HIS", "atom_name": "ND1"},
                        "distance_ideal": 2.1,
                        "sigma": 0.1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    bonds = _load_restraint_bonds(data_dir, "model")
    assert len(bonds) == 1
    assert bonds[0]["distance_ideal"] == 2.1


def test_nucleic_n1_n3_angle_params_follow_atom_order():
    assert _na_hbond_angle_params("N1", "N3", "G", "C") == ((119.1, 2.59), (116.3, 2.66))
    assert _na_hbond_angle_params("N3", "N1", "C", "G") == ((116.3, 2.66), (119.1, 2.59))
    assert _na_hbond_angle_params("N1", "N3", "A", "U") == ((116.2, 3.46), (115.8, 2.88))
    assert _na_hbond_angle_params("N3", "N1", "U", "A") == ((115.8, 2.88), (116.2, 3.46))


def test_secondary_structure_plane_parallelity_uses_cctbx_radian_weight():
    class _Resid:
        def __init__(self, chain, name, resseq):
            self.chain = chain
            self.name = name
            self.resseq = resseq
            self.icode = ""

        def resid(self):
            return str(self.resseq)

        def selection(self):
            return f"chain '{self.chain}' and resid {self.resseq}"

    class _Residue:
        def __init__(self, chain, name, resseq):
            self.resid = _Resid(chain, name, resseq)

        def atom(self, atom_name):
            return object() if atom_name in {"N1", "C2", "N3"} else None

    payload = {"plane_parallelities": []}
    _add_plane_parallelity(
        payload,
        accepted_planes=set(),
        residue1=_Residue("A", "G", 1),
        residue2=_Residue("B", "C", 2),
        sigma_rad=NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD,
        ss_kind="nucleic_base_pair",
        parent={"kind": "base_pair"},
    )
    plane = payload["plane_parallelities"][0]
    assert np.isclose(plane["sigma"], np.degrees(NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD))
    assert np.isclose(plane["weight"], 1.0 / (NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD**2))


def _run_as_script() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        test_load_and_resolve_user_restraints_json(tmp_path)
        test_resolve_user_restraints_duplicate_match_error(tmp_path)
        test_secondary_structure_restraint_bonds_not_exported_as_struct_conn(tmp_path)
    test_compute_user_restraint_losses_zero_at_ideal()
    test_compute_user_plane_parallelity_loss_zero_at_ideal()
    test_merge_user_restraints_specs_user_overrides_default_duplicates()
    test_refine_loss_user_restraints_reduce_violation()
    test_nucleic_n1_n3_angle_params_follow_atom_order()
    test_secondary_structure_plane_parallelity_uses_cctbx_radian_weight()
    print("All user restraint tests passed.")


if __name__ == "__main__":
    _run_as_script()
