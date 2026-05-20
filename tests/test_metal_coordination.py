from __future__ import annotations

import numpy as np

from CryoNetRefine.data import const
from CryoNetRefine.data.parse.metal_coordination import (
    AutoMetalRestraintOptions,
    build_default_metal_restraints,
)
from CryoNetRefine.data.types import AtomV2, BondV2, Chain, Coords, Ensemble, Interface, Residue, StructureV2


def _build_structure(
    residue_specs: list[dict[str, object]],
    chain_types: dict[str, int],
    bonds: list[tuple[int, int]] | None = None,
) -> StructureV2:
    atom_rows: list[tuple[object, ...]] = []
    residue_rows: list[tuple[object, ...]] = []
    chain_rows: list[tuple[object, ...]] = []
    coords_rows: list[tuple[tuple[float, float, float],]] = []

    chain_order = list(chain_types.keys())
    residue_offsets: dict[str, int] = {chain_id: 0 for chain_id in chain_order}
    atom_offsets: dict[str, int] = {chain_id: 0 for chain_id in chain_order}
    chain_residue_counts: dict[str, int] = {chain_id: 0 for chain_id in chain_order}
    chain_atom_counts: dict[str, int] = {chain_id: 0 for chain_id in chain_order}

    atom_idx = 0
    for residue_idx, spec in enumerate(residue_specs):
        chain_id = str(spec["chain"])
        residue_atoms = list(spec["atoms"])
        atom_start = atom_idx
        for atom_name, atom_coords in residue_atoms:
            atom_rows.append((atom_name, atom_coords, True, 20.0, 0.0))
            coords_rows.append((atom_coords,))
            atom_idx += 1
        residue_rows.append(
            (
                spec["name"],
                0,
                residue_idx + 1,
                atom_start,
                len(residue_atoms),
                atom_start,
                atom_start,
                bool(spec.get("is_standard", True)),
                True,
                str(spec["auth_seq_id"]),
                "",
                str(spec.get("auth_comp_id", spec["name"])),
            )
        )
        chain_residue_counts[chain_id] += 1
        chain_atom_counts[chain_id] += len(residue_atoms)

    running_res_idx = 0
    running_atom_idx = 0
    for chain_id in chain_order:
        chain_rows.append(
            (
                chain_id,
                chain_types[chain_id],
                running_res_idx,
                0,
                chain_residue_counts[chain_id],
                running_atom_idx,
                chain_atom_counts[chain_id],
                running_res_idx,
                chain_residue_counts[chain_id],
                0,
                chain_id,
            )
        )
        running_res_idx += chain_residue_counts[chain_id]
        running_atom_idx += chain_atom_counts[chain_id]

    bond_rows = []
    for atom1, atom2 in bonds or []:
        bond_rows.append((0, 0, 0, 0, atom1, atom2, const.bond_type_ids["COVALENT"]))

    return StructureV2(
        atoms=np.array(atom_rows, dtype=AtomV2),
        bonds=np.array(bond_rows, dtype=BondV2),
        residues=np.array(residue_rows, dtype=Residue),
        chains=np.array(chain_rows, dtype=Chain),
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(len(chain_rows), dtype=bool),
        coords=np.array(coords_rows, dtype=Coords),
        ensemble=np.array([(0, len(atom_rows))], dtype=Ensemble),
        pocket=None,
    )


def _make_generic_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "ZN",
                "chain": "A",
                "auth_seq_id": "1",
                "auth_comp_id": "ZN",
                "is_standard": False,
                "atoms": [("ZN", (0.0, 0.0, 0.0))],
            },
            {
                "name": "HIS",
                "chain": "B",
                "auth_seq_id": "2",
                "atoms": [("ND1", (2.4, 0.0, 0.0)), ("CE1", (1.8, 0.0, 0.0))],
            },
            {
                "name": "ASP",
                "chain": "B",
                "auth_seq_id": "3",
                "atoms": [("OD1", (0.0, 2.1, 0.0))],
            },
            {
                "name": "GLY",
                "chain": "B",
                "auth_seq_id": "4",
                "atoms": [("C", (5.0, 5.0, 5.0))],
            },
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "B": int(const.chain_type_ids["PROTEIN"])},
    )


def _make_rna_fallback_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "MG",
                "chain": "A",
                "auth_seq_id": "1",
                "auth_comp_id": "MG",
                "is_standard": False,
                "atoms": [("MG", (0.0, 0.0, 0.0))],
            },
            {
                "name": "U",
                "chain": "C",
                "auth_seq_id": "24",
                "auth_comp_id": "U",
                "atoms": [("OP1", (2.4, 0.0, 0.0))],
            },
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "C": int(const.chain_type_ids["RNA"])},
    )


def _make_protein_link_budget_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "MG",
                "chain": "A",
                "auth_seq_id": "1",
                "auth_comp_id": "MG",
                "is_standard": False,
                "atoms": [("MG", (0.0, 0.0, 0.0))],
            },
            {"name": "ASP", "chain": "B", "auth_seq_id": "2", "atoms": [("OD1", (2.0, 0.0, 0.0))]},
            {"name": "GLU", "chain": "B", "auth_seq_id": "3", "atoms": [("OE1", (0.0, 2.1, 0.0))]},
            {"name": "SER", "chain": "B", "auth_seq_id": "4", "atoms": [("OG", (0.0, 0.0, 2.2))]},
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "B": int(const.chain_type_ids["PROTEIN"])},
    )


def _make_nad_name_collision_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "NAD",
                "chain": "A",
                "auth_seq_id": "481",
                "auth_comp_id": "NAD",
                "is_standard": False,
                "atoms": [("N1A", (0.0, 0.0, 0.0))],
            },
            {
                "name": "VAL",
                "chain": "B",
                "auth_seq_id": "243",
                "atoms": [("N", (2.9, 0.0, 0.0)), ("O", (0.0, 3.0, 0.0))],
            },
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "B": int(const.chain_type_ids["PROTEIN"])},
    )


def _make_zn_mcl_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "ZN",
                "chain": "A",
                "auth_seq_id": "1",
                "auth_comp_id": "ZN",
                "is_standard": False,
                "atoms": [("ZN", (0.0, 0.0, 0.0))],
            },
            {"name": "CYS", "chain": "B", "auth_seq_id": "10", "atoms": [("SG", (2.25, 0.0, 0.0))]},
            {"name": "CYS", "chain": "B", "auth_seq_id": "11", "atoms": [("SG", (-2.25, 0.0, 0.0))]},
            {"name": "HIS", "chain": "B", "auth_seq_id": "12", "atoms": [("ND1", (0.0, 2.0, 0.0))]},
            {"name": "HIS", "chain": "B", "auth_seq_id": "13", "atoms": [("ND1", (0.0, -2.0, 0.0))]},
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "B": int(const.chain_type_ids["PROTEIN"])},
    )


def _make_fes_mcl_structure() -> StructureV2:
    return _build_structure(
        residue_specs=[
            {
                "name": "FES",
                "chain": "A",
                "auth_seq_id": "1",
                "auth_comp_id": "FES",
                "is_standard": False,
                "atoms": [("FE", (0.0, 0.0, 0.0))],
            },
            {"name": "CYS", "chain": "B", "auth_seq_id": "20", "atoms": [("SG", (2.3, 0.0, 0.0))]},
            {"name": "CYS", "chain": "B", "auth_seq_id": "21", "atoms": [("SG", (0.0, 2.3, 0.0))]},
        ],
        chain_types={"A": int(const.chain_type_ids["NONPOLYMER"]), "B": int(const.chain_type_ids["PROTEIN"])},
    )


def test_auto_metal_restraints_input_strategy_uses_model_distance():
    structure = _make_generic_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="input",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 2
    bond_by_atom = {bond["atom2"]["atom_name"]: bond for bond in restraints["bonds"]}
    assert "ND1" in bond_by_atom
    assert "OD1" in bond_by_atom
    assert abs(bond_by_atom["ND1"]["distance_ideal"] - 2.4) < 1e-6
    assert abs(bond_by_atom["OD1"]["distance_ideal"] - 2.1) < 1e-6


def test_auto_metal_restraints_library_strategy_uses_cctbx_like_defaults():
    structure = _make_generic_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    bond_by_atom = {bond["atom2"]["atom_name"]: bond for bond in restraints["bonds"]}
    assert abs(bond_by_atom["ND1"]["distance_ideal"] - 2.3) < 1e-6
    assert abs(bond_by_atom["ND1"]["sigma"] - 0.03) < 1e-6
    assert abs(bond_by_atom["OD1"]["distance_ideal"] - 1.99) < 1e-6


def test_generic_library_uses_non_protein_fallback_for_rna_oxygen():
    structure = _make_rna_fallback_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 1
    bond = restraints["bonds"][0]
    assert bond["atom2"]["atom_name"] == "OP1"
    assert abs(bond["distance_ideal"] - 2.09) < 1e-6
    assert abs(bond["sigma"] - 0.25) < 1e-6


def test_generic_library_limits_protein_links_per_metal():
    structure = _make_protein_link_budget_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 2
    selected = {bond["atom2"]["atom_name"] for bond in restraints["bonds"]}
    assert selected == {"OD1", "OE1"}


def test_nad_atom_names_are_not_inferred_as_sodium():
    structure = _make_nad_name_collision_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    assert restraints == {"bonds": [], "angles": []}


def test_mcl_zn_tetrahedral_adds_bonds_and_angles():
    structure = _make_zn_mcl_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 4
    assert len(restraints["angles"]) == 2
    bond_by_atom = {(bond["atom2"]["auth_comp_id"], bond["atom2"]["atom_name"]): bond for bond in restraints["bonds"]}
    assert abs(bond_by_atom[("CYS", "SG")]["distance_ideal"] - 2.306) < 1e-6
    assert abs(bond_by_atom[("CYS", "SG")]["sigma"] - 0.029) < 1e-6
    assert abs(bond_by_atom[("HIS", "ND1")]["distance_ideal"] - 2.04) < 1e-6
    angle_by_atoms = {
        (angle["atom1"]["atom_name"], angle["atom3"]["atom_name"]): angle for angle in restraints["angles"]
    }
    assert abs(angle_by_atoms[("SG", "SG")]["angle_ideal_deg"] - 116.23) < 1e-6
    assert abs(angle_by_atoms[("ND1", "ND1")]["angle_ideal_deg"] - 102.38) < 1e-6


def test_mcl_fes_cluster_adds_bonds_and_angles():
    structure = _make_fes_mcl_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=None,
        options=AutoMetalRestraintOptions(
            enabled=True,
            ideal_distance_strategy="library",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 2
    assert len(restraints["angles"]) == 1
    assert abs(restraints["bonds"][0]["distance_ideal"] - 2.305) < 1e-6
    assert abs(restraints["angles"][0]["angle_ideal_deg"] - 107.77) < 1e-6


def test_explicit_pairs_are_kept_when_auto_detection_disabled():
    structure = _make_generic_structure()
    restraints = build_default_metal_restraints(
        structure,
        explicit_pairs=[(0, 3)],
        options=AutoMetalRestraintOptions(
            enabled=False,
            ideal_distance_strategy="input",
            coordination_cutoff=3.0,
        ),
    )
    assert len(restraints["bonds"]) == 1
    bond = restraints["bonds"][0]
    assert bond["atom1"]["atom_name"] == "ZN"
    assert bond["atom2"]["atom_name"] == "OD1"
    assert abs(bond["distance_ideal"] - 2.1) < 1e-6


def _run_as_script() -> None:
    test_auto_metal_restraints_input_strategy_uses_model_distance()
    test_auto_metal_restraints_library_strategy_uses_cctbx_like_defaults()
    test_generic_library_uses_non_protein_fallback_for_rna_oxygen()
    test_generic_library_limits_protein_links_per_metal()
    test_nad_atom_names_are_not_inferred_as_sodium()
    test_mcl_zn_tetrahedral_adds_bonds_and_angles()
    test_mcl_fes_cluster_adds_bonds_and_angles()
    test_explicit_pairs_are_kept_when_auto_detection_disabled()
    print("All metal coordination tests passed.")


if __name__ == "__main__":
    _run_as_script()
