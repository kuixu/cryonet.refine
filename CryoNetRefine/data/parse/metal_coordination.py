from __future__ import annotations

"""cctbx-like automatic metal restraint generation.

The checked-out cctbx tree uses two layers for metal-related restraints:
1. Generic automatic linking in monomer_library.
2. Specialized dynamic augmentation in MCL for Zn tetrahedral and Fe-S clusters.

This module mirrors that split for CryoNet's default metal restraints when
`ideal_distance_strategy == "library"`. See the adjacent parity note for the
full source mapping and scope boundaries.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.neighbors import KDTree

from CryoNetRefine.data import const
from CryoNetRefine.data.types import StructureV2


CCTBX_METAL_ELEMENTS = {
    "ZN",
    "CA",
    "MG",
    "NA",
    "MN",
    "K",
    "FE",
    "CU",
    "CD",
    "HG",
    "NI",
    "CO",
    "SR",
    "CS",
    "PT",
    "BA",
    "TL",
    "PB",
    "SM",
    "AU",
    "RB",
    "YB",
    "LI",
    "MO",
    "LU",
    "CR",
    "OS",
    "GD",
    "TB",
    "LA",
    "AG",
    "HO",
    "GA",
    "CE",
    "W",
    "RU",
    "RE",
    "PR",
    "IR",
    "EU",
    "AL",
    "V",
    "PD",
    "U",
    "SB",
    "SE",
    "TE",
}
CCTBX_NON_LINKING_ELEMENTS = {"H", "D", "F", "CL", "BR", "I", "AT", "HE", "NE", "AR", "KR", "XE"}
CCTBX_FIRST_ROW_ELEMENTS = {"LI", "BE", "B", "C", "N", "O", "F"}
CCTBX_WATER_RESNAMES = {"HOH", "WAT", "TIP"}
CCTBX_PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
CCTBX_GENERIC_SKIP_RESNAMES = {"SF4", "F3S", "FES"}
CCTBX_GENERIC_SKIP_RESNAME_PAIRS = {
    frozenset({"ZN", "CYS"}),
    frozenset({"ZN", "HIS"}),
}
CCTBX_EXCLUDED_METAL_ATOM_NAMES = {("HIS", "CE1"), ("HIS", "CD2"), ("HIS", "CB")}
CCTBX_MAX_INTER_RESIDUE_LINKS = {
    ("common_element", "other"): 8,
    ("metal", "other"): 6,
    ("common_amino_acid", "metal"): 2,
    ("common_saccharide", "metal"): 3,
    ("common_rna_dna", "metal"): 2,
    ("common_rna_dna", "common_rna_dna"): 5,
}
CCTBX_MAX_PER_ATOM_LINKS = {
    "common_saccharide": 1,
    "common_rna_dna": 1,
    "common_amino_acid": 1,
    "other": 1,
}
CCTBX_SECOND_ROW_BUFFER = 0.5
CCTBX_FALLBACK_DISTANCE_IDEAL = 2.3
CCTBX_FALLBACK_SIGMA = 0.02
CCTBX_QM_FALLBACK_SIGMA = 0.01
CCTBX_MCL_ZN_CUTOFF = 3.0
CCTBX_MCL_CLUSTER_CUTOFF = 3.5

BASE_METAL_DEFAULTS: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "NA": {"O": {"HOH": (2.41, 0.10), "ASP": (2.41, 0.10)}},
    "MG": {"O": {"HOH": (2.07, 0.05), "ASP": (2.07, 0.10)}},
    "K": {"O": {"HOH": (2.81, 0.15), "ASP": (2.82, 0.10)}},
    "CA": {"O": {"HOH": (2.39, 0.10), "ASP": (2.36, 0.10)}},
    "MN": {
        "O": {"HOH": (2.19, 0.05), "ASP": (2.15, 0.05)},
        "N": {"HIS": (2.21, 0.10)},
        "S": {"CYS": (2.35, 0.25)},
    },
    "FE": {
        "O": {"HOH": (2.09, 0.10), "ASP": (2.04, 0.10)},
        "N": {"HIS": (2.16, 0.15)},
        "S": {"CYS": (2.30, 0.05)},
    },
    "CO": {
        "O": {"HOH": (2.09, 0.10), "ASP": (2.05, 0.05)},
        "N": {"HIS": (2.14, 0.10)},
        "S": {"CYS": (2.25, 0.15)},
    },
    "CU": {
        "O": {"HOH": (2.13, 0.25), "ASP": (1.99, 0.15)},
        "N": {"HIS": (2.02, 0.10)},
        "S": {"CYS": (2.15, 0.25)},
    },
    "ZN": {
        "O": {"HOH": (2.09, 0.05), "ASP": (1.99, 0.05)},
        "N": {"HIS": (2.03, 0.05)},
        "S": {"CYS": (2.31, 0.10)},
    },
}
CARBONYL_DEFAULTS: dict[str, tuple[float, float]] = {
    "NA": (2.38, 0.10),
    "MG": (2.26, 0.25),
    "K": (2.74, 0.15),
    "CA": (2.36, 0.10),
    "MN": (2.19, 0.25),
    "FE": (2.04, 0.25),
    "CO": (2.08, 0.25),
    "CU": (2.04, 0.25),
    "ZN": (2.07, 0.25),
}
NON_PROTEIN_SIGMA = 0.25

MCL_ZN_DATABASE: dict[tuple[int, int], dict[tuple[str, ...], tuple[float, float]]] = {
    (4, 0): {
        ("ZN", "SG"): (2.330, 0.029),
        ("SG", "ZN", "SG"): (109.45, 5.46),
    },
    (3, 1): {
        ("ZN", "SG"): (2.318, 0.027),
        ("SG", "ZN", "SG"): (112.15, 3.96),
        ("ZN", "ND1"): (2.074, 0.056),
    },
    (2, 2): {
        ("ZN", "SG"): (2.306, 0.029),
        ("SG", "ZN", "SG"): (116.23, 4.58),
        ("ZN", "ND1"): (2.040, 0.050),
        ("ND1", "ZN", "ND1"): (102.38, 5.44),
    },
    (1, 3): {
        ("ZN", "SG"): (2.298, 0.017),
        ("ZN", "ND1"): (2.002, 0.045),
        ("ND1", "ZN", "ND1"): (107.23, 4.78),
    },
    (0, 4): {},
}
for coordination in MCL_ZN_DATABASE.values():
    for key, value in list(coordination.items()):
        if "ND1" in key:
            coordination[tuple("NE2" if item == "ND1" else item for item in key)] = value

SF_CLUSTER_COORDINATION = {
    "SF4": {
        "CYS": {
            ("FE", "S"): (2.268, 0.034),
            ("S", "FE", "S"): (114.24, 11.5),
        },
        "MET": {
            ("FE", "S"): (2.311, 0.012),
            ("S", "FE", "S"): (113.97, 17.528),
        },
        "HIS": {
            ("FE", "N"): (2.04, 0.05),
        },
    },
    "F3S": {
        "CYS": {
            ("FE", "S"): (2.318, 0.016),
            ("S", "FE", "S"): (112.23, 12.06),
        },
    },
    "FES": {
        "CYS": {
            ("FE", "S"): (2.305, 0.044),
            ("S", "FE", "S"): (111.20, 8.10),
            ("SG", "FE", "SG"): (107.77, 8.16),
            ("CYS", "CYS"): ("SG", "FE", "SG"),
        },
        "HIS": {
            ("FE", "N"): (2.14, 0.05),
        },
    },
}
SF_CLUSTERS = {"SF4", "F3S", "FES"}
F3S_NAMING = {1: 4, 3: 2, 4: 1}


def _expand_metal_defaults() -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    expanded: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for metal_name, donor_table in BASE_METAL_DEFAULTS.items():
        expanded[metal_name] = {donor: dict(values) for donor, values in donor_table.items()}
    for metal_name, donor_table in expanded.items():
        oxygen_defaults = donor_table.get("O")
        if oxygen_defaults is None:
            continue
        oxygen_defaults["WAT"] = oxygen_defaults["HOH"]
        oxygen_defaults["TIP"] = oxygen_defaults["HOH"]
        oxygen_defaults["GLU"] = oxygen_defaults["ASP"]
        oxygen_defaults["ASN"] = (oxygen_defaults["ASP"][0] + 0.02, oxygen_defaults["ASP"][1])
        oxygen_defaults["GLN"] = (oxygen_defaults["GLU"][0] + 0.02, oxygen_defaults["GLU"][1])
        ser_thr = ((oxygen_defaults["ASP"][0] + oxygen_defaults["HOH"][0]) / 2.0, oxygen_defaults["ASP"][1])
        oxygen_defaults["SER"] = ser_thr
        oxygen_defaults["THR"] = ser_thr
        oxygen_defaults["TYR"] = (oxygen_defaults["ASP"][0] - 0.1, oxygen_defaults["ASP"][1])
    return expanded


METAL_DEFAULTS = _expand_metal_defaults()
NON_PROTEIN_DEFAULTS: dict[tuple[str, str], tuple[float, float]] = {}
for metal_name, fallback in CARBONYL_DEFAULTS.items():
    oxygen_default = METAL_DEFAULTS[metal_name]["O"]["ASN"]
    NON_PROTEIN_DEFAULTS[(metal_name, "O")] = (oxygen_default[0], NON_PROTEIN_SIGMA)
    if "N" in METAL_DEFAULTS[metal_name]:
        NON_PROTEIN_DEFAULTS[(metal_name, "N")] = (METAL_DEFAULTS[metal_name]["N"]["HIS"][0], NON_PROTEIN_SIGMA)
    if "S" in METAL_DEFAULTS[metal_name]:
        NON_PROTEIN_DEFAULTS[(metal_name, "S")] = (METAL_DEFAULTS[metal_name]["S"]["CYS"][0], NON_PROTEIN_SIGMA)


@dataclass(frozen=True)
class AutoMetalRestraintOptions:
    enabled: bool = True
    ideal_distance_strategy: str = "input"
    coordination_cutoff: float = 3.0
    input_distance_sigma: float = 0.2


@dataclass(frozen=True)
class _AtomInfo:
    atom_idx: int
    chain_idx: int
    residue_idx: int
    chain_name: str
    auth_asym_id: str
    resname: str
    auth_comp_id: str
    auth_seq_id: str
    atom_name: str
    element: str
    mol_type: int
    coords: np.ndarray

    @property
    def residue_name(self) -> str:
        return self.auth_comp_id or self.resname

    @property
    def residue_key(self) -> tuple[int, int]:
        return (self.chain_idx, self.residue_idx)


def _infer_element(atom_name: str, residue_name: str | None = None) -> str:
    atom_text = "".join(ch for ch in str(atom_name).strip().upper() if ch.isalpha())
    residue_text = str(residue_name or "").strip().upper()
    if residue_text in CCTBX_METAL_ELEMENTS and atom_text.startswith(residue_text):
        return residue_text
    if len(atom_text) >= 2 and atom_text[:2] in CCTBX_METAL_ELEMENTS:
        # Do not infer two-letter metals from arbitrary ligand atom names:
        # NAD atom names such as N1A/N3A/N6A collapse to "NA" after stripping
        # digits and must remain nitrogen, not sodium. Multi-letter metal
        # inference is only safe for actual metal residues or supported metal
        # clusters.
        if residue_text in CCTBX_METAL_ELEMENTS or residue_text in SF_CLUSTERS:
            return atom_text[:2]
    if not atom_text:
        return residue_text[:1]
    return atom_text[:1]


def _build_atom_infos(structure: StructureV2) -> list[_AtomInfo]:
    atom_infos: list[_AtomInfo] = []
    for chain_idx, chain in enumerate(structure.chains):
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for residue_idx in range(res_start, res_end):
            residue = structure.residues[residue_idx]
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            for atom_idx in range(atom_start, atom_end):
                atom = structure.atoms[atom_idx]
                atom_infos.append(
                    _AtomInfo(
                        atom_idx=atom_idx,
                        chain_idx=chain_idx,
                        residue_idx=residue_idx,
                        chain_name=str(chain["name"]).strip(),
                        auth_asym_id=str(chain["auth_asym_id"]).strip(),
                        resname=str(residue["name"]).strip(),
                        auth_comp_id=str(residue["auth_comp_id"]).strip(),
                        auth_seq_id=str(residue["auth_seq_id"]).strip(),
                        atom_name=str(atom["name"]).strip(),
                        element=_infer_element(atom["name"], residue["auth_comp_id"]),
                        mol_type=int(chain["mol_type"]),
                        coords=np.asarray(atom["coords"], dtype=np.float32),
                    )
                )
    return atom_infos


def _build_bond_graph(structure: StructureV2) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for bond in structure.bonds:
        atom_idx1 = int(bond["atom_1"])
        atom_idx2 = int(bond["atom_2"])
        adjacency.setdefault(atom_idx1, set()).add(atom_idx2)
        adjacency.setdefault(atom_idx2, set()).add(atom_idx1)
    return adjacency


def _classify_atom(atom: _AtomInfo) -> str:
    if atom.element in CCTBX_METAL_ELEMENTS:
        return "metal"
    if atom.residue_name.upper() in CCTBX_WATER_RESNAMES:
        return "common_water"
    if atom.mol_type == int(const.chain_type_ids["PROTEIN"]):
        return "common_amino_acid"
    if atom.mol_type in {int(const.chain_type_ids["DNA"]), int(const.chain_type_ids["RNA"])}:
        return "common_rna_dna"
    return "other"


def _pair_key(atom_idx1: int, atom_idx2: int) -> tuple[int, int]:
    return (atom_idx1, atom_idx2) if atom_idx1 < atom_idx2 else (atom_idx2, atom_idx1)


def _bond_entry(atom1: _AtomInfo, atom2: _AtomInfo, distance_ideal: float, sigma: float) -> dict[str, Any]:
    return {
        "atom1": {
            "auth_asym_id": atom1.auth_asym_id,
            "auth_seq_id": atom1.auth_seq_id,
            "auth_comp_id": atom1.auth_comp_id,
            "atom_name": atom1.atom_name,
        },
        "atom2": {
            "auth_asym_id": atom2.auth_asym_id,
            "auth_seq_id": atom2.auth_seq_id,
            "auth_comp_id": atom2.auth_comp_id,
            "atom_name": atom2.atom_name,
        },
        "distance_ideal": float(distance_ideal),
        "sigma": float(sigma),
    }


def _angle_entry(atom1: _AtomInfo, atom2: _AtomInfo, atom3: _AtomInfo, angle_ideal: float, sigma: float) -> dict[str, Any]:
    return {
        "atom1": {
            "auth_asym_id": atom1.auth_asym_id,
            "auth_seq_id": atom1.auth_seq_id,
            "auth_comp_id": atom1.auth_comp_id,
            "atom_name": atom1.atom_name,
        },
        "atom2": {
            "auth_asym_id": atom2.auth_asym_id,
            "auth_seq_id": atom2.auth_seq_id,
            "auth_comp_id": atom2.auth_comp_id,
            "atom_name": atom2.atom_name,
        },
        "atom3": {
            "auth_asym_id": atom3.auth_asym_id,
            "auth_seq_id": atom3.auth_seq_id,
            "auth_comp_id": atom3.auth_comp_id,
            "atom_name": atom3.atom_name,
        },
        "angle_ideal_deg": float(angle_ideal),
        "sigma": float(sigma),
    }


def _generic_library_params(metal: _AtomInfo, other: _AtomInfo) -> tuple[float, float]:
    metal_name = metal.element.upper()
    donor_element = other.element.upper()
    residue_name = other.residue_name.upper()
    atom_name = other.atom_name.upper()

    if donor_element == "O" and residue_name in CCTBX_PROTEIN_RESNAMES and atom_name == "O":
        carbonyl_default = CARBONYL_DEFAULTS.get(metal_name)
        if carbonyl_default is not None:
            return carbonyl_default

    defaults = METAL_DEFAULTS.get(metal_name)
    if defaults is None:
        return (CCTBX_FALLBACK_DISTANCE_IDEAL, CCTBX_FALLBACK_SIGMA)

    element_defaults = defaults.get(donor_element)
    if element_defaults is not None:
        residue_default = element_defaults.get(residue_name)
        if residue_default is not None:
            return residue_default

    non_protein = NON_PROTEIN_DEFAULTS.get((metal_name, donor_element))
    if non_protein is not None:
        return non_protein

    return (CCTBX_FALLBACK_DISTANCE_IDEAL, CCTBX_FALLBACK_SIGMA)


def _generic_params(
    metal: _AtomInfo,
    other: _AtomInfo,
    model_distance: float,
    options: AutoMetalRestraintOptions,
) -> tuple[float, float]:
    if options.ideal_distance_strategy == "library":
        return _generic_library_params(metal, other)
    return (model_distance, options.input_distance_sigma)


def _bonded_to_selected_donor(
    metal_idx: int,
    donor_idx: int,
    selected_donors_by_metal: dict[int, set[int]],
    bond_graph: dict[int, set[int]],
) -> bool:
    selected = selected_donors_by_metal.get(metal_idx, set())
    for neighbor in bond_graph.get(donor_idx, set()):
        if neighbor in selected:
            return True
    return False


def _generic_cutoff_for_pair(metal: _AtomInfo, other: _AtomInfo, options: AutoMetalRestraintOptions) -> float:
    cutoff = float(options.coordination_cutoff)
    if metal.element.upper() not in CCTBX_FIRST_ROW_ELEMENTS or other.element.upper() not in CCTBX_FIRST_ROW_ELEMENTS:
        cutoff = float(np.sqrt(cutoff * cutoff + CCTBX_SECOND_ROW_BUFFER * CCTBX_SECOND_ROW_BUFFER))
    return cutoff


def _is_generic_auto_candidate(
    metal: _AtomInfo,
    other: _AtomInfo,
    options: AutoMetalRestraintOptions,
) -> bool:
    if metal.element.upper() not in CCTBX_METAL_ELEMENTS:
        return False
    if metal.atom_idx == other.atom_idx:
        return False
    if metal.residue_key == other.residue_key:
        return False
    if options.ideal_distance_strategy == "library":
        if metal.residue_name.upper() in CCTBX_GENERIC_SKIP_RESNAMES or other.residue_name.upper() in CCTBX_GENERIC_SKIP_RESNAMES:
            return False
        if frozenset({metal.residue_name.upper(), other.residue_name.upper()}) in CCTBX_GENERIC_SKIP_RESNAME_PAIRS:
            return False
    if other.element.upper() in CCTBX_NON_LINKING_ELEMENTS:
        return False
    if other.element.upper() == "C":
        return False
    if (other.residue_name.upper(), other.atom_name.upper()) in CCTBX_EXCLUDED_METAL_ATOM_NAMES:
        return False
    return True


def _collect_generic_candidates(
    atom_infos: list[_AtomInfo],
    options: AutoMetalRestraintOptions,
) -> list[tuple[float, int, int]]:
    metal_atoms = [atom for atom in atom_infos if atom.element.upper() in CCTBX_METAL_ELEMENTS]
    other_atoms = [atom for atom in atom_infos if atom.element.upper() not in CCTBX_NON_LINKING_ELEMENTS]
    if not metal_atoms or not other_atoms:
        return []

    donor_coords = np.stack([atom.coords for atom in other_atoms], axis=0)
    donor_tree = KDTree(donor_coords)
    candidates: list[tuple[float, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for metal in metal_atoms:
        local_cutoff = float(options.coordination_cutoff)
        neighbor_ids = donor_tree.query_radius(
            metal.coords.reshape(1, -1),
            r=local_cutoff + CCTBX_SECOND_ROW_BUFFER,
            return_distance=False,
        )[0]
        for donor_list_idx in neighbor_ids.tolist():
            donor = other_atoms[donor_list_idx]
            key = _pair_key(metal.atom_idx, donor.atom_idx)
            if key in seen_pairs:
                continue
            if not _is_generic_auto_candidate(metal, donor, options):
                continue
            distance = float(np.linalg.norm(metal.coords - donor.coords))
            if not np.isfinite(distance) or distance <= 0:
                continue
            if distance > _generic_cutoff_for_pair(metal, donor, options):
                continue
            seen_pairs.add(key)
            candidates.append((distance, metal.atom_idx, donor.atom_idx))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _check_generic_link_limits(
    metal: _AtomInfo,
    other: _AtomInfo,
    donor_link_counts: dict[int, int],
    metal_class_counts: dict[tuple[int, tuple[str, str]], int],
    selected_donors_by_metal: dict[int, set[int]],
    bond_graph: dict[int, set[int]],
) -> bool:
    donor_class = _classify_atom(other)
    donor_limit = CCTBX_MAX_PER_ATOM_LINKS.get(donor_class)
    if donor_limit is not None and donor_link_counts.get(other.atom_idx, 0) >= donor_limit:
        return False

    class_pair = tuple(sorted(("metal", donor_class)))
    class_limit = CCTBX_MAX_INTER_RESIDUE_LINKS.get(class_pair)
    if class_limit is not None and metal_class_counts.get((metal.atom_idx, class_pair), 0) >= class_limit:
        return False

    if _bonded_to_selected_donor(metal.atom_idx, other.atom_idx, selected_donors_by_metal, bond_graph):
        return False
    return True


def _add_bond(
    bonds: list[dict[str, Any]],
    accepted_pairs: set[tuple[int, int]],
    atom1: _AtomInfo,
    atom2: _AtomInfo,
    distance_ideal: float,
    sigma: float,
) -> None:
    bonds.append(_bond_entry(atom1, atom2, distance_ideal, sigma))
    accepted_pairs.add(_pair_key(atom1.atom_idx, atom2.atom_idx))


def _add_angle(
    angles: list[dict[str, Any]],
    accepted_angles: set[tuple[int, int, int]],
    atom1: _AtomInfo,
    atom2: _AtomInfo,
    atom3: _AtomInfo,
    angle_ideal: float,
    sigma: float,
) -> None:
    key = tuple(sorted((atom1.atom_idx, atom2.atom_idx, atom3.atom_idx)))
    if key in accepted_angles:
        return
    angles.append(_angle_entry(atom1, atom2, atom3, angle_ideal, sigma))
    accepted_angles.add(key)


def _apply_explicit_pairs(
    explicit_pairs: list[tuple[int, int]],
    atom_by_idx: dict[int, _AtomInfo],
    options: AutoMetalRestraintOptions,
    bonds: list[dict[str, Any]],
    accepted_pairs: set[tuple[int, int]],
    donor_link_counts: dict[int, int],
    metal_class_counts: dict[tuple[int, tuple[str, str]], int],
    selected_donors_by_metal: dict[int, set[int]],
) -> None:
    for atom_idx1, atom_idx2 in explicit_pairs:
        atom1 = atom_by_idx.get(atom_idx1)
        atom2 = atom_by_idx.get(atom_idx2)
        if atom1 is None or atom2 is None:
            continue
        if atom1.element.upper() in CCTBX_METAL_ELEMENTS:
            metal, other = atom1, atom2
        elif atom2.element.upper() in CCTBX_METAL_ELEMENTS:
            metal, other = atom2, atom1
        else:
            continue
        distance = float(np.linalg.norm(metal.coords - other.coords))
        if not np.isfinite(distance) or distance <= 0:
            continue
        key = _pair_key(metal.atom_idx, other.atom_idx)
        if key in accepted_pairs:
            continue
        distance_ideal, sigma = _generic_params(metal, other, distance, options)
        _add_bond(bonds, accepted_pairs, metal, other, distance_ideal, sigma)
        donor_class = _classify_atom(other)
        donor_link_counts[other.atom_idx] = donor_link_counts.get(other.atom_idx, 0) + 1
        class_pair = tuple(sorted(("metal", donor_class)))
        metal_class_counts[(metal.atom_idx, class_pair)] = metal_class_counts.get((metal.atom_idx, class_pair), 0) + 1
        selected_donors_by_metal.setdefault(metal.atom_idx, set()).add(other.atom_idx)


def _apply_generic_auto_linking(
    atom_infos: list[_AtomInfo],
    atom_by_idx: dict[int, _AtomInfo],
    bond_graph: dict[int, set[int]],
    options: AutoMetalRestraintOptions,
    bonds: list[dict[str, Any]],
    accepted_pairs: set[tuple[int, int]],
    donor_link_counts: dict[int, int],
    metal_class_counts: dict[tuple[int, tuple[str, str]], int],
    selected_donors_by_metal: dict[int, set[int]],
) -> None:
    for distance, metal_idx, donor_idx in _collect_generic_candidates(atom_infos, options):
        metal = atom_by_idx[metal_idx]
        donor = atom_by_idx[donor_idx]
        if _pair_key(metal_idx, donor_idx) in accepted_pairs:
            continue
        if not _check_generic_link_limits(
            metal,
            donor,
            donor_link_counts,
            metal_class_counts,
            selected_donors_by_metal,
            bond_graph,
        ):
            continue
        distance_ideal, sigma = _generic_params(metal, donor, distance, options)
        _add_bond(bonds, accepted_pairs, metal, donor, distance_ideal, sigma)
        donor_class = _classify_atom(donor)
        donor_link_counts[donor.atom_idx] = donor_link_counts.get(donor.atom_idx, 0) + 1
        class_pair = tuple(sorted(("metal", donor_class)))
        metal_class_counts[(metal.atom_idx, class_pair)] = metal_class_counts.get((metal.atom_idx, class_pair), 0) + 1
        selected_donors_by_metal.setdefault(metal.atom_idx, set()).add(donor.atom_idx)


def _collect_mcl_zn_coordination(atom_infos: list[_AtomInfo]) -> dict[int, list[_AtomInfo]]:
    zinc_atoms = [atom for atom in atom_infos if atom.element.upper() == "ZN"]
    donor_atoms = [atom for atom in atom_infos if atom.atom_name.upper() in {"SG", "ND1", "NE2"}]
    if not zinc_atoms or not donor_atoms:
        return {}

    donor_coords = np.stack([atom.coords for atom in donor_atoms], axis=0)
    donor_tree = KDTree(donor_coords)
    coordination: dict[int, list[_AtomInfo]] = {}
    for zinc in zinc_atoms:
        neighbor_ids = donor_tree.query_radius(
            zinc.coords.reshape(1, -1),
            r=CCTBX_MCL_ZN_CUTOFF,
            return_distance=False,
        )[0]
        residue_seen: set[tuple[int, int]] = set()
        local: list[tuple[float, _AtomInfo]] = []
        for donor_list_idx in neighbor_ids.tolist():
            donor = donor_atoms[donor_list_idx]
            if donor.residue_key == zinc.residue_key:
                continue
            if donor.residue_name.upper() not in {"CYS", "HIS"}:
                continue
            distance = float(np.linalg.norm(zinc.coords - donor.coords))
            if not np.isfinite(distance) or distance <= 0:
                continue
            local.append((distance, donor))
        local.sort(key=lambda item: item[0])
        for _, donor in local:
            if donor.residue_key in residue_seen:
                continue
            coordination.setdefault(zinc.atom_idx, []).append(donor)
            residue_seen.add(donor.residue_key)
    return coordination


def _get_zn_bond_params(coordination_key: tuple[int, int], donor: _AtomInfo) -> tuple[float, float] | None:
    ideals = MCL_ZN_DATABASE.get(coordination_key)
    if ideals is None:
        return None
    return ideals.get(("ZN", donor.atom_name.upper()))


def _get_zn_angle_params(coordination_key: tuple[int, int], donor1: _AtomInfo, donor2: _AtomInfo) -> tuple[float, float] | None:
    ideals = MCL_ZN_DATABASE.get(coordination_key)
    if ideals is None:
        return None
    return ideals.get((donor1.atom_name.upper(), "ZN", donor2.atom_name.upper()))


def _apply_mcl_zn_layer(
    atom_by_idx: dict[int, _AtomInfo],
    atom_infos: list[_AtomInfo],
    bonds: list[dict[str, Any]],
    angles: list[dict[str, Any]],
    accepted_pairs: set[tuple[int, int]],
    accepted_angles: set[tuple[int, int, int]],
) -> None:
    coordination = _collect_mcl_zn_coordination(atom_infos)
    for zinc_idx, donors in coordination.items():
        zinc = atom_by_idx[zinc_idx]
        if len(donors) < 4:
            for donor in donors:
                if _pair_key(zinc.atom_idx, donor.atom_idx) in accepted_pairs:
                    continue
                _add_bond(bonds, accepted_pairs, zinc, donor, 2.3, 0.03)
            continue

        num_cys = sum(1 for donor in donors if donor.residue_name.upper() == "CYS")
        num_his = sum(1 for donor in donors if donor.residue_name.upper() == "HIS")
        coordination_key = (num_cys, num_his)
        if coordination_key not in MCL_ZN_DATABASE:
            continue

        for donor in donors:
            params = _get_zn_bond_params(coordination_key, donor)
            if params is None or _pair_key(zinc.atom_idx, donor.atom_idx) in accepted_pairs:
                continue
            _add_bond(bonds, accepted_pairs, zinc, donor, params[0], params[1])
        for donor1, donor2 in combinations(donors, 2):
            params = _get_zn_angle_params(coordination_key, donor1, donor2)
            if params is None:
                params = _get_zn_angle_params(coordination_key, donor2, donor1)
            if params is None:
                continue
            _add_angle(angles, accepted_angles, donor1, zinc, donor2, params[0], params[1])


def _get_sf_cluster_name(atom1: _AtomInfo, atom2: _AtomInfo, atom3: _AtomInfo | None = None, other: bool = False) -> str | None:
    residue_names = {atom1.residue_name.upper(), atom2.residue_name.upper()}
    if atom3 is not None:
        residue_names.add(atom3.residue_name.upper())
    if other:
        residue_names = residue_names.difference(SF_CLUSTERS)
    else:
        residue_names = SF_CLUSTERS.intersection(residue_names)
    if len(residue_names) == 1:
        return next(iter(residue_names))
    return None


def _collect_sf_cluster_coordination(atom_infos: list[_AtomInfo]) -> list[tuple[_AtomInfo, _AtomInfo]]:
    cluster_fe_atoms = [
        atom for atom in atom_infos if atom.residue_name.upper() in SF_CLUSTERS and atom.element.upper() == "FE"
    ]
    donor_atoms = [atom for atom in atom_infos if atom.element.upper() in {"S", "N"}]
    if not cluster_fe_atoms or not donor_atoms:
        return []
    donor_coords = np.stack([atom.coords for atom in donor_atoms], axis=0)
    donor_tree = KDTree(donor_coords)
    coordination: list[tuple[_AtomInfo, _AtomInfo]] = []
    linked_residues: set[tuple[int, int]] = set()
    for cluster_fe in cluster_fe_atoms:
        neighbor_ids = donor_tree.query_radius(
            cluster_fe.coords.reshape(1, -1),
            r=CCTBX_MCL_CLUSTER_CUTOFF,
            return_distance=False,
        )[0]
        local: list[tuple[float, _AtomInfo]] = []
        for donor_list_idx in neighbor_ids.tolist():
            donor = donor_atoms[donor_list_idx]
            if donor.residue_name.upper() in SF_CLUSTERS:
                continue
            distance = float(np.linalg.norm(cluster_fe.coords - donor.coords))
            if not np.isfinite(distance) or distance <= 0:
                continue
            local.append((distance, donor))
        local.sort(key=lambda item: item[0])
        for _, donor in local:
            if donor.residue_key in linked_residues:
                continue
            coordination.append((cluster_fe, donor))
            linked_residues.add(donor.residue_key)
    return coordination


def _get_sf_lookup(atom1: _AtomInfo, atom2: _AtomInfo, atom3: _AtomInfo | None = None) -> dict[tuple[str, ...], Any] | None:
    cluster_name = _get_sf_cluster_name(atom1, atom2, atom3)
    ligand_name = _get_sf_cluster_name(atom1, atom2, atom3, other=True)
    if cluster_name is None:
        return None
    cluster_lookup = SF_CLUSTER_COORDINATION.get(cluster_name)
    if cluster_lookup is None:
        return None
    ligand_lookup = cluster_lookup.get(ligand_name)
    if ligand_lookup is None:
        ligand_lookup = cluster_lookup.get("CYS")
    return ligand_lookup


def _get_sf_bond_params(cluster_atom: _AtomInfo, donor: _AtomInfo) -> tuple[float, float] | None:
    ligand_lookup = _get_sf_lookup(cluster_atom, donor)
    if ligand_lookup is None:
        return None
    return ligand_lookup.get((cluster_atom.element.upper(), donor.element.upper()))


def _get_sf_angle_params(atom1: _AtomInfo, atom2: _AtomInfo, atom3: _AtomInfo) -> tuple[float, float] | None:
    ligand_lookup = _get_sf_lookup(atom1, atom2, atom3)
    if ligand_lookup is None:
        return None
    key = (atom1.element.upper(), atom2.element.upper(), atom3.element.upper())
    value = ligand_lookup.get(key)
    if value is None:
        return None
    parent_pair = (atom1.residue_name.upper(), atom3.residue_name.upper())
    if parent_pair in ligand_lookup:
        value = ligand_lookup[ligand_lookup[parent_pair]]
    return value


def _get_sf_angle_atoms(
    cluster_atom: _AtomInfo,
    donor: _AtomInfo,
    coordination: list[tuple[_AtomInfo, _AtomInfo]],
    atom_by_identity: dict[tuple[tuple[int, int], str], _AtomInfo],
) -> list[_AtomInfo]:
    cluster_name = _get_sf_cluster_name(cluster_atom, donor)
    if cluster_name is None:
        return []
    atoms: list[_AtomInfo] = []
    if cluster_name == "F3S":
        try:
            atom_index = int(cluster_atom.atom_name.strip()[-1])
        except ValueError:
            atom_index = -1
        for sulfur_idx in range(1, 5):
            if sulfur_idx == F3S_NAMING.get(atom_index, -1):
                continue
            sulfur = atom_by_identity.get((cluster_atom.residue_key, f"S{sulfur_idx}"))
            if sulfur is not None:
                atoms.append(sulfur)
    elif cluster_name in {"SF4", "FES"}:
        try:
            atom_index = int(cluster_atom.atom_name.strip()[-1])
        except ValueError:
            atom_index = -1
        for sulfur_idx in range(1, 5):
            if sulfur_idx == atom_index:
                continue
            sulfur = atom_by_identity.get((cluster_atom.residue_key, f"S{sulfur_idx}"))
            if sulfur is not None:
                atoms.append(sulfur)
    if cluster_name == "FES":
        for other_cluster_atom, other_donor in coordination:
            if other_cluster_atom.atom_idx != cluster_atom.atom_idx:
                continue
            if other_donor.residue_key == donor.residue_key:
                continue
            sulfur = atom_by_identity.get((other_donor.residue_key, "SG"))
            if sulfur is not None:
                atoms.append(sulfur)
    return atoms


def _apply_mcl_sf_cluster_layer(
    atom_infos: list[_AtomInfo],
    atom_by_idx: dict[int, _AtomInfo],
    bonds: list[dict[str, Any]],
    angles: list[dict[str, Any]],
    accepted_pairs: set[tuple[int, int]],
    accepted_angles: set[tuple[int, int, int]],
) -> None:
    coordination = _collect_sf_cluster_coordination(atom_infos)
    if not coordination:
        return
    atom_by_identity = {(atom.residue_key, atom.atom_name.upper()): atom for atom in atom_infos}
    for cluster_atom, donor in coordination:
        params = _get_sf_bond_params(cluster_atom, donor)
        if params is None or _pair_key(cluster_atom.atom_idx, donor.atom_idx) in accepted_pairs:
            continue
        _add_bond(bonds, accepted_pairs, cluster_atom, donor, params[0], params[1])
    for cluster_atom, donor in coordination:
        for angle_atom in _get_sf_angle_atoms(cluster_atom, donor, coordination, atom_by_identity):
            params = _get_sf_angle_params(angle_atom, cluster_atom, donor)
            if params is None:
                continue
            _add_angle(angles, accepted_angles, angle_atom, cluster_atom, donor, params[0], params[1])


def build_default_metal_restraints(
    structure: StructureV2,
    explicit_pairs: list[tuple[int, int]] | None = None,
    options: AutoMetalRestraintOptions | None = None,
) -> dict[str, Any]:
    options = options or AutoMetalRestraintOptions()
    atom_infos = _build_atom_infos(structure)
    atom_by_idx = {atom.atom_idx: atom for atom in atom_infos}
    bond_graph = _build_bond_graph(structure)

    bonds: list[dict[str, Any]] = []
    angles: list[dict[str, Any]] = []
    accepted_pairs: set[tuple[int, int]] = set()
    accepted_angles: set[tuple[int, int, int]] = set()
    donor_link_counts: dict[int, int] = {}
    metal_class_counts: dict[tuple[int, tuple[str, str]], int] = {}
    selected_donors_by_metal: dict[int, set[int]] = {}

    _apply_explicit_pairs(
        explicit_pairs or [],
        atom_by_idx,
        options,
        bonds,
        accepted_pairs,
        donor_link_counts,
        metal_class_counts,
        selected_donors_by_metal,
    )

    if not options.enabled:
        return {"bonds": bonds, "angles": angles}

    _apply_generic_auto_linking(
        atom_infos,
        atom_by_idx,
        bond_graph,
        options,
        bonds,
        accepted_pairs,
        donor_link_counts,
        metal_class_counts,
        selected_donors_by_metal,
    )

    if options.ideal_distance_strategy == "library":
        _apply_mcl_zn_layer(atom_by_idx, atom_infos, bonds, angles, accepted_pairs, accepted_angles)
        _apply_mcl_sf_cluster_layer(atom_infos, atom_by_idx, bonds, angles, accepted_pairs, accepted_angles)

    return {"bonds": bonds, "angles": angles}
