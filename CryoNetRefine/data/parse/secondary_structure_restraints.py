from __future__ import annotations

"""Convert CryoNet.Refine secondary-structure detections into restraints."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from CryoNetRefine.secondary_structure.constants import NA_ONE_LETTER, RING_ATOMS
from CryoNetRefine.secondary_structure.detector import detect_secondary_structure
from CryoNetRefine.secondary_structure.io import iter_chain_residues, read_structure
from CryoNetRefine.secondary_structure.protein import (
    BETA,
    ProteinSegment,
    align_strands,
    get_ind_h_bond_sheet,
)


PROTEIN_HBOND_DISTANCE_IDEAL = 2.9
PROTEIN_HBOND_DISTANCE_CUT = 3.5
PROTEIN_HBOND_SIGMA = 0.05
NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD = 0.0335
NUCLEIC_STACKING_PARALLELITY_SIGMA_RAD = 0.027
NA_HBOND_ANCHORS = {
    "N1": "C2",
    "N2": "C2",
    "N3": "C2",
    "O2": "C2",
    "N4": "C4",
    "O4": "C4",
    "N6": "C6",
    "O6": "C6",
}


def _selector(resid: Any, atom_name: str) -> dict[str, Any]:
    return {
        "auth_asym_id": resid.chain,
        "auth_seq_id": str(resid.resseq),
        "ins_code": resid.icode,
        "auth_comp_id": resid.name,
        "atom_name": atom_name,
        "chain_id": resid.chain,
        "resid": resid.resid(),
        "selection": f"{resid.selection()} and name {atom_name}",
    }


def _range_selection(start: Any, end: Any) -> str:
    if start.chain == end.chain:
        return f"chain '{start.chain}' and resid {start.resid()} through {end.resid()}"
    return f"{start.selection()} or {end.selection()}"


def _bond_key(resid1: Any, atom1: str, resid2: Any, atom2: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    atoms = sorted(
        (
            (resid1.chain, resid1.resid(), atom1),
            (resid2.chain, resid2.resid(), atom2),
        )
    )
    return atoms[0], atoms[1]


def _angle_key(
    resid1: Any,
    atom1: str,
    resid2: Any,
    atom2: str,
    resid3: Any,
    atom3: str,
) -> tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]:
    center = (resid2.chain, resid2.resid(), atom2)
    outer = sorted(((resid1.chain, resid1.resid(), atom1), (resid3.chain, resid3.resid(), atom3)))
    return center, (outer[0], outer[1])


def _add_bond(
    payload: dict[str, Any],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    resid1: Any,
    atom1: str,
    resid2: Any,
    atom2: str,
    distance_ideal: float,
    sigma: float,
    source: str,
    ss_kind: str,
    parent: dict[str, Any] | None = None,
    slack: float = 0.0,
) -> None:
    key = _bond_key(resid1, atom1, resid2, atom2)
    if key in accepted_bonds:
        return
    entry = {
        "atom1": _selector(resid1, atom1),
        "atom2": _selector(resid2, atom2),
        "distance_ideal": float(distance_ideal),
        "sigma": float(sigma),
        "slack": float(slack),
        "restraint_source": source,
        "secondary_structure_type": ss_kind,
    }
    if parent:
        entry["secondary_structure"] = parent
    payload["bonds"].append(entry)
    accepted_bonds.add(key)


def _add_angle(
    payload: dict[str, Any],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
    resid1: Any,
    atom1: str,
    resid2: Any,
    atom2: str,
    resid3: Any,
    atom3: str,
    angle_ideal_deg: float,
    sigma: float,
    source: str,
    ss_kind: str,
    parent: dict[str, Any] | None = None,
) -> None:
    key = _angle_key(resid1, atom1, resid2, atom2, resid3, atom3)
    if key in accepted_angles:
        return
    entry = {
        "atom1": _selector(resid1, atom1),
        "atom2": _selector(resid2, atom2),
        "atom3": _selector(resid3, atom3),
        "angle_ideal_deg": float(angle_ideal_deg),
        "sigma": float(sigma),
        "restraint_source": source,
        "secondary_structure_type": ss_kind,
    }
    if parent:
        entry["secondary_structure"] = parent
    payload["angles"].append(entry)
    accepted_angles.add(key)


def _residue_sequences(st: Any) -> tuple[dict[tuple[int, str, int, str], Any], dict[tuple[int, str], list[Any]]]:
    lookup = {}
    by_chain: dict[tuple[int, str], list[Any]] = {}
    for model_index, chain_name, residues in iter_chain_residues(st):
        by_chain[(model_index, chain_name)] = residues
        for residue in residues:
            lookup[residue.resid.key()] = residue
    return lookup, by_chain


def _range_residues(by_chain: dict[tuple[int, str], list[Any]], start: Any, end: Any) -> list[Any]:
    if start.model != end.model or start.chain != end.chain:
        return []
    residues = by_chain.get((start.model, start.chain), [])
    start_idx = next((i for i, residue in enumerate(residues) if residue.resid.key() == start.key()), None)
    end_idx = next((i for i, residue in enumerate(residues) if residue.resid.key() == end.key()), None)
    if start_idx is None or end_idx is None:
        return []
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return residues[start_idx : end_idx + 1]


def _previous_residue(by_chain: dict[tuple[int, str], list[Any]], residue: Any) -> Any | None:
    residues = by_chain.get((residue.resid.model, residue.resid.chain), [])
    idx = next((i for i, item in enumerate(residues) if item.resid.key() == residue.resid.key()), None)
    if idx is None or idx == 0:
        return None
    return residues[idx - 1]


def _distance(atom1: Any, atom2: Any) -> float:
    return float(np.linalg.norm(np.asarray(atom1.xyz, dtype=float) - np.asarray(atom2.xyz, dtype=float)))


def _add_protein_hbond(
    payload: dict[str, Any],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
    by_chain: dict[tuple[int, str], list[Any]],
    acceptor_residue: Any,
    donor_residue: Any,
    ss_kind: str,
    parent: dict[str, Any],
    angle_params: tuple[float, float, float, float, float, float] | None,
) -> bool:
    if donor_residue.resid.name == "PRO":
        return False
    atom_o = acceptor_residue.atom("O")
    atom_n = donor_residue.atom("N")
    if atom_o is None or atom_n is None:
        return False
    if _distance(atom_o, atom_n) > PROTEIN_HBOND_DISTANCE_CUT:
        return False

    _add_bond(
        payload,
        accepted_bonds,
        acceptor_residue.resid,
        "O",
        donor_residue.resid,
        "N",
        PROTEIN_HBOND_DISTANCE_IDEAL,
        PROTEIN_HBOND_SIGMA,
        "secondary_structure",
        ss_kind,
        parent,
    )

    if angle_params is None:
        return True
    co_ideal, co_sigma, ca_ideal, ca_sigma, cprev_ideal, cprev_sigma = angle_params
    if acceptor_residue.atom("C") is not None:
        _add_angle(
            payload,
            accepted_angles,
            acceptor_residue.resid,
            "C",
            acceptor_residue.resid,
            "O",
            donor_residue.resid,
            "N",
            co_ideal,
            co_sigma,
            "secondary_structure",
            ss_kind,
            parent,
        )
    if donor_residue.atom("CA") is not None:
        _add_angle(
            payload,
            accepted_angles,
            donor_residue.resid,
            "CA",
            donor_residue.resid,
            "N",
            acceptor_residue.resid,
            "O",
            ca_ideal,
            ca_sigma,
            "secondary_structure",
            ss_kind,
            parent,
        )
    previous = _previous_residue(by_chain, donor_residue)
    if previous is not None and previous.atom("C") is not None:
        _add_angle(
            payload,
            accepted_angles,
            previous.resid,
            "C",
            donor_residue.resid,
            "N",
            acceptor_residue.resid,
            "O",
            cprev_ideal,
            cprev_sigma,
            "secondary_structure",
            ss_kind,
            parent,
        )
    return True


def _add_helix_restraints(
    payload: dict[str, Any],
    result: Any,
    by_chain: dict[tuple[int, str], list[Any]],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
) -> None:
    step_by_type = {"alpha": 4, "pi": 5, "3_10": 3}
    for helix in result.helices:
        residues = _range_residues(by_chain, helix.start, helix.end)
        step = step_by_type.get(helix.helix_type)
        if not residues or step is None:
            continue
        parent = {
            "kind": "helix",
            "type": helix.helix_type,
            "selection": _range_selection(helix.start, helix.end),
            "source": helix.source,
        }
        hbond_count = max(0, len(residues) - step)
        for idx in range(hbond_count):
            angle_params = None
            if helix.helix_type == "alpha":
                if idx == 0 or idx == hbond_count - 1:
                    angle_params = (155.0, 10.0, 116.0, 10.0, 121.0, 10.0)
                else:
                    angle_params = (155.0, 5.0, 116.0, 5.0, 121.0, 5.0)
            _add_protein_hbond(
                payload,
                accepted_bonds,
                accepted_angles,
                by_chain,
                residues[idx],
                residues[idx + step],
                "protein_helix",
                parent,
                angle_params,
            )


def _sheet_segment(strand: Any, by_chain: dict[tuple[int, str], list[Any]]) -> Any | None:
    residues = _range_residues(by_chain, strand.start, strand.end)
    if len(residues) < 2:
        return None
    return ProteinSegment(residues=residues, params=BETA)


def _add_sheet_pair_restraints(
    payload: dict[str, Any],
    by_chain: dict[tuple[int, str], list[Any]],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
    prev_segment: Any,
    cur_segment: Any,
    parent: dict[str, Any],
) -> None:
    rel = align_strands(prev_segment, cur_segment, tol=6.0, min_len=min(4, prev_segment.length(), cur_segment.length()))
    if rel is None:
        return
    first1, last1, first2, last2, parallel = rel
    i_index, j_index = get_ind_h_bond_sheet(prev_segment, cur_segment, rel)
    if i_index is None or j_index is None:
        return
    beta_angles = (155.0, 9.0, 124.0, 7.0, 113.0, 6.0)
    for i in range(prev_segment.length()):
        if (i - i_index) % 2 != 0:
            continue
        li = i_index + (i - i_index)
        lj = j_index + (i - i_index if parallel else -(i - i_index))
        if not (0 <= li < prev_segment.length()):
            continue
        if li < first1 or li > last1 or lj < first2 or lj > last2:
            continue
        if 0 <= lj < cur_segment.length():
            _add_protein_hbond(
                payload,
                accepted_bonds,
                accepted_angles,
                by_chain,
                prev_segment.residues[li],
                cur_segment.residues[lj],
                "protein_sheet",
                parent,
                beta_angles,
            )
        lj2 = lj - 2 if parallel else lj
        if 0 <= lj2 < cur_segment.length():
            _add_protein_hbond(
                payload,
                accepted_bonds,
                accepted_angles,
                by_chain,
                cur_segment.residues[lj2],
                prev_segment.residues[li],
                "protein_sheet",
                parent,
                beta_angles,
            )


def _add_sheet_restraints(
    payload: dict[str, Any],
    result: Any,
    by_chain: dict[tuple[int, str], list[Any]],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
) -> None:
    for sheet in result.sheets:
        segments = [_sheet_segment(strand, by_chain) for strand in sheet.strands]
        parent = {
            "kind": "sheet",
            "sheet_id": sheet.sheet_id,
            "source": sheet.source,
            "strands": [
                {
                    "selection": _range_selection(strand.start, strand.end),
                    "sense": strand.sense,
                    "start_chain": strand.start.chain,
                    "start_resid": strand.start.resid(),
                    "end_chain": strand.end.chain,
                    "end_resid": strand.end.resid(),
                }
                for strand in sheet.strands
            ],
        }
        for prev_segment, cur_segment in zip(segments[:-1], segments[1:]):
            if prev_segment is None or cur_segment is None:
                continue
            _add_sheet_pair_restraints(
                payload,
                by_chain,
                accepted_bonds,
                accepted_angles,
                prev_segment,
                cur_segment,
                parent,
            )


def _base_letter(resid: Any) -> str | None:
    base = NA_ONE_LETTER.get(resid.name)
    return "U" if base == "T" else base


def _ordered_base_residues(
    residue1: Any,
    residue2: Any,
) -> tuple[Any, Any, str | None, str | None]:
    base1 = _base_letter(residue1.resid)
    base2 = _base_letter(residue2.resid)
    if base1 is not None and base2 is not None and base1 > base2:
        return residue2, residue1, base2, base1
    return residue1, residue2, base1, base2


def _na_hbond_target(atom1: str, atom2: str, base1: str | None, base2: str | None) -> tuple[float, float]:
    atoms = frozenset({atom1, atom2})
    if atoms == frozenset({"N6", "O4"}):
        return 3.00, 0.11
    if atoms == frozenset({"O6", "N4"}):
        return 2.93, 0.10
    if atoms == frozenset({"N2", "O2"}):
        return 2.78, 0.10
    if atoms == frozenset({"N1", "N3"}):
        bases = frozenset({base1, base2})
        if bases == frozenset({"G", "C"}):
            return 2.88, 0.07
        if bases == frozenset({"A", "U"}):
            return 2.82, 0.08
    return 2.91, 0.15


def _na_hbond_angle_params(atom1: str, atom2: str, base1: str | None, base2: str | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    exact = {
        ("O6", "N4"): ((122.8, 3.00), (117.3, 2.86)),
        ("N4", "O6"): ((117.3, 2.86), (122.8, 3.00)),
        ("N2", "O2"): ((122.2, 2.88), (120.7, 2.20)),
        ("O2", "N2"): ((120.7, 2.20), (122.2, 2.88)),
        ("N6", "O4"): ((115.6, 8.34), (121.2, 4.22)),
        ("O4", "N6"): ((121.2, 4.22), (115.6, 8.34)),
    }
    if (atom1, atom2) in exact:
        return exact[(atom1, atom2)]
    if frozenset({atom1, atom2}) == frozenset({"N1", "N3"}):
        bases = frozenset({base1, base2})
        if bases == frozenset({"G", "C"}):
            if atom1 == "N1":
                return ((119.1, 2.59), (116.3, 2.66))
            return ((116.3, 2.66), (119.1, 2.59))
        if bases == frozenset({"A", "U"}):
            if atom1 == "N1":
                return ((116.2, 3.46), (115.8, 2.88))
            return ((115.8, 2.88), (116.2, 3.46))
    return None


def _ring_selectors(residue: Any) -> list[dict[str, Any]]:
    selectors = []
    for atom_name in RING_ATOMS:
        if residue.atom(atom_name) is not None:
            selectors.append(_selector(residue.resid, atom_name))
    return selectors


def _add_plane_parallelity(
    payload: dict[str, Any],
    accepted_planes: set[tuple[tuple[str, str], tuple[str, str], str]],
    residue1: Any,
    residue2: Any,
    sigma_rad: float,
    ss_kind: str,
    parent: dict[str, Any],
) -> None:
    plane1 = _ring_selectors(residue1)
    plane2 = _ring_selectors(residue2)
    if len(plane1) < 3 or len(plane2) < 3:
        return
    key_items = sorted(((residue1.resid.chain, residue1.resid.resid()), (residue2.resid.chain, residue2.resid.resid())))
    key = (key_items[0], key_items[1], ss_kind)
    if key in accepted_planes:
        return
    payload["plane_parallelities"].append(
        {
            "plane1": plane1,
            "plane2": plane2,
            "angle_ideal_deg": 0.0,
            "sigma": float(math.degrees(sigma_rad)),
            "weight": float(1.0 / (sigma_rad * sigma_rad)),
            "restraint_source": "secondary_structure",
            "secondary_structure_type": ss_kind,
            "secondary_structure": parent,
        }
    )
    accepted_planes.add(key)


def _add_nucleic_restraints(
    payload: dict[str, Any],
    result: Any,
    lookup: dict[tuple[int, str, int, str], Any],
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]],
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ],
    accepted_planes: set[tuple[tuple[str, str], tuple[str, str], str]],
) -> None:
    for bp in result.base_pairs:
        residue1 = lookup.get(bp.base1.key())
        residue2 = lookup.get(bp.base2.key())
        if residue1 is None or residue2 is None:
            continue
        # The detector's classify_basepair() stores hydrogen bonds in the
        # internally ordered base order. Preserve that order when converting atom
        # names to selectors, otherwise A/G/C/U pairs whose order was swapped by
        # the classifier get atom names applied to the wrong residue.
        hbond_residue1, hbond_residue2, base1, base2 = _ordered_base_residues(residue1, residue2)
        parent = {
            "kind": "base_pair",
            "saenger_class": bp.saenger_class,
            "base1_chain": bp.base1.chain,
            "base1_resid": bp.base1.resid(),
            "base2_chain": bp.base2.chain,
            "base2_resid": bp.base2.resid(),
            "source": bp.source,
        }
        for atom1, atom2, _distance_model in bp.hbonds:
            if hbond_residue1.atom(atom1) is None or hbond_residue2.atom(atom2) is None:
                continue
            target, sigma = _na_hbond_target(atom1, atom2, base1, base2)
            _add_bond(
                payload,
                accepted_bonds,
                hbond_residue1.resid,
                atom1,
                hbond_residue2.resid,
                atom2,
                target,
                sigma,
                "secondary_structure",
                "nucleic_base_pair",
                parent,
            )
            angle_params = _na_hbond_angle_params(atom1, atom2, base1, base2)
            anchor1 = NA_HBOND_ANCHORS.get(atom1)
            anchor2 = NA_HBOND_ANCHORS.get(atom2)
            if angle_params is not None and anchor1 is not None and anchor2 is not None:
                if hbond_residue1.atom(anchor1) is not None:
                    _add_angle(
                        payload,
                        accepted_angles,
                        hbond_residue1.resid,
                        anchor1,
                        hbond_residue1.resid,
                        atom1,
                        hbond_residue2.resid,
                        atom2,
                        angle_params[0][0],
                        angle_params[0][1],
                        "secondary_structure",
                        "nucleic_base_pair",
                        parent,
                    )
                if hbond_residue2.atom(anchor2) is not None:
                    _add_angle(
                        payload,
                        accepted_angles,
                        hbond_residue2.resid,
                        anchor2,
                        hbond_residue2.resid,
                        atom2,
                        hbond_residue1.resid,
                        atom1,
                        angle_params[1][0],
                        angle_params[1][1],
                        "secondary_structure",
                        "nucleic_base_pair",
                        parent,
                    )
        _add_plane_parallelity(
            payload,
            accepted_planes,
            residue1,
            residue2,
            NUCLEIC_BASEPAIR_PARALLELITY_SIGMA_RAD,
            "nucleic_base_pair",
            parent,
        )

    for stacking in result.stacking_pairs:
        residue1 = lookup.get(stacking.base1.key())
        residue2 = lookup.get(stacking.base2.key())
        if residue1 is None or residue2 is None:
            continue
        parent = {
            "kind": "stacking_pair",
            "base1_chain": stacking.base1.chain,
            "base1_resid": stacking.base1.resid(),
            "base2_chain": stacking.base2.chain,
            "base2_resid": stacking.base2.resid(),
            "distance": stacking.distance,
            "normal_angle": stacking.normal_angle,
            "source": stacking.source,
        }
        _add_plane_parallelity(
            payload,
            accepted_planes,
            residue1,
            residue2,
            NUCLEIC_STACKING_PARALLELITY_SIGMA_RAD,
            "nucleic_stacking_pair",
            parent,
        )


def build_default_secondary_structure_restraints(
    path: str | Path,
    protein_enabled: bool = False,
    nucleic_enabled: bool = False,
    mode: str = "auto",
    include_single_strands: bool = False,
) -> dict[str, Any]:
    if not protein_enabled and not nucleic_enabled:
        return {"bonds": [], "angles": [], "plane_parallelities": []}

    st = read_structure(path)
    result = detect_secondary_structure(
        path,
        mode=mode,
        detect_protein=protein_enabled,
        detect_nucleic=nucleic_enabled,
        include_single_strands=include_single_strands,
    )
    lookup, by_chain = _residue_sequences(st)

    payload: dict[str, Any] = {
        "bonds": [],
        "angles": [],
        "plane_parallelities": [],
        "secondary_structure": result.to_dict(),
        "secondary_structure_restraints": {
            "enabled": True,
            "protein_enabled": protein_enabled,
            "nucleic_enabled": nucleic_enabled,
            "mode": mode,
            "used_existing_protein": result.used_existing_protein,
        },
    }
    accepted_bonds: set[tuple[tuple[str, str, str], tuple[str, str, str]]] = set()
    accepted_angles: set[
        tuple[tuple[str, str, str], tuple[tuple[str, str, str], tuple[str, str, str]]]
    ] = set()
    accepted_planes: set[tuple[tuple[str, str], tuple[str, str], str]] = set()

    if protein_enabled:
        _add_helix_restraints(payload, result, by_chain, accepted_bonds, accepted_angles)
        _add_sheet_restraints(payload, result, by_chain, accepted_bonds, accepted_angles)
    if nucleic_enabled:
        _add_nucleic_restraints(payload, result, lookup, accepted_bonds, accepted_angles, accepted_planes)

    payload["secondary_structure_restraints"].update(
        {
            "bond_count": len(payload["bonds"]),
            "angle_count": len(payload["angles"]),
            "plane_parallelity_count": len(payload["plane_parallelities"]),
        }
    )
    return payload


def merge_restraint_payloads(*payloads: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {"bonds": [], "angles": [], "plane_parallelities": []}
    for payload in payloads:
        if not payload:
            continue
        merged["bonds"].extend(payload.get("bonds", []))
        merged["angles"].extend(payload.get("angles", []))
        merged["plane_parallelities"].extend(payload.get("plane_parallelities", []))
        for key, value in payload.items():
            if key not in {"bonds", "angles", "plane_parallelities"}:
                merged[key] = value
    return merged
