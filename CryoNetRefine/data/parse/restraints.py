from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from CryoNetRefine.data.types import StructureV2

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


@dataclass(frozen=True)
class AtomSelector:
    chain: str | None = None
    auth_asym_id: str | None = None
    resseq: str | int | None = None
    auth_seq_id: str | None = None
    icode: str | None = None
    ins_code: str | None = None
    resname: str | None = None
    auth_comp_id: str | None = None
    atom_name: str | None = None
    altloc: str | None = None


@dataclass(frozen=True)
class BondRestraintSpec:
    atom1: AtomSelector
    atom2: AtomSelector
    distance_ideal: float
    sigma: float | None = None
    weight: float | None = None
    slack: float = 0.0


@dataclass(frozen=True)
class AngleRestraintSpec:
    atom1: AtomSelector
    atom2: AtomSelector
    atom3: AtomSelector
    angle_ideal_deg: float
    sigma: float | None = None
    weight: float | None = None
    slack_deg: float = 0.0


@dataclass(frozen=True)
class UserRestraintsSpec:
    bonds: tuple[BondRestraintSpec, ...] = ()
    angles: tuple[AngleRestraintSpec, ...] = ()


@dataclass(frozen=True)
class ResolvedBondRestraint:
    atom_idx1: int
    atom_idx2: int
    distance_ideal: float
    sigma: float | None
    weight: float
    slack: float


@dataclass(frozen=True)
class ResolvedAngleRestraint:
    atom_idx1: int
    atom_idx2: int
    atom_idx3: int
    angle_ideal_deg: float
    sigma: float | None
    weight: float
    slack_deg: float


@dataclass(frozen=True)
class ResolvedUserRestraints:
    bonds: tuple[ResolvedBondRestraint, ...] = ()
    angles: tuple[ResolvedAngleRestraint, ...] = ()
    atom_lookup: dict[int, str] | None = None


def _selector_identity(selector: AtomSelector) -> tuple[str | None, ...]:
    resseq = selector.auth_seq_id if selector.auth_seq_id is not None else selector.resseq
    ins_code = selector.ins_code if selector.ins_code is not None else selector.icode
    resname = selector.auth_comp_id if selector.auth_comp_id is not None else selector.resname
    return (
        _normalize_string(selector.chain),
        _normalize_string(selector.auth_asym_id),
        None if resseq is None else str(resseq).strip(),
        _normalize_string(ins_code),
        _normalize_string(resname),
        _normalize_string(selector.atom_name),
        _normalize_string(selector.altloc),
    )


def _bond_identity(bond: BondRestraintSpec) -> tuple[tuple[str | None, ...], tuple[str | None, ...]]:
    atom_keys = sorted((_selector_identity(bond.atom1), _selector_identity(bond.atom2)))
    return (atom_keys[0], atom_keys[1])


def _angle_identity(
    angle: AngleRestraintSpec,
) -> tuple[tuple[str | None, ...], tuple[tuple[str | None, ...], tuple[str | None, ...]]]:
    center = _selector_identity(angle.atom2)
    outer = sorted((_selector_identity(angle.atom1), _selector_identity(angle.atom3)))
    return (center, (outer[0], outer[1]))


def merge_user_restraints_specs(
    first: UserRestraintsSpec | None,
    second: UserRestraintsSpec | None,
) -> UserRestraintsSpec | None:
    if first is None and second is None:
        return None
    if first is None:
        return second
    if second is None:
        return first
    merged_bonds: dict[tuple[tuple[str | None, ...], tuple[str | None, ...]], BondRestraintSpec] = {
        _bond_identity(bond): bond for bond in first.bonds
    }
    for bond in second.bonds:
        merged_bonds[_bond_identity(bond)] = bond
    merged_angles: dict[
        tuple[tuple[str | None, ...], tuple[tuple[str | None, ...], tuple[str | None, ...]]],
        AngleRestraintSpec,
    ] = {_angle_identity(angle): angle for angle in first.angles}
    for angle in second.angles:
        merged_angles[_angle_identity(angle)] = angle
    return UserRestraintsSpec(
        bonds=tuple(merged_bonds.values()),
        angles=tuple(merged_angles.values()),
    )


@dataclass(frozen=True)
class _AtomRecord:
    atom_idx: int
    chain_name: str
    auth_asym_id: str
    resseq: str
    ins_code: str
    resname: str
    auth_comp_id: str
    atom_name: str
    altloc: str | None = None

    def describe(self) -> str:
        ins = self.ins_code or "."
        return (
            f"chain={self.chain_name!r} auth_asym_id={self.auth_asym_id!r} "
            f"resseq={self.resseq!r} ins_code={ins!r} resname={self.resname!r} "
            f"atom_name={self.atom_name!r} atom_idx={self.atom_idx}"
        )


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_selector(data: dict[str, Any], context: str) -> AtomSelector:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping.")
    selector = AtomSelector(
        chain=_normalize_string(data.get("chain")),
        auth_asym_id=_normalize_string(data.get("auth_asym_id")),
        resseq=data.get("resseq"),
        auth_seq_id=_normalize_string(data.get("auth_seq_id")),
        icode=_normalize_string(data.get("icode")),
        ins_code=_normalize_string(data.get("ins_code")),
        resname=_normalize_string(data.get("resname")),
        auth_comp_id=_normalize_string(data.get("auth_comp_id")),
        atom_name=_normalize_string(data.get("atom_name")),
        altloc=_normalize_string(data.get("altloc")),
    )
    if selector.atom_name is None:
        raise ValueError(f"{context}.atom_name is required.")
    return selector


def _resolve_weight_sigma(
    weight: Any,
    sigma: Any,
    context: str,
) -> tuple[float, float | None]:
    parsed_weight = None if weight is None else float(weight)
    parsed_sigma = None if sigma is None else float(sigma)
    if parsed_weight is not None and parsed_weight <= 0:
        raise ValueError(f"{context}.weight must be > 0.")
    if parsed_sigma is not None and parsed_sigma <= 0:
        raise ValueError(f"{context}.sigma must be > 0.")
    if parsed_weight is None and parsed_sigma is None:
        raise ValueError(f"{context} requires either weight or sigma.")
    if parsed_weight is None:
        parsed_weight = 1.0 / (parsed_sigma * parsed_sigma)
    return parsed_weight, parsed_sigma


def _parse_bond(entry: dict[str, Any], idx: int) -> BondRestraintSpec:
    context = f"bonds[{idx}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{context} must be a mapping.")
    distance_ideal = float(entry["distance_ideal"])
    if distance_ideal <= 0:
        raise ValueError(f"{context}.distance_ideal must be > 0.")
    weight, sigma = _resolve_weight_sigma(entry.get("weight"), entry.get("sigma"), context)
    slack = float(entry.get("slack", 0.0))
    if slack < 0:
        raise ValueError(f"{context}.slack must be >= 0.")
    return BondRestraintSpec(
        atom1=_parse_selector(entry.get("atom1"), f"{context}.atom1"),
        atom2=_parse_selector(entry.get("atom2"), f"{context}.atom2"),
        distance_ideal=distance_ideal,
        sigma=sigma,
        weight=weight,
        slack=slack,
    )


def _parse_angle(entry: dict[str, Any], idx: int) -> AngleRestraintSpec:
    context = f"angles[{idx}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{context} must be a mapping.")
    angle_ideal_deg = float(entry["angle_ideal_deg"])
    weight, sigma = _resolve_weight_sigma(entry.get("weight"), entry.get("sigma"), context)
    slack_deg = float(entry.get("slack_deg", 0.0))
    if slack_deg < 0:
        raise ValueError(f"{context}.slack_deg must be >= 0.")
    return AngleRestraintSpec(
        atom1=_parse_selector(entry.get("atom1"), f"{context}.atom1"),
        atom2=_parse_selector(entry.get("atom2"), f"{context}.atom2"),
        atom3=_parse_selector(entry.get("atom3"), f"{context}.atom3"),
        angle_ideal_deg=angle_ideal_deg,
        sigma=sigma,
        weight=weight,
        slack_deg=slack_deg,
    )


def parse_user_restraints_dict(raw: dict[str, Any] | None) -> UserRestraintsSpec:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Restraints file root must be a mapping.")
    bonds = tuple(_parse_bond(entry, i) for i, entry in enumerate(raw.get("bonds", [])))
    angles = tuple(_parse_angle(entry, i) for i, entry in enumerate(raw.get("angles", [])))
    return UserRestraintsSpec(bonds=bonds, angles=angles)


def load_user_restraints(path: str | Path | None) -> UserRestraintsSpec | None:
    if path is None:
        return None
    restraints_path = Path(path)
    if not restraints_path.exists():
        raise FileNotFoundError(f"Restraints file not found: {restraints_path}")
    suffix = restraints_path.suffix.lower()
    text = restraints_path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML restraints files.")
        raw = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported restraints file format {suffix!r}. Use .json, .yaml, or .yml."
        )
    return parse_user_restraints_dict(raw)


def _build_atom_records(structure: StructureV2) -> list[_AtomRecord]:
    records: list[_AtomRecord] = []
    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains
    for chain in chains:
        chain_name = str(chain["name"]).strip()
        auth_asym_id = str(chain["auth_asym_id"]).strip()
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for residue in residues[res_start:res_end]:
            atom_start = int(residue["atom_idx"])
            atom_end = atom_start + int(residue["atom_num"])
            resseq = str(residue["auth_seq_id"]).strip()
            ins_code = str(residue["ins_code"]).strip()
            resname = str(residue["name"]).strip()
            auth_comp_id = str(residue["auth_comp_id"]).strip()
            for atom_idx in range(atom_start, atom_end):
                atom = atoms[atom_idx]
                records.append(
                    _AtomRecord(
                        atom_idx=atom_idx,
                        chain_name=chain_name,
                        auth_asym_id=auth_asym_id,
                        resseq=resseq,
                        ins_code=ins_code,
                        resname=resname,
                        auth_comp_id=auth_comp_id,
                        atom_name=str(atom["name"]).strip(),
                    )
                )
    return records


def _selector_matches(selector: AtomSelector, atom: _AtomRecord) -> bool:
    if selector.altloc not in (None, "", "."):
        raise ValueError("altloc matching is not supported by the current CryoNet.Refine structure tables.")
    chain = _normalize_string(selector.chain)
    if chain is not None and chain != atom.chain_name:
        return False
    auth_asym_id = _normalize_string(selector.auth_asym_id)
    if auth_asym_id is not None and auth_asym_id != atom.auth_asym_id:
        return False
    atom_name = _normalize_string(selector.atom_name)
    if atom_name is not None and atom_name != atom.atom_name:
        return False
    resseq = selector.auth_seq_id if selector.auth_seq_id is not None else selector.resseq
    if resseq is not None and str(resseq).strip() != atom.resseq:
        return False
    ins_code = selector.ins_code if selector.ins_code is not None else selector.icode
    if _normalize_string(ins_code) is not None and _normalize_string(ins_code) != atom.ins_code:
        return False
    resname = selector.auth_comp_id if selector.auth_comp_id is not None else selector.resname
    if _normalize_string(resname) is not None:
        if _normalize_string(resname) not in {atom.resname, atom.auth_comp_id}:
            return False
    return True


def _resolve_selector(
    selector: AtomSelector,
    atom_records: list[_AtomRecord],
    context: str,
) -> _AtomRecord:
    matches = [record for record in atom_records if _selector_matches(selector, record)]
    if len(matches) == 0:
        raise ValueError(f"{context} did not match any atom: {selector}")
    if len(matches) > 1:
        preview = "; ".join(match.describe() for match in matches[:5])
        raise ValueError(f"{context} matched multiple atoms: {preview}")
    return matches[0]


def resolve_user_restraints(
    spec: UserRestraintsSpec | None,
    structure: StructureV2,
) -> ResolvedUserRestraints | None:
    if spec is None:
        return None
    atom_records = _build_atom_records(structure)
    atom_lookup = {record.atom_idx: record.describe() for record in atom_records}
    resolved_bonds: list[ResolvedBondRestraint] = []
    for idx, bond in enumerate(spec.bonds):
        atom1 = _resolve_selector(bond.atom1, atom_records, f"bonds[{idx}].atom1")
        atom2 = _resolve_selector(bond.atom2, atom_records, f"bonds[{idx}].atom2")
        resolved_bonds.append(
            ResolvedBondRestraint(
                atom_idx1=atom1.atom_idx,
                atom_idx2=atom2.atom_idx,
                distance_ideal=float(bond.distance_ideal),
                sigma=bond.sigma,
                weight=float(bond.weight if bond.weight is not None else 0.0),
                slack=float(bond.slack),
            )
        )
    resolved_angles: list[ResolvedAngleRestraint] = []
    for idx, angle in enumerate(spec.angles):
        atom1 = _resolve_selector(angle.atom1, atom_records, f"angles[{idx}].atom1")
        atom2 = _resolve_selector(angle.atom2, atom_records, f"angles[{idx}].atom2")
        atom3 = _resolve_selector(angle.atom3, atom_records, f"angles[{idx}].atom3")
        if len({atom1.atom_idx, atom2.atom_idx, atom3.atom_idx}) != 3:
            raise ValueError(f"angles[{idx}] must reference three distinct atoms.")
        resolved_angles.append(
            ResolvedAngleRestraint(
                atom_idx1=atom1.atom_idx,
                atom_idx2=atom2.atom_idx,
                atom_idx3=atom3.atom_idx,
                angle_ideal_deg=float(angle.angle_ideal_deg),
                sigma=angle.sigma,
                weight=float(angle.weight if angle.weight is not None else 0.0),
                slack_deg=float(angle.slack_deg),
            )
        )
    return ResolvedUserRestraints(
        bonds=tuple(resolved_bonds),
        angles=tuple(resolved_angles),
        atom_lookup=atom_lookup,
    )
