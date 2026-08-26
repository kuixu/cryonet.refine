from __future__ import annotations

import itertools
import math

import numpy as np

from .constants import BASEPAIR_LENGTHS, NA_ONE_LETTER, RING_ATOMS
from .geometry import angle_between_abs, angle_degrees_from_cos, distance, norm, unit, vec
from .io import na_segments
from .models import BasePairRecord, ResidueInfo, StackingPairRecord


def one_letter_base(residue: ResidueInfo) -> str | None:
    base = NA_ONE_LETTER.get(residue.resid.name)
    if base == "T":
        return "U"
    return base


def ordered_bases(r1: ResidueInfo, r2: ResidueInfo) -> tuple[ResidueInfo, ResidueInfo, str, str] | None:
    b1 = one_letter_base(r1)
    b2 = one_letter_base(r2)
    if b1 is None or b2 is None:
        return None
    if b1 > b2:
        return r2, r1, b2, b1
    return r1, r2, b1, b2


def is_consecutive(r1: ResidueInfo, r2: ResidueInfo) -> bool:
    return r1.resid.chain == r2.resid.chain and abs(r1.resid.resseq - r2.resid.resseq) < 2


def base_contact_atoms(residue: ResidueInfo):
    for atom in residue.atoms.values():
        if atom.element not in ("N", "O"):
            continue
        if "P" in atom.name or "'" in atom.name or "*" in atom.name:
            continue
        yield atom


def final_link_direction_check(atom1, residue1: ResidueInfo, atom2, cutoff: float = 35.0) -> bool:
    a1 = residue1.atom("C4")
    a2 = residue1.atom("C5")
    a3 = residue1.atom("C6")
    if a1 is None or a2 is None or a3 is None:
        return False
    v1 = vec(a1.xyz) - vec(a2.xyz)
    v2 = vec(a2.xyz) - vec(a3.xyz)
    normal = np.cross(v1, v2)
    link = vec(atom2.xyz) - vec(atom1.xyz)
    normal_u = unit(normal)
    link_u = unit(link)
    if normal_u is None or link_u is None:
        return False
    angle_from_plane = 90.0 - angle_degrees_from_cos(abs(float(np.dot(normal_u, link_u))))
    return angle_from_plane < cutoff


def classify_basepair(r1_in: ResidueInfo, r2_in: ResidueInfo, cutoff: float = 3.4) -> tuple[int | None, list[tuple[str, str, float]]]:
    ordered = ordered_bases(r1_in, r2_in)
    if ordered is None:
        return None, []
    r1, r2, b1, b2 = ordered
    best_class: int | None = None
    best_score = 1e9
    best_links: list[tuple[str, str, float, float, float]] = []
    for class_number, data in BASEPAIR_LENGTHS.items():
        if (b1, b2) != data[0]:
            continue
        score = 0.0
        n_seen = 0
        links = []
        for atom1_name, atom2_name, target, sigma, slack in data[1:]:
            a1 = r1.atom(atom1_name)
            a2 = r2.atom(atom2_name)
            if a1 is None or a2 is None:
                continue
            score += abs(distance(a1.xyz, a2.xyz) - 2.89)
            n_seen += 1
            links.append((atom1_name, atom2_name, target, sigma, slack))
        if n_seen == 0:
            continue
        score /= len(data[1:])
        if score < best_score:
            best_score = score
            best_class = class_number
            best_links = links
    if best_class is None:
        return None, []
    hbonds: list[tuple[str, str, float]] = []
    for atom1_name, atom2_name, *_ in best_links:
        a1 = r1.atom(atom1_name)
        a2 = r2.atom(atom2_name)
        if a1 is None or a2 is None:
            continue
        d = distance(a1.xyz, a2.xyz)
        if d < cutoff:
            hbonds.append((atom1_name, atom2_name, d))
    return best_class, hbonds


def detect_base_pairs(st, cutoff: float = 3.4) -> list[BasePairRecord]:
    residues = [residue for segment in na_segments(st) for residue in segment]
    records: list[BasePairRecord] = []
    seen: set[tuple[tuple[int, str, int, str], tuple[int, str, int, str]]] = set()
    for r1, r2 in itertools.combinations(residues, 2):
        if is_consecutive(r1, r2):
            continue
        candidate = False
        for a1 in base_contact_atoms(r1):
            if candidate:
                break
            for a2 in base_contact_atoms(r2):
                if a1.altloc and a2.altloc and a1.altloc != a2.altloc:
                    continue
                if distance(a1.xyz, a2.xyz) >= cutoff:
                    continue
                if final_link_direction_check(a1, r1, a2) or final_link_direction_check(a2, r2, a1):
                    candidate = True
                    break
        if not candidate:
            continue
        klass, hbonds = classify_basepair(r1, r2, cutoff=cutoff)
        if klass is None or len(hbonds) <= 1:
            continue
        key = tuple(sorted([r1.resid.key(), r2.resid.key()]))
        if key in seen:
            continue
        seen.add(key)
        records.append(BasePairRecord(base1=r1.resid, base2=r2.resid, saenger_class=klass, hbonds=hbonds))
    return records


def ring_points(residue: ResidueInfo) -> list[np.ndarray]:
    points = []
    for name in RING_ATOMS:
        atom = residue.atom(name)
        if atom is not None:
            points.append(vec(atom.xyz))
    return points


def ring_center_and_normal(residue: ResidueInfo) -> tuple[np.ndarray, np.ndarray] | None:
    points = ring_points(residue)
    if len(points) <= 2:
        return None
    center = np.mean(points, axis=0)
    normal = None
    for i in range(len(points) - 1):
        n = np.cross(points[i] - center, points[i + 1] - center)
        if norm(n) > 1e-8:
            normal = n
            break
    if normal is None:
        return None
    return center, normal


def stacking_geometry(r1: ResidueInfo, r2: ResidueInfo) -> tuple[float, float] | None:
    g1 = ring_center_and_normal(r1)
    g2 = ring_center_and_normal(r2)
    if g1 is None or g2 is None:
        return None
    c1, n1 = g1
    c2, n2 = g2
    center_vector = c2 - c1
    center_dist = norm(center_vector)
    if center_dist < 1e-10:
        return None
    normal_angle = angle_between_abs(n1, n2)
    center_normal_angle = angle_between_abs(center_vector, n1)
    if normal_angle is None or center_normal_angle is None:
        return None
    if center_dist < 5.5 and normal_angle < 30.0 and center_normal_angle < 40.0:
        return center_dist, normal_angle
    return None


def detect_stacking_pairs(st) -> list[StackingPairRecord]:
    records: list[StackingPairRecord] = []
    for segment in na_segments(st):
        for r1, r2 in zip(segment[:-1], segment[1:]):
            geom = stacking_geometry(r1, r2)
            if geom is None:
                continue
            center_dist, normal_angle = geom
            records.append(
                StackingPairRecord(
                    base1=r1.resid,
                    base2=r2.resid,
                    distance=center_dist,
                    normal_angle=normal_angle,
                )
            )
    return records
