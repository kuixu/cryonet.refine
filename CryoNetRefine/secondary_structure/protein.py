from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import distance, norm, unit
from .io import protein_segments
from .models import HelixRecord, ResidueInfo, SheetRecord, StrandRecord


@dataclass(frozen=True)
class SegmentParams:
    name: str
    kind: str
    span: float
    rise: float
    minimum_length: int
    dot_min: float
    dot_min_single: float | None
    target_i_ip3: float | None = None
    tol_i_ip3: float | None = None
    n_link_min: int = 0


ALPHA = SegmentParams("alpha", "helix", 3.5, 1.54, 6, 0.90, 0.30)
THREE_TEN = SegmentParams("3_10", "helix", 3.0, 2.00, 6, 0.90, 0.50)
PI = SegmentParams("pi", "helix", 4.0, 0.95, 6, 0.90, 0.10)
BETA = SegmentParams("beta", "strand", 2.0, 3.30, 4, 0.75, 0.50, 10.0, 1.5, 3)


@dataclass
class ProteinSegment:
    residues: list[ResidueInfo]
    params: SegmentParams

    @property
    def start(self):
        return self.residues[0].resid

    @property
    def end(self):
        return self.residues[-1].resid

    def sites(self) -> np.ndarray:
        return np.asarray([r.atom("CA").xyz for r in self.residues], dtype=float)

    def length(self) -> int:
        return len(self.residues)

    def average_direction(self) -> np.ndarray | None:
        diffs, _ = segment_diffs(self.sites(), self.params)
        if len(diffs) == 0:
            return None
        return average_direction(diffs)


def segment_diffs(sites: np.ndarray, params: SegmentParams) -> tuple[np.ndarray, np.ndarray]:
    if params.kind == "strand":
        if len(sites) < 3:
            return np.empty((0, 3)), np.empty((0,))
        raw = sites[2:] - sites[:-2]
    else:
        if len(sites) < 5:
            return np.empty((0, 3)), np.empty((0,))
        offset3 = sites[3:-1]
        offset4 = sites[4:]
        if params.span <= 3:
            target = offset3
        elif params.span >= 4:
            target = offset4
        else:
            target = 0.5 * (offset3 + offset4)
        raw = target - sites[:-4]
    norms = np.linalg.norm(raw, axis=1)
    safe = np.where(norms < 1e-10, 1e-10, norms)
    return raw / safe[:, None], norms


def norms_i_i3(sites: np.ndarray) -> np.ndarray:
    if len(sites) < 4:
        return np.empty((0,))
    return np.linalg.norm(sites[3:] - sites[:-3], axis=1)


def average_direction(diffs: np.ndarray, start: int | None = None, end: int | None = None) -> np.ndarray | None:
    if len(diffs) == 0:
        return None
    if start is None and end is None:
        start = 0
        end = len(diffs) - 1
    assert start is not None and end is not None
    if start < 0 or start >= len(diffs):
        return None
    stop = max(start + 1, min(end, len(diffs)))
    return unit(np.mean(diffs[start:stop], axis=0))


def segment_is_ok(residues: list[ResidueInfo], params: SegmentParams) -> bool:
    if len(residues) < params.minimum_length:
        return False
    sites = np.asarray([r.atom("CA").xyz for r in residues], dtype=float)
    diffs, norms = segment_diffs(sites, params)
    if len(diffs) == 0:
        return False
    target = params.rise * params.span
    tol = 0.5 * params.span
    rise = float(np.mean(norms)) / params.span
    avg = average_direction(diffs)
    if avg is None:
        return False
    mean_dot = float(np.mean(diffs @ avg))
    single = sites[1:] - sites[:-1]
    single_norm = np.linalg.norm(single, axis=1)
    single = single / np.where(single_norm < 1e-10, 1e-10, single_norm)[:, None]
    mean_dot_single = float(np.mean(single @ avg))
    if mean_dot < params.dot_min:
        return False
    if abs(float(np.mean(norms)) - target) > tol:
        return False
    if params.dot_min_single is not None and mean_dot_single < params.dot_min_single:
        return False
    if params.target_i_ip3 is not None and params.tol_i_ip3 is not None:
        n3 = norms_i_i3(sites)
        if len(n3) and float(np.min(n3)) < params.target_i_ip3 - params.tol_i_ip3:
            return False
    return rise > 0


def find_segments_in_chain(residues: list[ResidueInfo], params: SegmentParams, used: set[int] | None = None) -> list[ProteinSegment]:
    if used is None:
        used = set()
    if len(residues) < params.minimum_length:
        return []
    sites = np.asarray([r.atom("CA").xyz for r in residues], dtype=float)
    diffs, norms = segment_diffs(sites, params)
    n = len(diffs)
    if n == 0:
        return []
    target = params.rise * params.span
    tol = 0.5 * params.span
    last_offset = int(params.span + 0.99)
    n3 = norms_i_i3(sites)
    candidates: dict[int, int] = {}
    occupied: set[int] = set()
    for i in range(n):
        if i in occupied or i in used or abs(norms[i] - target) > tol:
            continue
        start = i
        end = i
        occupied.add(i)
        for j in range(i + 1, n):
            if j in used:
                break
            if abs(norms[j] - target) > tol:
                break
            if params.target_i_ip3 is not None and j < len(n3):
                if abs(n3[j] - params.target_i_ip3) > (params.tol_i_ip3 or 0):
                    break
            if float(np.dot(diffs[j], diffs[j - 1])) < params.dot_min:
                break
            end = j
            occupied.add(j)
        if end + 1 + last_offset - start >= params.minimum_length:
            candidates[start] = end

    candidates = _remove_bad_residues(candidates, diffs, params, last_offset)
    candidates = _merge_segments(candidates, residues, params, last_offset)
    candidates = _trim_short_linkages(candidates, residues, params, last_offset)
    candidates = _merge_segments(candidates, residues, params, last_offset)
    segments: list[ProteinSegment] = []
    for start, end in sorted(candidates.items()):
        end_res = min(len(residues) - 1, end + last_offset)
        seg_res = residues[start : end_res + 1]
        if segment_is_ok(seg_res, params):
            segments.append(ProteinSegment(seg_res, params))
    return segments


def _remove_bad_residues(candidates: dict[int, int], diffs: np.ndarray, params: SegmentParams, last_offset: int) -> dict[int, int]:
    result = dict(candidates)
    cycles = 0
    changed = True
    while changed and cycles < 2:
        changed = False
        new: dict[int, int] = {}
        for start, end in sorted(result.items()):
            # cctbx get_average_direction(diffs, i, j) intentionally averages
            # diffs[i:j], excluding j.  This affects beta boundary trimming.
            avg = average_direction(diffs, start, end)
            if avg is None:
                continue
            current_start: int | None = None
            for j in range(start, end + 1):
                if float(np.dot(diffs[j], avg)) >= params.dot_min:
                    if current_start is None:
                        current_start = j
                    new[current_start] = j
                else:
                    current_start = None
                    changed = True
        result = {
            s: e
            for s, e in new.items()
            if e + 1 + last_offset - s >= params.minimum_length
        }
        cycles += 1
    return result


def _trim_short_linkages(candidates: dict[int, int], residues: list[ResidueInfo], params: SegmentParams, last_offset: int) -> dict[int, int]:
    result = dict(candidates)
    changed = True
    cycles = 0
    while changed and cycles <= len(result):
        changed = False
        cycles += 1
        keys = sorted(result)
        for a, b in zip(keys[:-1], keys[1:]):
            delta = b - (result[a] + last_offset)
            if delta >= params.n_link_min:
                continue
            seg1_res = residues[a : min(len(residues), result[a] + last_offset + 1)]
            seg2_res = residues[b : min(len(residues), result[b] + last_offset + 1)]
            if not seg1_res or not seg2_res:
                continue
            dir1 = ProteinSegment(seg1_res, params).average_direction()
            dir2 = ProteinSegment(seg2_res, params).average_direction()
            if dir1 is None or dir2 is None or float(np.dot(dir1, dir2)) > 0:
                continue

            residues_to_cut = (delta + 1) // 2
            old_end_b = result[b]
            changed = True

            new_end_a = result[a] - residues_to_cut
            if new_end_a >= a and segment_is_ok(residues[a : min(len(residues), new_end_a + last_offset + 1)], params):
                result[a] = new_end_a
            else:
                del result[a]

            new_b = b + residues_to_cut
            if old_end_b >= new_b and segment_is_ok(residues[new_b : min(len(residues), old_end_b + last_offset + 1)], params):
                result[new_b] = old_end_b
            if new_b != b and b in result:
                del result[b]
            break
    return result


def _merge_segments(candidates: dict[int, int], residues: list[ResidueInfo], params: SegmentParams, last_offset: int) -> dict[int, int]:
    result = dict(candidates)
    changed = True
    cycles = 0
    while changed and cycles <= len(result):
        changed = False
        cycles += 1
        keys = sorted(result)
        for a, b in zip(keys[:-1], keys[1:]):
            if b <= result[a] + last_offset + 1:
                end_res = min(len(residues) - 1, result[b] + last_offset)
                if segment_is_ok(residues[a : end_res + 1], params):
                    result[a] = result[b]
                    if a != b:
                        del result[b]
                    changed = True
                    break
    return result


def hbond_counts_for_helix(segment: ProteinSegment, max_length: float = 3.5) -> tuple[int, int]:
    offset = {"alpha": 4, "pi": 5, "3_10": 3}.get(segment.params.name, 4)
    good = poor = 0
    residues = segment.residues
    for i in range(0, len(residues) - offset):
        o_atom = residues[i].atom("O")
        n_atom = residues[i + offset].atom("N")
        if o_atom is None or n_atom is None or residues[i + offset].resid.name == "PRO":
            continue
        if distance(o_atom.xyz, n_atom.xyz) <= max_length:
            good += 1
        else:
            poor += 1
    return good, poor


def make_unique_segments(
    chain: list[ResidueInfo],
    groups: list[list[ProteinSegment]],
) -> list[list[ProteinSegment]]:
    index_by_key = {residue.resid.key(): i for i, residue in enumerate(chain)}
    used = [False] * len(chain)
    unique_groups: list[list[ProteinSegment]] = []
    for segments in groups:
        unique: list[ProteinSegment] = []
        for segment in segments:
            first = index_by_key.get(segment.residues[0].resid.key())
            last = index_by_key.get(segment.residues[-1].resid.key())
            if first is None or last is None:
                continue
            start_pos, end_pos = _first_unused_range(used[first : last + 1])
            if start_pos is None or end_pos is None:
                continue
            if start_pos != 0 or end_pos != last - first:
                trimmed = segment.residues[start_pos : end_pos + 1]
                segment = ProteinSegment(trimmed, segment.params)
                if not segment_is_ok(trimmed, segment.params):
                    continue
            for i in range(first + start_pos, first + end_pos + 1):
                used[i] = True
            unique.append(segment)
        unique_groups.append(unique)
    return unique_groups


def _first_unused_range(already_used: list[bool]) -> tuple[int | None, int | None]:
    new_start = None
    new_end = None
    for i, is_used in enumerate(already_used):
        if is_used:
            if new_end is not None:
                return new_start, new_end
        else:
            if new_end is None:
                new_start = i
                new_end = i
            else:
                new_end = i
    return new_start, new_end


def detect_helices(chains: list[list[ResidueInfo]], include_310: bool = True, include_pi: bool = True) -> list[HelixRecord]:
    records: list[HelixRecord] = []
    params = [ALPHA]
    if include_310:
        params.append(THREE_TEN)
    if include_pi:
        params.append(PI)
    for chain in chains:
        for par in params:
            for segment in find_segments_in_chain(chain, par):
                good, poor = hbond_counts_for_helix(segment)
                records.append(
                    HelixRecord(
                        helix_type=par.name,
                        start=segment.start,
                        end=segment.end,
                        length=segment.length(),
                        good_hbonds=good,
                        poor_hbonds=poor,
                    )
                )
    return records


def detect_beta_segments(chains: list[list[ResidueInfo]], exclude: set[tuple[int, str, int, str]] | None = None) -> list[ProteinSegment]:
    segments: list[ProteinSegment] = []
    for chain in chains:
        used = set()
        if exclude:
            used = {i for i, residue in enumerate(chain) if residue.resid.key() in exclude}
        segments.extend(find_segments_in_chain(chain, BETA, used))
    return segments


SheetInfo = tuple[int, int, int, int, bool, int | None, int | None]


def detect_sheets(
    strands: list[ProteinSegment],
    max_ca_ca: float = 6.0,
    min_sheet_length: int = 4,
    include_single_strands: bool = False,
) -> list[SheetRecord]:
    pair_dict: dict[int, list[int]] = {i: [] for i in range(len(strands))}
    info: dict[tuple[int, int], SheetInfo] = {}
    for i in range(len(strands)):
        for j in range(i + 1, len(strands)):
            aligned = align_strands(strands[i], strands[j], max_ca_ca, min_sheet_length)
            if aligned is None:
                continue
            i_index, j_index = get_ind_h_bond_sheet(strands[i], strands[j], aligned)
            pair_dict[i].append(j)
            pair_dict[j].append(i)
            a1, b1, a2, b2, parallel = aligned
            info[(i, j)] = (a1, b1, a2, b2, parallel, i_index, j_index)
            ri_index, rj_index = get_ind_h_bond_sheet(strands[j], strands[i], (a2, b2, a1, b1, parallel))
            info[(j, i)] = (a2, b2, a1, b1, parallel, ri_index, rj_index)

    single_strands = _get_strands_by_pairs(pair_dict, len(strands), [], 0)
    used_for_classification = list(single_strands)
    pair_strands = _get_strands_by_pairs(pair_dict, len(strands), used_for_classification, 1)
    triple_strands = _get_strands_by_pairs(pair_dict, len(strands), used_for_classification, 2)
    _get_strands_by_pairs(pair_dict, len(strands), used_for_classification, None)

    used: list[int] = list(single_strands)
    sheet_lists: list[list[int]] = []
    if include_single_strands:
        sheet_lists.extend([i] for i in single_strands)
    sheet_lists.extend(_get_sheets_from_edges(pair_strands, pair_dict, used))
    sheet_lists.extend(_get_sheets_from_edges(triple_strands, pair_dict, used))

    existing_pairs: set[tuple[int, int]] = set()
    for sheet in sheet_lists:
        for i, j in zip(sheet[:-1], sheet[1:]):
            existing_pairs.add((i, j))
            existing_pairs.add((j, i))
    missing_pairs: set[tuple[int, int]] = set()
    for i in range(len(strands)):
        for j in pair_dict.get(i, []):
            if (i, j) not in existing_pairs and (i, j) not in missing_pairs and (j, i) not in missing_pairs:
                missing_pairs.add((i, j))
                sheet_lists.append([i, j])
                if i not in used:
                    used.append(i)
                if j not in used:
                    used.append(j)

    records: list[SheetRecord] = []
    for sheet_id, sheet in enumerate(sheet_lists, start=1):
        strand_records: list[StrandRecord] = []
        good = poor = 0
        start_dict, end_dict = _required_start_end(sheet, info)
        for pos, idx in enumerate(sheet):
            segment = strands[idx]
            sense = 0
            if pos > 0:
                rel = info.get((sheet[pos - 1], idx))
                if rel:
                    sense = 1 if rel[4] else -1
                    g, p = hbond_counts_for_strand_pair(strands[sheet[pos - 1]], segment, rel)
                    good += g
                    poor += p
            start_index = start_dict.get(idx)
            end_index = end_dict.get(idx)
            start_res = segment.residues[start_index] if start_index is not None else segment.residues[0]
            end_res = segment.residues[end_index] if end_index is not None else segment.residues[-1]
            strand_records.append(StrandRecord(start=start_res.resid, end=end_res.resid, sense=sense))
        records.append(SheetRecord(sheet_id=sheet_id, strands=strand_records, good_hbonds=good, poor_hbonds=poor))
    return records


def align_strands(s1: ProteinSegment, s2: ProteinSegment, tol: float, min_len: int) -> tuple[int, int, int, int, bool] | None:
    sites1 = s1.sites()
    sites2 = s2.sites()
    close = ca_pair_is_close(sites1, sites2, tol)
    if close is None:
        return None
    center1, center2 = close
    _dists, keep1, keep2 = _residue_pairs_in_sheet(sites1, sites2, center1, center2, tol)
    _rdists, rkeep1, rkeep2 = _residue_pairs_in_sheet(
        sites1, sites2[::-1], center1, len(sites2) - center2 - 1, tol
    )
    if len(keep1) < min_len and len(rkeep1) < min_len:
        return None
    if len(rkeep1) > len(keep1):
        return (rkeep1[0], rkeep1[-1], len(sites2) - rkeep2[-1] - 1, len(sites2) - rkeep2[0] - 1, False)
    return (keep1[0], keep1[-1], keep2[0], keep2[-1], True)


def ca_pair_is_close(
    sites1: np.ndarray,
    sites2: np.ndarray,
    tol: float,
    dist_per_residue: float = 3.5,
    jump: int = 4,
) -> tuple[int, int] | None:
    best_dist_sq: float | None = None
    best: tuple[int, int] | None = None
    while jump > 0:
        for i in range(jump // 2, len(sites1), jump):
            for j in range(jump // 2, len(sites2), jump):
                dist_sq = float(np.sum((sites1[i] - sites2[j]) ** 2))
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best = (i, j)
        if best_dist_sq is None or best_dist_sq**0.5 > (jump + 1) * dist_per_residue + tol:
            break
        if jump == 1:
            break
        jump = max(1, jump // 2)
    if best_dist_sq is not None and best_dist_sq <= tol**2:
        return best
    return None


def _residue_pairs_in_sheet(
    sites1: np.ndarray,
    sites2: np.ndarray,
    center1: int,
    center2: int,
    tol: float,
) -> tuple[list[float], list[int], list[int]]:
    keep1: list[int] = []
    keep2: list[int] = []
    dists: list[float] = []
    if float(np.sum((sites1[center1] - sites2[center2]) ** 2)) <= tol**2:
        start_offset = max(-center1, -center2)
        end_offset = min(len(sites1) - (center1 + 1), len(sites2) - (center2 + 1))
        for offset in range(start_offset, end_offset + 1):
            i1 = center1 + offset
            i2 = center2 + offset
            dist = float(np.linalg.norm(sites1[i1] - sites2[i2]))
            if dist <= tol:
                keep1.append(i1)
                keep2.append(i2)
                dists.append(dist)
            elif offset > 0:
                break
            else:
                dists = []
    return dists, keep1, keep2


def get_ind_h_bond_sheet(
    strand_i: ProteinSegment,
    strand_j: ProteinSegment,
    rel: tuple[int, int, int, int, bool],
) -> tuple[int | None, int | None]:
    first1, _last1, first2, last2, parallel = rel
    i_index = first1
    j_index = first2 + 1 if parallel else last2
    if not (0 <= i_index < strand_i.length()) or not (0 <= j_index < strand_j.length()):
        return None, None
    inter = strand_j.sites()[j_index] - strand_i.sites()[i_index]
    inter_u = unit(inter)
    avg_dir = strand_i.average_direction()
    if inter_u is None or avg_dir is None:
        return None, None
    up = unit(np.cross(inter_u, avg_dir))
    if up is None:
        return None, None

    n_dot = 0
    sum_dot = 0.0
    last_offset_index = len(strand_i.sites()) - i_index - 2
    for i in range(last_offset_index // 2 + 1):
        offset = 2 * i
        delta = strand_i.sites()[i_index + 1 + offset] - strand_i.sites()[i_index + offset]
        sum_dot += float(np.dot(up, delta))
        n_dot += 1
    if not n_dot:
        return None, None
    if sum_dot / n_dot > 0:
        i_index += 1
        j_index = j_index + 1 if parallel else j_index - 1
    if i_index + 1 > strand_i.length() or j_index + 1 > strand_j.length() or j_index < 0:
        return None, None
    return i_index, j_index


def _get_strands_by_pairs(pair_dict: dict[int, list[int]], n: int, used: list[int], pairs: int | None) -> list[int]:
    result: list[int] = []
    while True:
        found = None
        for i in range(n):
            if i in used:
                continue
            if pairs is None or len(pair_dict.get(i, [])) == pairs:
                found = i
                break
        if found is None:
            break
        used.append(found)
        result.append(found)
    return result


def _get_sheets_from_edges(pair_strands: list[int], pair_dict: dict[int, list[int]], used: list[int]) -> list[list[int]]:
    sheets: list[list[int]] = []
    for i in pair_strands:
        if i in used:
            continue
        strand_list = [i]
        current = i
        while current is not None:
            current = next((x for x in pair_dict.get(current, []) if x not in used and x not in strand_list), None)
            if current is not None:
                strand_list.append(current)
        if len(strand_list) > 1:
            used.extend(strand_list)
            sheets.append(strand_list)
    return sheets


def _required_start_end(sheet: list[int], info: dict[tuple[int, int], SheetInfo]) -> tuple[dict[int, int | None], dict[int, int | None]]:
    start: dict[int, int | None] = {i: None for i in sheet}
    end: dict[int, int | None] = {i: None for i in sheet}
    for i, j in zip(sheet[:-1], sheet[1:]):
        first1, last1, first2, last2, _parallel, _i_index, _j_index = info[(i, j)]
        if start[i] is None or start[i] > first1:
            start[i] = first1
        if end[i] is None or end[i] < last1:
            end[i] = last1
        if start[j] is None or start[j] > first2:
            start[j] = first2
        if end[j] is None or end[j] < last2:
            end[j] = last2
    return start, end


def hbond_counts_for_strand_pair(prev: ProteinSegment, cur: ProteinSegment, rel: SheetInfo, max_length: float = 3.5) -> tuple[int, int]:
    first1, last1, first2, last2, parallel, i_index, j_index = rel
    if i_index is None or j_index is None:
        return 0, 0
    good = poor = 0
    for i in range(prev.length()):
        if (i - i_index) % 2 != 0:
            continue
        li = i_index + (i - i_index)
        lj = j_index + (i - i_index if parallel else -(i - i_index))
        if not (0 <= li < prev.length()):
            continue
        if li < first1 or li > last1 or lj < first2 or lj > last2:
            continue
        for o_to_n in (True, False):
            if o_to_n:
                if not (0 <= lj < cur.length()):
                    continue
                a = prev.residues[li].atom("O")
                b = cur.residues[lj].atom("N")
            else:
                lj2 = lj - 2 if parallel else lj
                if not (0 <= lj2 < cur.length()):
                    continue
                a = prev.residues[li].atom("N")
                b = cur.residues[lj2].atom("O")
            if a is None or b is None:
                continue
            if distance(a.xyz, b.xyz) <= max_length:
                good += 1
            else:
                poor += 1
    return good, poor


def detect_protein_secondary_structure(st, include_single_strands: bool = False) -> tuple[list[HelixRecord], list[SheetRecord]]:
    chains = protein_segments(st)
    alpha_segments: list[ProteinSegment] = []
    three_ten_segments: list[ProteinSegment] = []
    pi_segments: list[ProteinSegment] = []
    beta_segments: list[ProteinSegment] = []
    for chain in chains:
        groups = make_unique_segments(
            chain,
            [
                find_segments_in_chain(chain, ALPHA),
                find_segments_in_chain(chain, THREE_TEN),
                find_segments_in_chain(chain, PI),
                find_segments_in_chain(chain, BETA),
            ],
        )
        alpha_segments.extend(groups[0])
        three_ten_segments.extend(groups[1])
        pi_segments.extend(groups[2])
        beta_segments.extend(groups[3])

    helices: list[HelixRecord] = []
    for par, segments in [(ALPHA, alpha_segments), (THREE_TEN, three_ten_segments), (PI, pi_segments)]:
        for segment in segments:
            good, poor = hbond_counts_for_helix(segment)
            helices.append(
                HelixRecord(
                    helix_type=par.name,
                    start=segment.start,
                    end=segment.end,
                    length=segment.length(),
                    good_hbonds=good,
                    poor_hbonds=poor,
                )
            )
    sheets = detect_sheets(beta_segments, include_single_strands=include_single_strands)
    return helices, sheets
