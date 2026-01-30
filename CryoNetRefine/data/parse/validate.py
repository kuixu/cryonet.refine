# CryoNetRefine/data/parse/validate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import gemmi

from CryoNetRefine.data import const
from CryoNetRefine.libs.density.density import DensityInfo
from CryoNetRefine.loss.loss import compute_overall_cc_loss


PathLike = Union[str, Path]
PathInput = Union[PathLike, Sequence[PathLike]]  # kept for backward-compat


@dataclass
class GapInfo:
    has_gap: bool
    chain_to_ranges: Dict[str, List[Tuple[int, int]]]
    total_missing: int


@dataclass
class ValidateReport:
    unsupported_residues: List[str]
    gap_info: GapInfo
    initial_cc: Optional[float]
    cc_ok: Optional[bool]
    messages: List[str]


def _mols_dir() -> Path:
    # .../CryoNetRefine/data/parse/validate.py -> .../CryoNetRefine/data/mols
    return Path(__file__).resolve().parents[1] / "mols"


def _supported_residue_codes() -> set[str]:
    # 用 data/mols 下的 *.pkl 作为“支持列表”，最贴合你项目实际情况
    mols = _mols_dir()
    supported = {p.stem.upper() for p in mols.glob("*.pkl")}
    # 解析里会把 MSE 视作 MET（mmcif.py 里有映射），所以这里也认为支持
    supported.add("MSE")
    return supported


def _normalize_input_paths(input_path: PathInput) -> List[Path]:
    if isinstance(input_path, (str, Path)):
        return [Path(input_path)]
    # Backward-compat: allow callers to pass a list/tuple of paths
    paths = [Path(p) for p in input_path]
    if len(paths) == 0:
        raise ValueError("input_path is empty")
    return paths


def _read_structure_any(path: Path) -> gemmi.Structure:
    if path.suffix.lower() == ".pdb":
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        return st
    # treat as cif/mmcif
    block = gemmi.cif.read(str(path))[0]
    st = gemmi.make_structure_from_block(block)
    st.merge_chain_parts()
    st.remove_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    return st


def _find_unsupported_residues(st: gemmi.Structure, supported: set[str]) -> List[str]:
    bad = set()
    model = st[0]
    for chain in model:
        for res in chain:
            name = res.name.upper()
            # skip waters (稳妥起见双保险)
            if getattr(res, "is_water", None) and res.is_water():
                continue
            if name in ("HOH", "WAT"):
                continue
            if name not in supported:
                bad.add(name)
    return sorted(bad)


def _detect_gaps(st: gemmi.Structure) -> GapInfo:
    """
    以 residue.seqid.num 的不连续作为 gap 定义（与你 mmcif.py 的 gap 逻辑一致：只看 num，不看 icode）。
    只在“像 polymer 的残基”上统计（避免 ligand 乱入编号）。
    """
    polymer_like = set(const.tokens) | {"MSE"}  # const.tokens 包含 20AA + A/C/G/U + DA/DC/DG/DT + UNK等
    chain_to_ranges: Dict[str, List[Tuple[int, int]]] = {}
    total_missing = 0

    model = st[0]
    for chain in model:
        nums = []
        for res in chain:
            name = res.name.upper()
            if name not in polymer_like:
                continue
            nums.append(int(res.seqid.num))
        nums = sorted(set(nums))
        ranges: List[Tuple[int, int]] = []
        for a, b in zip(nums, nums[1:]):
            if b - a > 1:
                start = a + 1
                end = b - 1
                ranges.append((start, end))
                total_missing += (end - start + 1)
        if ranges:
            chain_to_ranges[chain.name] = ranges

    return GapInfo(has_gap=(total_missing > 0), chain_to_ranges=chain_to_ranges, total_missing=total_missing)


def _build_atom_weight_table(device: torch.device) -> torch.Tensor:
    """
    根据 const.atomic_to_symbol + const.atom_weight 构造 [num_elements] 的原子权重表，
    用来从 one-hot ref_element 得到每个 atom 的 weight。
    """
    table = torch.zeros(const.num_elements, dtype=torch.float32, device=device)
    default_w = 12.0
    for atomic_num, symbol in const.atomic_to_symbol.items():
        if atomic_num < const.num_elements:
            table[int(atomic_num)] = float(const.atom_weight.get(symbol, default_w))
    return table


def _extract_atom_coords_and_weights_from_structure(
    st: gemmi.Structure,
    supported_residue_names: Optional[set[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract atom coordinates and per-atom weights from a parsed structure.

    - If supported_residue_names is provided, atoms belonging to residues NOT in this set
      will be skipped (matches the "we skip unsupported residues" policy).
    - Returns:
        coords: [N, 3] float32
        atom_weights: [N] float32
    """
    coords: List[List[float]] = []
    weights: List[float] = []
    default_w = 12.0

    model = st[0]
    for chain in model:
        for res in chain:
            resname = res.name.upper()
            if supported_residue_names is not None and resname not in supported_residue_names:
                continue
            for atom in res:
                p = atom.pos
                coords.append([float(p.x), float(p.y), float(p.z)])
                sym = atom.element.name.upper() if hasattr(atom, "element") else ""
                weights.append(float(const.atom_weight.get(sym, default_w)))

    if len(coords) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )

    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def validate_initial_cc_from_structure_paths(
    input_paths: Sequence[Path],
    target_density_obj: Sequence[DensityInfo],
    device: Union[str, torch.device] = "cpu",
    cc_threshold: float = 0.0,
    *,
    supported_residue_names: Optional[set[str]] = None,
) -> Tuple[float, bool, str]:
    """
    Compute global initial CC from the *input structure files* (no batch required).

    This uses the same CC definition as diffusion's initial_cc:
    `compute_overall_cc_loss()` + `mol_atom_density()` + density overlap cosine similarity.

    Notes:
    - Missing residues (gaps) naturally contribute no atoms here.
    - Unsupported residues can be excluded by passing supported_residue_names.
    """
    dev = torch.device(device)

    # Merge all atoms from all input paths into a single coordinate set.
    # (Typical usage is one structure file.)
    all_coords = []
    all_weights = []
    for p in input_paths:
        st = _read_structure_any(p)
        coords_i, weights_i = _extract_atom_coords_and_weights_from_structure(
            st,
            supported_residue_names=supported_residue_names,
        )
        if coords_i.numel() == 0:
            continue
        all_coords.append(coords_i)
        all_weights.append(weights_i)

    if not all_coords:
        cc_val = 0.0
        ok = False
        msg = (
            "Initial CC check: no atoms extracted from input structure "
            "(possibly empty structure or all residues unsupported)."
        )
        return cc_val, ok, msg

    coords = torch.cat(all_coords, dim=0).to(dev)  # [N, 3]
    atom_weights = torch.cat(all_weights, dim=0).to(dev)  # [N]

    # Build minimal feats for compute_overall_cc_loss
    atom_pad = torch.ones((1, coords.shape[0]), dtype=torch.bool, device=dev)
    atom_resolved = torch.ones((1, coords.shape[0]), dtype=torch.bool, device=dev)
    feats = {"atom_pad_mask": atom_pad, "atom_resolved_mask": atom_resolved}

    cc, _ = compute_overall_cc_loss(
        predicted_coords=coords.unsqueeze(0),  # [1, N, 3]
        target_density=target_density_obj,
        feats=feats,
        atom_weights=atom_weights,
    )

    cc_val = float(cc.detach().cpu().item())
    ok = cc_val > cc_threshold
    msg = (
        f"Initial CC check: CC={cc_val:.4f} (threshold>{cc_threshold})."
        + ("" if ok else " CC <= threshold, likely misalignment or invalid input.")
    )
    return cc_val, ok, msg


def validate_inputs(
    input_path: PathLike,
    target_density: Optional[Sequence[PathLike]] = None,
    resolution: Optional[Sequence[float]] = None,
    device: Union[str, torch.device] = "cpu",
    cc_threshold: float = 0.0,
) -> ValidateReport:
    """
    做三类检查：
    1) 非20AA/非核酸碱基（也即不在 data/mols/*.pkl 支持列表）-> 提示用户会跳过
    2) 初始全局 CC（直接从 input_path 解析原子坐标；需要 target_density+resolution 或传入 target_density_obj）
    3) gap（missing residue）检测
    """
    paths = _normalize_input_paths(input_path)
    supported = _supported_residue_codes()

    messages: List[str] = []
    unsupported_all = set()
    gap_agg: Dict[str, List[Tuple[int, int]]] = {}
    total_missing = 0

    for p in paths:
        st = _read_structure_any(p)

        bad = _find_unsupported_residues(st, supported)
        unsupported_all.update(bad)

        gap = _detect_gaps(st)
        if gap.has_gap:
            # 合并到总表里（不同文件同名 chain 也会合并；你目前一次只处理一个 input，所以够用）
            for ch, ranges in gap.chain_to_ranges.items():
                gap_agg.setdefault(ch, []).extend(ranges)
            total_missing += gap.total_missing

    unsupported_list = sorted(unsupported_all)
    if unsupported_list:
        messages.append(
            "Warning: found unsupported residue codes (not standard 20AA / not nucleic bases). "
            f"We will skip these residues if they cannot be parsed: {unsupported_list}"
        )

    gap_info = GapInfo(has_gap=(total_missing > 0), chain_to_ranges=gap_agg, total_missing=total_missing)
    if gap_info.has_gap:
        messages.append(
            "Warning: missing residues (gaps) detected in polymer chains. "
            f"total_missing={gap_info.total_missing}, chains={gap_info.chain_to_ranges}. "
            "Refinement quality may degrade."
        )

    # CC：只取第一个密度图（与 main.py 允许 multiple 的接口兼容）
    cc_val = None
    cc_ok = None
    td_obj: Optional[Sequence[DensityInfo]] = None
    if target_density is not None and resolution is not None:
        if len(target_density) == 0 or len(resolution) == 0:
            td_obj = None
        else:
            dev = torch.device(device)
            td_obj = [
                DensityInfo(
                    mrc_path=str(target_density[0]),
                    resolution=float(resolution[0]),
                    datatype="torch",
                    device=dev,
                )
            ]

    if td_obj:
        cc_val, cc_ok, cc_msg = validate_initial_cc_from_structure_paths(
            input_paths=paths,
            target_density_obj=td_obj,
            device=device,
            cc_threshold=cc_threshold,
            supported_residue_names=supported,
        )
        messages.append(cc_msg)

    return ValidateReport(
        unsupported_residues=unsupported_list,
        gap_info=gap_info,
        initial_cc=cc_val,
        cc_ok=cc_ok,
        messages=messages,
    )