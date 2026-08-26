from __future__ import annotations

from pathlib import Path

from .io import existing_protein_annotation, read_structure
from .models import DetectionResult
from .nucleic import detect_base_pairs, detect_stacking_pairs
from .protein import detect_protein_secondary_structure


def detect_secondary_structure(
    path: str | Path,
    mode: str = "auto",
    detect_nucleic: bool = True,
    include_single_strands: bool = False,
    detect_protein: bool = True,
) -> DetectionResult:
    if mode not in {"auto", "detect", "existing"}:
        raise ValueError("mode must be one of: auto, detect, existing")
    st = read_structure(path)
    result = DetectionResult(input_path=str(path), mode=mode)

    if detect_protein:
        existing_helices, existing_sheets = existing_protein_annotation(st)
        has_existing = bool(existing_helices or existing_sheets)
        if mode == "existing":
            result.helices = existing_helices
            result.sheets = existing_sheets
            result.used_existing_protein = has_existing
        elif mode == "auto" and has_existing:
            result.helices = existing_helices
            result.sheets = existing_sheets
            result.used_existing_protein = True
        else:
            result.helices, result.sheets = detect_protein_secondary_structure(
                st,
                include_single_strands=include_single_strands,
            )
    if detect_nucleic:
        result.base_pairs = detect_base_pairs(st)
        result.stacking_pairs = detect_stacking_pairs(st)
    return result
