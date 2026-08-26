from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResidueId:
    model: int
    chain: str
    resseq: int
    icode: str = ""
    name: str = ""

    def key(self) -> tuple[int, str, int, str]:
        return (self.model, self.chain, self.resseq, self.icode)

    def label(self) -> str:
        ins = self.icode.strip()
        return f"{self.chain}:{self.name}{self.resseq}{ins}"

    def resid(self) -> str:
        return f"{self.resseq}{self.icode.strip()}"

    def selection(self) -> str:
        return f"chain '{self.chain}' and resid {self.resid()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label(),
            "model": self.model,
            "chain": self.chain,
            "resid": self.resid(),
            "resseq": self.resseq,
            "icode": self.icode,
            "name": self.name,
            "selection": self.selection(),
        }


@dataclass
class AtomInfo:
    name: str
    element: str
    xyz: tuple[float, float, float]
    altloc: str = ""


@dataclass
class ResidueInfo:
    resid: ResidueId
    atoms: dict[str, AtomInfo]
    residue: Any = None

    def atom(self, name: str) -> AtomInfo | None:
        return self.atoms.get(name.strip().upper())

    def has_atom(self, name: str) -> bool:
        return self.atom(name) is not None


@dataclass
class HelixRecord:
    helix_type: str
    start: ResidueId
    end: ResidueId
    length: int
    source: str = "detected"
    good_hbonds: int = 0
    poor_hbonds: int = 0


@dataclass
class StrandRecord:
    start: ResidueId
    end: ResidueId
    sense: int = 0


@dataclass
class SheetRecord:
    sheet_id: int
    strands: list[StrandRecord]
    source: str = "detected"
    good_hbonds: int = 0
    poor_hbonds: int = 0


@dataclass
class BasePairRecord:
    base1: ResidueId
    base2: ResidueId
    saenger_class: int
    hbonds: list[tuple[str, str, float]]
    source: str = "detected"


@dataclass
class StackingPairRecord:
    base1: ResidueId
    base2: ResidueId
    distance: float
    normal_angle: float
    source: str = "detected"


@dataclass
class DetectionResult:
    input_path: str
    mode: str
    used_existing_protein: bool = False
    helices: list[HelixRecord] = field(default_factory=list)
    sheets: list[SheetRecord] = field(default_factory=list)
    base_pairs: list[BasePairRecord] = field(default_factory=list)
    stacking_pairs: list[StackingPairRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def range_selection(start: ResidueId, end: ResidueId) -> str:
            if start.chain == end.chain:
                return f"chain '{start.chain}' and resid {start.resid()} through {end.resid()}"
            return f"{start.selection()} or {end.selection()}"

        return {
            "input_path": self.input_path,
            "mode": self.mode,
            "used_existing_protein": self.used_existing_protein,
            "helices": [
                {
                    "type": h.helix_type,
                    "start": h.start.label(),
                    "end": h.end.label(),
                    "start_residue": h.start.to_dict(),
                    "end_residue": h.end.to_dict(),
                    "selection": range_selection(h.start, h.end),
                    "length": h.length,
                    "source": h.source,
                    "good_hbonds": h.good_hbonds,
                    "poor_hbonds": h.poor_hbonds,
                }
                for h in self.helices
            ],
            "sheets": [
                {
                    "sheet_id": s.sheet_id,
                    "source": s.source,
                    "good_hbonds": s.good_hbonds,
                    "poor_hbonds": s.poor_hbonds,
                    "strands": [
                        {
                            "start": strand.start.label(),
                            "end": strand.end.label(),
                            "start_residue": strand.start.to_dict(),
                            "end_residue": strand.end.to_dict(),
                            "selection": range_selection(strand.start, strand.end),
                            "sense": strand.sense,
                        }
                        for strand in s.strands
                    ],
                }
                for s in self.sheets
            ],
            "base_pairs": [
                {
                    "base1": bp.base1.label(),
                    "base2": bp.base2.label(),
                    "base1_residue": bp.base1.to_dict(),
                    "base2_residue": bp.base2.to_dict(),
                    "base1_selection": bp.base1.selection(),
                    "base2_selection": bp.base2.selection(),
                    "saenger_class": bp.saenger_class,
                    "hbonds": [
                        {"atom1": a1, "atom2": a2, "distance": d}
                        for a1, a2, d in bp.hbonds
                    ],
                    "source": bp.source,
                }
                for bp in self.base_pairs
            ],
            "stacking_pairs": [
                {
                    "base1": sp.base1.label(),
                    "base2": sp.base2.label(),
                    "base1_residue": sp.base1.to_dict(),
                    "base2_residue": sp.base2.to_dict(),
                    "base1_selection": sp.base1.selection(),
                    "base2_selection": sp.base2.selection(),
                    "distance": sp.distance,
                    "normal_angle": sp.normal_angle,
                    "source": sp.source,
                }
                for sp in self.stacking_pairs
            ],
            "warnings": self.warnings,
        }
