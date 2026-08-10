from __future__ import annotations

import json

from .models import DetectionResult


def as_json(result: DetectionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def as_records(result: DetectionResult) -> str:
    lines: list[str] = []
    for i, helix in enumerate(result.helices, start=1):
        lines.append(
            "HELIX {idx:4d} {typ:5s} {sc:>2s} {sr:>4d}{si:1s}  {ec:>2s} {er:>4d}{ei:1s}  {length:4d}  {source}".format(
                idx=i,
                typ=helix.helix_type,
                sc=helix.start.chain,
                sr=helix.start.resseq,
                si=(helix.start.icode or " ")[:1],
                ec=helix.end.chain,
                er=helix.end.resseq,
                ei=(helix.end.icode or " ")[:1],
                length=helix.length,
                source=helix.source,
            )
        )
    for sheet in result.sheets:
        for i, strand in enumerate(sheet.strands, start=1):
            lines.append(
                "SHEET {sid:4d} {strand:3d} {sc:>2s} {sr:>4d}{si:1s}  {ec:>2s} {er:>4d}{ei:1s}  sense={sense:2d} {source}".format(
                    sid=sheet.sheet_id,
                    strand=i,
                    sc=strand.start.chain,
                    sr=strand.start.resseq,
                    si=(strand.start.icode or " ")[:1],
                    ec=strand.end.chain,
                    er=strand.end.resseq,
                    ei=(strand.end.icode or " ")[:1],
                    sense=strand.sense,
                    source=sheet.source,
                )
            )
    for bp in result.base_pairs:
        hb = ",".join(f"{a1}-{a2}:{d:.2f}" for a1, a2, d in bp.hbonds)
        lines.append(
            f"BASEPAIR {bp.base1.label()} {bp.base2.label()} saenger={bp.saenger_class} hbonds={hb}"
        )
    for sp in result.stacking_pairs:
        lines.append(
            f"STACKING {sp.base1.label()} {sp.base2.label()} distance={sp.distance:.2f} normal_angle={sp.normal_angle:.1f}"
        )
    return "\n".join(lines)


def as_phil(result: DetectionResult) -> str:
    lines = ["secondary_structure {"]
    if result.helices or result.sheets:
        lines.append("  protein {")
        for helix in result.helices:
            lines.extend(
                [
                    "    helix {",
                    f"      selection = chain '{helix.start.chain}' and resid {helix.start.resseq} through {helix.end.resseq}",
                    f"      helix_type = {helix.helix_type}",
                    "    }",
                ]
            )
        for sheet in result.sheets:
            if not sheet.strands:
                continue
            first = sheet.strands[0]
            lines.extend(
                [
                    "    sheet {",
                    f"      first_strand = chain '{first.start.chain}' and resid {first.start.resseq} through {first.end.resseq}",
                    f"      sheet_id = {sheet.sheet_id}",
                ]
            )
            for strand in sheet.strands[1:]:
                sense = "parallel" if strand.sense == 1 else "antiparallel" if strand.sense == -1 else "unknown"
                lines.extend(
                    [
                        "      strand {",
                        f"        selection = chain '{strand.start.chain}' and resid {strand.start.resseq} through {strand.end.resseq}",
                        f"        sense = {sense}",
                        "      }",
                    ]
                )
            lines.append("    }")
        lines.append("  }")
    if result.base_pairs or result.stacking_pairs:
        lines.append("  nucleic_acid {")
        for bp in result.base_pairs:
            lines.extend(
                [
                    "    base_pair {",
                    f"      base1 = chain '{bp.base1.chain}' and resid {bp.base1.resseq}",
                    f"      base2 = chain '{bp.base2.chain}' and resid {bp.base2.resseq}",
                    f"      saenger_class = {bp.saenger_class}",
                    "    }",
                ]
            )
        for sp in result.stacking_pairs:
            lines.extend(
                [
                    "    stacking_pair {",
                    f"      base1 = chain '{sp.base1.chain}' and resid {sp.base1.resseq}",
                    f"      base2 = chain '{sp.base2.chain}' and resid {sp.base2.resseq}",
                    "    }",
                ]
            )
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
