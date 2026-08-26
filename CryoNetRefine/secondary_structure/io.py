from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gemmi

from .constants import AA3, NA_ONE_LETTER
from .models import AtomInfo, HelixRecord, ResidueId, ResidueInfo, SheetRecord, StrandRecord


def read_structure(path: str | Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def atom_key(name: str) -> str:
    return name.strip().upper().replace("*", "'")


def _seqid_parts(seqid) -> tuple[int, str]:
    num = getattr(seqid, "num", None)
    if num is None:
        try:
            num = int(str(seqid).strip())
        except ValueError:
            num = 0
    icode = getattr(seqid, "icode", "") or ""
    return int(num), str(icode).strip()


def make_residue_id(model_index: int, chain_name: str, residue) -> ResidueId:
    resseq, icode = _seqid_parts(residue.seqid)
    return ResidueId(
        model=model_index,
        chain=chain_name,
        resseq=resseq,
        icode=icode,
        name=residue.name.strip().upper(),
    )


def make_residue_info(model_index: int, chain_name: str, residue) -> ResidueInfo:
    atoms: dict[str, AtomInfo] = {}
    for atom in residue:
        key = atom_key(atom.name)
        altloc = "" if atom.altloc == "\0" else str(atom.altloc).strip()
        current = atoms.get(key)
        if current is not None and current.altloc == "" and altloc != "":
            continue
        if current is not None and current.altloc == "A" and altloc not in ("", "A"):
            continue
        atoms[key] = AtomInfo(
            name=key,
            element=atom.element.name,
            xyz=(atom.pos.x, atom.pos.y, atom.pos.z),
            altloc=altloc,
        )
    return ResidueInfo(make_residue_id(model_index, chain_name, residue), atoms, residue)


def iter_chain_residues(st: gemmi.Structure) -> Iterable[tuple[int, str, list[ResidueInfo]]]:
    for model_index, model in enumerate(st):
        for chain in model:
            residues = [make_residue_info(model_index, chain.name, residue) for residue in chain]
            yield model_index, chain.name, residues


def is_protein_residue(residue: ResidueInfo) -> bool:
    return residue.resid.name in AA3 and residue.has_atom("CA")


def is_na_residue(residue: ResidueInfo) -> bool:
    return residue.resid.name in NA_ONE_LETTER


def protein_segments(st: gemmi.Structure) -> list[list[ResidueInfo]]:
    segments: list[list[ResidueInfo]] = []
    for _, _, residues in iter_chain_residues(st):
        cur: list[ResidueInfo] = []
        prev: ResidueInfo | None = None
        for residue in residues:
            if not is_protein_residue(residue):
                if cur:
                    segments.append(cur)
                    cur = []
                prev = None
                continue
            if prev is not None and residue.resid.resseq - prev.resid.resseq > 1:
                if cur:
                    segments.append(cur)
                cur = []
            cur.append(residue)
            prev = residue
        if cur:
            segments.append(cur)
    return segments


def na_segments(st: gemmi.Structure) -> list[list[ResidueInfo]]:
    segments: list[list[ResidueInfo]] = []
    for _, _, residues in iter_chain_residues(st):
        cur: list[ResidueInfo] = []
        for residue in residues:
            if is_na_residue(residue):
                cur.append(residue)
            elif cur:
                segments.append(cur)
                cur = []
        if cur:
            segments.append(cur)
    return segments


def residue_lookup(st: gemmi.Structure) -> dict[tuple[int, str, int, str], ResidueInfo]:
    result = {}
    for _, _, residues in iter_chain_residues(st):
        for residue in residues:
            result[residue.resid.key()] = residue
    return result


def _address_to_resid(model_index: int, address, lookup: dict[tuple[int, str, int, str], ResidueInfo]) -> ResidueId:
    res_id = address.res_id
    resseq, icode = _seqid_parts(res_id.seqid)
    chain = address.chain_name
    key = (model_index, chain, resseq, icode)
    if key in lookup:
        return lookup[key].resid
    return ResidueId(model_index, chain, resseq, icode, getattr(res_id, "name", "") or "")


def existing_protein_annotation(st: gemmi.Structure) -> tuple[list[HelixRecord], list[SheetRecord]]:
    lookup = residue_lookup(st)
    helices: list[HelixRecord] = []
    for helix in st.helices:
        start = _address_to_resid(0, helix.start, lookup)
        end = _address_to_resid(0, helix.end, lookup)
        hclass = str(helix.pdb_helix_class).split(".")[-1].lower()
        if "310" in hclass or "3" in hclass and "10" in hclass:
            helix_type = "3_10"
        elif "pi" in hclass:
            helix_type = "pi"
        else:
            helix_type = "alpha"
        length = int(getattr(helix, "length", 0) or max(1, end.resseq - start.resseq + 1))
        helices.append(HelixRecord(helix_type, start, end, length, source="existing"))

    sheets: list[SheetRecord] = []
    for i_sheet, sheet in enumerate(st.sheets, start=1):
        strands: list[StrandRecord] = []
        for strand in sheet.strands:
            start = _address_to_resid(0, strand.start, lookup)
            end = _address_to_resid(0, strand.end, lookup)
            sense = int(getattr(strand, "sense", 0) or 0)
            strands.append(StrandRecord(start=start, end=end, sense=sense))
        sheets.append(SheetRecord(sheet_id=i_sheet, strands=strands, source="existing"))
    return helices, sheets
