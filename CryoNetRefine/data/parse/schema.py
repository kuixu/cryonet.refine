from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from Bio import Align
from rdkit.Chem.rdchem import Mol
from scipy.optimize import linear_sum_assignment
from CryoNetRefine.data.parse.mmcif import parse_mmcif
from CryoNetRefine.data.parse.pdb import parse_pdb

from CryoNetRefine.data.types import (

    ChainInfo,
    Record,
    StructureInfo,
    Target,
    TemplateInfo,
)

####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedRDKitBoundsConstraint:
    """A parsed RDKit bounds constraint object."""

    atom_idxs: tuple[int, int]
    is_bond: bool
    is_angle: bool
    upper_bound: float
    lower_bound: float


@dataclass(frozen=True)
class ParsedChiralAtomConstraint:
    """A parsed chiral atom constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_reference: bool
    is_r: bool


@dataclass(frozen=True)
class ParsedStereoBondConstraint:
    """A parsed stereo bond constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_check: bool
    is_e: bool


@dataclass(frozen=True)
class ParsedPlanarBondConstraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing5Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing6Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool
    rdkit_bounds_constraints: Optional[list[ParsedRDKitBoundsConstraint]] = None
    chiral_atom_constraints: Optional[list[ParsedChiralAtomConstraint]] = None
    stereo_bond_constraints: Optional[list[ParsedStereoBondConstraint]] = None
    planar_bond_constraints: Optional[list[ParsedPlanarBondConstraint]] = None
    planar_ring_5_constraints: Optional[list[ParsedPlanarRing5Constraint]] = None
    planar_ring_6_constraints: Optional[list[ParsedPlanarRing6Constraint]] = None


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: int
    residues: list[ParsedResidue]
    cyclic_period: int
    sequence: Optional[str] = None


@dataclass(frozen=True)
class Alignment:
    """A parsed alignment object."""

    query_st: int
    query_en: int
    template_st: int
    template_en: int


####################################################################################################
# HELPERS
####################################################################################################

def get_global_alignment_score(query: str, template: str) -> float:
    """Align a sequence to a template.

    Parameters
    ----------
    query : str
        The query sequence.
    template : str
        The template sequence.

    Returns
    -------
    float
        The global alignment score.

    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    score = aligner.align(query, template)[0].score
    return score


def get_local_alignments(query: str, template: str) -> list[Alignment]:
    """Align a sequence to a template.

    Parameters
    ----------
    query : str
        The query sequence.
    template : str
        The template sequence.

    Returns
    -------
    Alignment
        The alignment between the query and template.

    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aligner.open_gap_score = -1000
    aligner.extend_gap_score = -1000

    alignments = []
    for result in aligner.align(query, template):
        coordinates = result.coordinates
        alignment = Alignment(
            query_st=int(coordinates[0][0]),
            query_en=int(coordinates[0][1]),
            template_st=int(coordinates[1][0]),
            template_en=int(coordinates[1][1]),
        )
        alignments.append(alignment)

    return alignments


def get_template_records_from_search(
    template_id: str,
    chain_ids: list[str],
    sequences: dict[str, str],
    template_chain_ids: list[str],
    template_sequences: dict[str, str],
    force: bool = False,
    threshold: Optional[float] = None,
) -> list[TemplateInfo]:
    """Get template records from an alignment."""
    # Compute pairwise scores
    score_matrix = []
    for chain_id in chain_ids:
        row = []
        for template_chain_id in template_chain_ids:
            chain_seq = sequences[chain_id]
            template_seq = template_sequences[template_chain_id]
            score = get_global_alignment_score(chain_seq, template_seq)
            row.append(score)
        score_matrix.append(row)

    # Find optimal mapping
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    # Get alignment records
    template_records = []

    for row_idx, col_idx in zip(row_ind, col_ind):
        chain_id = chain_ids[row_idx]
        template_chain_id = template_chain_ids[col_idx]
        chain_seq = sequences[chain_id]
        template_seq = template_sequences[template_chain_id]
        alignments = get_local_alignments(chain_seq, template_seq)

        for alignment in alignments:
            template_record = TemplateInfo(
                name=template_id,
                query_chain=chain_id,
                query_st=alignment.query_st,
                query_en=alignment.query_en,
                template_chain=template_chain_id,
                template_st=alignment.template_st,
                template_en=alignment.template_en,
                force=force,
                threshold=threshold,
            )
            template_records.append(template_record)

    return template_records

def parse_refine_schema(
    cif_path: Path,
    ccd: Mapping[str, Mol],
    mol_dir: Optional[Path] = None,
) -> Target:
    """Parse a cif/pdb file for refinement.

    This simplified version:
    - Directly parses the input PDB/mmCIF as a template structure.
    - Uses the parsed template `StructureV2` as `Target.structure`.
    - Reuses sequences from the parsed template (no reconstruction from schema).
    - Does NOT rebuild a new Structure from `schema["sequences"]`.
    """

    if not cif_path.exists():
        raise ValueError(f"File not found: {cif_path}")
    # Parse the input as a template structure
    path_str = str(cif_path)
    if cif_path.suffix.lower() == ".pdb":
        parsed = parse_pdb(
            path_str,
            mols=ccd,
            moldir=mol_dir,
            use_assembly=False,
            compute_interfaces=False,
        )
    else:
        parsed = parse_mmcif(
            path_str,
            mols=ccd,
            moldir=mol_dir,
            use_assembly=False,
            compute_interfaces=False,
        )
    # Use the parsed template data directly
    data = parsed.data          # StructureV2
    sequences = parsed.sequences  # dict[chain_name -> sequence] if available

    templates = {cif_path.stem: data}
    extra_mols: dict[str, Mol] = {}

    # ---------- Construct template_records (each chain aligned to itself) ----------
    template_id = cif_path.stem
    template_records: list[TemplateInfo] = []
    if isinstance(sequences, dict):
        for chain in data.chains:
            chain_name = str(chain["name"]) 
            seq = sequences.get(chain_name)
            if not seq:
                continue
            seq_len = len(seq)
            template_records.append(
                TemplateInfo(
                    name=template_id,
                    query_chain=chain_name,
                    query_st=0,
                    query_en=seq_len,
                    template_chain=chain_name,
                    template_st=0,
                    template_en=seq_len,
                    force=False,
                    threshold=float("inf"),
                )
            )
    # -------------------------------------------------------------
    # Build metadata for Record
    struct_info = StructureInfo(num_chains=len(data.chains))
    chain_infos = []
    for chain in data.chains:
        chain_info = ChainInfo(
            chain_id=int(chain["asym_id"]),
            chain_name=chain["name"],
            mol_type=int(chain["mol_type"]),
            cluster_id=-1,
            num_residues=int(chain["res_num"]),
            valid=True,
            entity_id=int(chain["entity_id"]),
        )
        chain_infos.append(chain_info)

    record = Record(
        id=cif_path.stem,
        structure=struct_info,
        chains=chain_infos,
        interfaces=[],
        inference_options=None,
        templates=template_records,  
    )
    return Target(
        record=record,
        structure=data,
        sequences=sequences,
        templates=templates,
        extra_mols=extra_mols,
    )