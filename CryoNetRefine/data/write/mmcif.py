from __future__ import annotations
from typing import Optional, Any
from itertools import product
import re
from CryoNetRefine.data import const
from CryoNetRefine.data.types import Structure, StructureV2
import io
import re
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor

def to_mmcif_old(
    structure: Structure,
    plddts: Optional[Tensor] = None,
) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output MMCIF file

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Group entities by (sequence, mol_type) to avoid duplicate Entity objects
    # ihm/modelcif library treats entities with same sequence as duplicates
    sequence_to_entity_obj = {}  # Maps (seq_tuple, mol_type) -> Entity object
    entities_map = {}  # Maps chain_idx -> Entity object

    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]
        seq_tuple = tuple(sequence)
        cache_key = (seq_tuple, mol_type)

        if cache_key in sequence_to_entity_obj:
            # Reuse existing Entity object for same sequence
            model_e = sequence_to_entity_obj[cache_key]
        else:
            # Create new Entity object
            if mol_type == const.chain_type_ids["PROTEIN"]:
                alphabet = ihm.LPeptideAlphabet()
                chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
            elif mol_type == const.chain_type_ids["DNA"]:
                alphabet = ihm.DNAAlphabet()
                chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
            elif mol_type == const.chain_type_ids["RNA"]:
                alphabet = ihm.RNAAlphabet()
                chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
            elif len(sequence) > 1:
                alphabet = {}
                chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
            else:
                alphabet = {}
                chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            model_e = Entity(seq)
            sequence_to_entity_obj[cache_key] = model_e

        # Map all chains of this entity to the Entity object
        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        chain_tag = str(chain["name"])
        entity = entities_map[chain_idx]
        if entity.type == "water":
            asym = ihm.WaterAsymUnit(
                entity,
                1,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        else:
            asym = AsymUnit(
                entity,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Index into plddt tensor for current residue.
            res_num = 0
            # Tracks non-ligand plddt tensor indices,
            # Initializing to -1 handles case where ligand is resnum 0
            prev_polymer_resnum = -1
            # Tracks ligand indices.
            ligand_index_offset = 0

            # Add all atom sites.
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    res_name = str(residue["name"])
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        atom_name = str(atom["name"])
                        atom_key = re.sub(r"\d", "", atom_name)
                        if atom_key in const.ambiguous_atoms:
                            if isinstance(const.ambiguous_atoms[atom_key], str):
                                element = const.ambiguous_atoms[atom_key]
                            elif res_name in const.ambiguous_atoms[atom_key]:
                                element = const.ambiguous_atoms[atom_key][res_name]
                            else:
                                element = const.ambiguous_atoms[atom_key]["*"]
                        else:
                            element = atom_key[0]
     
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]

                        if record_type != "HETATM":
                            # # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                            # biso = (
                            #     0.00 # change from 100.00 to 0.00 by huangfuyao
                            #     if plddts is None
                            #     else round(
                            #         plddts[res_num + ligand_index_offset].item() * 100,
                            #         3,
                            #     )
                            # )
                                                        # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                            biso = atom['bfactor']
                            prev_polymer_resnum = res_num
                        else:
                            # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                            ligand_index_offset += 1
                            # biso = (
                            #     0.00 # change from 100.00 to 0.00 by huangfuyao
                            #     if plddts is None
                            #     else round(
                            #         plddts[
                            #             prev_polymer_resnum + ligand_index_offset
                            #         ].item()
                            #         * 100,
                            #         3,
                            #     )
                            # )
                            biso = atom['bfactor']
                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=atom_name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    if record_type != "HETATM":
                        res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            prev_polymer_resnum = (
                -1
            )  # -1 handles case where ligand is the first residue
            ligand_index_offset = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1

                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]

                    if record_type != "HETATM":
                        # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[res_num + ligand_index_offset].item() * 100,
                                    3,
                                ),
                            )
                        )
                        prev_polymer_resnum = res_num
                    else:
                        # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[
                                        prev_polymer_resnum
                                        + ligand_index_offset
                                        + 1 : prev_polymer_resnum
                                        + ligand_index_offset
                                        + residue["atom_num"]
                                        + 1
                                    ]
                                    .mean()
                                    .item()
                                    * 100,
                                    2,
                                ),
                            )
                        )
                        ligand_index_offset += residue["atom_num"]

                    if record_type != "HETATM":
                        res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)
    ihm.dumper.set_line_wrap(False)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()

def _short_id_generator():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for c in alphabet:
        yield c
    for a, b in product(alphabet, repeat=2):
        yield a + b


def _build_chain_id_maps(chains):
    """
    Build:
      - label_asym_id  (long id, your original)
      - auth_asym_id   (short id <=2 chars)
    """
    gen = _short_id_generator()
    mapping: dict[str, str] = {}
    used = {str(c["name"]) for c in chains if len(str(c["name"])) <= 2}

    def next_free() -> str:
        while True:
            cid = next(gen)
            if cid not in used:
                used.add(cid)
                return cid

    for chain in chains:
        long_id = str(chain["name"])
        if len(long_id) <= 2:
            mapping[long_id] = long_id
        else:
            if long_id not in mapping:
                mapping[long_id] = next_free()
    return mapping


def _entity_key(structure, chain):
    """
    Group entity by (sequence tuple, mol_type)
    """
    res_start = chain["res_idx"]
    res_end = chain["res_idx"] + chain["res_num"]
    residues = structure.residues[res_start:res_end]
    seq = tuple(str(r["name"]) for r in residues)
    return (seq, chain["mol_type"])


def _infer_element_symbol(atom_name: str, res_name: str) -> str:
    """
    Infer element symbol from atom name, using the same ambiguous atom mapping
    as the PDB writer.
    """
    atom_key = re.sub(r"\d", "", atom_name.strip())
    if atom_key in const.ambiguous_atoms:
        v = const.ambiguous_atoms[atom_key]
        if isinstance(v, str):
            element = v
        elif res_name in v:
            element = v[res_name]
        else:
            element = v["*"]
    else:
        element = atom_key[0] if atom_key else "C"
    return str(element).upper()

def to_mmcif(
    structure: Structure | StructureV2,
    plddts: Optional[Any] = None,
) -> str:
    """
    Write an mmCIF string directly (no modelcif/ihm, no gemmi mmCIF writer),
    always emitting a complete `_atom_site` loop compatible with iotbx/mmtbx.

    Key semantics:
    - `_atom_site.label_asym_id` uses your original (possibly long) chain IDs.
    - `_atom_site.auth_asym_id` uses a short unique ID (<=2 chars) for cctbx.
    """

    # -------------------------
    # 1️⃣ Build entity grouping (sequence, mol_type) -> entity_id
    # -------------------------
    entity_map: dict[tuple[tuple[str, ...], int], int] = {}
    entity_id_counter = 1
    for chain in structure.chains:
        key = _entity_key(structure, chain)
        if key not in entity_map:
            entity_map[key] = entity_id_counter
            entity_id_counter += 1

    # label_asym_id -> short auth_asym_id
    label_to_auth = _build_chain_id_maps(structure.chains)

    # label_asym_id -> entity_id
    label_to_entity_id: dict[str, int] = {}
    for chain in structure.chains:
        long_id = str(chain["name"])
        label_to_entity_id[long_id] = int(entity_map[_entity_key(structure, chain)])

    # -------------------------
    # 2️⃣ Emit minimal header
    # -------------------------
    lines: list[str] = []
    lines.append("data_model\n")
    lines.append("_entry.id model\n")
    lines.append("\n")
    # IMPORTANT (cryo-EM): do NOT emit crystallographic unit cell / symmetry.
    # A fake small unit cell (e.g. 1x1x1) makes mmtbx/pdb_interpretation reject the
    # model ("Unit cell volume is incompatible with number of atoms").
    # When crystal symmetry is absent, callers typically use:
    #   substitute_non_crystallographic_unit_cell_if_necessary=True
    # which will create a large non-crystallographic unit cell automatically.

    # -------------------------
    # 3️⃣ _atom_site loop (iotbx-required)
    # -------------------------
    tags = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines.append("loop_\n")
    for t in tags:
        lines.append(f"{t}\n")

    atom_id = 1
    for chain in structure.chains:
        long_id = str(chain["name"])
        auth_id = label_to_auth.get(long_id, long_id[:2])
        ent_id = label_to_entity_id.get(long_id, 1)

        is_nonpoly = int(chain["mol_type"]) == int(const.chain_type_ids["NONPOLYMER"])
        group_pdb = "HETATM" if is_nonpoly else "ATOM"

        res_start = int(chain["res_idx"])
        res_end = int(chain["res_idx"] + chain["res_num"])
        residues = structure.residues[res_start:res_end]

        for residue in residues:
            res_name_full = str(residue["name"])
            comp_id = "LIG" if is_nonpoly else res_name_full[:3]
            seq_id = int(residue["res_idx"]) + 1

            atom_start = int(residue["atom_idx"])
            atom_end = int(residue["atom_idx"] + residue["atom_num"])
            atoms = structure.atoms[atom_start:atom_end]

            for atom in atoms:
                if "is_present" in atoms.dtype.names and not bool(atom["is_present"]):
                    continue

                atom_name = str(atom["name"]).strip()
                if atom_name == "OXT":
                    continue

                coords = atom["coords"]
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                b = float(atom["bfactor"]) if "bfactor" in atoms.dtype.names else 1.0
                element = _infer_element_symbol(atom_name, res_name_full)

                row = [
                    group_pdb,
                    str(atom_id),
                    element,
                    atom_name,
                    ".",
                    comp_id,
                    long_id,
                    str(ent_id),
                    str(seq_id),
                    "?",
                    f"{x:.5f}",
                    f"{y:.5f}",
                    f"{z:.5f}",
                    "1",
                    f"{b:.2f}",
                    str(seq_id),
                    comp_id,
                    auth_id,
                    "1",
                ]
                lines.append(" ".join(row) + "\n")
                atom_id += 1

    return "".join(lines)