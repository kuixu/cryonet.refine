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
import numpy as np

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

def _token(value: Any) -> str:
    if value is None:
        return "?"
    text = str(value).strip()
    if text == "":
        return "?"
    if re.search(r"\s", text) or "'" in text or '"' in text or text.startswith("_") or "#" in text:
        return "'" + text.replace("'", "''") + "'"
    return text


def _is_metal_element(element: str) -> bool:
    return str(element).upper() in {
        "ZN", "CA", "MG", "NA", "MN", "K", "FE", "CU", "CD", "HG", "NI", "CO", "SR", "CS",
        "PT", "BA", "TL", "PB", "SM", "AU", "RB", "YB", "LI", "MO", "LU", "CR", "OS", "GD",
        "TB", "LA", "AG", "HO", "GA", "CE", "W", "RU", "RE", "PR", "IR", "EU", "AL", "V",
        "PD", "U", "SB", "SE", "TE",
    }


def to_mmcif(
    structure: Structure | StructureV2,
    plddts: Optional[Any] = None,
    restraint_bonds: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Write an mmCIF string directly (no modelcif/ihm, no gemmi mmCIF writer),
    always emitting a complete `_atom_site` loop compatible with iotbx/mmtbx.
    """

    entity_map: dict[tuple[tuple[str, ...], int], int] = {}
    entity_id_counter = 1
    for chain in structure.chains:
        key = _entity_key(structure, chain)
        if key not in entity_map:
            entity_map[key] = entity_id_counter
            entity_id_counter += 1

    label_to_auth = _build_chain_id_maps(structure.chains)
    chain_fields = set(structure.chains.dtype.names or [])
    residue_fields = set(structure.residues.dtype.names or [])

    label_to_entity_id: dict[str, int] = {}
    for chain in structure.chains:
        long_id = str(chain["name"])
        label_to_entity_id[long_id] = int(entity_map[_entity_key(structure, chain)])
    lines: list[str] = []
    lines.append(f"data_model\n")
    lines.append("_entry.id model\n")
    lines.append("\n")

    lines.append("loop_\n")
    lines.append("_entity.id\n")
    lines.append("_entity.type\n")
    for (seq, mol_type), ent_id in sorted(entity_map.items(), key=lambda kv: kv[1]):
        if int(mol_type) == int(const.chain_type_ids["NONPOLYMER"]):
            ent_type = "non-polymer"
        else:
            ent_type = "polymer"
        lines.append(f"{ent_id} {ent_type}\n")
    lines.append("\n")

    polymer_rows: list[tuple[int, str, int, tuple[str, ...]]] = []
    for (seq, mol_type), ent_id in sorted(entity_map.items(), key=lambda kv: kv[1]):
        mt = int(mol_type)
        if mt == int(const.chain_type_ids["PROTEIN"]):
            poly_type = "polypeptide(L)"
        elif mt == int(const.chain_type_ids["DNA"]):
            poly_type = "polydeoxyribonucleotide"
        elif mt == int(const.chain_type_ids["RNA"]):
            poly_type = "polyribonucleotide"
        else:
            continue
        polymer_rows.append((ent_id, poly_type, mt, seq))

    if polymer_rows:
        lines.append("loop_\n")
        lines.append("_entity_poly.entity_id\n")
        lines.append("_entity_poly.type\n")
        for ent_id, poly_type, _, _ in polymer_rows:
            lines.append(f"{ent_id} {poly_type}\n")
        lines.append("\n")
        lines.append("loop_\n")
        lines.append("_entity_poly_seq.entity_id\n")
        lines.append("_entity_poly_seq.num\n")
        lines.append("_entity_poly_seq.mon_id\n")
        for ent_id, _, _, seq in polymer_rows:
            for i, mon in enumerate(seq, start=1):
                lines.append(f"{ent_id} {i} {str(mon)[:3]}\n")
        lines.append("\n")

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
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines.append("loop_\n")
    for t in tags:
        lines.append(f"{t}\n")

    atom_id = 1
    atom_meta_by_idx: dict[int, dict[str, Any]] = {}
    atom_lookup: dict[tuple[str, str, str, str], int] = {}
    for chain in structure.chains:
        long_id = str(chain["name"])
        if "auth_asym_id" in chain_fields:
            auth_id = str(chain["auth_asym_id"]).strip() or long_id
        else:
            auth_id = label_to_auth.get(long_id, long_id[:2])
        label_id = auth_id
        ent_id = label_to_entity_id.get(long_id, 1)

        is_nonpoly = int(chain["mol_type"]) == int(const.chain_type_ids["NONPOLYMER"])
        group_pdb = "HETATM" if is_nonpoly else "ATOM"

        res_start = int(chain["res_idx"])
        res_end = int(chain["res_idx"] + chain["res_num"])
        residues = structure.residues[res_start:res_end]

        for residue in residues:
            res_name_full = str(residue["name"])
            comp_id = res_name_full[:3]
            label_seq_id = int(residue["res_idx"]) + 1
            auth_seq_id = str(label_seq_id)
            if "auth_seq_id" in residue_fields:
                auth_seq_id = str(residue["auth_seq_id"]).strip() or auth_seq_id
            ins_code = "?"
            if "ins_code" in residue_fields:
                ins_code = str(residue["ins_code"]).strip() or "?"
            auth_comp_id = comp_id
            if "auth_comp_id" in residue_fields:
                auth_comp_id = str(residue["auth_comp_id"]).strip() or comp_id
            if is_nonpoly:
                # For ligands/ions, keep label_seq_id aligned with auth numbering
                # so each ion is represented as a distinct residue in downstream parsers.
                label_seq_id_token = auth_seq_id
            else:
                label_seq_id_token = str(label_seq_id)

            atom_start = int(residue["atom_idx"])
            atom_end = int(residue["atom_idx"] + residue["atom_num"])
            atoms = structure.atoms[atom_start:atom_end]

            for atom_offset, atom in enumerate(atoms):
                if "is_present" in atoms.dtype.names and not bool(atom["is_present"]):
                    continue

                atom_name = str(atom["name"]).strip()
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
                    label_id,
                    str(ent_id),
                    label_seq_id_token,
                    ins_code,
                    f"{x:.5f}",
                    f"{y:.5f}",
                    f"{z:.5f}",
                    "1",
                    f"{b:.2f}",
                    "?",
                    auth_seq_id,
                    auth_id,
                    "1",
                ]
                lines.append(" ".join(row) + "\n")
                atom_idx_global = atom_start + atom_offset
                atom_meta_by_idx[atom_idx_global] = {
                    "label_asym_id": label_id,
                    "label_comp_id": comp_id,
                    "label_seq_id": label_seq_id_token,
                    "auth_asym_id": auth_id,
                    "auth_comp_id": auth_comp_id,
                    "auth_seq_id": auth_seq_id,
                    "atom_name": atom_name,
                    "element": element,
                }
                atom_lookup[(auth_id, auth_seq_id, auth_comp_id, atom_name)] = atom_idx_global
                atom_id += 1

    struct_conn_rows: list[list[str]] = []
    seen_conn: set[tuple[int, int, str]] = set()

    def append_struct_conn(atom_idx1: int, atom_idx2: int, conn_type: str, distance: float | None = None) -> None:
        if atom_idx1 not in atom_meta_by_idx or atom_idx2 not in atom_meta_by_idx:
            return
        key = (min(atom_idx1, atom_idx2), max(atom_idx1, atom_idx2), conn_type)
        if key in seen_conn:
            return
        seen_conn.add(key)
        m1 = atom_meta_by_idx[atom_idx1]
        m2 = atom_meta_by_idx[atom_idx2]
        dist_val = "?"
        if distance is not None and np.isfinite(distance):
            dist_val = f"{float(distance):.3f}"
        p1_label_seq = m1["auth_seq_id"] 
        p2_label_seq = m2["auth_seq_id"] 
        struct_conn_rows.append(
            [
                f"{conn_type}{len(struct_conn_rows)+1}",
                conn_type,
                _token(m1["label_asym_id"]),
                _token(m1["label_comp_id"]),
                _token(p1_label_seq),
                _token(m1["atom_name"]),
                ".",
                _token(m1["auth_asym_id"]),
                _token(m1["auth_comp_id"]),
                _token(m1["auth_seq_id"]),
                _token(m2["label_asym_id"]),
                _token(m2["label_comp_id"]),
                _token(p2_label_seq),
                _token(m2["atom_name"]),
                ".",
                _token(m2["auth_asym_id"]),
                _token(m2["auth_comp_id"]),
                _token(m2["auth_seq_id"]),
                dist_val,
            ]
        )

    for bond in getattr(structure, "bonds", []):
        atom_idx1 = int(bond["atom_1"])
        atom_idx2 = int(bond["atom_2"])
        m1 = atom_meta_by_idx.get(atom_idx1)
        m2 = atom_meta_by_idx.get(atom_idx2)
        if m1 is None or m2 is None:
            continue
        coords1 = np.asarray(structure.atoms[atom_idx1]["coords"], dtype=float)
        coords2 = np.asarray(structure.atoms[atom_idx2]["coords"], dtype=float)
        dist = float(np.linalg.norm(coords1 - coords2))
        conn_type = "metalc" if (_is_metal_element(m1["element"]) or _is_metal_element(m2["element"])) else "covale"
        append_struct_conn(atom_idx1, atom_idx2, conn_type, dist)

    for rb in restraint_bonds or []:
        a1 = rb.get("atom1", {})
        a2 = rb.get("atom2", {})
        key1 = (
            str(a1.get("auth_asym_id", "")).strip(),
            str(a1.get("auth_seq_id", "")).strip(),
            str(a1.get("auth_comp_id", "")).strip(),
            str(a1.get("atom_name", "")).strip(),
        )
        key2 = (
            str(a2.get("auth_asym_id", "")).strip(),
            str(a2.get("auth_seq_id", "")).strip(),
            str(a2.get("auth_comp_id", "")).strip(),
            str(a2.get("atom_name", "")).strip(),
        )
        atom_idx1 = atom_lookup.get(key1)
        atom_idx2 = atom_lookup.get(key2)
        if atom_idx1 is None or atom_idx2 is None:
            continue
        m1 = atom_meta_by_idx.get(atom_idx1)
        m2 = atom_meta_by_idx.get(atom_idx2)
        if m1 is None or m2 is None:
            continue
        conn_type = "metalc" if (_is_metal_element(m1["element"]) or _is_metal_element(m2["element"])) else "covale"
        dist = rb.get("distance_ideal")
        append_struct_conn(atom_idx1, atom_idx2, conn_type, float(dist) if dist is not None else None)

    if struct_conn_rows:
        lines.append("#\n")
        lines.append("loop_\n")
        struct_conn_tags = [
            "_struct_conn.id",
            "_struct_conn.conn_type_id",
            "_struct_conn.ptnr1_label_asym_id",
            "_struct_conn.ptnr1_label_comp_id",
            "_struct_conn.ptnr1_label_seq_id",
            "_struct_conn.ptnr1_label_atom_id",
            "_struct_conn.pdbx_ptnr1_label_alt_id",
            "_struct_conn.ptnr1_auth_asym_id",
            "_struct_conn.ptnr1_auth_comp_id",
            "_struct_conn.ptnr1_auth_seq_id",
            "_struct_conn.ptnr2_label_asym_id",
            "_struct_conn.ptnr2_label_comp_id",
            "_struct_conn.ptnr2_label_seq_id",
            "_struct_conn.ptnr2_label_atom_id",
            "_struct_conn.pdbx_ptnr2_label_alt_id",
            "_struct_conn.ptnr2_auth_asym_id",
            "_struct_conn.ptnr2_auth_comp_id",
            "_struct_conn.ptnr2_auth_seq_id",
            "_struct_conn.pdbx_dist_value",
        ]
        for tag in struct_conn_tags:
            lines.append(f"{tag}\n")
        for row in struct_conn_rows:
            lines.append(" ".join(row) + "\n")

    return "".join(lines)