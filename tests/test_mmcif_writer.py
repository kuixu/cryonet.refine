from types import SimpleNamespace

import gemmi
import numpy as np

from CryoNetRefine.data import const
from CryoNetRefine.data.write.mmcif import to_mmcif


def _structure_with_repeated_polymer_entity():
    chains = np.array(
        [
            ("A_1", const.chain_type_ids["PROTEIN"], 0, 1, "A"),
            ("B_1", const.chain_type_ids["PROTEIN"], 1, 1, "B"),
            ("B_2", const.chain_type_ids["NONPOLYMER"], 2, 1, "B"),
        ],
        dtype=[
            ("name", "<U5"),
            ("mol_type", "i1"),
            ("res_idx", "i4"),
            ("res_num", "i4"),
            ("auth_asym_id", "<U16"),
        ],
    )
    residues = np.array(
        [
            ("ALA", 0, 0, 1, "1", "", "ALA"),
            ("ALA", 0, 1, 1, "1", "", "ALA"),
            ("ZN", 0, 2, 1, "1", "", "ZN"),
        ],
        dtype=[
            ("name", "<U5"),
            ("res_idx", "i4"),
            ("atom_idx", "i4"),
            ("atom_num", "i4"),
            ("auth_seq_id", "<U16"),
            ("ins_code", "<U4"),
            ("auth_comp_id", "<U8"),
        ],
    )
    atoms = np.array(
        [
            ("CA", (0.0, 0.0, 0.0), True, 20.0),
            ("CA", (1.0, 0.0, 0.0), True, 20.0),
            ("ZN", (2.0, 0.0, 0.0), True, 20.0),
        ],
        dtype=[
            ("name", "<U4"),
            ("coords", "3f4"),
            ("is_present", "?"),
            ("bfactor", "f4"),
        ],
    )
    bonds = np.array([], dtype=[("atom_1", "i4"), ("atom_2", "i4")])
    return SimpleNamespace(chains=chains, residues=residues, atoms=atoms, bonds=bonds)


def test_deposition_entity_and_chain_categories_are_complete():
    block = gemmi.cif.read_string(to_mmcif(_structure_with_repeated_polymer_entity())).sole_block()

    entity_poly = block.get_mmcif_category("_entity_poly.")
    assert entity_poly["pdbx_strand_id"] == ["A,B"]
    assert entity_poly["pdbx_seq_one_letter_code"] == ["A"]
    assert entity_poly["pdbx_seq_one_letter_code_can"] == ["A"]

    struct_asym = block.get_mmcif_category("_struct_asym.")
    assert struct_asym["id"] == ["A", "B", "C"]
    assert struct_asym["entity_id"] == ["1", "1", "2"]

    atom_site = block.get_mmcif_category("_atom_site.")
    assert atom_site["label_asym_id"] == ["A", "B", "C"]
    assert atom_site["label_entity_id"] == ["1", "1", "2"]
    assert atom_site["auth_asym_id"] == ["A", "B", "B"]
    assert atom_site["label_seq_id"] == ["1", "1", False]
