from typing import Optional
from rdkit.Chem.rdchem import Mol
import gemmi
from pathlib import Path
from tempfile import NamedTemporaryFile
from CryoNetRefine.data.utils import update_status
from CryoNetRefine.data.parse.mmcif import parse_mmcif, ParsedStructure

def sanitize_models(st: gemmi.Structure, path: Path) -> gemmi.Structure:
    valid_model = None
    for model in st:
        if len(model) > 0:
            valid_model = model
            break
    new_st = gemmi.Structure()
    new_st.name = st.name
    new_st.cell = st.cell
    new_st.spacegroup_hm = st.spacegroup_hm
    new_st.add_model(valid_model.clone())
    new_st.setup_entities()

    return new_st

def parse_pdb(
    path: str,
    mols: Optional[dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    compute_interfaces: bool = True,
    auto_metal_restraints: bool = True,
    metal_restraint_distance_strategy: str = "input",
    metal_coordination_cutoff: float = 3.0,
) -> ParsedStructure:
    with NamedTemporaryFile(suffix=".cif") as tmp_cif_file:
        tmp_cif_path = tmp_cif_file.name
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()

        if len(structure) > 1:
            update_status(Path(path).parent, {'msg': f"Refining...Warning: Multi-model PDB (with MODEL-ENDMDL) detected, only the first valid model will be refined. We suggest you to combine all models into a single model.", 'error_code':0, "progress": 10})
            structure = sanitize_models(structure, Path(path))
        subchain_counts: dict[str, int] = {}
        subchain_renaming: dict[str, str] = {}
        used_new_subchains: set[str] = set()
        for chain in structure[0]:
            subchain_counts[chain.name] = 0
            for res in chain:
                if res.subchain not in subchain_renaming:
                    # Use a collision-safe synthetic subchain id.
                    # `chain.name + N` can collide with real chain names like C1/C11,
                    # which later corrupts subchain -> auth_asym_id mapping.
                    while True:
                        subchain_counts[chain.name] += 1
                        candidate = f"{chain.name}_{subchain_counts[chain.name]}"
                        if candidate not in used_new_subchains:
                            subchain_renaming[res.subchain] = candidate
                            used_new_subchains.add(candidate)
                            break
                res.subchain = subchain_renaming[res.subchain]
        for entity in structure.entities:
            entity.subchains = [subchain_renaming.get(subchain, subchain) for subchain in entity.subchains]

        doc = structure.make_mmcif_document()
        doc.write_file(tmp_cif_path)

        return parse_mmcif(
            path=tmp_cif_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces,
            auto_metal_restraints=auto_metal_restraints,
            metal_restraint_distance_strategy=metal_restraint_distance_strategy,
            metal_coordination_cutoff=metal_coordination_cutoff,
        )