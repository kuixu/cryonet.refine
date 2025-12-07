import string
from pathlib import Path
from dataclasses import replace
from collections.abc import Iterator
import numpy as np
from CryoNetRefine.data.write.pdb import to_pdb
from CryoNetRefine.data.write.mmcif import to_mmcif
from CryoNetRefine.data.types import Coords, Interface, StructureV2

def generate_tags() -> Iterator[str]:
    """Generate chain tags.

    Yields
    ------
    str
        The next chain tag

    """
    for i in range(1, 4):
        for j in range(len(string.ascii_uppercase) ** i):
            tag = ""
            for k in range(i):
                tag += string.ascii_uppercase[
                    j
                    // (len(string.ascii_uppercase) ** k)
                    % len(string.ascii_uppercase)
                ]
            yield tag



def write_refined_structure_pdb(predicted_coords, feats, data_dir, output_path):
        """Write refined structure and refinement info."""
        
        def _resolve_crop_bounds(structure, feats):
            """Return residue/atom span for current crop."""
            total_res = len(structure.residues)
            if total_res == 0:
                raise ValueError("Structure has no residues to export.")
            crop_start = int(feats.get("crop_start", 0) or 0)
            crop_start = max(0, min(crop_start, total_res - 1))
            crop_size = feats.get("crop_size")
            if crop_size is None:
                crop_end = total_res
            else:
                crop_end = min(total_res, max(crop_start + int(crop_size), crop_start + 1))
            first_residue = structure.residues[crop_start]
            last_residue = structure.residues[crop_end - 1]
            atom_start = int(first_residue["atom_idx"])
            atom_end = int(last_residue["atom_idx"]) + int(last_residue["atom_num"])
            return crop_start, crop_end, atom_start, atom_end
        
        # Ensure data_dir and output_path are Path objects
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        
        # Get record and pad masks
        record = feats["record"][0]
        pad_masks = feats["atom_pad_mask"].squeeze(0)
        
        # Load the structure from data_dir
        path = data_dir / f"{record.id}.npz"
        structure: StructureV2 = StructureV2.load(path)
        
        # Compute chain map with masked removed, to be used later
        chain_map = {}
        for i, mask in enumerate(structure.mask):
            if mask:
                chain_map[len(chain_map)] = i
        
        # Remove masked chains completely
        structure = structure.remove_invalid_chains()
        
        # Determine crop span
        crop_start, crop_end, atom_start, atom_end = _resolve_crop_bounds(structure, feats)
        crop_atom_count = atom_end - atom_start
        
        # Get refined coordinates
        model_coord = predicted_coords[0]  # Take first model

        # Unpad
        coord_unpad = model_coord[pad_masks.bool()]
        coord_unpad = coord_unpad.detach().cpu().numpy()  # Detach gradients for saving
        if coord_unpad.shape[0] != crop_atom_count:
            raise ValueError(
                f"Predicted atom count ({coord_unpad.shape[0]}) does not match crop span ({crop_atom_count})."
            )
        
        # Update atom/residue tables with crop awareness
        atoms = np.array(structure.atoms, copy=True)
        residues = np.array(structure.residues, copy=True)
        coords_field = np.array(structure.coords, copy=True)
        
        atoms["is_present"] = False
        residues["is_present"] = False
        atoms["coords"][atom_start:atom_end] = coord_unpad
        atoms["is_present"][atom_start:atom_end] = True
        coords_field["coords"][atom_start:atom_end] = coord_unpad
        residues["is_present"][crop_start:crop_end] = True
        
        # Update the structure (interfaces cleared for PDB writing)
        interfaces = np.array([], dtype=Interface)
        new_structure: StructureV2 = replace(
            structure,
            atoms=atoms,
            residues=residues,
            interfaces=interfaces,
            coords=coords_field,
        )
        
        # Update chain info
        chain_info = []
        for chain in new_structure.chains:
            old_chain_idx = chain_map[chain["asym_id"]]
            old_chain_info = record.chains[old_chain_idx]
            chain_has_atoms = (
                (chain["res_idx"] < crop_end)
                and (chain["res_idx"] + chain["res_num"] > crop_start)
            )
            new_chain_info = replace(
                old_chain_info,
                chain_id=int(chain["asym_id"]),
                valid=bool(chain_has_atoms),
            )
            chain_info.append(new_chain_info)
        
        # Save the PDB structure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(to_pdb(new_structure, plddts=None))


def write_refined_structure_pdb_by_crop(predicted_coords, feats, data_dir, output_path):
        """Write refined structure for the current crop only (not the full structure)."""
        
        def _resolve_crop_bounds(structure, feats):
            """Return residue/atom span for current crop."""
            total_res = len(structure.residues)
            if total_res == 0:
                raise ValueError("Structure has no residues to export.")
            crop_start = int(feats.get("crop_start", 0) or 0)
            crop_start = max(0, min(crop_start, total_res - 1))
            crop_size = feats.get("crop_size")
            if crop_size is None:
                crop_end = total_res
            else:
                crop_end = min(total_res, max(crop_start + int(crop_size), crop_start + 1))
            first_residue = structure.residues[crop_start]
            last_residue = structure.residues[crop_end - 1]
            atom_start = int(first_residue["atom_idx"])
            atom_end = int(last_residue["atom_idx"]) + int(last_residue["atom_num"])
            return crop_start, crop_end, atom_start, atom_end
        
        # Ensure data_dir and output_path are Path objects
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        
        # Get record and pad masks
        record = feats["record"][0]
        pad_masks = feats["atom_pad_mask"].squeeze(0)
        
        # Load the structure from data_dir
        path = data_dir / f"{record.id}.npz"
        structure: StructureV2 = StructureV2.load(path)
        
        # Compute chain map with masked removed, to be used later
        chain_map = {}
        for i, mask in enumerate(structure.mask):
            if mask:
                chain_map[len(chain_map)] = i
        
        # Remove masked chains completely
        structure = structure.remove_invalid_chains()
        
        # Determine crop span
        crop_start, crop_end, atom_start, atom_end = _resolve_crop_bounds(structure, feats)
        crop_atom_count = atom_end - atom_start
        
        # Get refined coordinates
        model_coord = predicted_coords[0]  # Take first model

        # Unpad
        coord_unpad = model_coord[pad_masks.bool()]
        coord_unpad = coord_unpad.detach().cpu().numpy()  # Detach gradients for saving
        if coord_unpad.shape[0] != crop_atom_count:
            raise ValueError(
                f"Predicted atom count ({coord_unpad.shape[0]}) does not match crop span ({crop_atom_count})."
            )
        
        # Extract ONLY the crop region from the structure
        # 1. Extract crop residues
        crop_residues = structure.residues[crop_start:crop_end].copy()
        
        # 2. Extract crop atoms
        crop_atoms = structure.atoms[atom_start:atom_end].copy()
        
        # 3. Update coordinates for crop atoms
        crop_atoms["coords"] = coord_unpad
        crop_atoms["is_present"] = True
        
        # 4. Prepare coords field (convert to Coords dtype)
        crop_coords = [(x,) for x in coord_unpad]
        crop_coords = np.array(crop_coords, dtype=Coords)
        
        # 5. Update crop residues
        crop_residues["is_present"] = True
        # Reindex atom_idx in residues to be relative to crop
        for i, res in enumerate(crop_residues):
            old_atom_idx = res["atom_idx"]
            res["atom_idx"] = old_atom_idx - atom_start
            res["atom_center"] = max(0, res["atom_center"] - atom_start)
            res["atom_disto"] = max(0, res["atom_disto"] - atom_start)
        
        # 6. Extract crop chains (only chains that have residues in crop)
        crop_chains = []
        new_chain_map = {}
        res_offset = 0
        atom_offset = 0
        
        for chain in structure.chains:
            chain_res_start = chain["res_idx"]
            chain_res_end = chain["res_idx"] + chain["res_num"]
            
            # Check if chain overlaps with crop
            if chain_res_end > crop_start and chain_res_start < crop_end:
                # Calculate overlap
                overlap_start = max(crop_start, chain_res_start)
                overlap_end = min(crop_end, chain_res_end)
                overlap_num = overlap_end - overlap_start
                
                # Get atom range for this chain's residues in crop
                first_res_in_crop = structure.residues[overlap_start]
                last_res_in_crop = structure.residues[overlap_end - 1]
                chain_atom_start = int(first_res_in_crop["atom_idx"])
                chain_atom_end = int(last_res_in_crop["atom_idx"]) + int(last_res_in_crop["atom_num"])
                chain_atom_num = chain_atom_end - chain_atom_start
                
                # Create new chain with updated indices
                new_chain = chain.copy()
                new_chain["res_idx"] = res_offset
                new_chain["res_num"] = overlap_num
                new_chain["atom_idx"] = atom_offset
                new_chain["atom_num"] = chain_atom_num
                
                crop_chains.append(new_chain)
                new_chain_map[chain["asym_id"]] = len(crop_chains) - 1
                
                res_offset += overlap_num
                atom_offset += chain_atom_num
        
        crop_chains = np.array(crop_chains) if crop_chains else structure.chains[:0]
        
        # 7. Extract crop bonds (only bonds within crop atoms)
        crop_bonds = []
        for bond in structure.bonds:
            atom1_idx = bond["atom_1"]
            atom2_idx = bond["atom_2"]
            if atom_start <= atom1_idx < atom_end and atom_start <= atom2_idx < atom_end:
                new_bond = bond.copy()
                new_bond["atom_1"] = atom1_idx - atom_start
                new_bond["atom_2"] = atom2_idx - atom_start
                crop_bonds.append(new_bond)
        crop_bonds = np.array(crop_bonds, dtype=structure.bonds.dtype) if crop_bonds else structure.bonds[:0]
        
        # 8. Create mask for crop chains
        crop_mask = np.ones(len(crop_chains), dtype=bool)
        
        # 9. Create ensemble (empty or single model)
        crop_ensemble = structure.ensemble[:0] if len(structure.ensemble) > 0 else structure.ensemble
        
        # 10. Create new structure with ONLY crop data
        interfaces = np.array([], dtype=Interface)
        crop_structure: StructureV2 = StructureV2(
            atoms=crop_atoms,
            bonds=crop_bonds,
            residues=crop_residues,
            chains=crop_chains,
            interfaces=interfaces,
            mask=crop_mask,
            coords=crop_coords,
            ensemble=crop_ensemble,
            pocket=None
        )
        
        # Save the PDB structure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(to_pdb(crop_structure, plddts=None))




def write_refined_structure(batch, refined_coords,data_dir,output_path):
        """Write refined structure and refinement info."""
        try:
            # Get record and pad masks
            record = batch["record"][0]
            pad_masks = batch["atom_pad_mask"].squeeze(0)   # [N_atoms_total], 0/1
            atom_pad = pad_masks.bool()
            # Load the structure from data_dir
            path = data_dir / f"{record.id}.npz"
            structure: StructureV2 = StructureV2.load(path)
            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i
            # Remove masked chains completely
            structure = structure.remove_invalid_chains()
            # Get refined coordinates
            model_coord = refined_coords[0]  # Take first model
            # Map from padded-atom indices -> structure atoms
            atom_indices = atom_pad.nonzero().squeeze(-1)           # indices of real atoms
            coord_unpad = model_coord[atom_indices]                 # [N_real_atoms, 3]
            coord_unpad = coord_unpad.detach().cpu().numpy()        # to numpy
            # Update atom table
            atoms = structure.atoms
            atoms["coords"] = coord_unpad
           
            # -------- NEW: use resolved mask to decide is_present --------
            # Prefer atom_resolved_mask if available
            if "atom_resolved_mask" in batch:
                atom_resolved = batch["atom_resolved_mask"].squeeze(0).bool()  # [N_atoms_total]
            # Fallback: derive atom_resolved from token_resolved_mask + atom_to_token
            elif "token_resolved_mask" in batch and "atom_to_token" in batch:
                token_resolved = batch["token_resolved_mask"].squeeze(0).bool()  # [N_tokens]
                atom_to_token = batch["atom_to_token"].squeeze(0)               # [N_atoms_total, N_tokens]
                atom_resolved = atom_to_token[:, token_resolved].any(dim=1)
            else:
                # If no resolved info, treat all non-pad atoms as resolved (old behavior)
                atom_resolved = atom_pad

            resolved_for_atoms = atom_resolved[atom_indices]        # [N_real_atoms]
            atoms["is_present"] = resolved_for_atoms.detach().cpu().numpy()
            # -------- END NEW -------- 

            # Prepare coordinates for structure (coords table)
            coords_field = [(x,) for x in coord_unpad]
            coords_field = np.array(coords_field, dtype=Coords)

            residues = structure.residues
            residues["is_present"] = True

            # Update the structure
            interfaces = np.array([], dtype=Interface)
            new_structure: StructureV2 = replace(
                structure,
                atoms=atoms,
                residues=residues,
                interfaces=interfaces,
                coords=coords_field,
            )
            # Update chain info
            chain_info = []
            for chain in new_structure.chains:
                old_chain_idx = chain_map[chain["asym_id"]]
                old_chain_info = record.chains[old_chain_idx]
                new_chain_info = replace(
                    old_chain_info,
                    chain_id=int(chain["asym_id"]),
                    valid=True,
                )
                chain_info.append(new_chain_info)
            # Save the CIF structure
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                f.write(to_mmcif(new_structure, plddts=None))
        except Exception as e:
            print(f"Error saving CIF structure: {e}")
            import traceback
            traceback.print_exc()
