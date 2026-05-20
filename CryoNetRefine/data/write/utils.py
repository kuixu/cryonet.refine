import string
import json
from pathlib import Path
from dataclasses import replace
from collections.abc import Iterator
from collections import defaultdict
import numpy as np
from CryoNetRefine.data.write.pdb import to_pdb
from CryoNetRefine.data.write.mmcif import to_mmcif
from CryoNetRefine.data.types import Coords, Interface, StructureV2, Residue, Chain

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


def _load_restraint_bonds(data_dir: Path, record_id: str) -> list[dict]:
    constraints_path = data_dir.parent / "constraints" / f"{record_id}.json"
    if not constraints_path.exists():
        return []
    try:
        payload = json.loads(constraints_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    bonds = payload.get("bonds", [])
    if isinstance(bonds, list):
        return [b for b in bonds if isinstance(b, dict)]
    return []


def _tensor_or_array_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _build_atom_to_residue_index(structure: StructureV2) -> np.ndarray:
    atom_to_residue = np.full(len(structure.atoms), -1, dtype=np.int64)
    for residue_idx, residue in enumerate(structure.residues):
        atom_start = int(residue["atom_idx"])
        atom_end = atom_start + int(residue["atom_num"])
        atom_to_residue[atom_start:atom_end] = residue_idx
    return atom_to_residue


def _build_atom_to_chain_index(structure: StructureV2) -> np.ndarray:
    atom_to_chain = np.full(len(structure.atoms), -1, dtype=np.int64)
    for chain_idx, chain in enumerate(structure.chains):
        atom_start = int(chain["atom_idx"])
        atom_end = atom_start + int(chain["atom_num"])
        atom_to_chain[atom_start:atom_end] = chain_idx
    return atom_to_chain


def _build_molecule_aware_crop_from_atom_indices(
    structure: StructureV2,
    atom_indices: np.ndarray,
    coord_unpad: np.ndarray,
) -> StructureV2:
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    if atom_indices.ndim != 1:
        atom_indices = atom_indices.reshape(-1)
    if atom_indices.size == 0:
        raise ValueError("Molecule-aware crop has no selected atom indices.")
    if coord_unpad.shape[0] != atom_indices.size:
        raise ValueError(
            f"[molecule_aware] Predicted atom count ({coord_unpad.shape[0]}) "
            f"does not match selected atom indices ({atom_indices.size})."
        )
    if np.any(atom_indices < 0) or np.any(atom_indices >= len(structure.atoms)):
        raise ValueError("Molecule-aware crop contains atom indices outside the structure atom table.")

    atom_to_residue = _build_atom_to_residue_index(structure)
    atom_to_chain = _build_atom_to_chain_index(structure)

    selected_atoms_by_residue: dict[int, list[int]] = defaultdict(list)
    for atom_idx in atom_indices.tolist():
        residue_idx = int(atom_to_residue[atom_idx])
        if residue_idx < 0:
            raise ValueError(f"Atom index {atom_idx} is not assigned to any residue.")
        selected_atoms_by_residue[residue_idx].append(int(atom_idx))

    selected_residues_by_chain: dict[int, list[int]] = defaultdict(list)
    for residue_idx in sorted(selected_atoms_by_residue):
        orig_residue = structure.residues[residue_idx]
        chain_idx = int(atom_to_chain[int(orig_residue["atom_idx"])])
        if chain_idx < 0:
            raise ValueError(f"Residue index {residue_idx} is not assigned to any chain.")
        selected_residues_by_chain[chain_idx].append(residue_idx)

    coord_by_atom = {int(atom_idx): coord_unpad[pos] for pos, atom_idx in enumerate(atom_indices.tolist())}
    crop_atoms_list = []
    crop_residues_list = []
    crop_chains_list = []
    atom_offset = 0
    res_offset = 0

    for chain_idx, residue_indices in sorted(selected_residues_by_chain.items()):
        orig_chain = structure.chains[chain_idx]
        chain_res_start = res_offset
        chain_atom_start = atom_offset

        for residue_idx in residue_indices:
            orig_residue = structure.residues[residue_idx]
            selected_atom_indices = sorted(selected_atoms_by_residue[residue_idx])
            atoms_seg = np.array(structure.atoms[selected_atom_indices], copy=True)
            atoms_seg["coords"] = np.array([coord_by_atom[i] for i in selected_atom_indices], dtype=atoms_seg["coords"].dtype)

            res = np.array(orig_residue, copy=True)
            res["atom_idx"] = atom_offset
            res["atom_num"] = len(atoms_seg)
            res["atom_center"] = atom_offset
            res["atom_disto"] = atom_offset
            res["is_present"] = bool(np.any(atoms_seg["is_present"]))

            crop_atoms_list.append(atoms_seg)
            crop_residues_list.append(res)
            atom_offset += len(atoms_seg)
            res_offset += 1

        new_chain = np.array(orig_chain, copy=True)
        new_chain["res_idx"] = chain_res_start
        new_chain["res_num"] = res_offset - chain_res_start
        new_chain["atom_idx"] = chain_atom_start
        new_chain["atom_num"] = atom_offset - chain_atom_start
        crop_chains_list.append(new_chain)

    crop_atoms = np.concatenate(crop_atoms_list).astype(structure.atoms.dtype, copy=False)
    crop_residues = np.array(crop_residues_list, dtype=Residue)
    crop_chains = np.array(crop_chains_list, dtype=Chain)
    crop_coords = np.array([(x,) for x in crop_atoms["coords"]], dtype=Coords)
    crop_ensemble = (
        np.array([(0, len(crop_atoms))], dtype=structure.ensemble.dtype)
        if len(structure.ensemble) > 0
        else structure.ensemble
    )

    return StructureV2(
        atoms=crop_atoms,
        bonds=structure.bonds[:0],
        residues=crop_residues,
        chains=crop_chains,
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(len(crop_chains), dtype=bool),
        coords=crop_coords,
        ensemble=crop_ensemble,
        pocket=None,
    )



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

def write_refined_structure(batch, refined_coords,data_dir,output_path):
        """Write refined structure and refinement info."""
        try:
            # Ensure data_dir and output_path are Path objects
            if not isinstance(data_dir, Path):
                data_dir = Path(data_dir)
            if not isinstance(output_path, Path):
                output_path = Path(output_path)
            
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
            restraint_bonds = _load_restraint_bonds(data_dir, record.id)
            # Save the CIF structure
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                # f.write(to_pdb(new_structure, plddts=None))
                f.write(to_mmcif(new_structure, plddts=None, restraint_bonds=restraint_bonds))
        except Exception as e:
            print(f"Error saving CIF structure: {e}")
            import traceback
            traceback.print_exc()
def write_refined_structure_cif_by_crop(predicted_coords, feats, data_dir, output_path):
    """Write refined structure for the current crop only (not the full structure).

    - For legacy contiguous crops (no molecule-aware), use a single
      [crop_start, crop_end) residue span.
    - For molecule-aware crops (crop_type == 'molecule_aware'), reconstruct
      a mini-StructureV2 using (asym_id, residue_index) pairs and
      crop_metadata['sequences'], so that only the residues actually present
      in this crop are exported (no mixed molecule types, no duplicate atom labels).
    """
    # Ensure data_dir and output_path are Path objects
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    # Get record and crop atom mask (inside this crop)
    record = feats["record"][0]
    pad_masks = feats["atom_pad_mask"].squeeze(0)  # [N_cropped_atoms] bool

    # Load the full structure
    path = data_dir / f"{record.id}.npz"
    structure_full: StructureV2 = StructureV2.load(path)

    # Remove masked chains completely (same as featurizer)
    chain_map = {}
    for i, mask in enumerate(structure_full.mask):
        if mask:
            chain_map[len(chain_map)] = i
    structure = structure_full.remove_invalid_chains()

    # Predicted coordinates for this crop (in the same order as cropped atoms)
    model_coord = predicted_coords[0]  # [N_cropped_atoms_padded, 3]
    coord_unpad = model_coord[pad_masks.bool()]
    coord_unpad = coord_unpad.detach().cpu().numpy()  # [N_cropped_atoms, 3]

    crop_type = feats.get("crop_type", None)

    # ==========================================================
    # 1) Molecule-aware crops: build structure from crop_metadata['sequences']
    # ==========================================================
    if crop_type == "molecule_aware":
        crop_meta = feats["crop_metadata"]
        global_atom_indices = feats.get("global_atom_indices")
        if global_atom_indices is None:
            global_atom_indices = crop_meta.get("global_atom_indices")

        if global_atom_indices is not None:
            atom_indices = _tensor_or_array_to_numpy(global_atom_indices).astype(np.int64).reshape(-1)
            crop_structure = _build_molecule_aware_crop_from_atom_indices(
                structure=structure_full,
                atom_indices=atom_indices,
                coord_unpad=coord_unpad,
            )
            restraint_bonds = _load_restraint_bonds(data_dir, record.id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                f.write(to_mmcif(crop_structure, plddts=None, restraint_bonds=restraint_bonds))
            return
        seq_dict = crop_meta["sequences"]  # {asym_id: [local_token_pos, ...]}
        # Token-level info for this crop (already cropped)
        token_mask = feats["token_pad_mask"].squeeze(0).bool()         # [N_crop_tokens]
        asym_ids_all = feats["asym_id"].squeeze(0)[token_mask]         # [N_crop_tokens]
        res_indices_all = feats["residue_index"].squeeze(0)[token_mask]  # [N_crop_tokens]

        # (asym_id, local_res_idx) -> global residue index in StructureV2
        asym_res_to_global = {}
        for chain in structure.chains:
            asym = int(chain["asym_id"])
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for gidx in range(res_start, res_end):
                res = structure.residues[gidx]
                local_res_idx = int(res["res_idx"])
                asym_res_to_global[(asym, local_res_idx)] = gidx

        # Collect global residue indices per asym_id, driven by crop_metadata['sequences']
        chain_to_global_res = defaultdict(set)
        for asym_id_in_meta, local_positions in seq_dict.items():
            asym_id_in_meta = int(asym_id_in_meta)
            if not local_positions:
                continue
            local_positions = np.array(local_positions, dtype=np.int64)
            # local_positions are positions in this crop's token array
            res_idx_for_chain = res_indices_all[local_positions].cpu().tolist()
            for r_idx in res_idx_for_chain:
                key = (asym_id_in_meta, int(r_idx))
                if key in asym_res_to_global:
                    gidx = asym_res_to_global[key]
                    chain_to_global_res[asym_id_in_meta].add(gidx)

        crop_residues_list = []
        crop_atoms_list = []
        crop_chains_list = []

        atom_offset = 0
        res_offset = 0

        # Keep original chain order, but only include selected residues per chain
        for chain in structure.chains:
            asym = int(chain["asym_id"])
            if asym not in chain_to_global_res:
                continue

            selected_globals = sorted(chain_to_global_res[asym])
            if not selected_globals:
                continue

            chain_res_start_new = res_offset
            chain_atom_start_new = atom_offset

            # Build residues (renumber res_idx locally within this chain)
            for local_idx, gidx in enumerate(selected_globals):
                orig_res = structure.residues[gidx]
                res = np.array(orig_res, copy=True)

                orig_atom_start = int(orig_res["atom_idx"])
                orig_atom_num = int(orig_res["atom_num"])

                # Copy atoms for this residue (keeps original atom ordering)
                atoms_seg = np.array(
                    structure.atoms[orig_atom_start: orig_atom_start + orig_atom_num],
                    copy=True,
                )
                crop_atoms_list.append(atoms_seg)

                # Update residue atom indices relative to the new concatenated array
                res["atom_idx"] = atom_offset
                res["atom_center"] = (
                    max(0, int(orig_res["atom_center"]) - orig_atom_start)
                    + atom_offset
                )
                res["atom_disto"] = (
                    max(0, int(orig_res["atom_disto"]) - orig_atom_start)
                    + atom_offset
                )
                # Renumber residue index to be local within this chain
                # res["res_idx"] = local_idx
                res["res_idx"] = orig_res["res_idx"]
                crop_residues_list.append(res)

                atom_offset += orig_atom_num
                res_offset += 1

            chain_res_num_new = len(selected_globals)
            chain_atom_num_new = atom_offset - chain_atom_start_new

            new_chain = np.array(chain, copy=True)
            new_chain["res_idx"] = chain_res_start_new
            new_chain["res_num"] = chain_res_num_new
            new_chain["atom_idx"] = chain_atom_start_new
            new_chain["atom_num"] = chain_atom_num_new

            crop_chains_list.append(new_chain)

        if not crop_atoms_list:
            raise ValueError(
                "Molecule-aware crop produced no residues/atoms for this structure."
            )

        # Concatenate atoms & residues
        crop_atoms = np.concatenate(crop_atoms_list).astype(
            structure.atoms.dtype, copy=False
        )
        crop_residues = np.array(crop_residues_list, dtype=Residue)
        crop_chains = np.array(crop_chains_list, dtype=Chain)

        # Crop-first featurization keeps every atom slot referenced by selected
        # tokens, including slots marked is_present=False in the input/template.
        # The model therefore returns one coordinate per selected atom slot, not
        # one coordinate per present atom. Keep original is_present flags so the
        # mmCIF writer still suppresses unresolved atoms.
        crop_atoms = crop_atoms.copy()
        if coord_unpad.shape[0] != crop_atoms.shape[0]:
            raise ValueError(
                f"[molecule_aware] Predicted atom count ({coord_unpad.shape[0]}) "
                f"does not match selected atom slots ({crop_atoms.shape[0]})."
            )
        crop_atoms["coords"] = crop_atoms["coords"].copy()
        crop_atoms["coords"] = coord_unpad
        crop_coords = np.array([(x,) for x in crop_atoms["coords"]], dtype=Coords)
        crop_residues["is_present"] = True

        # No bonds / ensemble needed for geometry RMSD; mmtbx will infer topology
        crop_bonds = structure.bonds[:0]
        crop_mask = np.ones(len(crop_chains), dtype=bool)
        crop_ensemble = (
            structure.ensemble[:0]
            if len(structure.ensemble) > 0
            else structure.ensemble
        )
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
            pocket=None,
        )
        restraint_bonds = _load_restraint_bonds(data_dir, record.id)
        # Write PDB
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(to_mmcif(crop_structure, plddts=None, restraint_bonds=restraint_bonds))

        return  # Molecule-aware path completed




def write_refined_structure_cif(batch, refined_coords,data_dir,output_path):
        """Write refined structure and refinement info."""
        try:
            # Ensure data_dir and output_path are Path objects
            if not isinstance(data_dir, Path):
                data_dir = Path(data_dir)
            if not isinstance(output_path, Path):
                output_path = Path(output_path)
            
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
            restraint_bonds = _load_restraint_bonds(data_dir, record.id)
            # Save the CIF structure
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                f.write(to_mmcif(new_structure, plddts=None, restraint_bonds=restraint_bonds))
        except Exception as e:
            print(f"Error saving CIF structure: {e}")
            import traceback
            traceback.print_exc()
