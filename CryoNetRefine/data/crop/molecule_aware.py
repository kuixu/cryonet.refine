"""Molecule-type-aware cropper for handling mixed protein/nucleic acid structures. (designed by Cryonet.Refine)"""

from typing import Dict, List, Tuple
import torch
from CryoNetRefine.data import const


class MoleculeTypeAwareSlidingWindowCropper:
    """
    Simplified molecule-type-aware cropper that ensures:
    1. Each crop contains only ONE molecule type (protein OR nucleic acid, not mixed)
    2. Each crop has strictly <= max_token tokens
    """
    
    def __init__(self, crop_size: int, overlap_size: int = 0, min_crop_size: int = 0):
        """
        Initialize the molecule-type-aware cropper.
        
        Parameters
        ----------
        crop_size : int
            Maximum number of tokens per crop (strict upper bound)
        overlap_size : int
            Unused, kept for API compatibility
        min_crop_size : int
            Unused, kept for API compatibility
        """
        self.crop_size = crop_size
        self.overlap_size = overlap_size  # Unused but kept for compatibility
        self.min_crop_size = min_crop_size  # Unused but kept for compatibility
    
    def get_molecule_type_aware_crops(self, batch: Dict) -> List[Tuple]:
        """
        Get the list of molecule-type-aware crops.
        
        Each crop contains tokens from only ONE molecule type and has <= crop_size tokens.
        
        Parameters
        ----------
        batch : Dict
            Batch data containing token and atom information
            
        Returns
        -------
        List[Tuple]
            Each element is (crop_idx, crop_token_indices, molecule_type, crop_metadata)
        """
        # Get valid tokens (non-padding and resolved)
        token_pad = batch['token_pad_mask'].squeeze(0).bool()  # [N_tokens]
        if 'token_resolved_mask' in batch:
            token_resolved = batch['token_resolved_mask'].squeeze(0).bool()
            valid_token_mask = token_pad & token_resolved
        else:
            valid_token_mask = token_pad
        valid_token_indices = valid_token_mask.nonzero().squeeze(-1)
        
        if valid_token_indices.numel() == 0:
            return []
        
        # Get molecule types
        if 'mol_type' in batch:
            mol_types = batch['mol_type'].squeeze(0)[valid_token_indices]
        else:
            # Assume all protein if mol_type not available
            mol_types = torch.zeros(len(valid_token_indices), dtype=torch.long, device=valid_token_indices.device)
        
        # Get asym_ids for sequence tracking
        if 'asym_id' in batch:
            asym_ids = batch['asym_id'].squeeze(0)[valid_token_indices]
        else:
            asym_ids = torch.zeros(len(valid_token_indices), dtype=torch.long, device=valid_token_indices.device)
        
        # Group tokens by molecule type
        type_id_to_name = {v: k for k, v in const.chain_type_ids.items()}
        molecule_groups = self._group_by_molecule_type(valid_token_indices, mol_types, asym_ids, type_id_to_name)
        
        # Generate crops for each molecule type
        all_crops = []
        global_crop_idx = 0
        
        for mol_type_name, group_data in molecule_groups.items():
            token_indices = group_data['token_indices']
            sequences = group_data['sequences']
            num_tokens = len(token_indices)
            
            # Simple sliding window: split into chunks of at most crop_size
            start = 0
            while start < num_tokens:
                end = min(start + self.crop_size, num_tokens)
                crop_token_indices = token_indices[start:end]
                
                # Extract sequence info for this crop
                crop_sequences = {}
                for asym_id, positions in sequences.items():
                    crop_positions = [p - start for p in positions if start <= p < end]
                    if crop_positions:
                        crop_sequences[asym_id] = crop_positions
                
                crop_metadata = {
                    'is_complete': (start == 0 and end == num_tokens),
                    'num_tokens': len(crop_token_indices),
                    'sequences': crop_sequences,
                    'local_start': start,
                    'local_end': end,
                }
                
                all_crops.append((global_crop_idx, crop_token_indices, mol_type_name, crop_metadata))
                global_crop_idx += 1
                start = end
        
        return all_crops
    
    def _group_by_molecule_type(
        self, 
        valid_token_indices: torch.Tensor,
        mol_types: torch.Tensor,
        asym_ids: torch.Tensor,
        type_id_to_name: Dict
    ) -> Dict[str, Dict]:
        """
        Group tokens by molecule type, maintaining original order within each type.
        """
        molecule_groups = {}
        
        for i, (token_idx, mol_type, asym_id) in enumerate(zip(valid_token_indices, mol_types, asym_ids)):
            mol_type_id = mol_type.item()
            mol_type_name = type_id_to_name.get(mol_type_id, f"UNKNOWN_{mol_type_id}")
            asym_id_val = asym_id.item()
            
            if mol_type_name not in molecule_groups:
                molecule_groups[mol_type_name] = {
                    'token_indices': [],
                    'sequences': {}  # asym_id -> list of positions within this group
                }
            
            group = molecule_groups[mol_type_name]
            pos_in_group = len(group['token_indices'])
            group['token_indices'].append(token_idx.item())
            
            if asym_id_val not in group['sequences']:
                group['sequences'][asym_id_val] = []
            group['sequences'][asym_id_val].append(pos_in_group)
        
        # Convert token_indices to tensor
        for mol_type_name, group in molecule_groups.items():
            group['token_indices'] = torch.tensor(
                group['token_indices'], 
                device=valid_token_indices.device,
                dtype=torch.long
            )
        
        return molecule_groups
    
    def get_num_crops(self, batch: Dict) -> int:
        """Get the number of crops needed for the given batch."""
        return len(self.get_molecule_type_aware_crops(batch))
    
    def get_crop_info(self, batch: Dict) -> Dict:
        """Get detailed information about the crop strategy."""
        crops = self.get_molecule_type_aware_crops(batch)
        
        mol_type_counts = {}
        for crop in crops:
            mol_type = crop[2]
            mol_type_counts[mol_type] = mol_type_counts.get(mol_type, 0) + 1
        
        return {
            "total_crops": len(crops),
            "crop_size": self.crop_size,
            "molecule_type_counts": mol_type_counts,
            "crops": [
                {
                    "crop_idx": crop[0],
                    "molecule_type": crop[2],
                    "num_tokens": crop[3]['num_tokens'],
                    "is_complete": crop[3]['is_complete'],
                }
                for crop in crops
            ]
        }

    def extract_sequence_crop_from_batch(self, batch, token_indices, chain_id, sequence_type="protein"):
        """Extract a sequence-based crop from the batch data."""
        crop_batch = {}
        crop_size = len(token_indices)
        crop_batch['token_pad_mask'] = torch.ones(1, crop_size, dtype=torch.bool, device=token_indices.device)
        
        # Get dimensions
        batch_token_dim = batch['token_pad_mask'].shape[1]
        batch_atom_dim = batch['atom_pad_mask'].shape[1]
        
        # Handle atom-level data
        if 'atom_to_token' in batch:
            atom_to_token = batch['atom_to_token'].squeeze(0)  # [N_atoms, N_tokens]
            atom_mask = batch['atom_pad_mask'].squeeze(0)  # [N_atoms]
            
            # Create mask for selected tokens
            crop_token_mask = torch.zeros(atom_to_token.shape[1], dtype=torch.bool, device=atom_to_token.device)
            crop_token_mask[token_indices] = True
            
            # Find atoms belonging to selected tokens
            crop_atom_mask = atom_to_token[:, crop_token_mask].any(dim=1) & atom_mask.bool()
            
            crop_batch['atom_pad_mask'] = torch.ones(1, crop_atom_mask.sum().item(), dtype=torch.bool, device=crop_atom_mask.device)
            
            # Crop atom_to_token
            crop_atom_to_token = atom_to_token[crop_atom_mask][:, token_indices]
            crop_batch['atom_to_token'] = crop_atom_to_token.unsqueeze(0)
        else:
            crop_atom_mask = batch['atom_pad_mask'].squeeze(0).bool()
            crop_batch['atom_pad_mask'] = batch['atom_pad_mask']

        # Handle token bond tensors
        for key in ['token_bonds', 'type_bonds']:
            if key in batch:
                original_tensor = batch[key]
                if len(original_tensor.shape) >= 3 and original_tensor.shape[1] == original_tensor.shape[2]:
                    crop_batch[key] = original_tensor[:, token_indices][:, :, token_indices]
                else:
                    crop_batch[key] = original_tensor

        # Atom-level keys to handle explicitly
        atom_level_keys = ['ref_pos', 'atom_resolved_mask', 'ref_atom_name_chars', 'ref_element', 
                           'ref_charge', 'ref_chirality', 'atom_backbone_feat', 'ref_space_uid',
                           'bfactor', 'plddt', 'coords', 'template_coords', 'template_resolved_mask']
        
        for key in atom_level_keys:
            if key in batch and key not in crop_batch:
                value = batch[key]
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    if value.shape[1] == batch_atom_dim:
                        if len(value.shape) == 2:
                            crop_batch[key] = value[:, crop_atom_mask]
                        elif len(value.shape) == 3:
                            crop_batch[key] = value[:, crop_atom_mask, :]
                        elif len(value.shape) == 4:
                            crop_batch[key] = value[:, crop_atom_mask, :, :]
                    elif len(value.shape) == 4 and value.shape[1] == 1 and value.shape[2] == batch_atom_dim:
                        crop_batch[key] = value[:, :, crop_atom_mask, :]
                    elif len(value.shape) == 3 and value.shape[1] == 1 and value.shape[2] == batch_atom_dim:
                        crop_batch[key] = value[:, :, crop_atom_mask]

        # Copy remaining tensors
        for key, value in batch.items():
            if key not in crop_batch:
                if isinstance(value, torch.Tensor):
                    if (len(value.shape) >= 3 and 
                        value.shape[1] == batch_token_dim and 
                        value.shape[2] == batch_token_dim):
                        crop_batch[key] = value[:, token_indices][:, :, token_indices]
                    elif len(value.shape) >= 4 and value.shape[2] == batch_token_dim and value.shape[3] == batch_token_dim:
                        crop_batch[key] = value[:, :, token_indices][:, :, :, token_indices]
                    elif (len(value.shape) >= 3 and 
                          value.shape[1] == batch_token_dim and 
                          value.shape[2] == batch_atom_dim):
                        crop_batch[key] = value[:, token_indices][:, :, crop_atom_mask]
                    elif len(value.shape) >= 3 and value.shape[2] == batch_token_dim and value.shape[1] != batch_atom_dim:
                        crop_batch[key] = value[:, :, token_indices]
                    elif len(value.shape) >= 2 and value.shape[1] == batch_token_dim:
                        crop_batch[key] = value[:, token_indices]
                    elif len(value.shape) >= 2 and value.shape[1] == batch_atom_dim:
                        crop_batch[key] = value[:, crop_atom_mask]
                    elif len(value.shape) == 4 and value.shape[3] == 3 and value.shape[1] == 1:
                        crop_batch[key] = value[:, :, crop_atom_mask, :]
                    else:
                        crop_batch[key] = value
                else:
                    crop_batch[key] = value

        # Pad atoms to window size
        window_size = 128
        current_atom_count = crop_batch['atom_pad_mask'].sum().item()
        target_atom_count = ((current_atom_count + window_size - 1) // window_size) * window_size
        
        if target_atom_count > current_atom_count:
            padding_size = target_atom_count - current_atom_count
            padding_mask = torch.zeros(1, padding_size, dtype=torch.bool, device=crop_batch['atom_pad_mask'].device)
            crop_batch['atom_pad_mask'] = torch.cat([crop_batch['atom_pad_mask'], padding_mask], dim=1)
            
            atom_padding_keys = ['ref_pos', 'atom_resolved_mask', 'ref_atom_name_chars', 'ref_element', 
                                'ref_charge', 'ref_chirality', 'atom_backbone_feat', 'ref_space_uid',
                                'bfactor', 'plddt', 'template_coords', 'template_resolved_mask', 'coords']
            
            for key in atom_padding_keys:
                if key in crop_batch and isinstance(crop_batch[key], torch.Tensor):
                    if len(crop_batch[key].shape) == 2:
                        padding = torch.zeros(1, padding_size, device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
                    elif len(crop_batch[key].shape) == 3:
                        padding = torch.zeros(1, padding_size, crop_batch[key].shape[2], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
                    elif len(crop_batch[key].shape) == 4 and key in ['template_coords', 'coords']:
                        padding = torch.zeros(1, crop_batch[key].shape[1], padding_size, crop_batch[key].shape[3], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=2)
                    elif len(crop_batch[key].shape) == 4:
                        padding = torch.zeros(1, padding_size, crop_batch[key].shape[2], crop_batch[key].shape[3], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
            
            token_atom_padding_keys = ['token_to_rep_atom', 'r_set_to_rep_atom', 'token_to_center_atom']
            for key in token_atom_padding_keys:
                if key in crop_batch and isinstance(crop_batch[key], torch.Tensor):
                    if len(crop_batch[key].shape) == 3:
                        padding = torch.zeros(1, crop_batch[key].shape[1], padding_size, device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=2)
            
            if 'atom_to_token' in crop_batch:
                if len(crop_batch['atom_to_token'].shape) == 3:
                    padding_atom_to_token = torch.zeros(1, padding_size, crop_batch['atom_to_token'].shape[2], device=crop_batch['atom_to_token'].device, dtype=crop_batch['atom_to_token'].dtype)
                    crop_batch['atom_to_token'] = torch.cat([crop_batch['atom_to_token'], padding_atom_to_token], dim=1)

        # Calculate crop_start
        crop_start = token_indices.min().item() if len(token_indices) > 0 else 0

        # Add metadata
        crop_batch['record'] = batch['record']
        crop_batch['is_cropped'] = True
        crop_batch['crop_type'] = 'sequence'
        crop_batch['chain_id'] = chain_id
        crop_batch['crop_size'] = crop_size
        crop_batch['crop_start'] = crop_start
        crop_batch['sequence_type'] = sequence_type
        
        return crop_batch, token_indices, crop_atom_mask

    def extract_molecule_aware_crop_from_batch(self, batch, crop_info):
        """
        Extract batch crop based on molecule-type-aware crop information.
        
        Args:
            batch: Original batch
            crop_info: tuple (crop_idx, crop_token_indices, molecule_type, crop_metadata)
        
        Returns:
            crop_batch: Cropped batch
            crop_token_indices: Cropped token indices
            crop_atom_mask: Cropped atom mask
        """
        crop_idx, crop_token_indices, molecule_type, crop_metadata = crop_info
        
        crop_batch, _, crop_atom_mask = self.extract_sequence_crop_from_batch(
            batch, crop_token_indices, chain_id=crop_idx, sequence_type=molecule_type
        )
        
        # Add molecule-type-aware specific information
        crop_batch['molecule_type'] = molecule_type
        crop_batch['crop_metadata'] = crop_metadata
        crop_batch['crop_idx'] = crop_idx
        crop_batch['crop_type'] = 'molecule_aware'
        
        return crop_batch, crop_token_indices, crop_atom_mask
