"""Molecule-type-aware cropper for handling mixed protein/nucleic acid structures. (desined by Cryonet.Refine)"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from CryoNetRefine.data import const


class MoleculeTypeAwareSlidingWindowCropper:
    """
    Molecule-type-aware sliding window cropper that ensures each crop contains only one molecule type.
    
    This cropper addresses the issue where traditional sliding windows may mix proteins and nucleic acids
    in the same crop. It first groups by molecule type, then performs sliding window cropping within each type.
    """
    
    def __init__(self, crop_size: int, overlap_size: int = 0, min_crop_size: Optional[int] = None):
        """
        Initialize the molecule-type-aware cropper.
        
        Parameters
        ----------
        crop_size : int
            Target size for each crop (number of tokens)
        overlap_size : int
            Number of overlapping tokens between adjacent crops, default is 0
        min_crop_size : int, optional
            Minimum crop size, remaining parts smaller than this will be merged into the previous crop.
            Defaults to crop_size // 4
        """
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        self.min_crop_size = min_crop_size or (crop_size // 4)
    
    def get_molecule_type_aware_crops(self, batch: Dict) -> List[Tuple]:
        """
        Get the list of molecule-type-aware crops.
        
        Parameters
        ----------
        batch : Dict
            Batch data containing token and atom information
            
        Returns
        -------
        List[Tuple]
            Each element is (crop_idx, crop_token_indices, molecule_type, crop_metadata)
            - crop_idx: Global crop index
            - crop_token_indices: Token indices included in this crop (positions in the original batch)
            - molecule_type: Molecule type name (PROTEIN/DNA/RNA/NONPOLYMER)
            - crop_metadata: Dictionary containing detailed crop information
        """
        # Get token information - only select tokens that are both non-padding and resolved
        token_pad = batch['token_pad_mask'].squeeze(0).bool()  # [N_tokens], bool
        if 'token_resolved_mask' in batch:
            token_resolved = batch['token_resolved_mask'].squeeze(0).bool()  # [N_tokens], bool
            # Only select tokens that are both non-padding and resolved (have actual structure)
            valid_token_mask = token_pad & token_resolved
        else:
            # Fallback: if token_resolved_mask is not available, use only token_pad_mask
            valid_token_mask = token_pad
        valid_token_indices = valid_token_mask.nonzero().squeeze(-1)  # Indices of valid tokens
        
        # Get asym_id and mol_type
        if 'asym_id' in batch:
            asym_ids = batch['asym_id'].squeeze(0)[valid_token_indices]
        else:
            print("⚠️ asym_id not found, treating all as single sequence")
            asym_ids = torch.zeros(len(valid_token_indices), dtype=torch.long, device=valid_token_indices.device)
            
        if 'mol_type' in batch:
            mol_types = batch['mol_type'].squeeze(0)[valid_token_indices]
        else:
            print("⚠️ mol_type not found, assuming all are protein")
            mol_types = torch.zeros(len(valid_token_indices), dtype=torch.long, device=valid_token_indices.device)
        
        # Group tokens by molecule type
        molecule_groups = self._group_tokens_by_molecule_type(
            valid_token_indices, asym_ids, mol_types
        )
        
        # Generate crops for each molecule type
        all_crops = []
        global_crop_idx = 0
        
        for mol_type, group_info in molecule_groups.items():
            crops = self._generate_crops_for_molecule_type(
                group_info, mol_type, global_crop_idx
            )
            all_crops.extend(crops)
            global_crop_idx += len(crops)
        
        return all_crops
    
    def _group_tokens_by_molecule_type(
        self, 
        valid_token_indices: torch.Tensor,
        asym_ids: torch.Tensor, 
        mol_types: torch.Tensor
    ) -> Dict[str, Dict]:
        """
        Group tokens by molecule type, preserving the order within each molecule type.
        
        Parameters
        ----------
        valid_token_indices : torch.Tensor
            Indices of valid tokens
        asym_ids : torch.Tensor
            Chain ID for each token
        mol_types : torch.Tensor
            Molecule type ID for each token
            
        Returns
        -------
        Dict[str, Dict]
            Mapping from molecule type name to group information
        """
        molecule_groups = {}
        
        # Get mapping from molecule type ID to name
        type_id_to_name = {v: k for k, v in const.chain_type_ids.items()}
        
        # Group according to original token order
        for i, (token_idx, asym_id, mol_type) in enumerate(
            zip(valid_token_indices, asym_ids, mol_types)
        ):
            mol_type_item = mol_type.item()
            mol_type_name = type_id_to_name.get(mol_type_item, f"UNKNOWN_{mol_type_item}")
            
            if mol_type_name not in molecule_groups:
                molecule_groups[mol_type_name] = {
                    'token_indices': [],
                    'original_positions': [],  # Position in valid_token_indices
                    'asym_ids': [],
                    'sequences': {}  # asym_id -> list of token positions within this group
                }
            
            group = molecule_groups[mol_type_name]
            group['token_indices'].append(token_idx.item())
            group['original_positions'].append(i)
            group['asym_ids'].append(asym_id.item())
            
            # Record token positions for each sequence (positions within this molecule type group)
            asym_id_item = asym_id.item()
            if asym_id_item not in group['sequences']:
                group['sequences'][asym_id_item] = []
            group['sequences'][asym_id_item].append(len(group['token_indices']) - 1)
        
        # Convert to tensors
        for mol_type_name, group in molecule_groups.items():
            group['token_indices'] = torch.tensor(
                group['token_indices'], 
                device=valid_token_indices.device
            )
            group['original_positions'] = torch.tensor(
                group['original_positions'],
                device=valid_token_indices.device
            )
            group['asym_ids'] = torch.tensor(
                group['asym_ids'],
                device=valid_token_indices.device
            )
        
        return molecule_groups
    
    def _generate_crops_for_molecule_type(
        self, 
        group_info: Dict, 
        mol_type: str, 
        start_crop_idx: int
    ) -> List[Tuple]:
        """
        Generate crops for a single molecule type.
        
        Parameters
        ----------
        group_info : Dict
            Information for the molecule type group
        mol_type : str
            Molecule type name
        start_crop_idx : int
            Starting crop index
            
        Returns
        -------
        List[Tuple]
            All crops for this molecule type
        """
        token_indices = group_info['token_indices']
        num_tokens = len(token_indices)
        
        # If token count is less than or equal to crop_size, treat the whole as one crop
        if num_tokens <= self.crop_size:
            return [(
                start_crop_idx,
                token_indices,
                mol_type,
                {
                    'is_complete': True,
                    'num_tokens': num_tokens,
                    'sequences': group_info['sequences'],
                    'original_positions': group_info['original_positions'],
                    'local_start': 0,
                    'local_end': num_tokens
                }
            )]
        
        # Generate sliding window crops
        crops = []
        crop_idx = start_crop_idx
        
        # If sequence boundary information is available, try to split at sequence boundaries
        sequence_boundaries = self._get_sequence_boundaries(group_info['sequences'])
        
        i = 0
        while i < num_tokens:
            crop_end = min(i + self.crop_size, num_tokens)
            
            # Try to adjust crop_end at sequence boundaries to avoid truncating sequences
            if crop_end < num_tokens:  # Not the last crop
                crop_end = self._adjust_crop_end_at_sequence_boundary(
                    i, crop_end, sequence_boundaries, num_tokens
                )
            
            crop_token_indices = token_indices[i:crop_end]
            crop_original_positions = group_info['original_positions'][i:crop_end]
            
            # Extract sequence information involved in this crop
            crop_sequences = self._extract_crop_sequences(
                group_info['sequences'], i, crop_end
            )
            
            crops.append((
                crop_idx,
                crop_token_indices,
                mol_type,
                {
                    'is_complete': (crop_end == num_tokens and i == 0),
                    'num_tokens': len(crop_token_indices),
                    'sequences': crop_sequences,
                    'original_positions': crop_original_positions,
                    'local_start': i,
                    'local_end': crop_end
                }
            ))
            
            crop_idx += 1
            
            # Calculate starting position for next crop (considering overlap)
            if self.overlap_size > 0 and crop_end < num_tokens:
                i = crop_end - self.overlap_size
            else:
                i = crop_end
            
            # If remaining tokens are too few, merge into current crop or treat as last crop
            if i < num_tokens and (num_tokens - i) < self.min_crop_size:
                if len(crops) > 0 and (crop_end - i) <= self.crop_size:
                    # Extend the last crop
                    last_crop = crops[-1]
                    extended_indices = token_indices[last_crop[3]['local_start']:num_tokens]
                    extended_positions = group_info['original_positions'][
                        last_crop[3]['local_start']:num_tokens
                    ]
                    extended_sequences = self._extract_crop_sequences(
                        group_info['sequences'], last_crop[3]['local_start'], num_tokens
                    )
                    
                    crops[-1] = (
                        last_crop[0],
                        extended_indices,
                        last_crop[2],
                        {
                            'is_complete': last_crop[3]['local_start'] == 0,
                            'num_tokens': len(extended_indices),
                            'sequences': extended_sequences,
                            'original_positions': extended_positions,
                            'local_start': last_crop[3]['local_start'],
                            'local_end': num_tokens
                        }
                    )
                break
        
        return crops
    
    def _get_sequence_boundaries(self, sequences: Dict) -> List[int]:
        """
        Get sequence boundary positions.
        
        Parameters
        ----------
        sequences : Dict
            Mapping from asym_id to list of token positions
            
        Returns
        -------
        List[int]
            Sorted list of boundary positions
        """
        boundaries = set()
        for asym_id, positions in sequences.items():
            if positions:
                boundaries.add(positions[0])       # Sequence start
                boundaries.add(positions[-1] + 1)  # Position after sequence end
        return sorted(boundaries)
    
    def _adjust_crop_end_at_sequence_boundary(
        self, 
        crop_start: int, 
        crop_end: int, 
        boundaries: List[int], 
        max_pos: int
    ) -> int:
        """
        Adjust crop end position at sequence boundaries.
        
        Parameters
        ----------
        crop_start : int
            Crop start position
        crop_end : int
            Crop end position (to be adjusted)
        boundaries : List[int]
            List of sequence boundary positions
        max_pos : int
            Maximum position
            
        Returns
        -------
        int
            Adjusted crop end position
        """
        # Find boundaries near crop_end
        tolerance = min(self.crop_size // 10, 20)  # Allowed adjustment range
        
        for boundary in boundaries:
            if crop_end - tolerance <= boundary <= crop_end + tolerance:
                if boundary > crop_start and boundary <= max_pos:
                    return boundary
        
        return crop_end
    
    def _extract_crop_sequences(
        self, 
        all_sequences: Dict, 
        start: int, 
        end: int
    ) -> Dict:
        """
        Extract sequence information contained in the crop.
        
        Parameters
        ----------
        all_sequences : Dict
            Information for all sequences
        start : int
            Crop start position
        end : int
            Crop end position
            
        Returns
        -------
        Dict
            Sequence information in the crop
        """
        crop_sequences = {}
        for asym_id, positions in all_sequences.items():
            crop_positions = [p - start for p in positions if start <= p < end]
            if crop_positions:
                crop_sequences[asym_id] = crop_positions
        return crop_sequences
    
    def get_num_crops(self, batch: Dict) -> int:
        """
        Get the number of crops needed for the given batch.
        
        Parameters
        ----------
        batch : Dict
            Input batch data
            
        Returns
        -------
        int
            Number of crops needed
        """
        crops = self.get_molecule_type_aware_crops(batch)
        return len(crops)
    
    def get_crop_info(self, batch: Dict) -> Dict:
        """
        Get detailed information about the crop strategy.
        
        Parameters
        ----------
        batch : Dict
            Input batch data
            
        Returns
        -------
        Dict
            Dictionary containing crop information
        """
        crops = self.get_molecule_type_aware_crops(batch)
        
        # Count crops for each molecule type
        mol_type_counts = {}
        for crop in crops:
            mol_type = crop[2]
            mol_type_counts[mol_type] = mol_type_counts.get(mol_type, 0) + 1
        
        return {
            "total_crops": len(crops),
            "crop_size": self.crop_size,
            "overlap_size": self.overlap_size,
            "min_crop_size": self.min_crop_size,
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
        # Create a new token mask for this sequence
        crop_size = len(token_indices)
        crop_batch['token_pad_mask'] = torch.ones(1, crop_size, dtype=torch.bool, device=token_indices.device)
        # Handle atom-level data by finding atoms that belong to the sequence tokens
        if 'atom_to_token' in batch:
            atom_to_token = batch['atom_to_token'].squeeze(0)  # [N_atoms, N_tokens]
            atom_mask = batch['atom_pad_mask'].squeeze(0)  # [N_atoms]
            # Create a boolean mask for sequence tokens
            crop_token_mask = torch.zeros(atom_to_token.shape[1], dtype=torch.bool, device=atom_to_token.device)
            crop_token_mask[token_indices] = True
            # Find atoms belonging to sequence tokens
            crop_atom_mask = atom_to_token[:, crop_token_mask].any(dim=1) & atom_mask.bool()
            # Create a properly cropped atom_pad_mask with only the selected atoms
            crop_batch['atom_pad_mask'] = torch.ones(1, crop_atom_mask.sum().item(), dtype=torch.bool, device=crop_atom_mask.device)
            # Crop atom_to_token: [N_atoms, N_tokens] -> [N_cropped_atoms, N_cropped_tokens]
            crop_atom_to_token = atom_to_token[crop_atom_mask]  # [N_cropped_atoms, N_tokens]
            crop_atom_to_token = crop_atom_to_token[:, token_indices]  # [N_cropped_atoms, N_cropped_tokens]
            crop_batch['atom_to_token'] = crop_atom_to_token.unsqueeze(0)  # [1, N_cropped_atoms, N_cropped_tokens]
        else:
            # Fallback: use all atoms (not ideal but prevents crashes)
            crop_atom_mask = batch['atom_pad_mask'].squeeze(0).bool()
            crop_batch['atom_pad_mask'] = batch['atom_pad_mask']

        # Handle token-level data
        for key in ['token_bonds', 'type_bonds']:
            if key in batch:
                original_tensor = batch[key]
                if len(original_tensor.shape) >= 3 and original_tensor.shape[1] == original_tensor.shape[2]:
                    crop_batch[key] = original_tensor[:, token_indices][:, :, token_indices]
                else:
                    crop_batch[key] = original_tensor
        # Get dimensions for efficient processing
        batch_token_dim = batch['token_pad_mask'].shape[1]
        batch_atom_dim = batch['atom_pad_mask'].shape[1]
        # Explicitly handle atom-level features first to avoid misclassification
        # This is critical when token_dim happens to match feature dimensions
        atom_level_keys = ['ref_pos', 'atom_resolved_mask', 'ref_atom_name_chars', 'ref_element', 
                           'ref_charge', 'ref_chirality', 'atom_backbone_feat', 'ref_space_uid',
                           'bfactor', 'plddt', 'coords', 'template_coords', 'template_resolved_mask']
        for key in atom_level_keys:
            if key in batch and key not in crop_batch:
                value = batch[key]
                if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                    if value.shape[1] == batch_atom_dim:
                        # Atom-level tensor: crop using atom mask
                        if len(value.shape) == 2:  # [B, N_atoms]
                            crop_batch[key] = value[:, crop_atom_mask]
                        elif len(value.shape) == 3:  # [B, N_atoms, D]
                            crop_batch[key] = value[:, crop_atom_mask, :]
                        elif len(value.shape) == 4:  # [B, N_atoms, D1, D2]
                            crop_batch[key] = value[:, crop_atom_mask, :, :]
                    elif len(value.shape) == 4 and value.shape[1] == 1 and value.shape[2] == batch_atom_dim:
                        # Template coordinates: [B, 1, N_atoms, 3] -> [B, 1, crop_atoms, 3]
                        crop_batch[key] = value[:, :, crop_atom_mask, :]
                    elif len(value.shape) == 3 and value.shape[1] == 1 and value.shape[2] == batch_atom_dim:
                        # Template resolved mask: [B, 1, N_atoms] -> [B, 1, crop_atoms]
                        crop_batch[key] = value[:, :, crop_atom_mask]
        # Copy and crop all other tensors
        for key, value in batch.items():
            if key not in crop_batch:  # Only copy if not already handled
                if isinstance(value, torch.Tensor):
                    if (len(value.shape) >= 3 and 
                        value.shape[1] == batch_token_dim and 
                        value.shape[2] == batch_token_dim):
                        # Pair-wise token tensor: [B, N, N, ...] -> [B, crop_size, crop_size, ...]
                        crop_batch[key] = value[:, token_indices][:, :, token_indices]
                    elif len(value.shape) >= 4 and value.shape[2] == batch_token_dim and value.shape[3] == batch_token_dim:
                        # 4D pair-wise token tensor: [B, M, N, N, ...] -> [B, M, crop_size, crop_size, ...]
                        crop_batch[key] = value[:, :, token_indices][:, :, :, token_indices]
                    elif (len(value.shape) >= 3 and 
                          value.shape[1] == batch_token_dim and 
                          value.shape[2] == batch_atom_dim):
                        # Token-atom tensor: [B, N_tokens, N_atoms, ...] -> [B, crop_tokens, crop_atoms, ...]
                        crop_batch[key] = value[:, token_indices][:, :, crop_atom_mask]
                    elif len(value.shape) >= 3 and value.shape[2] == batch_token_dim and value.shape[1] != batch_atom_dim:
                        # MSA-like tensor: [B, M, N, ...] -> [B, M, crop_size, ...]
                        # Exclude atom-level tensors that might have matching dimensions
                        crop_batch[key] = value[:, :, token_indices]
                    elif len(value.shape) >= 2 and value.shape[1] == batch_token_dim:
                        # Token-level tensor: [B, N, ...] -> [B, crop_size, ...]
                        crop_batch[key] = value[:, token_indices]
                    elif len(value.shape) >= 2 and value.shape[1] == batch_atom_dim:
                        # Atom-level tensor: [B, N_atoms, ...] -> [B, crop_atoms, ...]
                        crop_batch[key] = value[:, crop_atom_mask]
                    elif len(value.shape) == 4 and value.shape[3] == 3 and value.shape[1] == 1:
                        # Template coordinates: [B, 1, N_atoms, 3] -> [B, 1, crop_atoms, 3]
                        crop_batch[key] = value[:, :, crop_atom_mask, :]
                    else:
                        # Other tensor: copy as is
                        crop_batch[key] = value
                else:
                    # Non-tensor data: copy as is
                    crop_batch[key] = value
        # Handle padding for atom count divisibility by window size
        window_size = 128  # atoms_per_window_queries
        actual_atom_pad_mask = crop_batch['atom_pad_mask']
        current_atom_count = actual_atom_pad_mask.sum().item()
        target_atom_count = ((current_atom_count + window_size - 1) // window_size) * window_size
        
        if target_atom_count > current_atom_count:
            padding_size = target_atom_count - current_atom_count
            padding_mask = torch.zeros(1, padding_size, dtype=torch.bool, device=actual_atom_pad_mask.device)
            crop_batch['atom_pad_mask'] = torch.cat([actual_atom_pad_mask, padding_mask], dim=1)
            
            # Pad atom-level tensors
            atom_padding_keys = ['ref_pos', 'atom_resolved_mask', 'ref_atom_name_chars', 'ref_element', 
                                'ref_charge', 'ref_chirality', 'atom_backbone_feat', 'ref_space_uid',
                                'bfactor', 'plddt', 'template_coords','template_resolved_mask', 'coords']
            
            for key in atom_padding_keys:
                if key in crop_batch and isinstance(crop_batch[key], torch.Tensor):
                    if len(crop_batch[key].shape) == 2:  # [B, N]
                        padding = torch.zeros(1, padding_size, device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
                    elif len(crop_batch[key].shape) == 3:  # [B, N, D]
                        padding = torch.zeros(1, padding_size, crop_batch[key].shape[2], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
                    elif len(crop_batch[key].shape) == 4 and key in ['template_coords', 'coords']:  # [B, 1, N_atoms, 3]
                        padding = torch.zeros(1, crop_batch[key].shape[1], padding_size, crop_batch[key].shape[3], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=2)
                    elif len(crop_batch[key].shape) == 4:  # [B, N, D1, D2]
                        padding = torch.zeros(1, padding_size, crop_batch[key].shape[2], crop_batch[key].shape[3], device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=1)
            
            # Pad token-atom tensors
            token_atom_padding_keys = ['token_to_rep_atom', 'r_set_to_rep_atom', 'token_to_center_atom']
            for key in token_atom_padding_keys:
                if key in crop_batch and isinstance(crop_batch[key], torch.Tensor):
                    if len(crop_batch[key].shape) == 3:  # [B, N_tokens, N_atoms]
                        padding = torch.zeros(1, crop_batch[key].shape[1], padding_size, device=crop_batch[key].device, dtype=crop_batch[key].dtype)
                        crop_batch[key] = torch.cat([crop_batch[key], padding], dim=2)
            
            # Update atom_to_token padding
            if 'atom_to_token' in crop_batch:
                if len(crop_batch['atom_to_token'].shape) == 3:  # [B, N_cropped_atoms, N_cropped_tokens]
                    padding_atom_to_token = torch.zeros(1, padding_size, crop_batch['atom_to_token'].shape[2], device=crop_batch['atom_to_token'].device, dtype=crop_batch['atom_to_token'].dtype)
                    crop_batch['atom_to_token'] = torch.cat([crop_batch['atom_to_token'], padding_atom_to_token], dim=1)
        
        # Calculate crop_start for this sequence
        # For sequence cropping, crop_start should be the position of the first token
        # in the original token sequence (before any padding/masking)
        if len(token_indices) > 0:
            # Find the minimum token index in this sequence
            crop_start = token_indices.min().item()
        else:
            crop_start = 0
        
        # Mark as cropped and add metadata
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
        # Use extract_sequence_crop_from_batch method, as the logic is similar
        # Both are based on given token_indices for cropping
        crop_batch, _, crop_atom_mask = self.extract_sequence_crop_from_batch(
            batch, crop_token_indices, chain_id=crop_idx, sequence_type=molecule_type
        )
        
        # Add molecule-type-aware specific information
        crop_batch['molecule_type'] = molecule_type
        crop_batch['crop_metadata'] = crop_metadata
        crop_batch['crop_idx'] = crop_idx
        crop_batch['crop_type'] = 'molecule_aware'
        
        return crop_batch, crop_token_indices, crop_atom_mask
    
