import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Optional
import torch
from CryoNetRefine.data import const
from CryoNetRefine.data.types import StructureV2
from CryoNetRefine.libs.geometry.GeoMetric import GeoMetric  # type: ignore
from CryoNetRefine.libs.prot_utils import residue_constants


class GeometricAdapter:
    """Adapter to convert Boltz all-atom format to GeoMetric_v2 expected format.

    Inputs (Boltz):
      - coords: [B, N_atom, 3]
      - feats: dict with keys including
          atom_to_token: [B, N_atom, R]
          res_type: [B, R, 33]
          atom_pad_mask: [B, N_atom]
          token_pad_mask or token_resolved_mask: [B, R]
          ref_atom_name_chars: [B, N_atom, K] (optional, used to map atom names)

    Outputs (GeoMetric_v2 expected minimal set):
      - coords_geo: [B, R, A_max, 3] (A_max backbone-first ordering)
      - seq_idx: [B, R] (0-19 AA indices)
      - res_mask: [B, R]
      - atom_mask_geo: [B, R, A_max]
    """

    # Atom14 naming comes from residue_constants.restype_name_to_atom14_names

    def __init__(self, device: str = "cuda", data_dir: str = None, model_to_af_map: Optional[torch.Tensor] = None) -> None:
        self.device = device
        self.data_dir = data_dir
        
        self.structure_cache = {}
        self.structure_cache_info = {}
        self._protected_structures = []
        self._setup_optimized_mappings()
        self._crop_cache = {}
        self.residue_atom14_map = {}
        for res_name, atom14_names in residue_constants.restype_name_to_atom14_names.items():
            self.residue_atom14_map[res_name] = {
                atom_name: idx for idx, atom_name in enumerate(atom14_names) if atom_name
            }
    def _setup_optimized_mappings(self):

        self.boltz_to_geometric_map = {}
        for boltz_3letter in const.canonical_tokens[:-1]:  # Exclude 'UNK'
            if boltz_3letter in const.prot_token_to_letter:
                one_letter = const.prot_token_to_letter[boltz_3letter]
                if one_letter in residue_constants.restype_order:
                    geometric_idx = residue_constants.restype_order[one_letter]
                    self.boltz_to_geometric_map[boltz_3letter] = geometric_idx
                else:
                    self.boltz_to_geometric_map[boltz_3letter] = 20  # UNK
            else:
                self.boltz_to_geometric_map[boltz_3letter] = 20  # UNK
        
        self.atom14_names_cache = {}
        for res_name in self.boltz_to_geometric_map.keys():
            self.atom14_names_cache[res_name] = residue_constants.restype_name_to_atom14_names.get(res_name, [""] * 14)
        
        self.required_atoms = {'N', 'CA', 'C'}

    def _get_cached_structure(self, record_id: str, data_dir: str) -> StructureV2:
        if record_id not in self.structure_cache:
            path = Path(data_dir) / f"{record_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Structure file not found: {path}")
            
            structure = StructureV2.load(path)
            self.structure_cache[record_id] = structure
            self.structure_cache_info[record_id] = {
                'path': str(path),
                'load_time': time.time(),
                'num_atoms': len(structure.atoms),
                'num_residues': len(structure.residues)
            }
            
            self._protected_structures.append(structure)

        
        return self.structure_cache[record_id]
    
    def clear_cache(self, record_id: str = None):
        if record_id is None:
            self.structure_cache.clear()
            self.structure_cache_info.clear()
            self._crop_cache.clear()
        elif record_id in self.structure_cache:
            del self.structure_cache[record_id]
            del self.structure_cache_info[record_id]
            keys_to_remove = [key for key in self._crop_cache.keys() if key.startswith(record_id)]
            for key in keys_to_remove:
                del self._crop_cache[key]
    
    def get_cache_info(self) -> dict:
        return {
            'cached_structures': list(self.structure_cache.keys()),
            'cache_info': self.structure_cache_info.copy(),
            'crop_cache_size': len(self._crop_cache),
            'crop_cache_keys': list(self._crop_cache.keys())
        }

    def _infer_res_mask(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Infer residue mask from features."""
        if "token_resolved_mask" in feats:
            return feats["token_resolved_mask"]
        elif "token_pad_mask" in feats:
            return feats["token_pad_mask"]
        else:
            # Fallback: all residues are valid
            B, R, _ = feats["res_type"].shape
            return torch.ones((B, R), dtype=torch.bool, device=feats["res_type"].device)

    def convert(self, coords: torch.Tensor, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Boltz all-atom format to GeoMetric format.
        
        For PROTEIN crops:
        - All tokens are protein, directly convert to atom14
        - Atoms are stored in const.ref_atoms order, map by position to atom14
        
        Returns:
            coords_geo: [B, R, 14, 3] coordinates in atom14 format
            seq_idx: [B, R] amino acid indices (0-19 for GeoMetric)
            res_mask: [B, R] valid residue mask
            atom_mask_geo: [B, R, 14] atom presence mask
        """
        B, N_atom, _ = coords.shape
        device = coords.device
        A_max = 14
        
        # Get num_tokens from crop_metadata
        crop_metadata = feats.get('crop_metadata', {})
        num_tokens = crop_metadata.get('num_tokens', 0)
        
        if num_tokens == 0:
            token_pad_mask = feats["token_pad_mask"].squeeze(0).bool()
            num_tokens = token_pad_mask.sum().item()
        
        if num_tokens == 0:
            return (
                torch.zeros((B, 1, A_max, 3), device=device, dtype=coords.dtype),
                torch.zeros((B, 1), device=device, dtype=torch.long),
                torch.zeros((B, 1), device=device, dtype=torch.bool),
                torch.zeros((B, 1, A_max), device=device, dtype=torch.bool),
            )

        # Get basic tensors
        atom_pad_mask = feats["atom_pad_mask"].squeeze(0).bool()    # [N_atoms]
        res_type = feats["res_type"].squeeze(0)                     # [N_tokens, 33]
        
        # Get token indices from res_type (const.tokens ordering)
        # const.tokens: ["<pad>", "-", "ALA", "ARG", ..., "VAL", "UNK", ...]
        # Protein AAs are at indices 2-21
        token_indices = res_type[:num_tokens, :].argmax(dim=-1)  # [num_tokens]
        
        # Convert to GeoMetric indices (0-19 for standard AAs)
        # const.tokens[2:22] = canonical_tokens[0:20] = ["ALA", ..., "VAL"]
        geo_indices = (token_indices - 2).clamp(min=0, max=20)
        
        # Create output tensors
        coords_geo = torch.zeros((B, num_tokens, A_max, 3), device=device, dtype=coords.dtype)
        atom_mask_geo = torch.zeros((B, num_tokens, A_max), device=device, dtype=torch.bool)
        seq_idx = geo_indices.unsqueeze(0)  # [1, num_tokens]
        res_mask = torch.ones((B, num_tokens), device=device, dtype=torch.bool)
        
        # Get atom_to_token mapping
        if 'atom_to_token' not in feats:
            return coords_geo, seq_idx, res_mask, atom_mask_geo
        
        atom_to_token = feats['atom_to_token'].squeeze(0)  # [N_atoms, N_tokens]
        
        # Build ref_atoms to atom14 mapping for each residue type
        # This maps position in const.ref_atoms to position in atom14
        ref_to_atom14_maps = {}
        for res_name in const.ref_atoms:
            ref_atom_list = const.ref_atoms[res_name]
            atom14_names = residue_constants.restype_name_to_atom14_names.get(res_name, [])
            if not atom14_names:
                continue
            # Create mapping: ref_atom position -> atom14 position
            pos_map = {}
            for ref_pos, atom_name in enumerate(ref_atom_list):
                if atom_name in atom14_names:
                    atom14_pos = atom14_names.index(atom_name)
                    pos_map[ref_pos] = atom14_pos
            ref_to_atom14_maps[res_name] = pos_map
        
        # Map atoms to atom14 positions by their position within the token
        res_idx_list, atom14_idx_list, coord_idx_list = [], [], []
        
        for token_idx in range(num_tokens):
            tok_idx = token_indices[token_idx].item()
            
            # Get 3-letter residue name from const.tokens
            res_name_3 = const.tokens[tok_idx] if tok_idx < len(const.tokens) else 'UNK'
            
            # Get position mapping for this residue type
            pos_map = ref_to_atom14_maps.get(res_name_3, {})
            if not pos_map:
                continue
            
            # Find atoms belonging to this token (sorted by global index = ref_atoms order)
            token_atoms = atom_to_token[:, token_idx].bool()
            atom_positions = token_atoms.nonzero().squeeze(-1)
            
            if atom_positions.numel() == 0:
                continue
            if atom_positions.dim() == 0:
                atom_positions = atom_positions.unsqueeze(0)
            
            # Sort atom positions (should already be in order, but ensure it)
            atom_positions = atom_positions.sort()[0]
            
            # Map each atom by its position within the token to atom14
            for ref_pos, atom_idx in enumerate(atom_positions):
                atom_idx_val = atom_idx.item()
                
                # Check if this atom is valid
                if atom_idx_val >= N_atom or not atom_pad_mask[atom_idx_val]:
                    continue
                
                # Map ref_pos to atom14 position
                if ref_pos in pos_map:
                    atom14_idx = pos_map[ref_pos]
                    res_idx_list.append(token_idx)
                    atom14_idx_list.append(atom14_idx)
                    coord_idx_list.append(atom_idx_val)
        
        # Fill output tensors
        if res_idx_list:
            res_idx_t = torch.tensor(res_idx_list, device=device, dtype=torch.long)
            atom14_idx_t = torch.tensor(atom14_idx_list, device=device, dtype=torch.long)
            coord_idx_t = torch.tensor(coord_idx_list, device=device, dtype=torch.long)
            
            coords_geo[0, res_idx_t, atom14_idx_t] = coords[0, coord_idx_t]
            atom_mask_geo[0, res_idx_t, atom14_idx_t] = True
        return coords_geo, seq_idx, res_mask, atom_mask_geo

class GeometricMetricWrapper:
    """Strict wrapper around vendored GeoMetric: constructs protein-like input and calls its methods."""

    def __init__(self, geom_root: str, pdb_id: str,  device: torch.device, top8000_path: str | None = None) -> None:
        self.device = device
        self.pdb_id = pdb_id
        # Import vendored GeoMetric
        self.GeoMetric = GeoMetric(protein=None)
        self._crop_cache = {}  # crop_key -> GeoMetric instance
    
            
    def compute(self,crop_idx, coords_geo: torch.Tensor, seq_idx: torch.Tensor, atom_mask_geo: torch.Tensor, weights: Dict[str, float] = None, output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Compute geometric losses using GeoMetric methods.
        Returns dict of individual losses: rama, rotamer, bond, angle, cbeta, ramaz,
        """
        start_time = time.time()
        loss_dict: Dict[str, torch.Tensor] = {}
        time_loss_dict = {}
        device = coords_geo.device
        assert coords_geo.shape[0] == 1, "Only batch size 1 supported"
        R = coords_geo.shape[1]

        crop_key = f"{crop_idx}"

        # Build minimal Protein-like object expected by GeoMetric
        prot = SimpleNamespace()
        prot.atom14_positions = coords_geo[0]              # [R,14,3]
        prot.atom14_mask = atom_mask_geo[0].to(torch.float32)  # [R,14]
        prot.aatype = seq_idx[0].to(torch.long)            # [R]
        prot.residue_index = torch.arange(R, device=device, dtype=torch.long)
        if crop_key in self._crop_cache:
            gm = self._crop_cache[crop_key]
            gm.prot = prot
            gm.build_protein()
            build_prot_time = time.time() - start_time
        else:
            gm = GeoMetric(protein=prot)  
            gm.pdb_id = self.pdb_id
            self._crop_cache[crop_key] = gm
            build_prot_time = time.time() - start_time

        if weights is None or weights.get("rama", 1.0) > 0:

            rama = gm.rama_outliers()
            loss_dict["rama"] = rama["rama_outliers"].mean()
        else:
            loss_dict["rama"] = torch.tensor(0.0, device=device)
        rama_time = time.time() - start_time - build_prot_time
        # 2) Rotamer
        if weights is None or weights.get("rotamer", 1.0) > 0:
            rot = gm.rotamer_outliers()
            loss_dict["rotamer"] = rot["rotamer_outliers"].mean()
        else:
            loss_dict["rotamer"] = torch.tensor(0.0, device=device)
        rot_time = time.time() - start_time - build_prot_time - rama_time
    
        bond_angle_time = 0 # we removed the bond and angle loss from this file,the new implementation is in the refine_loss.py file(prot + DNA/RNA)

        # 4) C-beta deviation

        if weights is None or weights.get("cbeta", 1.0) > 0:
            cb = gm.cbeta_dev_batch()
            loss_dict["cbeta"] = cb["cbeta_outliers"].mean()
        else:
            loss_dict["cbeta"] = torch.tensor(0.0, device=device)
        cb_time = time.time() - start_time - build_prot_time - rama_time - rot_time - bond_angle_time
        # 5) RamaZ 
        if weights is None or weights.get("ramaz", 1.0) > 0:
            # try:
            z = gm.ramaz_loss(output_path)
            loss_dict["ramaz"] =  torch.relu(torch.abs(z["whole_z_score"]))

        else:
            loss_dict["ramaz"] = torch.tensor(0.0, device=device)
        ramaz_time = time.time() - start_time - build_prot_time - rama_time - rot_time - bond_angle_time - cb_time

   
        clash_time = 0
        time_loss_dict = {
            "build_prot": build_prot_time,
            "rama": rama_time,
            "rotamer": rot_time,
            "bond_angle": bond_angle_time,
            "cbeta": cb_time,
            "ramaz": ramaz_time,
            "clash": clash_time,
        }
        return loss_dict, time_loss_dict


    def clear_cache(self):
        for gm in self._crop_cache.values():
            if hasattr(gm, 'clear_pdb_interpretation_cache'):
                gm.clear_pdb_interpretation_cache()
        self._crop_cache.clear()


