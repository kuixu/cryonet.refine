import os
import sys
import math
import pickle
import json
from typing import Dict, Optional
from io import StringIO
import numpy as np
import torch
import iotbx.pdb
from mmtbx.monomer_library import pdb_interpretation
from mmtbx.monomer_library import server as mon_lib_server
from mmtbx.monomer_library import cif_types
from CryoNetRefine.libs.protein import Protein
from CryoNetRefine.libs.prot_utils import residue_constants 
from CryoNetRefine.libs.prot_utils.residue_constants import restype3_to_atoms, index_to_restype_3, restype_3_to_index, chisPerAA
from .utils import (
    chiralty, construct_fourth_batch, calc_dihedral_batch, calc_dihedrals,
    interpolate_2d, aaTables, rama_tables, ramaz_db,
    SS_TYPE, ss2idx, RAMA_RESNAME, rama_res2idx, SS_CALIBRATION_VALUES
)
# Ramachandran type constants
RAMA_GENERAL = 0
RAMA_GLYCINE = 1
RAMA_CISPRO = 2
RAMA_TRANSPRO = 3
RAMA_PREPRO = 4
RAMA_ILE_VAL = 5

restype3_to_atoms_index = dict(
    [
        (res, dict([(a, i) for (i, a) in enumerate(atoms)]))
        for (res, atoms) in restype3_to_atoms.items()
    ]
)
for residue in restype3_to_atoms_index:
    restype3_to_atoms_index[residue]["OXT"] = restype3_to_atoms_index[residue]["O"]


class GeoMetric:

    def __init__(self, protein: Optional[Protein] = None):
        """Initialize GeoMetric class

        Args:
            protein: Optional Protein object. If provided, initializes directly with protein.
        """
        self.seq = None
        self.prot_len = None
        self.phi_psi = None
        self.atom_pos = None
        self.bond_proxies_cache = None
        self.angle_proxies_cache = None
        self.ss_types_res_cache = None
        self.ss_cache_seq = None  # Used to validate if cache is valid (based on sequence)
        self.pdb_id = None
        self._rmsd_grm_cache = None  # geometry_restraints_manager cache
        self._rmsd_sites_cart_cache = None  # sites_cart cache
        self._rmsd_perm_tensor_cache = None  # atom order mapping cache
        self._rmsd_cache_key = None  # cache key used for validity check (based on crop_idx and num_atoms)

        self._setup_static_attributes()

        if protein is not None:
            self.prot = protein
            self.build_protein()

    def _setup_static_attributes(self):
        """Setup static attributes to avoid redundant computation"""
        # Load idealized CA angles data from file
        with open(os.path.join(os.path.dirname(__file__), "data", "idealized_angles"), "r") as rf:
            self.idealized_ca_angles = json.load(rf)

        # Convert to tensor, shape = [21, 5] (assume 20 amino acids + X)
        self.idealized_ca_tensor = torch.tensor(
            [self.idealized_ca_angles[res] for res in index_to_restype_3],
            dtype=torch.float32,
            device='cpu'  # Default is on CPU, can later be moved to other device as needed
        )

        # Precompute chi atom indices for all amino acids
        self._chi_atoms_cache = {}
        for aa_idx in range(len(index_to_restype_3)):
            resname = index_to_restype_3[aa_idx]
            atom14_names = residue_constants.restype_name_to_atom14_names[resname]
            chi_groups = residue_constants.chi_angles_atoms.get(resname, [])

            # Precompute chi atom indices for each chi angle (up to 4 chis)
            chi_indices = []
            for chi_idx, atoms in enumerate(chi_groups[:4]):  # up to 4 chi angles
                if len(atoms) >= 4:
                    try:
                        idxs = [atom14_names.index(an) if an in atom14_names else -1 for an in atoms]
                        if all(k >= 0 for k in idxs):
                            chi_indices.append((chi_idx, idxs))
                    except Exception:
                        continue

            self._chi_atoms_cache[aa_idx] = chi_indices

        # Create 3-to-1 amino acid mapping
        self._restype_3to1_mapping = {
            index_to_restype_3[i]: residue_constants.restype_3to1.get(index_to_restype_3[i], 'X')
            for i in range(len(index_to_restype_3))
        }

        # Prepare Ramachandran ("ramaz") table
        # Create a fast mapping from index to one-letter code
        # Include index 20 for unknown/non-standard residues (mapped to 'X')
        self._idx_to_1letter = torch.tensor([
            ord(self._restype_3to1_mapping[index_to_restype_3[i]])
            for i in range(len(index_to_restype_3))
        ] + [ord('X')], dtype=torch.int32, device='cpu')  # Add index 20 -> 'X'

        # Process the Ramachandran Z-score database: convert to unified tensor format
        # Find maximum phi/psi bins across all tables
        max_phi_bins = max(len(arr) for ss in SS_TYPE for arr in ramaz_db[ss].values())
        max_psi_bins = max(len(arr[0]) for ss in SS_TYPE for arr in ramaz_db[ss].values())

        table_tensor = torch.zeros((len(SS_TYPE), len(index_to_restype_3), max_phi_bins, max_psi_bins), dtype=torch.float32)
        means_tensor = torch.zeros((len(SS_TYPE), len(index_to_restype_3)), dtype=torch.float32)
        stds_tensor  = torch.ones((len(SS_TYPE), len(index_to_restype_3)), dtype=torch.float32)
        
        def normalize_resname(resname: str) -> str:
            if resname == "MSE":
                return "MET"
            if resname in ["cisPRO", "transPRO", "prePRO"]:
                return "PRO"
            return resname

        for ss in SS_TYPE:
            for resname, mat in ramaz_db[ss].items():
                resname_norm = normalize_resname(resname)
                if resname_norm not in restype_3_to_index:
                    continue
                res_idx = restype_3_to_index[resname_norm]
                ss_idx = ss2idx[ss]

                arr = np.array(mat, dtype=np.float32)
                # Pad to maximum shape
                padded = np.zeros((max_phi_bins, max_psi_bins), dtype=np.float32)
                padded[:arr.shape[0], :arr.shape[1]] = arr
                table_tensor[ss_idx, res_idx] = torch.from_numpy(padded)

                mean_val = arr.mean()
                std_val  = arr.std() if arr.std() > 1e-6 else 1.0
                means_tensor[ss_idx, res_idx] = float(mean_val)
                stds_tensor[ss_idx, res_idx]  = float(std_val)

        self.ramaz_table_tensor = table_tensor
        self._ramaz_means = means_tensor
        self._ramaz_stds  = stds_tensor
        
        self._setup_rama_tables_cache()

    def _setup_rama_tables_cache(self):
        self._rama_tables_cache = {}
        self._rama_limits_cache = {}
        
        for type_id, ndt in enumerate(rama_tables):
            table, limits = self.convert_ndtable_to_numpy(ndt)
            self._rama_tables_cache[type_id] = torch.tensor(
                table, dtype=torch.float32, device='cpu'
            )
            self._rama_limits_cache[type_id] = limits

    def clear_build_cache(self):
        self.ss_types_res_cache = None
        self.ss_cache_seq = None
        # 🚀 Clear RMSD calculation cache
        self._rmsd_grm_cache = None
        self._rmsd_perm_tensor_cache = None
        self._rmsd_cache_key = None
        # 🚀 Clear rama table cache
        if hasattr(self, '_rama_tables_cache'):
            self._rama_tables_cache.clear()
        if hasattr(self, '_rama_limits_cache'):
            self._rama_limits_cache.clear()

    def _compute_sidechain_torsions_parallel(self) -> None:
        device = self.atom_pos.device
        tors = torch.zeros((self.prot_len, 7, 2), device=device, dtype=self.atom_pos.dtype)
        batch_coords = []
        batch_indices = []  # [(res_idx, chi_idx)]
        for i in range(self.prot_len):
            aa_idx = int(self.prot.aatype[i]) if isinstance(self.prot.aatype, (torch.Tensor, np.ndarray)) else int(self.prot.aatype[i])
            chi_indices = self._chi_atoms_cache.get(aa_idx, [])
            for chi_idx, idxs in chi_indices:
                idxs_tensor = torch.tensor(idxs, device=self.atom_pos.device, dtype=torch.long)
                atoms = self.atom_pos[i, idxs_tensor]
                if torch.any(atoms == 0): 
                    continue
                batch_coords.append(atoms)
                batch_indices.append((i, chi_idx))
        if batch_coords:
            coords = torch.stack(batch_coords, dim=0)   # (B, 4, 3) 
            angles = calc_dihedral_batch(coords)                   # (B,)
            sin, cos = torch.sin(angles), torch.cos(angles)
            for (res_idx, chi_idx), s, c in zip(batch_indices, sin, cos):
                tors[res_idx, 3 + chi_idx, 0] = s
                tors[res_idx, 3 + chi_idx, 1] = c
        # attach to prot-like object
        self.prot.torsion_angles_sin_cos = tors
        if not self.prot.torsion_angles_sin_cos.requires_grad:
            self.prot.torsion_angles_sin_cos.requires_grad_(True)

    def _interpolate_rotamer_scores_differentiable(self, chis_sel, ndt, num_chis, resname=None):
        """
        🚀 Differentiable N-dimensional interpolation following mmtbx algorithm
        Implements 2^n neighbor weighted interpolation for accurate rotamer scoring
        
        Args:
            chis_sel: [n_res, num_chis] tensor of chi angles in degrees
            ndt: NDimTable object with lookup data
            num_chis: number of chi angles
            resname: residue name for symmetry handling
        """
        device = chis_sel.device
        n_res = chis_sel.shape[0]
        
        if n_res == 0:
            return torch.zeros(0, device=device)
        
        # Get lookup table parameters
        # Safely extract parameters, handling potential length mismatches
        actual_dims = min(num_chis, len(ndt.minVal), len(ndt.wBin), len(ndt.nBins), len(ndt.doWrap))
        
        # If actual dimensions differ from num_chis, adjust input
        if actual_dims < num_chis:
            chis_sel = chis_sel[:, :actual_dims]
            num_chis = actual_dims
        
        minVal = torch.tensor(ndt.minVal[:num_chis], device=device, dtype=torch.float32)
        wBin = torch.tensor(ndt.wBin[:num_chis], device=device, dtype=torch.float32)
        nBins = torch.tensor(ndt.nBins[:num_chis], device=device, dtype=torch.long)
        doWrap = list(ndt.doWrap[:num_chis])  # Keep as list for wrapping logic
        lookup_table = ndt.lookupTable.clone().detach().to(device=device, dtype=torch.float32)
        
        # Apply symmetry for ASP, GLU, PHE, TYR (last chi angle mod 180)
        chis_normalized = chis_sel.clone()
        if resname and resname.lower() in ['asp', 'glu', 'phe', 'tyr'] and num_chis > 0:
            # Last chi angle needs mod 180 treatment
            last_chi = chis_normalized[:, num_chis-1]
            last_chi = last_chi % 360.0
            last_chi = torch.where(last_chi < 0, last_chi + 360.0, last_chi)
            last_chi = last_chi % 180.0  # Apply 180-degree symmetry
            chis_normalized[:, num_chis-1] = last_chi
        
        # Normalize all angles to [0, 360)
        chis_normalized = chis_normalized % 360.0
        chis_normalized = torch.where(chis_normalized < 0, chis_normalized + 360.0, chis_normalized)
        
        # Find home bin (whereIs in mmtbx)
        va_home_float = (chis_normalized - minVal.unsqueeze(0)) / wBin.unsqueeze(0)
        va_home = torch.floor(va_home_float).long()
        # Clamp to valid range [0, nBins-1] for each dimension
        va_home = torch.max(va_home, torch.zeros_like(va_home))
        va_home = torch.min(va_home, (nBins.unsqueeze(0) - 1).expand_as(va_home))
        
        # Calculate bin center coordinates
        va_home_ctr = minVal.unsqueeze(0) + wBin.unsqueeze(0) * (va_home.float() + 0.5)
        
        # Determine neighbor direction and contribution weight for each dimension
        # If point < bin_center, neighbor is lower (home-1), else higher (home+1)
        va_neighbor = torch.where(
            chis_normalized < va_home_ctr,
            va_home - 1,
            va_home + 1
        )
        
        # Calculate relative contribution from neighbor (always in [0, 0.5])
        va_contrib = torch.abs((chis_normalized - va_home_ctr) / wBin.unsqueeze(0))
        
        scores = torch.zeros(n_res, device=device)
        
        # Loop over all 2^num_chis neighboring bins
        num_bins_total = 1 << num_chis  # 2^num_chis
        
        for bin_mask in range(num_bins_total):
            # Initialize coefficient and current bin indices
            coeff = torch.ones(n_res, device=device)
            va_current = torch.zeros((n_res, num_chis), dtype=torch.long, device=device)
            
            # For each dimension, check the bit in bin_mask
            for dim in range(num_chis):
                if bin_mask & (1 << dim) == 0:
                    # Bit is off: use va_home and (1 - va_contrib)
                    va_current[:, dim] = va_home[:, dim]
                    coeff *= (1.0 - va_contrib[:, dim])
                else:
                    # Bit is on: use va_neighbor and va_contrib
                    va_current[:, dim] = va_neighbor[:, dim]
                    coeff *= va_contrib[:, dim]
            
            # Apply wrapping and clamping
            for dim in range(num_chis):
                # Safe access to doWrap with fallback to True (most chi angles are periodic)
                do_wrap_dim = doWrap[dim] if dim < len(doWrap) else True
                
                if do_wrap_dim:
                    # Apply wrapping for periodic dimensions
                    va_current[:, dim] = va_current[:, dim] % nBins[dim].item()
                else:
                    # Clamp to valid range for non-periodic dimensions
                    va_current[:, dim] = torch.clamp(va_current[:, dim], 0, nBins[dim].item() - 1)
            
            # Convert multidimensional indices to flat indices
            flat_indices = self._bin2index_batch(va_current, nBins)
            flat_indices = torch.clamp(flat_indices, 0, lookup_table.numel() - 1)
            
            # Lookup values and accumulate weighted contribution
            lookup_flat = lookup_table.flatten()
            values = lookup_flat[flat_indices]
            scores += coeff * values
        
        return scores
    
    def _bin2index_batch(self, bins, nBins):
        """
        Convert multidimensional bin indices to flat array indices (batch version)
        
        Args:
            bins: [n_res, n_dim] tensor of bin indices
            nBins: [n_dim] tensor of number of bins per dimension
        Returns:
            [n_res] tensor of flat indices
        """
        n_res, n_dim = bins.shape
        device = bins.device
        
        idx = torch.zeros(n_res, dtype=torch.long, device=device)
        
        # Build flat index: idx = bin[0] * nBins[1] * ... + bin[1] * nBins[2] * ... + bin[-1]
        for i in range(n_dim - 1):
            idx += bins[:, i]
            idx *= nBins[i + 1]
        
        # Add last dimension
        idx += bins[:, n_dim - 1]
        
        return idx

    def build_protein(self):
        device = self.prot.atom14_positions.device if isinstance(self.prot.atom14_positions, torch.Tensor) else 'cpu'
        if self.idealized_ca_tensor.device != device:
            self.idealized_ca_tensor = self.idealized_ca_tensor.to(device)
        if self._idx_to_1letter.device != device:
            self._idx_to_1letter = self._idx_to_1letter.to(device)
        
        def _build_seq_vectorized(aa_type: torch.Tensor):
            """🚀 Vectorized sequence construction, replacing Python loop"""
            # Use precomputed mapping table
            seq_chars = [self._restype_3to1_mapping[index_to_restype_3[idx.item()]] for idx in aa_type]
            return "".join(seq_chars)
        
        def _build_seq_torch_vectorized(aa_type: torch.Tensor):
            """🚀 Ultra-fast sequence building using PyTorch tensor operations"""
            # Use precomputed mapping by direct indexing
            char_codes = self._idx_to_1letter[aa_type]
            # Convert to string
            return "".join([chr(code.item()) for code in char_codes])
        
        # Preserve the gradient path to upstream coordinates
        self.atom_pos = self.prot.atom14_positions.clone()
        # 🚀 Set requires_grad in batch to avoid multiple calls
        if not self.atom_pos.requires_grad:
            self.atom_pos.requires_grad_(True)
        
        # 🚀 Set protein length
        self.prot_len = self.prot.aatype.shape[0]
        self.phi_psi = calc_dihedrals(self.atom_pos)   # Radians
        phi_psi_deg = torch.rad2deg(self.phi_psi)      # Convert to degrees
        self.phi_psi = phi_psi_deg
        self.phi_psi.requires_grad_(True)
        self._compute_sidechain_torsions_parallel()
        try:
            self.seq = _build_seq_torch_vectorized(self.prot.aatype)
        except (IndexError, RuntimeError):
            # Fallback to standard vectorized method if error
            self.seq = _build_seq_vectorized(self.prot.aatype)

        assert len(self.seq) == self.prot_len

    def is_cis_peptide(self, res_idx1, res_idx2):
        omega_sites = [
            self.atom_pos[res_idx1][1],
            self.atom_pos[res_idx1][2],
            self.atom_pos[res_idx2][0],
            self.atom_pos[res_idx2][1],
        ]
        omega = calc_dihedral_batch(omega_sites)
        if abs(omega) < 30:
            return True
        else:
            return False

    def get_rama_types(self) -> torch.Tensor:
        """
        🚀 Vectorized calculation of Ramachandran types.
        Returns a tensor of shape [L-2], where each residue is assigned a rama_type.
        """
        L = self.prot_len
        aatype = self.prot.aatype  # shape [L]

        # Initialize to general
        rama_types = torch.full((L - 2,), RAMA_GENERAL, device=aatype.device, dtype=torch.long)

        # Amino acid type of residue i+1
        mid_res = aatype[1:L-1]      # Corresponds to i+1 in loop
        next_res = aatype[2:L]       # Corresponds to i+2 in loop

        # Glycine
        rama_types[mid_res == 7] = RAMA_GLYCINE

        # Isoleucine/Valine
        mask_ile_val = (mid_res == 9) | (mid_res == 19)
        rama_types[mask_ile_val] = RAMA_ILE_VAL

        # Proline (trans/cis)
        mask_pro = (mid_res == 14)
        if mask_pro.any():
            idx = mask_pro.nonzero(as_tuple=False).squeeze(-1)  # Indices for proline
            # Default to trans
            rama_types[idx] = RAMA_TRANSPRO
            # Check every proline for cis/trans
            for j in idx.tolist():
                if self.is_cis_peptide(j, j + 1):
                    rama_types[j] = RAMA_CISPRO

        # Pre-proline (the next residue is proline)
        rama_types[next_res == 14] = RAMA_PREPRO

        return rama_types

    def convert_ndtable_to_numpy(self, ndt):
        """Convert custom NDimTable to numpy grid"""
        import numpy as np

        n_phi, n_psi = ndt.nBins
        phi_min, phi_max = ndt.minVal[0], ndt.maxVal[0]
        psi_min, psi_max = ndt.minVal[1], ndt.maxVal[1]

        values = ndt.lookupTable.cpu().numpy().astype(np.float32)
        values = values.reshape(n_phi, n_psi)
        return values, (phi_min, phi_max, psi_min, psi_max)

    def lookup_rama_scores_cached(self, phi_psi, type_id):
        """
        🚀 Query Ramachandran scores using cached tensors (bilinear interpolation, gradient-preserving)
        phi_psi: [N, 2] tensor, unit=degree
        type_id: rama type ID
        """
        table_torch = self._rama_tables_cache[type_id].to(phi_psi.device, dtype=phi_psi.dtype)
        limits = self._rama_limits_cache[type_id]
        
        phi_min, phi_max, psi_min, psi_max = limits
        n_phi, n_psi = table_torch.shape

        # Compute continuous index positions (preserving gradients)
        phi_pos = (phi_psi[:, 0] - phi_min) / (phi_max - phi_min) * (n_phi - 1)
        psi_pos = (phi_psi[:, 1] - psi_min) / (psi_max - psi_min) * (n_psi - 1)
        
        # Clamp within valid range
        phi_pos = torch.clamp(phi_pos, 0, n_phi - 1)
        psi_pos = torch.clamp(psi_pos, 0, n_psi - 1)

        # Bilinear interpolation
        phi_low = torch.floor(phi_pos).long()
        phi_high = torch.clamp(phi_low + 1, 0, n_phi - 1)
        psi_low = torch.floor(psi_pos).long()
        psi_high = torch.clamp(psi_low + 1, 0, n_psi - 1)

        # Compute interpolation weights
        phi_weight = phi_pos - phi_low.float()
        psi_weight = psi_pos - psi_low.float()

        # Get four grid corner values
        val_ll = table_torch[phi_low, psi_low]      # lower left
        val_lh = table_torch[phi_low, psi_high]     # upper left
        val_hl = table_torch[phi_high, psi_low]     # lower right
        val_hh = table_torch[phi_high, psi_high]    # upper right

        # Bilinear interpolation computation
        val_l = val_ll * (1 - psi_weight) + val_lh * psi_weight
        val_h = val_hl * (1 - psi_weight) + val_hh * psi_weight
        result = val_l * (1 - phi_weight) + val_h * phi_weight

        return result

    def rama_outliers(self) -> Dict[str, torch.Tensor]:
        """
        🚀 Fully vectorized GPU-accelerated calculation of Ramachandran outliers
        - rama_tables: list[NDimTable]
        - self.prot_len: protein length
        - self.phi_psi: [L, 2] backbone dihedrals
        - self.get_rama_types(): [L-2]
        """

        L = self.prot_len - 2
        phi_psi = self.phi_psi[:L]          # [L, 2] backbone dihedrals
        rama_types = self.get_rama_types()  # [L] per-residue rama type ids
        # Build rama_scores by type, keeping gradient. This is the simplest method.
        rama_scores = torch.zeros(L, device=phi_psi.device, dtype=phi_psi.dtype)
        
        for type_id in range(len(rama_tables)):
            mask = (rama_types == type_id)
            if mask.any():
                # Select residues of the current type
                scores = self.lookup_rama_scores_cached(phi_psi[mask], type_id)
                # Assign results by index (in-places, keeps gradient)
                rama_scores[mask] = scores

        rama_scores.requires_grad_(True)

        # ---- Fully vectorized outlier classification ----
        thresholds = torch.tensor([0.0005, 0.0020, 0.0010], device=rama_scores.device, dtype=rama_scores.dtype)
        rama_outliers = torch.zeros(L, device=phi_psi.device, dtype=phi_psi.dtype)

        is_general = (rama_types == RAMA_GENERAL)
        is_cispro  = (rama_types == RAMA_CISPRO)
        is_other   = ~(is_general | is_cispro)

        # Use Leaky ReLU for differentiable and numerically stable bounds
        # When x < 0, output is 0.01*x, when x >= 0, output is x
        rama_outliers = rama_outliers + torch.nn.functional.leaky_relu(thresholds[0] - rama_scores, negative_slope=0.01) * is_general.float() * 100
        rama_outliers = rama_outliers + torch.nn.functional.leaky_relu(thresholds[1] - rama_scores, negative_slope=0.01) * is_cispro.float()  * 100
        rama_outliers = rama_outliers + torch.nn.functional.leaky_relu(thresholds[2] - rama_scores, negative_slope=0.01) * is_other.float()   * 100

        # Combine favored mask: if favored, zero out; else, keep rama_outliers
        favored_mask = torch.relu(0.02 - rama_scores)
        rama_outliers = torch.where(favored_mask == 0, favored_mask, rama_outliers)

        rama_outliers.requires_grad_(True)

        return {
            "rama_types": rama_types,
            "rama_scores": rama_scores,
            "rama_outliers": rama_outliers
        }

    def get_ss_from_python(self, output_path: str = None)->torch.Tensor:
        """
        Use the Python script get_phenix_ss.py to compute secondary structure (conda environment switch may be required)

        Returns:
        --------
        ss_types_res : torch.Tensor
            Per-residue secondary structure labels:
            0 = loop (coil)
            1 = helix (including alpha, 3_10, pi)
            2 = sheet (beta strand)
        """

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        get_phenix_ss_script = os.path.join(script_dir, "compute_ss.py")
        # Find project root (cryonet.refine directory) by going up from current file
        # GeoMetric.py is at: cryonet.refine/CryoNetRefine/libs/geometry/GeoMetric.py
        # Project root is 3 levels up
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        # absolute path !
        pickle_path = os.path.splitext(output_path)[0] + ".pickle"
        # Run script directly with PYTHONPATH set to project root
        # Use absolute paths and ensure PYTHONPATH is set correctly
        env_pythonpath = os.environ.get('PYTHONPATH', '')
        pythonpath = f"{project_root}:{env_pythonpath}" if env_pythonpath else project_root
        cmd = f"cd {project_root} && PYTHONPATH={pythonpath} {sys.executable} {get_phenix_ss_script} {output_path} {pickle_path}"
        os.system(cmd)
        ss_types_res = torch.load(pickle_path, map_location=self.phi_psi.device, weights_only=True)
        os.system(f"rm {pickle_path}")
        return ss_types_res.to(self.phi_psi.device)

    def ramaz_loss(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        db = ramaz_db
        # Get basic information
        aa_type = self.prot.aatype.clone().detach().to(dtype=int)
        atom_mask = self.prot.atom14_mask.clone().detach()
        device = self.phi_psi.device
        atom2res = self.prot.atom14_mask.nonzero()[0].clone().detach()

        # 🚀 Use cached secondary structure to avoid redundant calculations
        # Check if cache is valid: based on sequence match
        if (
            self.ss_types_res_cache is not None and
            self.ss_cache_seq is not None and
            self.seq is not None and
            self.ss_cache_seq == self.seq and
            self.ss_types_res_cache.shape[0] == self.prot_len
        ):
            # Use cached secondary structure
            ss_types_res = self.ss_types_res_cache.to(device)
        else:
            # Compute and cache secondary structure
            ss_types_res = self.get_ss_from_python(output_path).to(device)  # cctbx

            # Cache result (store on CPU to save GPU memory)
            self.ss_types_res_cache = ss_types_res.clone().detach().cpu()
            self.ss_cache_seq = self.seq


        # z_scores=torch.zeros(self.prot_len-2,dtype=float,requires_grad=True)
        means=torch.zeros((3,22),dtype=float)
        stds=torch.zeros((3,22),dtype=float)

        def _get_mean(ss_type, resname):
            reg_sum = 0
            sq_sum = 0
            for i in db[ss_type][resname]:
                for j in i:
                    reg_sum += j
                    sq_sum += j * j
            if reg_sum > 0:
                mean = sq_sum / reg_sum
            else:
                mean = 0
            return mean

        def _get_std(ss_type, resname, mean):
            ch, zn = 0, 0
            for i in db[ss_type][resname]:
                for j in i:
                    ch += j * (j - mean) ** 2
                    zn += j
            zn -= 1
            if zn == 0:
                return 0
            std = math.sqrt(ch / zn)
            return std
        
        def _get_resname(rama_type, resname):
            rn = resname
            if resname == "MSE":
                rn = "MET"
            if rama_type == 2:
                rn = "cisPRO"
            if rama_type == 3:
                rn = "transPRO"
            if rama_type == 4:
                rn = "prePRO"
            return rn

        for i in range(len(SS_TYPE)):
            for j in range(len(RAMA_RESNAME)):
                mean=_get_mean(SS_TYPE[i],RAMA_RESNAME[j])
                means[i][j]=mean
                stds[i][j]=_get_std(SS_TYPE[i],RAMA_RESNAME[j],mean)

        rama_types=self.get_rama_types()

        vmin=-178
        step=4
        int_zsc=torch.zeros(self.prot_len-2,dtype=float,device=self.phi_psi.device)
        phi_psi_angles=self.phi_psi
        for idx in range(0,self.prot_len-2):
            phi,psi=float(phi_psi_angles[idx][0]),float(phi_psi_angles[idx][1])
            phi_psi=phi_psi_angles[idx]
            rama_type=rama_types[idx]
            resname=index_to_restype_3[aa_type[idx+1]]
            ss=ss_types_res[idx+1]
            ss=SS_TYPE[ss]

            '''get_z_score_point'''
            resname=_get_resname(int(rama_type),resname)
            if resname=="cisPRO": ss="L"
            table=db[ss][resname]
            if phi < -178:
                i = -1
                x1 = -182
                x2 = -178
            elif phi > 178:
                i = -1
                x1 = 178
                x2 = 182
            else:
                i = int(abs(-178 - phi) // 4)
                nsteps = abs(vmin - phi) // step
                x1 = vmin + nsteps * step
                x2 = x1 + 4

            if psi < -178:
                j = -1
                y1 = -182
                y2 = -178
            elif psi > 178:
                j = -1
                y1 = 178
                y2 = 182
            else:
                j = int(abs(-178 - psi) // 4)
                nsteps = abs(vmin - psi) // step
                y1 = vmin + nsteps * step
                y2 = y1 + 4

            xx = phi
            yy = psi
            # Get table dimensions to prevent index out of bounds
            table_nrows = len(table)
            table_ncols = len(table[0]) if table_nrows > 0 else 0
            
            # Clamp indices to valid range to prevent out-of-bounds access
            # When i = -1 (phi < -178), map to 0; when i is at max, clamp to max-2
            # This ensures i+1 is always within bounds
            i_safe = max(0, min(i, table_nrows - 2))  # Clamp to max-2 so i+1 is valid
            j_safe = max(0, min(j, table_ncols - 2))   # Clamp to max-2 so j+1 is valid
            i1_safe = i_safe + 1
            j1_safe = j_safe + 1
            
            v1 = table[i_safe][j_safe]
            v2 = table[i1_safe][j1_safe]
            v3 = table[i_safe][j1_safe]
            v4 = table[i1_safe][j_safe]
            zsc=interpolate_2d(x1, y1, x2, y2, v1, v2, v3, v4, phi_psi)
            ss_idx=ss2idx[ss]
            res_idx=rama_res2idx[resname]
            zsc=(zsc-means[ss_idx][res_idx])/stds[ss_idx][res_idx]
            int_zsc[idx]=zsc
        int_zsc.requires_grad_(True)
        ss_types_res=ss_types_res[1:-1]

        helix_zscores=int_zsc[ss_types_res==1]
        sheet_zscores=int_zsc[ss_types_res==2]
        loop_zscores=int_zsc[ss_types_res==0]
        
        # Handle empty secondary structure categories to avoid nan
        if helix_zscores.numel() > 0:
            helix_zsc=(helix_zscores.mean()-SS_CALIBRATION_VALUES["H"][0])/SS_CALIBRATION_VALUES["H"][1]
        else:
            helix_zsc = torch.tensor(0.0, device=int_zsc.device, dtype=int_zsc.dtype)
            
        if sheet_zscores.numel() > 0:
            sheet_zsc=(sheet_zscores.mean()-SS_CALIBRATION_VALUES["S"][0])/SS_CALIBRATION_VALUES["S"][1]
        else:
            sheet_zsc = torch.tensor(0.0, device=int_zsc.device, dtype=int_zsc.dtype)
            
        if loop_zscores.numel() > 0:
            loop_zsc=(loop_zscores.mean()-SS_CALIBRATION_VALUES["L"][0])/SS_CALIBRATION_VALUES["L"][1]
        else:
            loop_zsc = torch.tensor(0.0, device=int_zsc.device, dtype=int_zsc.dtype)

        whole_zsc=helix_zsc+sheet_zsc+loop_zsc

        return {
            "helix_z_score":helix_zsc,
            "sheet_z_score":sheet_zsc,
            "loop_z_score":loop_zsc,
            "whole_z_score":whole_zsc
        }

    def rotamer_outliers(self) -> Dict[str, torch.Tensor]:
        """
        Calculate rotamer outliers following mmtbx methodology with proper symmetry handling
        
        Returns:
            Dictionary containing:
            - rotamer_outliers: penalty values for outliers
            - chi_scores: probability scores from rotamer library (0.0-1.0)
            - chi_angles: computed chi angles
            - outliers_count: number of outliers (score < 0.003)
            - favored_count: number of favored rotamers (score >= 0.02)
        """
        OUTLIER_THRESHOLD = 0.003
        ALLOWED_THRESHOLD = 0.02
        device = self.atom_pos.device
        prot_len = self.prot_len

        # Get chi angles from torsion angles (sin/cos format)
        torsion_angles = (
            self.prot.torsion_angles_sin_cos
            if isinstance(self.prot.torsion_angles_sin_cos, torch.Tensor)
            else torch.tensor(self.prot.torsion_angles_sin_cos, device=device)
        )
        
        # Reconstruct angles from sin/cos with numerical stability
        y = torsion_angles[:, 3:7, 0]  # sin values for chi1-chi4
        x = torsion_angles[:, 3:7, 1]  # cos values for chi1-chi4
        eps = 1e-8
        x_safe = x + (x.abs() < eps).float() * eps
        chi_angles = torch.rad2deg(torch.atan2(y, x_safe))

        # Initialize scores to -1 (indicates not evaluated)
        chi_scores = torch.full((prot_len,), -1.0, device=device)

        if isinstance(self.prot.aatype, torch.Tensor):
            res_type_idx = self.prot.aatype.to(device)
        else:
            res_type_idx = torch.tensor(self.prot.aatype, device=device, dtype=torch.long)

        # Process each amino acid type
        for resname, ndt in aaTables.items():
            num_chis = chisPerAA[resname]
            
            # Skip residues with no chi angles (Gly, Ala)
            if num_chis == 0:
                continue
            
            res_index = restype_3_to_index[resname.upper()]
            mask = res_type_idx == res_index
            
            if not mask.any():
                continue

            # Select chi angles for this residue type
            chis_sel = chi_angles[mask, :num_chis]  # [n_res, num_chis]

            # Compute rotamer scores with proper interpolation and symmetry
            scores = self._interpolate_rotamer_scores_differentiable(
                chis_sel, ndt, num_chis, resname=resname
            )
            
            # Update chi_scores for matched residues
            if mask.any():
                mask_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if mask_indices.numel() > 0:
                    # Ensure scores is 1D
                    if scores.dim() == 0:  
                        scores = scores.unsqueeze(0)
                    elif scores.dim() > 1: 
                        scores = scores.flatten()
                    
                    # Create new chi_scores with gradient tracking
                    new_scores = chi_scores.clone()
                    new_scores[mask_indices] = scores
                    chi_scores = new_scores

        # Calculate outlier penalties (only for evaluated residues, chi_scores >= 0)
        # Use smooth penalty for gradient-based optimization
        rotamer_outliers = torch.where(
            chi_scores >= 0, 
            torch.nn.functional.leaky_relu(OUTLIER_THRESHOLD - chi_scores, negative_slope=0.01), 
            torch.zeros_like(chi_scores)
        )
        
        # Count outliers and favored rotamers
        n_outliers = torch.logical_and(chi_scores < OUTLIER_THRESHOLD, chi_scores >= 0).sum()
        n_favored = (chi_scores >= ALLOWED_THRESHOLD).sum()
        
        return {
            "rotamer_outliers": rotamer_outliers,
            "chi_scores": chi_scores,
            "chi_angles": chi_angles,
            "outliers_count": n_outliers,
            "favored_count": n_favored,
        }



    def cbeta_dev_batch(self) -> dict[str, torch.Tensor]:
        """Return C-beta deviations"""

        device = self.atom_pos.device
        chiralty_volume = chiralty(self.atom_pos).to(device)

        ideals = torch.zeros((self.prot_len, 3), dtype=torch.float32, device=device)
        dihedrals = torch.zeros(self.prot_len, dtype=torch.float32, device=device)
        devs = torch.zeros(self.prot_len, dtype=torch.float32, device=device)

        # Mask
        valid_mask = ~torch.all(self.atom_pos[:, 4] == 0, dim=1)
        mask_idx = torch.where(valid_mask)[0]

        N, CA, C, CB = self.atom_pos[:, 0], self.atom_pos[:, 1], self.atom_pos[:, 2], self.atom_pos[:, 4]

        angle_data = self.idealized_ca_tensor[self.prot.aatype]  

        dist           = angle_data[:, 0]                   
        angleCAB_deg   = angle_data[:, 1]                
        dihedralNCAB_deg = angle_data[:, 2]               
        angleNAB_deg   = angle_data[:, 3]                  
        dihedralCNAB_deg = angle_data[:, 4]                 

        dihedralNCAB_deg = torch.where(chiralty_volume > 0, -dihedralNCAB_deg, dihedralNCAB_deg)
        dihedralCNAB_deg = torch.where(chiralty_volume > 0, -dihedralCNAB_deg, dihedralCNAB_deg)

        angleCAB   = torch.deg2rad(angleCAB_deg)
        angleNAB   = torch.deg2rad(angleNAB_deg)
        dihedralNCAB = torch.deg2rad(dihedralNCAB_deg)
        dihedralCNAB = torch.deg2rad(dihedralCNAB_deg)

        betaNCAB = construct_fourth_batch(
            N[mask_idx], CA[mask_idx], C[mask_idx],
            dist[mask_idx], angleCAB[mask_idx], dihedralNCAB[mask_idx],
            method="NCAB"
        )
        betaCNAB = construct_fourth_batch(
            N[mask_idx], CA[mask_idx], C[mask_idx],
            dist[mask_idx], angleNAB[mask_idx], dihedralCNAB[mask_idx],
            method="CNAB"
        )


        betaxyz = (betaNCAB + betaCNAB) / 2

        betadist = torch.norm(CA[mask_idx] - betaxyz, dim=1)
        dist_valid = dist[mask_idx]
        scale_factors = torch.where(betadist != 0, dist_valid / betadist, torch.ones_like(betadist))
        betaxyz = CA[mask_idx] + (betaxyz - CA[mask_idx]) * scale_factors.unsqueeze(1)

        ideals[mask_idx] = betaxyz

        points = torch.stack([N[mask_idx], CA[mask_idx], betaxyz, CB[mask_idx]], dim=1)  # (N,4,3)
        dihedrals[mask_idx] = calc_dihedral_batch(points)

        devs[mask_idx] = torch.norm(CB[mask_idx] - betaxyz, dim=1)

        cbeta_outliers = torch.where(devs >= 0.25, devs, torch.zeros_like(devs))

        return {
            "cbeta_dev": devs,
            "cbeta_outliers": cbeta_outliers
        }



    def _setup_monomer_library_server(self):
        
        mon_lib_srv = mon_lib_server.server()
        ener_lib = mon_lib_server.ener_lib()
        
        if "SS" not in mon_lib_srv.link_link_id_dict:
            dummy_ss_link = cif_types.chem_link()
            dummy_ss_link.id = "SS"
            
            dummy_bond = cif_types.chem_link_bond()
            dummy_bond.atom_1_comp_id = 1
            dummy_bond.atom_id_1 = "SG"
            dummy_bond.atom_2_comp_id = 2
            dummy_bond.atom_id_2 = "SG"
            dummy_bond.value_dist = 2.0
            dummy_bond.value_dist_esd = 0.02
            dummy_ss_link.bond_list = [dummy_bond]
            
            dummy_angle = cif_types.chem_link_angle()
            dummy_angle.atom_1_comp_id = 1
            dummy_angle.atom_id_1 = "CB"
            dummy_angle.atom_2_comp_id = 1
            dummy_angle.atom_id_2 = "SG"
            dummy_angle.atom_3_comp_id = 2
            dummy_angle.atom_id_3 = "SG"
            dummy_angle.value_angle = 104.0
            dummy_angle.value_angle_esd = 2.0
            dummy_ss_link.angle_list = [dummy_angle]
            
            dummy_ss_link.tor_list = []
            
            mon_lib_srv.link_link_id_dict["SS"] = dummy_ss_link
        
        return mon_lib_srv, ener_lib
        
    def _get_atom_index(self, atom_name, residue_name):
        
        if len(residue_name) == 1:
            residue_3letter = residue_constants.restype_1to3.get(residue_name)
        else:
            residue_3letter = residue_name
        
        if residue_3letter is None:
            return None
            
        atom14_names = residue_constants.restype_name_to_atom14_names.get(residue_3letter, [])
        
        if atom_name == 'OXT':
            return 12 
        
        try:
            return atom14_names.index(atom_name)
        except ValueError:
            return None
    def compute_bond_angle_rmsd_from_pdb(self, pdb_path: str, pred_coords_unpad_tensor: torch.Tensor, cache_key: str = None) -> Dict[str, torch.Tensor]:
        """
        Read coordinates from a PDB file and use mmtbx proxies to compute differentiable geometric RMSD.
        Args:
            pdb_path: Path to the PDB file.
            pred_coords_unpad_tensor: Predicted coordinates tensor (N_atoms, 3).
            cache_key: Cache key (optional), used to identify structural topology. If not provided, defaults to pdb_path + num_atoms.
            
        Returns:
            A dictionary containing bond_rmsd and angle_rmsd, all as differentiable tensors.
        """
        if cache_key is None:
            cache_key = f"{pdb_path}_{pred_coords_unpad_tensor.shape[0]}"
        
        cache_valid = (
            self._rmsd_cache_key == cache_key and 
            self._rmsd_grm_cache is not None and 
            self._rmsd_sites_cart_cache is not None and
            self._rmsd_perm_tensor_cache is not None
        )
        if not cache_valid:
            try:
                pdb_inp = iotbx.pdb.input(file_name=pdb_path)
                
                # Setup monomer library server
                mon_lib_srv, ener_lib = self._setup_monomer_library_server()
                
                log_buffer = StringIO()
                
                params = pdb_interpretation.master_params.extract()
                params.restraints_library.cdl = False
                params.restraints_library.omega_cdl = False
                params.c_beta_restraints = False 
                
                processed_pdb_file = pdb_interpretation.process(
                    mon_lib_srv=mon_lib_srv,
                    ener_lib=ener_lib,
                    pdb_inp=pdb_inp,
                    params=params, 
                    substitute_non_crystallographic_unit_cell_if_necessary=True,
                    log=log_buffer 
                )
                
                # Generate geometry restraints manager
                # NOTE: Disable conformation-dependent restraints to avoid atom selection errors
                grm = processed_pdb_file.geometry_restraints_manager(
                    assume_hydrogens_all_missing=True,
                    show_energies=False,
                    plain_pairs_radius=5.0
                )
                
                # Get coordinates and atom order mapping
                pdb_hierarchy = processed_pdb_file.all_chain_proxies.pdb_hierarchy
                xray_structure = pdb_hierarchy.extract_xray_structure()
                sites_cart = xray_structure.sites_cart()
                
                # mmtbx reads PDB with sorted atoms (sort_atoms=True), need to create a mapping
                # Map from mmtbx hierarchy atoms (after sorting) to original pred_coords
                
                # Read unsorted PDB to get the original atom order
                pdb_inp_unsorted = iotbx.pdb.input(file_name=pdb_path)
                hierarchy_unsorted = pdb_inp_unsorted.construct_hierarchy(sort_atoms=False)
                
                # Build atom identifier -> pred_coords index mapping
                # Use (chain_id, resseq, icode, resname, altloc, atom_name) as unique identifier
                pred_idx_map = {}
                for i, atom in enumerate(hierarchy_unsorted.atoms()):
                    labels = atom.fetch_labels()
                    key = (
                        labels.chain_id.strip(),
                        labels.resseq.strip(),
                        labels.icode.strip(),
                        labels.resname.strip(),
                        labels.altloc.strip(),
                        atom.name.strip()
                    )
                    pred_idx_map[key] = i
                # Map from sorted hierarchy to pred_coords
                perm_indices = []
                for atom in pdb_hierarchy.atoms():
                    labels = atom.fetch_labels()
                    key = (
                        labels.chain_id.strip(),
                        labels.resseq.strip(),
                        labels.icode.strip(),
                        labels.resname.strip(),
                        labels.altloc.strip(),
                        atom.name.strip()
                    )
                    if key in pred_idx_map:
                        perm_indices.append(pred_idx_map[key])
                    else:
                        raise ValueError(f"Atom not found in original PDB: {key}")
                
                # for i in range(8000):
                #     atom = list(pdb_hierarchy.atoms())[i]
                #     atom_xyz = atom.xyz  # (x, y, z) tuple
                #     sites_xyz = (sites_cart[i][0], sites_cart[i][1], sites_cart[i][2])
                #     print(f"Atom {i}: hierarchy={atom_xyz}, sites_cart={sites_xyz}")
                #     print(f"  Match: {abs(atom_xyz[0] - sites_xyz[0]) < 0.001}")

                # Convert to tensor (keep on CPU to save GPU memory, move to GPU when using)
                perm_tensor_cpu = torch.tensor(perm_indices, dtype=torch.long, device='cpu')
                energies_sites = grm.energies_sites(
                    sites_cart=sites_cart,
                    compute_gradients=False
                )
                # 🚀 Cache grm, sites_cart, perm_tensor, energies_sites 
                self._rmsd_grm_cache = grm
                self._rmsd_sites_cart_cache = sites_cart
                self._rmsd_perm_tensor_cache = perm_tensor_cpu
                self._rmsd_cache_key = cache_key
                self._energies_sites = energies_sites
                # print(f"💾 Cached GRM and atom mapping (key: {cache_key})")
                
            except Exception as e:
                print(f"Warning: Error in RMSD calculation for {pdb_path}: {e}")
                import traceback
                traceback.print_exc()
                # Return zeros, allow training to continue
                return {
                    "bond_rmsd": torch.tensor(0.0, device=pred_coords_unpad_tensor.device, dtype=torch.float64, requires_grad=True),
                    "angle_rmsd": torch.tensor(0.0, device=pred_coords_unpad_tensor.device, dtype=torch.float64, requires_grad=True),
                }
        else:
            # 🚀 Use cache
            grm = self._rmsd_grm_cache
            sites_cart = self._rmsd_sites_cart_cache
            perm_tensor_cpu = self._rmsd_perm_tensor_cache
            energies_sites = self._energies_sites
            # print(f"🚀 Using cached GRM and atom mapping (key: {cache_key})")
        # Move perm_tensor to target device
        perm_tensor = perm_tensor_cpu.to(pred_coords_unpad_tensor.device)
        pred_coords_aligned = pred_coords_unpad_tensor[perm_tensor]
        
        # Upgrade precision to float64 to reduce numerical errors
        pred_coords_aligned_f64 = pred_coords_aligned.to(dtype=torch.float64)
        # Get energies_sites object (use mmtbx sites_cart to create proxies)
        # bond_deviations = energies_sites.bond_deviations()
        # angle_deviations = energies_sites.angle_deviations()
        # bond_rmsd = torch.tensor(bond_deviations[2], requires_grad=True)
        # angle_rmsd = torch.tensor(angle_deviations[2], requires_grad=True)
        # print(f'accurate bond_rmsd: {bond_rmsd}, angle_rmsd: {angle_rmsd}')
        # Perform RMSD calculation using differentiable, reordered coords tensor
        
        # Compute differentiable bond RMSD
        bond_deltas_sq = []
        bond_proxies = energies_sites.bond_proxies
        
        # Handle simple bonds (intra-residue bonds)
        for proxy in bond_proxies.simple:
            # Only process covalent bonds (origin_id=0)
            if hasattr(proxy, 'origin_id') and proxy.origin_id != 0:
                continue
                
            i_seq, j_seq = proxy.i_seqs
            
            # Compute actual bond length (differentiable, use more stable vector_norm)
            site1 = pred_coords_aligned_f64[i_seq]
            site2 = pred_coords_aligned_f64[j_seq]
            distance_model = torch.linalg.vector_norm(site2 - site1, ord=2)
            
            # Delta = ideal - model (consistent with mmtbx)
            delta = proxy.distance_ideal - distance_model
            bond_deltas_sq.append(delta * delta)
        
        # Handle ASU bonds (symmetry-related bonds, if any)
        for proxy in bond_proxies.asu:
            if hasattr(proxy, 'origin_id') and proxy.origin_id != 0:
                continue
                
            i_seq = proxy.i_seq
            j_seq = proxy.j_seq
            site1 = pred_coords_aligned_f64[i_seq]
            site2 = pred_coords_aligned_f64[j_seq]
            distance_model = torch.linalg.vector_norm(site2 - site1, ord=2)
            delta = proxy.distance_ideal - distance_model
            bond_deltas_sq.append(delta * delta)
        # Compute bond RMSD
        if bond_deltas_sq:
            bond_deltas_sq_tensor = torch.stack(bond_deltas_sq)
            bond_rmsd = torch.sqrt(torch.mean(bond_deltas_sq_tensor))
        else:
            bond_rmsd = torch.tensor(0.0, requires_grad=True)
        
        
        # Compute differentiable angle RMSD
        angle_deltas_sq = []
        angle_proxies = energies_sites.angle_proxies
        
        for proxy in angle_proxies:
            # Only process covalent angles (origin_id=0)
            if hasattr(proxy, 'origin_id') and proxy.origin_id != 0:
                continue
                
            i_seq, j_seq, k_seq = proxy.i_seqs
            
            # Compute actual angle (differentiable, use more stable atan2 method)
            site1 = pred_coords_aligned_f64[i_seq]
            site2 = pred_coords_aligned_f64[j_seq]  # central atom
            site3 = pred_coords_aligned_f64[k_seq]
            
            vec1 = site1 - site2
            vec2 = site3 - site2
            
            # Use atan2(||cross||, dot) to compute angle, consistent with cctbx
            # This is more stable than acos(dot), avoiding numerical issues
            cross_product = torch.cross(vec1, vec2)
            cross_norm = torch.linalg.vector_norm(cross_product, ord=2)
            dot_product = torch.dot(vec1, vec2)
            
            # Compute angle (radians -> degrees)
            angle_model_rad = torch.atan2(cross_norm, dot_product)
            angle_model = angle_model_rad * 180.0 / math.pi
            
            # Delta = model - ideal
            delta = angle_model - proxy.angle_ideal
            
            # Wrap into [-180, 180] (consistent with cctbx)
            # This ensures correct difference (e.g., 179° and -179° are only 2° apart)
            delta = delta - 360.0 * torch.round(delta / 360.0)
            
            angle_deltas_sq.append(delta * delta)
        
        # Compute angle RMSD
        if angle_deltas_sq:
            angle_deltas_sq_tensor = torch.stack(angle_deltas_sq)
            angle_rmsd = torch.sqrt(torch.mean(angle_deltas_sq_tensor))
        else:
            angle_rmsd = torch.tensor(0.0, requires_grad=True)
        # print(f'self debug bond_rmsd: {bond_rmsd}, angle_rmsd: {angle_rmsd}')
        return {
            "bond_rmsd": bond_rmsd,
            "angle_rmsd": angle_rmsd,
        }
