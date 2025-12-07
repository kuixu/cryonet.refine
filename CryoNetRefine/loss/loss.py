import time, os
import torch
import torch.nn.functional as F
from CryoNetRefine.data import const
from CryoNetRefine.data.write.utils import write_refined_structure_pdb, write_refined_structure_pdb_by_crop
from CryoNetRefine.libs.density.density import DensityInfo, mol_atom_density
from CryoNetRefine.loss.geometric import GeometricAdapter, GeometricMetricWrapper

def compute_overall_cc_loss(predicted_coords, target_density, feats, resolution=1.9, voxel_size=1.0, atom_weights=None):
    """
    Compute overall cross-correlation loss between predicted structure and target density.
    input: predicted_coords: [batch_size, num_atoms, 3]
    input: target_density: DensityInfo
    input: feats: dict, contains atom_pad_mask and atom_resolved_mask
    input: resolution: float, resolution of the target density
    input: voxel_size: float, voxel size of the target density
    output: cc_cos: float, mean cosine similarity
    output: loss_den_cos: float, mean 1-cosine similarity
    """
    batch_size, num_atoms, _ = predicted_coords.shape
    device = predicted_coords.device

    if hasattr(target_density, 'device') and target_density.device != device:
        target_density = target_density.to(device)
    if (batch_size != 1):
        raise ValueError("batch_size must be 1")
    
    for batch_idx in range(batch_size):
        # 
        current_atom_coords = predicted_coords[batch_idx]  # [num_atoms, 3]
        # unpad
        pad_masks = feats["atom_pad_mask"].squeeze(0)
        resolved_mask = feats.get("atom_resolved_mask", pad_masks).squeeze(0)
        # ensure same length
        L = min(pad_masks.shape[0], resolved_mask.shape[0], current_atom_coords.shape[0])
        pad_masks = pad_masks[:L]
        resolved_mask = resolved_mask[:L]
        current_atom_coords = current_atom_coords[:L]
        atom_mask = pad_masks.bool() & resolved_mask.bool()
        current_atom_coords = current_atom_coords[atom_mask]
        # atom_weight = 14.0
        atom_weight = atom_weights[atom_mask]
        mol_density, nxyz = mol_atom_density(
            current_atom_coords,
            atom_weight,
            resolution,
            voxel_size=voxel_size,
            datatype="torch"
        )

        offset = nxyz * voxel_size ## fix!!!
        mol_den = DensityInfo(
            density=mol_density,
            offset=offset,
            apix=voxel_size,
            datatype="torch",
            device=device
        )

        t_ov, m_ov = target_density.overlap_right(mol_den)
        cc_cos = F.cosine_similarity(t_ov.reshape(1, -1), m_ov.reshape(1, -1))[0]
        loss_den_cos = 1 - cc_cos
    return cc_cos, loss_den_cos



def probe_style_clash_loss(
    predicted_coords: torch.Tensor,
    feats: dict,
    clash_cutoff: float = -0.4,  
    softness: float = 10.0,       
    exclude_neighbor_distance: int = 1,  
    eps: float = 1e-6,
):

    device = predicted_coords.device
    B, N, _ = predicted_coords.shape

    atom_mask = feats["atom_pad_mask"].bool()               # [B, N]
    coords = predicted_coords                               # [B, N, 3]

    ref_element = feats["ref_element"].float().to(device)   # [B, N, E]

    vdw_radii_table = torch.zeros(
        const.num_elements, dtype=torch.float32, device=device
    )
    vdw_radii_table[1:1 + len(const.vdw_radii)] = torch.tensor(
        const.vdw_radii, dtype=torch.float32, device=device
    )
    # [B, N, E] @ [E] -> [B, N]
    atom_vdw_radii = (ref_element @ vdw_radii_table.unsqueeze(-1)).squeeze(-1)

    # coords: [B, N, 3]
    diffs = coords.unsqueeze(2) - coords.unsqueeze(1)       # [B, N, N, 3]
    dists = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + eps)  # [B, N, N]
    r_i = atom_vdw_radii.unsqueeze(2)                        # [B, N, 1]
    r_j = atom_vdw_radii.unsqueeze(1)                        # [B, 1, N]
    r_sum = r_i + r_j                                        # [B, N, N]

    gap = dists - r_sum  

    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # [B, N, N]
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1, N, N]
    pair_mask = pair_mask & (~eye)

    if "atom_to_token" in feats and "residue_index" in feats:
        atom_to_token = feats["atom_to_token"].float().to(device)        # [B, N, L]
        residue_index = feats["residue_index"].float().to(device)        # [B, L]
        atom_res_idx = torch.bmm(atom_to_token, residue_index.unsqueeze(-1)).squeeze(-1)  # [B, N]
        res_diff = atom_res_idx.unsqueeze(2) - atom_res_idx.unsqueeze(1)  # [B, N, N]
        neighbor_mask = torch.abs(res_diff) <= float(exclude_neighbor_distance)
        pair_mask = pair_mask & (~neighbor_mask.bool())


    x = gap - clash_cutoff
    clash_prob = torch.sigmoid(-softness * x) * pair_mask.float()  # [B, N, N]

    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    clash_prob = clash_prob * triu_mask.unsqueeze(0)

    soft_n_clashes = clash_prob.sum(dim=(1, 2))  # [B]
    n_atoms = atom_mask.sum(dim=1).clamp(min=1).float()  # [B]
    clashscore = soft_n_clashes * 1000.0 / n_atoms       # [B]

    return clashscore, soft_n_clashes

def compute_geometric_losses(crop_idx, predicted_coords, feats, device, geom_root=None, top8000_path=None, data_dir=None, geometric_adapter=None, geometric_wrapper=None, weights=None, is_nucleic_acid=False):
    """
    Compute geometric losses via simplified wrapper.
    Returns (loss_dict)
    """
    loss_dict = {}
    time_loss_dict = {}
    if geometric_wrapper is None:
        wrapper = GeometricMetricWrapper(geom_root=geom_root, device=device, top8000_path=top8000_path)
    else:
        wrapper = geometric_wrapper

    data_dir_str = str(data_dir)
    record_id = feats["record"][0].id
    output_path = data_dir_str + f"/{record_id}_crop{crop_idx}_temp.pdb"
    if feats.get("is_cropped", False):
        write_refined_structure_pdb_by_crop(predicted_coords, feats, data_dir, output_path)
    else:
        write_refined_structure_pdb(predicted_coords, feats, data_dir, output_path)
    
    if not is_nucleic_acid:
        if geometric_adapter is None:
            adapter = GeometricAdapter(device=str(device),data_dir=data_dir)
        else:
            adapter = geometric_adapter
        coords_geo, seq_idx, res_mask, atom_mask_geo = adapter.convert(predicted_coords, feats)
        loss_dict , time_loss_dict = wrapper.compute(crop_idx, coords_geo, seq_idx,  atom_mask_geo, weights, output_path)

    if weights.get("bond", 0.0) or weights.get("angle", 0.0) > 0.0:
        bond_angle_start_time = time.time()

        model_coord = predicted_coords[0]  # Take first model
        pad_masks = feats["atom_pad_mask"].squeeze(0)
        pred_coords_unpad_tensor = model_coord[pad_masks.bool()]
        crop_key = f"{crop_idx}"
        if not is_nucleic_acid:
            gm = wrapper._crop_cache[crop_key]
        else:
            gm = wrapper.GeoMetric 
        cache_key = f"{record_id}_crop{crop_idx}"

        result = gm.compute_bond_angle_rmsd_from_pdb(output_path, pred_coords_unpad_tensor, cache_key=cache_key)
        bond_rmsd = result["bond_rmsd"]
        angle_rmsd = result["angle_rmsd"]

        if weights.get("bond", 0.0) > 0.0:
            loss_dict["bond"] = bond_rmsd
        if weights.get("angle", 0.0) > 0.0:
            loss_dict["angle"] = angle_rmsd
        bond_angle_end_time = time.time()
        time_loss_dict["bond_angle"] = bond_angle_end_time - bond_angle_start_time
    else:
        loss_dict["bond"] = torch.tensor(0.0, device=device)
        loss_dict["angle"] = torch.tensor(0.0, device=device)
        time_loss_dict["bond_angle"] = 0.0


    if weights.get("clash", 0.0) > 0.0:
        clash_start_time = time.time()
        clashscore, _ = probe_style_clash_loss(
            predicted_coords,  # [B, N, 3]
            feats,
            clash_cutoff=-0.4,
            softness=10.0,
        )
        loss_dict["clash"] = clashscore.mean()
        time_loss_dict["clash"] = time.time() - clash_start_time
        
    else:
        loss_dict["clash"] = torch.zeros((), device=device)
        time_loss_dict["clash"] = 0.0
      


    os.system(f"rm {output_path}")
    return loss_dict, time_loss_dict




def refine_loss(crop_idx, predicted_coords, target_density, feats, args, geometric_adapter=None, geometric_wrapper=None, atom_weights=None):
    device = predicted_coords.device
    weights = args.weight_dict

    total_loss = torch.zeros((), device=device, dtype=predicted_coords.dtype)
    loss_dict = {}
    time_loss_dict = {}
    cc_value = torch.tensor(0.0, device=device, dtype=predicted_coords.dtype)

    # Density CC loss
    start_time = time.time()

    den_w = float(weights.get("den", 0.0))
    if den_w > 0.0 and target_density is not None:
        cc_value, cc_loss = compute_overall_cc_loss(
            predicted_coords, target_density, feats,
            resolution=args.resolution,
            voxel_size=target_density.voxel_size,
            atom_weights=atom_weights
        )

        total_loss = total_loss + den_w * cc_loss
        loss_dict["CC"] = cc_value
        loss_dict["cc_loss"] = cc_loss * den_w
        CC_time = time.time() - start_time
    else:
        cc_loss = torch.tensor(0.0, device=device, dtype=predicted_coords.dtype)
        cc_value = torch.tensor(0.0, device=device, dtype=predicted_coords.dtype)
        loss_dict["CC"] = cc_value
        loss_dict["cc_loss"] = cc_loss * den_w
        CC_time = time.time() - start_time
    
    geo_w = float(weights.get("geometric", 0.0))
    
    # Check if this is a nucleic acid sequence - skip geometric losses for DNA/RNA
    # feats is a dict (crop_batch), so use dict access
    sequence_type = feats.get('molecule_type', 'PROTEIN')
    is_nucleic_acid = sequence_type in ['DNA', 'RNA', 'NONPOLYMER']
    
    # if geo_w > 0.0 and not is_nucleic_acid:
    if geo_w > 0.0:
        # Geometric losses (only for protein sequences)
        geo_losses, time_loss_dict = compute_geometric_losses(
            crop_idx,
            predicted_coords,
            feats,
            device,
            geom_root=getattr(args, "geo_metric_root", None),
            data_dir=getattr(args, "data_dir", None),
            geometric_adapter=geometric_adapter,
            geometric_wrapper=geometric_wrapper,
            weights=weights,
            is_nucleic_acid=is_nucleic_acid
        )
        # Expose each component with weights
        for name, value in geo_losses.items():
            loss_dict[f"geom_{name}"] = value * weights.get(name, 0.0)

        # Combine with weights (allow per-term weights, fallback to global 'geometric')
        geom_weight = float(weights.get("geometric", 0.0))
        rama_w = float(weights.get("rama", geom_weight))
        rotamer_w = float(weights.get("rotamer", geom_weight))
        cbeta_w = float(weights.get("cbeta", geom_weight))
        bond_w = float(weights.get("bond", geom_weight))
        angle_w = float(weights.get("angle", geom_weight))
        ramaz_w = float(weights.get("ramaz", geom_weight))
        clash_w = float(weights.get("clash", geom_weight))
        total_loss = (
            total_loss
            + rama_w * geo_losses.get("rama", torch.zeros((), device=device))
            + rotamer_w * geo_losses.get("rotamer", torch.zeros((), device=device))
            + cbeta_w * geo_losses.get("cbeta", torch.zeros((), device=device))
            + bond_w * geo_losses.get("bond", torch.zeros((), device=device))
            + angle_w * geo_losses.get("angle", torch.zeros((), device=device))
            + ramaz_w * geo_losses.get("ramaz", torch.zeros((), device=device))
            + clash_w * geo_losses.get("clash", torch.zeros((), device=device))
        )
    if is_nucleic_acid:
        # For nucleic acid sequences, skip geometric losses but still populate loss_dict with zeros
        loss_dict["geom_rama"] = torch.zeros((), device=device)
        loss_dict["geom_rotamer"] = torch.zeros((), device=device)
        loss_dict["geom_cbeta"] = torch.zeros((), device=device)
        loss_dict["geom_ramaz"] = torch.zeros((), device=device)
        time_loss_dict["build_prot"] = 0.0
        time_loss_dict["rama"] = 0.0
        time_loss_dict["rotamer"] = 0.0
        time_loss_dict["cbeta"] = 0.0
        time_loss_dict["ramaz"] = 0.0
    
    time_loss_dict["CC_time"] = CC_time

    return cc_value, total_loss, loss_dict, time_loss_dict