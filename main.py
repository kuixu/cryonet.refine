#!/usr/bin/env python3
"""
CryoNet.Refine Refinement 

This script performs structure refinement using density-guided diffusion.
It freezes all modules except the diffusion module and uses CC loss for optimization.
"""
import os
import sys
# Ensure the project package is importable when this file is run directly.
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also set environment variable for subprocess calls
    if 'PYTHONPATH' not in os.environ:
        os.environ['PYTHONPATH'] = project_root
    elif project_root not in os.environ['PYTHONPATH']:
        os.environ['PYTHONPATH'] = f"{project_root}:{os.environ['PYTHONPATH']}"

import click,time, warnings
from tqdm import tqdm
import shutil
import tarfile
from pathlib import Path
from typing import  Optional
from dataclasses import asdict
import numpy as np
import torch
from CryoNetRefine.data import const
from CryoNetRefine.data.module.inference import prepare_inference_case
from CryoNetRefine.data.feature.featurizer import BoltzFeaturizer
from CryoNetRefine.data.mol import load_canonicals
from CryoNetRefine.data.tokenize.boltz import BoltzTokenizer
from CryoNetRefine.data.types import  Manifest
from CryoNetRefine.data.parse.input import (
    BoltzProcessedInput, DiffusionParams, model_args,
    PairformerArgs, check_inputs, process_inputs
)
from CryoNetRefine.data.parse.restraints import (
    ResolvedUserRestraints,
    load_user_restraints,
    merge_user_restraints_specs,
    resolve_user_restraints,
)
from CryoNetRefine.data.parse.validate import validate_inputs
from CryoNetRefine.data.output.metrics_validation import run_validation
from CryoNetRefine.libs.density.density import DensityInfo
from CryoNetRefine.model.model import CryoNetRefineModel
from CryoNetRefine.model.engine import Engine, RefineArgs, set_seed
from CryoNetRefine.data.write.utils import write_refined_structure
import urllib.request
warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")

MOL_URL = "https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/mols.tar"


def ensure_checkpoint(checkpoint: Optional[str]) -> Path:
    """
    Ensure checkpoint file exists, download if necessary.
    
    Args:
        checkpoint: Path to checkpoint file, or None to use default
    Returns:
        Path to checkpoint file
    """
    # Determine checkpoint path
    if checkpoint is None:
        # Use default location in params directory
        params_dir = Path(__file__).resolve().parent / "params"
        checkpoint_path = params_dir / "CryoNet.Refine_model.pt"
 
    else:
        checkpoint_path = Path(checkpoint)
    
    # Check if checkpoint exists and is not empty
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        # Create params directory if it doesn't exist
        params_dir = Path(__file__).resolve().parent / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if downloaded checkpoint already exists in params directory
        download_url = "https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/CryoNet.Refine_model.pt"
        downloaded_checkpoint = params_dir / "CryoNet.Refine_model.pt"
        click.echo(f"Checkpoint not found or empty. Try to download from {download_url}...")
        
        # If the downloaded checkpoint already exists and is not empty, use it
        if downloaded_checkpoint.exists() and downloaded_checkpoint.stat().st_size > 0:
            click.echo(f"Found existing downloaded checkpoint in params directory: {downloaded_checkpoint}")
            checkpoint_path = downloaded_checkpoint
        else:
            # Download checkpoint from URL
            click.echo(f"Downloading checkpoint from {download_url}...")
            try:
                # Download with progress bar
                response = urllib.request.urlopen(download_url)
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(downloaded_checkpoint, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading checkpoint") as pbar:
                        while True:
                            chunk = response.read(8192)  # 8KB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                click.echo(f"Downloaded checkpoint to {downloaded_checkpoint}")
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint: {e}")
            
            checkpoint_path = downloaded_checkpoint
    
    return checkpoint_path


def ensure_mols_dir(mol_dir: Path) -> Path:
    """Ensure local molecule library exists; download and extract if needed."""
    mol_dir = Path(mol_dir)
    mol_dir.mkdir(parents=True, exist_ok=True)
    ready_flag = mol_dir / ".download_complete"
    if ready_flag.exists() and any(p.name != ".download_complete" for p in mol_dir.iterdir()):
        return mol_dir

    # Directory exists but does not have completion marker: treat as partial/corrupt.
    for item in list(mol_dir.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    tar_path = mol_dir.parent / "mols.tar"
    extract_dir = mol_dir.parent / "_mols_extract_tmp"
    max_retries = 3

    def _validate_tar(path: Path) -> None:
        with tarfile.open(path, "r:*") as tar:
            for _ in tar:
                pass

    click.echo(f"Molecule directory is empty, downloading from {MOL_URL}...")
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if tar_path.exists():
                tar_path.unlink()
            with urllib.request.urlopen(MOL_URL) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with tar_path.open("wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading mols.tar (attempt {attempt}/{max_retries})",
                    ) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
            if total_size > 0 and downloaded != total_size:
                raise RuntimeError(
                    f"Incomplete download: expected {total_size} bytes, got {downloaded} bytes."
                )
            _validate_tar(tar_path)
            last_error = None
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            click.echo(f"Download/validation failed on attempt {attempt}: {e}")
            if tar_path.exists():
                tar_path.unlink()
            if attempt < max_retries:
                time.sleep(2)
    if last_error is not None:
        raise RuntimeError(f"Failed to download valid mols archive after {max_retries} attempts: {last_error}")

    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir)

        src_root = extract_dir / "mols" if (extract_dir / "mols").exists() else extract_dir
        copied = 0
        for item in src_root.iterdir():
            target = mol_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
            copied += 1
        if copied == 0:
            raise RuntimeError("Downloaded mols archive but extracted no files.")
        ready_flag.write_text("ok\n", encoding="utf-8")
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

    click.echo(f"Molecule library ready: {mol_dir}")
    return mol_dir


def build_lightweight_global_batch(case) -> dict:
    """Build a minimal full-structure batch for crop-first refinement."""
    structure = case.tokenized.structure
    tokens = case.tokenized.tokens
    if len(tokens) == 0:
        raise ValueError(f"Record {case.record.id} has no tokens.")
    atom_end = (tokens["atom_idx"].astype(np.int64) + tokens["atom_num"].astype(np.int64)).max()
    n_atoms = int(atom_end)
    if n_atoms == 0:
        raise ValueError(f"Record {case.record.id} has no atoms.")

    if len(structure.ensemble) == 0:
        raise ValueError(f"Record {case.record.id} has empty ensemble table.")
    coord_offset = int(structure.ensemble[0]["atom_coord_idx"])
    coords_np = structure.coords[coord_offset : coord_offset + n_atoms]["coords"].astype(np.float32)
    template_coords = torch.from_numpy(coords_np).unsqueeze(0).unsqueeze(0)

    atom_pad_mask = torch.ones((1, n_atoms), dtype=torch.bool)
    if "is_present" in structure.atoms.dtype.names:
        atom_present_np = structure.atoms[:n_atoms]["is_present"].astype(bool)
    else:
        atom_present_np = np.ones(n_atoms, dtype=bool)
    atom_present = torch.from_numpy(atom_present_np).unsqueeze(0)

    # Global clash auxiliary features (memory-light versions).
    residue_index = torch.from_numpy(tokens["res_idx"].astype(np.int64)).unsqueeze(0)
    atom_token_index_np = np.zeros(n_atoms, dtype=np.int64)
    for tok in tokens:
        tok_idx = int(tok["token_idx"])
        a0 = int(tok["atom_idx"])
        a1 = a0 + int(tok["atom_num"])
        atom_token_index_np[a0:a1] = tok_idx
    atom_token_index = torch.from_numpy(atom_token_index_np).unsqueeze(0)

    ref_element_idx_np = np.zeros(n_atoms, dtype=np.int64)
    # Prefer the same source as legacy featurizer: RDKit atom atomic number
    # via token res_name + atom name mapping.
    for tok in tokens:
        res_name = str(tok["res_name"])
        mol = case.molecules.get(res_name)
        if mol is None:
            continue
        atom_name_to_ref = {}
        for a in mol.GetAtoms():
            if a.HasProp("name"):
                atom_name_to_ref[a.GetProp("name")] = a
        a0 = int(tok["atom_idx"])
        a1 = a0 + int(tok["atom_num"])
        token_atoms = structure.atoms[a0:a1]
        for local_i, atom in enumerate(token_atoms):
            atom_name = str(atom["name"])
            if atom_name in atom_name_to_ref:
                ref_element_idx_np[a0 + local_i] = int(atom_name_to_ref[atom_name].GetAtomicNum())

    # Fallback for unresolved atoms from name heuristic (keeps full coverage).
    unresolved = ref_element_idx_np == 0
    if np.any(unresolved):
        symbol_to_atomic = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "P": 15,
            "S": 16,
            "CL": 17,
            "K": 19,
            "CA": 20,
            "MN": 25,
            "FE": 26,
            "CO": 27,
            "NI": 28,
            "CU": 29,
            "ZN": 30,
            "SE": 34,
            "BR": 35,
            "I": 53,
            "MG": 12,
            "NA": 11,
        }
        atom_names = structure.atoms[:n_atoms]["name"]
        for i in np.where(unresolved)[0]:
            name = str(atom_names[i]).strip().upper()
            alpha = "".join(ch for ch in name if ch.isalpha())
            atomic_num = 0
            if len(alpha) >= 2 and alpha[:2] in symbol_to_atomic:
                atomic_num = symbol_to_atomic[alpha[:2]]
            elif len(alpha) >= 1 and alpha[0] in symbol_to_atomic:
                atomic_num = symbol_to_atomic[alpha[0]]
            ref_element_idx_np[i] = atomic_num
    ref_element = torch.from_numpy(ref_element_idx_np).unsqueeze(0).clamp(0, const.num_elements - 1)

    return {
        "record": [case.record],
        "template_coords": template_coords,
        "atom_pad_mask": atom_pad_mask,
        "atom_resolved_mask": atom_present,
        "template_atom_present_mask": atom_present.unsqueeze(1),
        "residue_index": residue_index.long(),
        "atom_token_index": atom_token_index.long(),
        "ref_element": ref_element.long(),
    }


def report_restraint_deviation(
    record_id: str,
    coords: torch.Tensor,
    restraints: ResolvedUserRestraints | None,
) -> None:
    """Print final deviation between refined geometry and restraint ideals."""
    if restraints is None:
        return
    if coords.ndim == 3:
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] != 3:
        return

    bond_abs_deltas: list[tuple[float, str]] = []
    angle_abs_deltas: list[tuple[float, str]] = []
    atom_lookup = restraints.atom_lookup or {}

    for bond in restraints.bonds:
        v = coords[bond.atom_idx1] - coords[bond.atom_idx2]
        dist = torch.linalg.norm(v).item()
        delta = dist - float(bond.distance_ideal)
        desc = (
            f"{atom_lookup.get(bond.atom_idx1, str(bond.atom_idx1))} <-> "
            f"{atom_lookup.get(bond.atom_idx2, str(bond.atom_idx2))}"
        )
        bond_abs_deltas.append((abs(delta), desc))

    for angle in restraints.angles:
        v1 = coords[angle.atom_idx1] - coords[angle.atom_idx2]
        v2 = coords[angle.atom_idx3] - coords[angle.atom_idx2]
        n1 = torch.linalg.norm(v1)
        n2 = torch.linalg.norm(v2)
        if n1.item() == 0.0 or n2.item() == 0.0:
            continue
        cos_theta = torch.clamp(torch.dot(v1, v2) / (n1 * n2), min=-1.0 + 1e-7, max=1.0 - 1e-7)
        model_angle = torch.rad2deg(torch.acos(cos_theta)).item()
        delta = model_angle - float(angle.angle_ideal_deg)
        desc = (
            f"{atom_lookup.get(angle.atom_idx1, str(angle.atom_idx1))} - "
            f"{atom_lookup.get(angle.atom_idx2, str(angle.atom_idx2))} - "
            f"{atom_lookup.get(angle.atom_idx3, str(angle.atom_idx3))}"
        )
        angle_abs_deltas.append((abs(delta), desc))

    if bond_abs_deltas:
        bond_vals = [x[0] for x in bond_abs_deltas]
        click.echo(
            f"[{record_id}] Bond restraint deviation |mean abs|={np.mean(bond_vals):.4f} A, "
            f"max abs={np.max(bond_vals):.4f} A, n={len(bond_vals)}"
        )
        worst = sorted(bond_abs_deltas, key=lambda x: x[0], reverse=True)[:3]
        for value, desc in worst:
            click.echo(f"  - worst bond deviation for first 3 atoms: |delta|={value:.4f} A :: {desc}")
    if angle_abs_deltas:
        angle_vals = [x[0] for x in angle_abs_deltas]
        click.echo(
            f"[{record_id}] Angle restraint deviation |mean abs|={np.mean(angle_vals):.3f} deg, "
            f"max abs={np.max(angle_vals):.3f} deg, n={len(angle_vals)}"
        )
        worst = sorted(angle_abs_deltas, key=lambda x: x[0], reverse=True)[:3]
        for value, desc in worst:
            click.echo(f"  - worst angle deviation for first 3 atoms: |delta|={value:.3f} deg :: {desc}")

@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_suffix", type=str, help="Output suffix", default="CryoNet.Refine")
@click.option("--out_dir", type=click.Path(exists=False), help="Output directory", default="./refine_results")
@click.option("--checkpoint", type=click.Path(exists=False), help="Model checkpoint", default=Path(__file__).resolve().parent / "params/CryoNet.Refine_model.pt")
@click.option("--seed", type=int, help="Random seed", default=11)
@click.option("--target_density", multiple=True, type=click.Path(exists=True), help="Target density map (.mrc file)", default=None)
@click.option("--resolution", multiple=True, type=float, help="Resolution for density map operations", default=None)
@click.option("--max_tokens", type=int, help="Maximum number of tokens for cropping (0 to disable)", default=512)
@click.option("--gamma_0", type=float, help="Gamma 0 parameter", default=-0.5)
@click.option("--recycles", type=int, help="Number of refinement recycles", default=300)
@click.option("--enable_cropping", is_flag=True, help="Enable cropping for large structures", default=True)
@click.option("--cond_early_stop", type=str, help="Condition early stop", default="loss")
@click.option("--clash", type=float, help="Weight for clash loss", default=0.01)
@click.option("--nonbonded", type=float, help="Weight for nonbonded loss", default=50)
@click.option("--den", type=float, help="Weight for density loss", default=20.0)
@click.option("--rama", type=float, help="Weight for rama loss", default=500.0)
@click.option("--rotamer", type=float, help="Weight for rotamer loss", default=500.0)
@click.option("--bond", type=float, help="Weight for bond loss", default=50)
@click.option("--angle", type=float, help="Weight for angle loss", default=0.25)
@click.option("--restraints_file", type=click.Path(exists=True), help="User bond/angle restraints file (.json/.yaml)", default=None)
@click.option("--use_user_restraints/--no-use_user_restraints", is_flag=True, help="Enable explicit user bond/angle restraints", default=False)
@click.option("--user_bond", type=float, help="Weight for user bond restraint loss", default=1.0)
@click.option("--user_angle", type=float, help="Weight for user angle restraint loss", default=1.0)
@click.option("--user_plane_parallelity", type=float, help="Weight for user plane parallelity restraint loss", default=1.0)
@click.option("--cbeta", type=float, help="Weight for cbeta loss", default=50.0)
@click.option("--ramaz", type=float, help="Weight for ramaz loss", default=0.1)
@click.option("--learning_rate", type=float, help="Learning rate for refinement", default=1.8e-3)
@click.option("--max_norm_sigmas_value", type=float, help="max norm sigmas value", default=1.0)
@click.option("--num_workers", type=int, help="Number of data loader workers", default=0)
@click.option("--use_global_clash/--no-use_global_clash", is_flag=True, help="Global clash flag", default=True)
@click.option("--validate_output", is_flag=True, help="Validate output flag", default=False)
@click.option("--ignore_origin", is_flag=True, help="Ignore density origin flag", default=False)
@click.option("--progress", is_flag=True, help="Write progress values to job status updates", default=False)
@click.option(
    "--auto_metal_restraints/--no-auto_metal_restraints",
    is_flag=True,
    help="Detect default metal coordination bond restraints during preprocessing",
    default=True,
)
@click.option(
    "--metal_restraint_distance_strategy",
    type=click.Choice(["input", "library"]),
    help="Ideal-distance strategy for default metal bond restraints",
    default="library",
)
@click.option(
    "--metal_coordination_cutoff",
    type=float,
    help="Distance cutoff for automatic metal coordination detection",
    default=3.0,
)
@click.option(
    "--protein_secondary_structure_restraints/--no-protein_secondary_structure_restraints",
    is_flag=True,
    help="Generate protein helix and sheet restraints during preprocessing",
    default=False,
)
@click.option(
    "--nucleic_secondary_structure_restraints/--no-nucleic_secondary_structure_restraints",
    is_flag=True,
    help="Generate nucleic-acid base-pair and stacking restraints during preprocessing",
    default=False,
)
@click.option(
    "--secondary_structure_mode",
    type=click.Choice(["auto", "detect", "existing"]),
    help="Secondary-structure source: auto prefers existing protein annotation, detect forces search, existing reads only annotations",
    default="auto",
)
@click.option(
    "--secondary_structure_include_single_strands/--no-secondary_structure_include_single_strands",
    is_flag=True,
    help="Keep single beta strands from the built-in secondary-structure search",
    default=False,
)
def refine(
    data: str,
    out_dir: str,
    checkpoint: Optional[str] = None,
    out_suffix: str = "refine",
    seed: Optional[int] = 11,
    target_density: Optional[tuple] = None,
    resolution: Optional[tuple] = None,
    max_tokens: int = 512,
    recycles: int = 300,
    gamma_0: float = -0.5,
    enable_cropping: bool = True,
    cond_early_stop: str = "loss",
    clash: float = 0.01,
    nonbonded: float = 50.0,
    den: float = 20.0,
    rama: float = 500.0,
    rotamer: float = 500.0,
    bond: float = 50.0,
    angle: float = 0.25,
    restraints_file: Optional[str] = None,
    use_user_restraints: bool = True,
    user_bond: float = 1.0,
    user_angle: float = 1.0,
    user_plane_parallelity: float = 1.0,
    cbeta: float = 1.0,
    ramaz: float = 0.1,
    learning_rate: float = 1.8e-3,
    max_norm_sigmas_value: float = 1.0,
    use_global_clash: bool = True,
    validate_output: bool = False,
    ignore_origin: bool = False,
    progress: bool = False,
    auto_metal_restraints: bool = True,
    metal_restraint_distance_strategy: str = "input",
    metal_coordination_cutoff: float = 3.0,
    protein_secondary_structure_restraints: bool = False,
    nucleic_secondary_structure_restraints: bool = False,
    secondary_structure_mode: str = "auto",
    secondary_structure_include_single_strands: bool = False,
) -> None:
    """Run structure refinement with Boltz.""" 
    start_time = time.time()
    set_seed(seed)
    data = Path(data).expanduser()
    data_stem = data.stem
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = check_inputs(data)
    validate_inputs(
        input_path=data,
        target_density=target_density,
        resolution=resolution,
        enable_progress=progress,
    )
    mol_dir = Path(__file__).resolve().parent / "CryoNetRefine" / "data" / "mols"
    mol_dir = ensure_mols_dir(mol_dir)
    # Load processed data !!
    processed_dir = out_dir / f"processed_{data_stem}"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        click.echo(f"Removed intermediate directory: {processed_dir}")
    process_inputs(
        data=data,
        data_stem=data_stem,
        out_dir=out_dir,
        mol_dir=mol_dir,
        preprocessing_threads=1,
        auto_metal_restraints=auto_metal_restraints,
        metal_restraint_distance_strategy=metal_restraint_distance_strategy,
        metal_coordination_cutoff=metal_coordination_cutoff,
        protein_secondary_structure_restraints=protein_secondary_structure_restraints,
        nucleic_secondary_structure_restraints=nucleic_secondary_structure_restraints,
        secondary_structure_mode=secondary_structure_mode,
        secondary_structure_include_single_strands=secondary_structure_include_single_strands,
        enable_progress=progress,
    )
    # Load manifest
    manifest = Manifest.load(out_dir / f"processed_{data_stem}" / "manifest.json")
    processed = BoltzProcessedInput(
        manifest=manifest,
        constraints_dir=processed_dir / "constraints" if (processed_dir / "constraints").exists() else None,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    # Setup model parameters
    diffusion_params = DiffusionParams(gamma_0=gamma_0, max_norm_sigmas_value=max_norm_sigmas_value)
    pairformer_args = PairformerArgs()
    
    # Ensure checkpoint exists, download if necessary
    checkpoint = ensure_checkpoint(checkpoint)
    data_dir = str(processed.template_dir)
    # Try loading directly to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_module = CryoNetRefineModel.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=model_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,  # Auto-detect kernel availability
        pairformer_args=asdict(pairformer_args),
    )
    # Move to device with non_blocking=True for async transfer
    if device == "cuda":
        model_module = model_module.to(device, non_blocking=True)
    else:
        model_module = model_module.to(device)
    # Load target density map if provided
    if den == 0.0:
        target_density = None
        target_density_obj = None
        resolution = None
    else:
        assert target_density is not None and resolution is not None, "Target density and resolution must be provided"
        if len(target_density) == 1:
            target_density_obj = [DensityInfo(mrc_path=target_density[0], resolution=resolution[0], datatype="torch", device=device)]
        else:
            target_density_obj = [DensityInfo(mrc_path=td, resolution=res, datatype="torch", device=device) for td, res in zip(target_density, resolution)]
    
    refine_args = RefineArgs()
    refine_args.data_dir = data_dir
    refine_args.num_recycles = recycles
    refine_args.weight_dict["clash"] = clash
    refine_args.weight_dict["nonbonded"] = nonbonded
    refine_args.weight_dict["den"] = den
    refine_args.weight_dict["rama"] = rama
    refine_args.weight_dict["rotamer"] = rotamer
    refine_args.weight_dict["bond"] = bond
    refine_args.weight_dict["angle"] = angle
    refine_args.weight_dict["user_bond"] = user_bond
    refine_args.weight_dict["user_angle"] = user_angle
    refine_args.weight_dict["user_plane_parallelity"] = user_plane_parallelity
    refine_args.weight_dict["cbeta"] = cbeta
    refine_args.weight_dict["ramaz"] = ramaz
    refine_args.learning_rate = learning_rate
    refine_args.use_global_clash = use_global_clash
    refine_args.restraints_file = restraints_file
    refine_args.use_user_restraints = use_user_restraints or restraints_file is not None
    refine_args.auto_metal_restraints = auto_metal_restraints
    refine_args.metal_restraint_distance_strategy = metal_restraint_distance_strategy
    refine_args.metal_coordination_cutoff = metal_coordination_cutoff
    refine_args.protein_secondary_structure_restraints = protein_secondary_structure_restraints
    refine_args.nucleic_secondary_structure_restraints = nucleic_secondary_structure_restraints
    refine_args.secondary_structure_mode = secondary_structure_mode
    refine_args.secondary_structure_include_single_strands = secondary_structure_include_single_strands
    user_restraints_spec = None
    if refine_args.use_user_restraints:
        if restraints_file is None:
            raise click.UsageError("--use_user_restraints requires --restraints_file.")
        user_restraints_spec = load_user_restraints(restraints_file)
    # pdb_id = data[0].name.split('.')[0]
    input_name = data[0].name
    if input_name.endswith(".cif"):
        pdb_id = input_name[:-4]
    elif input_name.endswith(".pdb"):
        pdb_id = input_name[:-4]
    else:
        pdb_id = input_name
    refiner = Engine(
        model_module, 
        refine_args, 
        model_args,
        device, 
        target_density_obj, 
        max_tokens=max_tokens,
        enable_cropping=enable_cropping,
        pdb_id=pdb_id,  
    )
    tokenizer = BoltzTokenizer()
    canonicals = load_canonicals(mol_dir)
    crop_featurizer = BoltzFeaturizer()
    # Perform refinement for each structure (crop-first streaming path)
    for batch_idx, record in enumerate(tqdm(processed.manifest.records, desc="Refining structures")):
        click.echo(f"\nProcessing batch {batch_idx} (record={record.id})")
        case = prepare_inference_case(
            record=record,
            mol_dir=mol_dir,
            tokenizer=tokenizer,
            canonicals=canonicals,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
        )
        default_restraints_spec = None
        if processed.constraints_dir is not None:
            default_restraints_path = processed.constraints_dir / f"{record.id}.json"
            if default_restraints_path.exists():
                default_restraints_spec = load_user_restraints(default_restraints_path)
        combined_restraints_spec = merge_user_restraints_specs(default_restraints_spec, user_restraints_spec)
        if combined_restraints_spec is not None:
            resolved_user_restraints = resolve_user_restraints(combined_restraints_spec, case.tokenized.structure)
            refiner.set_user_restraints(resolved_user_restraints)
            click.echo(
                f"Loaded default atom restraints for {record.id}: "
                f"{len(resolved_user_restraints.bonds)} bonds, "
                f"{len(resolved_user_restraints.angles)} angles, "
                f"{len(resolved_user_restraints.plane_parallelities)} plane parallelities"
            )
        else:
            refiner.set_user_restraints(None)
        batch = build_lightweight_global_batch(case)
        crop_plans = refiner.molecule_aware_cropper.plan_crops_from_tokenized(case.tokenized)
        if len(crop_plans) == 0:
            raise RuntimeError(f"No valid crops found for record {record.id}")
        refiner.set_crop_first_context(
            tokenized=case.tokenized,
            molecules=case.molecules,
            record=case.record,
            crop_plans=crop_plans,
            override_method=None,
            random_seed=42,
            featurizer=crop_featurizer,
        )

        offset = None
        if ignore_origin and target_density_obj:
            offset = target_density_obj[0].offset.clone()
            batch["template_coords"] = batch["template_coords"] - offset.to(batch["template_coords"].device)
            target_density_obj[0].offset = torch.tensor([0.0, 0.0, 0.0], device=target_density_obj[0].device)
        refined_coords, _ = refiner.refine(batch, target_density_obj, processed.template_dir, out_dir, cond_early_stop=cond_early_stop)
        if ignore_origin and offset is not None:
            refined_coords = refined_coords + offset.to(refined_coords.device)
            target_density_obj[0].offset = offset
        report_restraint_deviation(record.id, refined_coords, refiner.user_restraints)
        
        # Get best results info from refiner
        best_iteration, best_loss, best_cc = (
            getattr(refiner, 'best_iteration', None),
            getattr(refiner, 'best_loss', None),
            getattr(refiner, 'best_cc', None),
        )
   
        # Save refined structure (best result)
        output_path = out_dir / f"{record.id}_{out_suffix}.cif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_refined_structure(batch, refined_coords, data_dir, output_path)
        if validate_output:
            map_path = str(Path(target_density_obj[0].path).expanduser().absolute())
            input_pdb_path = str(Path(data[0]).expanduser().resolve())
            run_validation(map_path, input_pdb_path, target_density_obj[0].resolution, "metrics_in")
            output_pdb_path = str(Path(output_path).expanduser().absolute())
            run_validation(map_path, output_pdb_path, target_density_obj[0].resolution, "metrics_out")
            click.echo(f"Validation completed for {output_path}")
        click.echo(f"Best Loss: {best_loss:.3f}, CC: {best_cc:.3f} at iteration {best_iteration}")
        click.echo(f"Refined structure {batch_idx} saved to {output_path}")
    click.echo("Refinement completed!")
    end_time = time.time()
    click.echo(f"Refinement completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    refine()
