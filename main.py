#!/usr/bin/env python3
"""
CryoNet.Refine Refinement 

This script performs structure refinement using density-guided diffusion.
It freezes all modules except the diffusion module and uses CC loss for optimization.
"""
import os
import sys
# Set PYTHONPATH to include project root if not already set
# This ensures CryoNetRefine package can be found when running compute_ss.py
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
from pathlib import Path
from typing import  Optional
from dataclasses import asdict
import torch
from CryoNetRefine.data.module.inference import BoltzInferenceDataModule
from CryoNetRefine.data.types import  Manifest
from CryoNetRefine.data.parse.input import (
    BoltzProcessedInput, DiffusionParams, model_args,
    PairformerArgs, check_inputs, process_inputs
)
from CryoNetRefine.libs.density.density import DensityInfo
from CryoNetRefine.model.model import CryoNetRefineModel
from CryoNetRefine.model.engine import Engine, RefineArgs, set_seed
from CryoNetRefine.data.write.utils import write_refined_structure
import urllib.request
warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")


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
        checkpoint_path = params_dir / "cryonet.refine_model_checkpoint_best26.pt"
 
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

@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_suffix", type=str, help="Output suffix", default="refine")
@click.option("--out_dir", type=click.Path(exists=False), help="Output directory", default="./refine_results")
@click.option("--checkpoint", type=click.Path(exists=False), help="Model checkpoint", default=None)
@click.option("--seed", type=int, help="Random seed", default=11)
@click.option("--target_density", multiple=True, type=click.Path(exists=True), help="Target density map (.mrc file)", default=None)
@click.option("--resolution", multiple=True, type=float, help="Resolution for density map operations", default=None)
@click.option("--max_tokens", type=int, help="Maximum number of tokens for cropping (0 to disable)", default=512)
@click.option("--gamma_0", type=float, help="Gamma 0 parameter", default=-0.5)
@click.option("--recycles", type=int, help="Number of refinement recycles", default=300)
@click.option("--enable_cropping", is_flag=True, help="Enable cropping for large structures", default=True)
@click.option("--cond_early_stop", type=str, help="Condition early stop", default="loss")
@click.option("--clash", type=float, help="Weight for clash loss", default=0.1)
@click.option("--den", type=float, help="Weight for density loss", default=20.0)
@click.option("--rama", type=float, help="Weight for rama loss", default=500.0)
@click.option("--rotamer", type=float, help="Weight for rotamer loss", default=500.0)
@click.option("--bond", type=float, help="Weight for bond loss", default=50)
@click.option("--angle", type=float, help="Weight for angle loss", default=1)
@click.option("--cbeta", type=float, help="Weight for cbeta loss", default=1.0)
@click.option("--ramaz", type=float, help="Weight for ramaz loss", default=1.00)
@click.option("--learning_rate", type=float, help="Learning rate for refinement", default=1.8e-4)
@click.option("--max_norm_sigmas_value", type=float, help="max norm sigmas value", default=1.0)
@click.option("--num_workers", type=int, help="Number of data loader workers", default=0)
@click.option("--use_global_clash", is_flag=True, help="Global clash flag", default=True)
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
    clash: int = 0.1,
    den: int = 20.0,
    rama: int = 500.0,
    rotamer: int = 500.0,
    bond: int = 50,
    angle: int = 1,
    cbeta: int = 1.0,
    ramaz: int = 1.00,
    learning_rate: float = 1.8e-4,
    max_norm_sigmas_value: float = 1.0,
    num_workers: int = 0,
    use_global_clash: bool = True,

) -> None:
    """Run structure refinement with Boltz.""" 
    start_time = time.time()
    set_seed(seed)
    data = Path(data).expanduser()
    data_stem = data.stem
    out_dir = Path(out_dir).expanduser()
    # out_dir = out_dir / f"{data.stem}_{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = check_inputs(data)
    mol_dir =Path(__file__).resolve().parent / "CryoNetRefine" / "data" / "mols"
    process_inputs(
        data=data,
        data_stem=data_stem,
        out_dir=out_dir,
        mol_dir=mol_dir,
        preprocessing_threads=1,
    )
    # Load manifest
    manifest = Manifest.load(out_dir / f"processed_{data_stem}" / "manifest.json")
    # Load processed data !!
    processed_dir = out_dir / f"processed_{data_stem}"
    processed = BoltzProcessedInput(
        manifest=manifest,
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
    refine_args.weight_dict["den"] = den
    refine_args.weight_dict["rama"] = rama
    refine_args.weight_dict["rotamer"] = rotamer
    refine_args.weight_dict["bond"] = bond
    refine_args.weight_dict["angle"] = angle
    refine_args.weight_dict["cbeta"] = cbeta
    refine_args.weight_dict["ramaz"] = ramaz
    refine_args.learning_rate = learning_rate
    refine_args.use_global_clash = use_global_clash
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
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        mol_dir=mol_dir,
        num_workers=num_workers,  # Single worker for stability
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )
    # Setup data loader - use original for stability
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()
    # Perform refinement for each structure
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Refining structures")):
        click.echo(f"\nProcessing batch {batch_idx}")
        # NOTE:
        # Do NOT move the whole batch to GPU here.
        # Each crop will be moved to GPU individually inside the Engine,
        # so that GPU memory is only used for the current crop.
        refined_coords, _ = refiner.refine(batch, target_density_obj, processed.template_dir, out_dir, cond_early_stop=cond_early_stop)
        # Get best results info from refiner
        best_iteration = getattr(refiner, 'best_iteration', None)
        best_loss = getattr(refiner, 'best_loss', None)
        best_cc = getattr(refiner, 'best_cc', None)
        # Save refined structure (best result)
        output_path = out_dir / f"{pdb_id}_{out_suffix}.cif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_refined_structure(batch, refined_coords, data_dir, output_path)
        click.echo(f"Best Loss: {best_loss:.3f}, CC: {best_cc:.3f} at iteration {best_iteration}")
        click.echo(f"Refined structure {batch_idx} saved to {output_path}")
    if 'refiner' in locals():
        refiner.clear_caches()
    if 'refiner' in locals() and hasattr(refiner, 'geometric_adapter'):
        cache_info = refiner.geometric_adapter.get_cache_info()
        click.echo(f"Structure cache info: {cache_info}")
    click.echo("Refinement completed!")
    end_time = time.time()
    click.echo(f"Refinement completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    refine()
