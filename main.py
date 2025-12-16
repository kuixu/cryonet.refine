#!/usr/bin/env python3
"""
CryoNet.Refine Refinement 

This script performs structure refinement using density-guided diffusion.
It freezes all modules except the diffusion module and uses CC loss for optimization.
"""
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
warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")

@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_suffix", type=str, help="Output suffix", default="CryoNet.Refine")
@click.option("--out_dir", type=click.Path(exists=False), help="Output directory", default="./refine_results")
@click.option("--cache", type=click.Path(exists=False), help="Cache directory")
@click.option("--checkpoint", type=click.Path(exists=True), help="Model checkpoint", default=None)
@click.option("--seed", type=int, help="Random seed", default=11)
@click.option("--target_density", type=click.Path(exists=True), help="Target density map (.mrc file)", default=None)
@click.option("--resolution", type=float, help="Resolution for density map operations", default=None)
@click.option("--max_tokens", type=int, help="Maximum number of tokens for cropping (0 to disable)", default=512)
@click.option("--gamma_0", type=float, help="Gamma 0 parameter", default=-0.5)
@click.option("--recycles", type=int, help="Number of refinement recycles", default=3)
@click.option("--enable_cropping", is_flag=True, help="Enable cropping for large structures", default=True)
@click.option("--cond_early_stop", type=str, help="Condition early stop", default="loss")
@click.option("--clash", type=float, help="Weight for clash loss", default=0.1)
@click.option("--den", type=float, help="Weight for density loss", default=20.0)
@click.option("--learning_rate", type=float, help="Learning rate for refinement", default=1.8e-4)
@click.option("--max_norm_sigmas_value", type=float, help="max norm sigmas value", default=1.0)
@click.option("--num_workers", type=int, help="Number of data loader workers", default=0)
def refine(
    data: str,
    out_dir: str,
    cache: str,
    checkpoint: Optional[str] = None,
    out_suffix: str = None,
    seed: Optional[int] = 11,
    target_density: Optional[str] = None,
    resolution: float = 1.9,
    max_tokens: int = 512,
    recycles: int = 3,
    gamma_0: float = -0.5,
    enable_cropping: bool = True,
    cond_early_stop: str = "loss",
    clash: int = 0.1,
    den: int = 20.0,
    learning_rate: float = 1.8e-4,
    max_norm_sigmas_value: float = 1.0,
    num_workers: int = 0,

) -> None:
    """Run structure refinement with Boltz.""" 
    start_time = time.time()
    set_seed(seed)
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    # out_dir = out_dir / f"{data.stem}_{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = check_inputs(data)
    mol_dir =Path(__file__).resolve().parent / "CryoNetRefine" / "data" / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        mol_dir=mol_dir,
        preprocessing_threads=1,
    )
    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    # Load processed data !!
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=manifest,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    # Setup model parameters
    diffusion_params = DiffusionParams(gamma_0=gamma_0, max_norm_sigmas_value=max_norm_sigmas_value)
    pairformer_args = PairformerArgs()
    # Load model
    if checkpoint is None:
        checkpoint = cache / "cryonet.refine_model_checkpoint_best26.pt"

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
    assert target_density is not None and resolution is not None, "Target density and resolution must be provided"
    target_density_obj = DensityInfo(mrc_path=target_density, resolution=resolution)
    refine_args = RefineArgs()
    refine_args.data_dir = data_dir
    refine_args.num_recycles = recycles
    refine_args.weight_dict["clash"] = clash
    refine_args.weight_dict["den"] = den
    refine_args.learning_rate = learning_rate
    pdb_id = data[0].name.split('.')[0]
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
