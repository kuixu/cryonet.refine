# Portions of this file are adapted from the original Boltz project:
# https://huggingface.co/boltz-community/boltz-2  (GNU GPL v3)
import pickle
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
import click
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from CryoNetRefine.data.mol import load_canonicals
from CryoNetRefine.data.types import Manifest, Record



@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False

@dataclass
class RefineArgs:
    """Refinement specific arguments."""
    
    learning_rate: float = 1.8e-4
    num_recycles: int = 300
    early_stopping_patience: int = 20
    resolution: float = 1.9
    weight_dict: dict = field(default_factory=lambda: {
        "den": 20.0, 
        "geometric": 1.0,
        "rama": 500.0,
        "rotamer": 500.0,
        "bond": 50,
        "angle": 1,
        "cbeta": 1.0,
        "ramaz": 1.00,
        "violation": 0.0,
        "clash": 0.1,
    })
    use_global_clash: bool = False
    data_dir: str | None = None
    use_molecule_aware_cropping: bool = True 
    min_improvement = 0




@dataclass
class DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = -0.5
    gamma_min: float = 0.2
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    norm_sigmas_flag: bool = True
    max_norm_sigmas_value: float = 1.0
    
model_args = {
    "recycling_steps": 1,  # Minimal recycling for speed
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "max_parallel_samples": 1,
    "write_confidence_summary": False,
    "write_full_pae": False,
    "write_full_pde": False,
}



def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    # if data.is_dir():
    #     data: list[Path] = list(data.glob("*"))
    #     for d in data:
    #         if d.is_dir():
    #             msg = f"Found directory {d} instead of .cif or .pdb."
    #             raise RuntimeError(msg)
    #         if d.suffix.lower() not in (".cif", ".pdb"):
    #             msg = (
    #                 f"Unable to parse filetype {d.suffix}, "
    #                 "please provide a .cif or .pdb file."
    #             )
    #             raise RuntimeError(msg)
    # else:
    #     data = [data]

    #if not endwith cif or pdb, raise error
    if data.suffix.lower() not in (".cif", ".pdb"):
        msg = f"Unable to parse filetype {data.suffix}, please provide a .cif or .pdb file."
        raise RuntimeError(msg)

    return [data]


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    mol_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    records_dir: Path,
) -> None:
    try:
        # Parse data
        if path.suffix.lower() in (".cif", ".pdb"):
            from CryoNetRefine.data.parse.schema import parse_refine_schema
            target = parse_refine_schema(path, ccd, mol_dir) # add by huangfuyao
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg) 
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a cif or pdb file."
            )
            raise RuntimeError(msg)  

        # Dump templates
        for template_id, template in target.templates.items():
            # name = f"{target.record.id}_{template_id}.npz"
            name = f"{target.record.id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)
        # # ================================
    except Exception as e:  # noqa: BLE001
        import traceback
    
        traceback.print_exc()
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201

@rank_zero_only
def process_inputs(
    data: list[Path], # [PosixPath('/home/huangfuyao/proj/boltz/examples/6cvm_A_200AA.yaml')]
    data_stem: str,
    out_dir: Path, # /home/huangfuyao/proj/boltz/out/boltz_results_6cvm_A_200AA
    mol_dir: Path,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.
    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
   
    # Check if records exist at output path
    records_dir = out_dir / f"processed_{data_stem}" / "records" # boltz/out/boltz_results_6cvm_A_200AA/processed/records
    if records_dir.exists():
        # Load existing records
        existing = [Record.load(p) for p in records_dir.glob("*.json")]
        processed_ids = {record.id for record in existing}
        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / f"processed_{data_stem}" / "manifest.json")

    # Create output directories
    records_dir = out_dir / f"processed_{data_stem}" / "records"
    processed_templates_dir = out_dir / f"processed_{data_stem}" / "templates"
    processed_mols_dir = out_dir / f"processed_{data_stem}" / "mols"

    out_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    ccd = load_canonicals(mol_dir)

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        mol_dir=mol_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / f"processed_{data_stem}" / "manifest.json")


