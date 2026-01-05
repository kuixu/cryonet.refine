# follow boltz/src/boltz/data/module/inferencev2.py
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from CryoNetRefine.data import const
from CryoNetRefine.data.feature.featurizer import BoltzFeaturizer
from CryoNetRefine.data.mol import load_canonicals, load_molecules
from CryoNetRefine.data.pad import pad_to_max
from CryoNetRefine.data.tokenize.boltz import BoltzTokenizer
from CryoNetRefine.data.types import (
    Input,
    Manifest,
    Record,
    StructureV2
)

def load_input(
    record: Record,
    template_dir: Optional[Path] = None,
    extra_mols_dir: Optional[Path] = None,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
    template_dir : Optional[Path]
        The path to the template directory.
    extra_mols_dir : Optional[Path]
        The path to the extra molecules directory.


    Returns
    -------
    Input
        The loaded input.

    """

    structure = StructureV2.load(template_dir / f"{record.id}.npz")


    # Load templates
    templates = None
    if record.templates and template_dir is not None:
        templates = {}
        for template_info in record.templates:
            template_id = template_info.name
            # template_path = template_dir / f"{record.id}_{template_id}.npz"
            template_path = template_dir / f"{record.id}.npz"
            template = StructureV2.load(template_path)
            templates[template_id] = template

    # Load extra molecules
    extra_mols = {}
    if extra_mols_dir is not None:
        extra_mol_path = extra_mols_dir / f"{record.id}.pkl"
        if extra_mol_path.exists():
            with extra_mol_path.open("rb") as f:
                extra_mols = pickle.load(f)  # noqa: S301
    return Input(
        structure,
        record=record,
        templates=templates,
        extra_mols=extra_mols,
    )


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "token_to_backbone_atoms",  
            "token_backbone_mask",    
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)
        elif key in ["token_to_backbone_atoms", "token_backbone_mask"]:
            values = values[0]
        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        mol_dir: Path,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.

        mol_dir : Path
            The path to the moldir.
        template_dir : Optional[Path]
            The path to the template directory.

        """
        super().__init__()
        self.manifest = manifest
        self.mol_dir = mol_dir
        self.template_dir = template_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.canonicals = load_canonicals(self.mol_dir)
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get record
        record = self.manifest.records[idx]

        # Finalize input data
        input_data = load_input(
            record=record,
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
        )
        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(  # noqa: T201
                f"Tokenizer failed on {record.id} with error {e}. Skipping."
            )
            return self.__getitem__(0)

        # Load conformers
        try:
            molecules = {}
            molecules.update(self.canonicals)
            molecules.update(input_data.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.mol_dir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                max_atoms=None,
                max_tokens=None,
                compute_frames=True,
                override_method=self.override_method,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)
    
    def get_sequence_length(self, idx: int) -> int:
        """Get the total sequence length for a given sample.
        
        Parameters
        ----------
        idx : int
            The index of the sample.
            
        Returns
        -------
        int
            The total sequence length (sum of all chains).
            
        """
        record = self.manifest.records[idx]
        # Calculate total sequence length across all chains
        # ChainInfo has num_residues attribute, not sequence
        total_length = sum(chain.num_residues for chain in record.chains)
        return total_length


class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        mol_dir: Path,
        num_workers: int,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        length_bin_size: int = 50,
        max_sequence_length: int = 2000,
        length_weight_power: float = 1.0,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
 
      
        mol_dir : Path
            The path to the moldir.
        num_workers : int
            The number of workers to use.
        template_dir : Optional[Path]
            The path to the template directory.
        extra_mols_dir : Optional[Path]
            The path to the extra molecules directory.
        override_method : Optional[str]
            The method to override.
        length_bin_size : int
            Size of each length bin for sampling.
        max_sequence_length : int
            Maximum sequence length to consider.
        length_weight_power : float
            Power to apply to length weights (1.0 = linear, 0.0 = uniform).

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.mol_dir = mol_dir
        self.template_dir = template_dir
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.rank = rank
        self.world_size = world_size

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader with length-balanced sampling.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            mol_dir=self.mol_dir,
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
            override_method=self.override_method,
        )
        
        # Use Length-Balanced DistributedSampler if in distributed mode
        sampler = None
        if self.world_size > 1:
            from CryoNetRefine.data.sampler.length_balanced import LengthBalancedDistributedSampler
            sampler = LengthBalancedDistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,  # Can be set to True to shuffle within each rank between epochs
                seed=0,
            )
            if self.rank == 0:
                print(f"✅ Using LengthBalancedDistributedSampler for load balancing")
        
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,  # Must be False when using sampler
            sampler=sampler,
            collate_fn=collate,
        )