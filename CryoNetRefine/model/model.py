# --------------------------------------------------------------------------------------
# Modified from boltz (https://github.com/jwohlwend/boltz/src/boltz/model/models/boltz.py)
# --------------------------------------------------------------------------------------

from typing import Any, Optional
from pathlib import Path 
import torch
import torch._dynamo
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

import CryoNetRefine.model.layers.initialize as init
from CryoNetRefine.data import const
from CryoNetRefine.model.layers.pairformer import PairformerModule
from CryoNetRefine.model.modules.diffusion_conditioning import DiffusionConditioning
from CryoNetRefine.model.modules.diffusion import AtomDiffusion
from CryoNetRefine.model.modules.encoders import RelativePositionEncoder
from CryoNetRefine.model.modules.trunk import (
    BFactorModule,
    InputEmbedder,
    TemplateV2Module,
)

class CryoNetRefineModel(LightningModule):
    """CryoNetRefine model."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        validators: Any = None,
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[dict] = None,
        confidence_prediction: bool = False,
        affinity_prediction: bool = False,
        affinity_ensemble: bool = False,
        affinity_mw_correction: bool = True,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        validate_structure: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        compile_affinity: bool = False,
        compile_msa: bool = False,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        use_no_atom_char: bool = False,
        no_random_recycling_training: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        steering_args: Optional[dict] = None,
        use_templates: bool = False,
        compile_templates: bool = False,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["validators"])
        # No random recycling
        self.no_random_recycling_training = no_random_recycling_training

        if validate_structure:
            # Late init at setup time
            self.val_group_mapper = {}  # maps a dataset index to a validation group name
            self.validator_mapper = {}  # maps a dataset index to a validator
            # Validators for each dataset keep track of all metrics,
            # compute validation, aggregate results and log
            self.validators = nn.ModuleList(validators)

        self.num_val_datasets = num_val_datasets
        self.log_loss_every_steps = log_loss_every_steps
        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay
        # Arguments
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args
        self.steering_args = steering_args
        # Training metrics
        if validate_structure:
            self.train_confidence_loss_logger = MeanMetric()
            self.train_confidence_loss_dict_logger = nn.ModuleDict()
            for m in [
                "plddt_loss",
                "resolved_loss",
                "pde_loss",
                "pae_loss",
            ]:
                self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.exclude_ions_from_lddt = exclude_ions_from_lddt

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Trunk
        self.is_pairformer_compiled = False
        self.is_msa_compiled = False
        self.is_template_compiled = False

        # Kernels
        self.use_kernels = use_kernels

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

     

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Set compile rules
        # Big models hit the default cache limit (8)
        torch._dynamo.config.cache_size_limit = 512  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001

        # Pairwise stack
        self.use_templates = use_templates
        if use_templates:
            self.template_module = TemplateV2Module(token_z, **template_args)
            if compile_templates:
                self.is_template_compiled = True
                self.template_module = torch.compile(
                    self.template_module,
                    dynamic=False,
                    fullgraph=False,
                )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            self.is_pairformer_compiled = True
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args[
                "conditioning_transition_layers"
            ],
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )
        # Output modules
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                **score_model_args,
            },
            compile_score=compile_structure,
            **diffusion_process_args,
        )
        self.predict_bfactor = predict_bfactor
        if predict_bfactor:
            self.bfactor_module = BFactorModule(token_s, num_bins)

        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.structure_prediction_training = structure_prediction_training
  
        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if (
                    name.split(".")[0] not in ["confidence_module", "affinity_module"]
                    and "out_token_feat_update" not in name
                ):
                    param.requires_grad = False

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0  # noqa: PLR2004
        ):
            self.use_kernels = False

        if (
            stage != "predict"
            and hasattr(self.trainer, "datamodule")
            and self.trainer.datamodule
            and self.validate_structure
        ):
            self.val_group_mapper.update(self.trainer.datamodule.val_group_mapper)

            l1 = len(self.val_group_mapper)
            l2 = self.num_val_datasets
            msg = (
                f"Number of validation datasets num_val_datasets={l2} "
                f"does not match the number of val_group_mapper entries={l1}."
            )
            assert l1 == l2, msg

            # Map an index to a validator, and double check val names
            # match from datamodule
            all_validator_names = []
            for validator in self.validators:
                for val_name in validator.val_names:
                    msg = f"Validator {val_name} duplicated in validators."
                    assert val_name not in all_validator_names, msg
                    all_validator_names.append(val_name)
                    for val_idx, val_group in self.val_group_mapper.items():
                        if val_name == val_group["label"]:
                            self.validator_mapper[val_idx] = validator

            msg = "Mismatch between validator names and val_group_mapper values."
            assert set(all_validator_names) == {
                x["label"] for x in self.val_group_mapper.values()
            }, msg

