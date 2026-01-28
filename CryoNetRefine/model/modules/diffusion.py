# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
from __future__ import annotations

from math import sqrt
from pathlib import Path
import torch
import torch.nn.functional as F 
from einops import rearrange
from torch import nn
from torch.nn import Module
import CryoNetRefine.model.layers.initialize as init
from CryoNetRefine.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from CryoNetRefine.model.modules.transformers import (
    DiffusionTransformer,
)
from CryoNetRefine.model.modules.utils import (
    LinearNoBias,
    default,
    log,
)
from CryoNetRefine.loss.loss import compute_overall_cc_loss
def deep_copy_tensors(obj):
    import copy

    if isinstance(obj, torch.Tensor):
        return obj.clone().detach()
    elif isinstance(obj, dict):
        return {k: deep_copy_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        out = [deep_copy_tensors(v) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    else:
        return copy.deepcopy(obj)
class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
        cc_movement_scale: float = 1.0,
        cc_min_scale: float = 0.25,
        cc_max_scale: float = 1.5,
    ):
        s_inputs_d = s_inputs.clone().detach()
        s_trunk_d = s_trunk.clone().detach()
        r_noisy_d = r_noisy.clone() if self.training else r_noisy.clone().detach()
        times_d = times.clone().detach()
        feats_d = deep_copy_tensors(feats)
        diffusion_conditioning_d = deep_copy_tensors(diffusion_conditioning)
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times_d,
                s_trunk_d.repeat_interleave(multiplicity, 0),
                s_inputs_d.repeat_interleave(multiplicity, 0),
                use_reentrant=False,
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times_d,
                s_trunk_d.repeat_interleave(multiplicity, 0),
                s_inputs_d.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats_d,
            q=diffusion_conditioning_d["q"].float(),
            c=diffusion_conditioning_d["c"].float(),
            atom_enc_bias=diffusion_conditioning_d["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning_d["to_keys"],
            r=r_noisy_d,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats_d["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update



class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5, 
        sigma_min: float = 0.0004,  
        sigma_max: float = 160.0, 
        sigma_data: float = 16.0,  
        rho: float = 7,  
        P_mean: float = -1.2,  
        P_std: float = 1.5,  
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,  # lamda λ
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        norm_sigmas_flag: bool = False, # add by hfy 20250826
        max_norm_sigmas_value: float = 1.0, # add by hfy 20250826
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        self.norm_sigmas_flag = norm_sigmas_flag # add by hfy 20250826
        self.max_norm_sigmas_value = max_norm_sigmas_value # add by hfy 20250826
    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25


    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1") # 1 -> 1,1,1

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def den_sample(
            self,
            atom_mask,
            num_sampling_steps=None,
            multiplicity=1,
            max_parallel_samples=None,
            target_density=None,
            use_eps_flag=False,
            iteration=0,
            atom_weights=None,
            **network_condition_kwargs,
        ):
            atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
      
            # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
            sigmas = self.sample_schedule(num_sampling_steps)
            if self.norm_sigmas_flag:
                sigmas = sigmas/sigmas.max()*self.max_norm_sigmas_value
            gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
            sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:])) 
            # atom position is noise at the beginning
            initial_atom_coords = network_condition_kwargs["feats"]["template_coords"].squeeze(0).to(self.device)
            if not initial_atom_coords.requires_grad and self.training:
                initial_atom_coords = initial_atom_coords.clone().requires_grad_(True)
            atom_coords = initial_atom_coords 
            atom_coords_denoised = None
            # compute initial cc
            if iteration == 0 and (target_density is not None):
                initial_cc, _ = compute_overall_cc_loss(
                    predicted_coords=initial_atom_coords,
                    target_density=target_density,
                    feats=network_condition_kwargs["feats"],
                    atom_weights=atom_weights
                )
            else:
                initial_cc = 0.0
            

            for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):

                sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item() # sigma_tm > sigma_t
                t_hat = sigma_tm * (1 + gamma) # line 5
                noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)  
           
                if use_eps_flag:
                    eps = sqrt(noise_var) * torch.randn_like(atom_coords)
                    atom_coords_noisy = atom_coords + eps
                else:
                    atom_coords_noisy = atom_coords
           
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )
                model_input = atom_coords_noisy # add by huangfuyao
                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward( #  0.05 seconds
                        model_input[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk 
                atom_coords = atom_coords_denoised
                if step_idx == 0:
                    break # one-step diffusion
            return dict(sample_atom_coords=atom_coords,initial_cc = initial_cc)
    