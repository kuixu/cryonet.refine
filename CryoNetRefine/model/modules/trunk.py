import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot

from CryoNetRefine.data import const
from CryoNetRefine.model.layers.pairformer import (
    PairformerNoSeqModule,
)
from CryoNetRefine.model.modules.encoders import (
    AtomAttentionEncoder,
    AtomEncoder,
    FourierEmbedding,
)

def log_cuda_mem(tag: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated() / 1024**2
        res = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[MEM][Boltz] {tag} | alloc={cur:.1f}MB reserved={res:.1f}MB peak={peak:.1f}MB")
    else:
        print(f"[MEM][Boltz] {tag} | CUDA not available")





class InputEmbedder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,
        add_method_conditioning: bool = False,
        add_modified_flag: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom embedding size.
        atom_z : int
            The atom pairwise embedding size.
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.token_s = token_s
        self.add_method_conditioning = add_method_conditioning
        self.add_modified_flag = add_modified_flag
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
            activation_checkpointing=activation_checkpointing,
        )

        self.res_type_encoding = nn.Linear(const.num_tokens, token_s, bias=False)

        if add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(
                const.num_method_types, token_s
            )
            self.method_conditioning_init.weight.data.fill_(0)
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
            self.modified_conditioning_init.weight.data.fill_(0)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
            self.cyclic_conditioning_init.weight.data.fill_(0)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(const.chain_type_ids), token_s
            )
            self.mol_type_conditioning_init.weight.data.fill_(0)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"].float()


        # Compute input embedding
        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = self.atom_enc_proj_z(p)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )

        s = (
            a
            + self.res_type_encoding(res_type)
        )

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])

        return s


class TemplateModule(nn.Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs,
    ) -> None:
        """Initialize the template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(
            const.num_tokens * 2 + num_bins + 5,
            template_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        self.pairformer = PairformerNoSeqModule(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        asym_id = feats["asym_id"]
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1)
        num_templates = num_templates.clamp(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
        asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Compute unit vector in each frame
            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot, (ca_coords - frame_t))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
            unit_vector = unit_vector.squeeze(-1)

            # Concatenate input features
            a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
            a_tij = torch.cat(a_tij, dim=-1)
            a_tij = a_tij * asym_mask.unsqueeze(-1)

            res_type_i = res_type[:, :, :, None]
            res_type_j = res_type[:, :, None, :]
            res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        # Record initial memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Input projection
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij

        # Reshape and prepare mask
        v = v.view(B * T, *v.shape[2:])
        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])
   
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)

        # Norm
        v = self.v_norm(v)
        
        v = v.view(B, T, *v.shape[1:])

        # Aggregate templates
        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # Output projection
        u = self.u_proj(self.relu(u))
        
        return u


class TemplateV2Module(nn.Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs,
    ) -> None:
        """Initialize the template module.
    
        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """

        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(
            const.num_tokens * 2 + num_bins + 5,
            template_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        self.pairformer = PairformerNoSeqModule(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )
        
    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = feats["template_mask"].any(dim=2).float()

        num_templates = template_mask.sum(dim=1)
        num_templates = num_templates.clamp(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        tmlp_pair_mask = (
            visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]
        ).float()

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Compute unit vector in each frame
            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot, (ca_coords - frame_t))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
            unit_vector = unit_vector.squeeze(-1)

            # Concatenate input features
            a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
            a_tij = torch.cat(a_tij, dim=-1)
            a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)

            res_type_i = res_type[:, :, :, None]
            res_type_j = res_type[:, :, None, :]
            res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        # Expand mask
        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])

        # Compute input projections
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, *v.shape[2:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, T, *v.shape[1:])

        # Aggregate templates
        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # Compute output projection
        u = self.u_proj(self.relu(u))
        return u


class BFactorModule(nn.Module):
    """BFactor Module."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        """Initialize the bfactor module.

        Parameters
        ----------
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def forward(self, s: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings

        Returns
        -------
        Tensor
            The predicted bfactor histogram.

        """
        return self.bfactor(s)


