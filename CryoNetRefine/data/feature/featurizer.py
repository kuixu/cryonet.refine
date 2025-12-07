"""
Copyright Notice:
This file is based on the original Boltz project (https://huggingface.co/boltz-community/boltz-2).
Portions of this code are adapted from the original Boltz-2 featurizer implementation.

Original License: GNU GPL v3
Source: https://huggingface.co/boltz-community/boltz-2

Modifications by: [huangfuyao/Cryonet.Refine]
Date: 2024-2025

Key Modifications:
1. Added template coordinate and mask processing for refinement
2. Modified padding logic to return padding lengths
3. Added template present mask and token pair mask generation
4. Adapted from Boltz-2 to Boltz-1 architecture
"""
import math
from typing import Optional
import numpy as np
import rdkit.Chem.Descriptors
import torch
from numba import types
from rdkit.Chem import Mol
from scipy.spatial.distance import cdist
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot

from CryoNetRefine.data import const
from CryoNetRefine.data.pad import pad_dim
from CryoNetRefine.data.types import (
    TemplateInfo,
    Tokenized,
)
from CryoNetRefine.model.modules.utils import center_random_augmentation

####################################################################################################
# HELPERS
####################################################################################################


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    name = str(name).strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def sample_d(
    min_d: float,
    max_d: float,
    n_samples: int,
    random: np.random.Generator,
) -> np.ndarray:
    """Generate samples from a 1/d distribution between min_d and max_d.

    Parameters
    ----------
    min_d : float
        Minimum value of d
    max_d : float
        Maximum value of d
    n_samples : int
        Number of samples to generate
    random : numpy.random.Generator
        Random number generator

    Returns
    -------
    numpy.ndarray
        Array of samples drawn from the distribution

    Notes
    -----
    The probability density function is:
    f(d) = 1/(d * ln(max_d/min_d)) for d in [min_d, max_d]

    The inverse CDF transform is:
    d = min_d * (max_d/min_d)**u where u ~ Uniform(0,1)

    """
    # Generate n_samples uniform random numbers in [0, 1]
    u = random.random(n_samples)
    # Transform u using the inverse CDF
    return min_d * (max_d / min_d) ** u


def compute_frames_nonpolymer(
    data: Tokenized,
    coords,
    resolved_mask,
    atom_to_token,
    frame_data: list,
    resolved_frame_data: list,
) -> tuple[list, list]:
    """Get the frames for non-polymer tokens.

    Parameters
    ----------
    data : Tokenized
        The input data to the model.
    frame_data : list
        The frame data.
    resolved_frame_data : list
        The resolved frame data.

    Returns
    -------
    tuple[list, list]
        The frame data and resolved frame data.

    """
    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0
    for id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == id
        mask_chain_atom = asym_id_atom == id
        num_tokens = mask_chain_token.sum()
        num_atoms = mask_chain_atom.sum()
        if (
            data.tokens[token_idx]["mol_type"] != const.chain_type_ids["NONPOLYMER"]
            or num_atoms < 3  # noqa: PLR2004
        ):
            token_idx += num_tokens
            atom_idx += num_atoms
            continue
        dist_mat = (
            (
                coords.reshape(-1, 3)[mask_chain_atom][:, None, :]
                - coords.reshape(-1, 3)[mask_chain_atom][None, :, :]
            )
            ** 2
        ).sum(-1) ** 0.5
        resolved_pair = 1 - (
            resolved_mask[mask_chain_atom][None, :]
            * resolved_mask[mask_chain_atom][:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )
        frame_data[token_idx : token_idx + num_atoms, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms] = resolved_mask[
            frames
        ].all(axis=1)
        token_idx += num_tokens
        atom_idx += num_atoms
    frames_expanded = coords.reshape(-1, 3)[frame_data]

    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    return frame_data, resolved_frame_data & mask_collinear


def compute_collinear_mask(v1, v2):
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2






deletions_dict_type = types.DictType(types.UniTuple(types.int64, 3), types.int64)




####################################################################################################
# FEATURES
####################################################################################################


def select_subset_from_mask(mask, p, random: np.random.Generator) -> np.ndarray:
    num_true = np.sum(mask)
    v = random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def get_range_bin(value: float, range_dict: dict[tuple[float, float], int], default=0):
    """Get the bin of a value given a range dictionary."""
    value = float(value)
    for k, idx in range_dict.items():
        if k == "other":
            continue
        low, high = k
        if low <= value < high:
            return idx
    return default


def process_token_features(  # noqa: C901, PLR0915, PLR0912
    data: Tokenized,
    random: np.random.Generator,
    max_tokens: Optional[int] = None,
    binder_pocket_conditioned_prop: Optional[float] = 0.0,
    contact_conditioned_prop: Optional[float] = 0.0,
    binder_pocket_cutoff_min: Optional[float] = 4.0,
    binder_pocket_cutoff_max: Optional[float] = 20.0,
    binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
    only_ligand_binder_pocket: Optional[bool] = False,
    only_pp_contact: Optional[bool] = False,
    inference_pocket_constraints: Optional[
        list[tuple[int, list[tuple[int, int]], float]]
    ] = False,
    inference_contact_constraints: Optional[
        list[tuple[tuple[int, int], tuple[int, int], float]]
    ] = False,
    override_method: Optional[str] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The input data to the model.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(np.ascontiguousarray(token_data["res_idx"])).long()
    asym_id = from_numpy(np.ascontiguousarray(token_data["asym_id"])).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    res_type = one_hot(res_type, num_classes=const.num_tokens)
    disto_center = from_numpy(token_data["disto_coords"].copy())
    modified = from_numpy(token_data["modified"].copy()).long()  # float()
    cyclic_period = from_numpy(token_data["cyclic_period"].copy())

    ## Conditioning features ##
    method = (
        np.zeros(len(token_data))
        + const.method_types_ids[
            (
                "x-ray diffraction"
                if override_method is None
                else override_method.lower()
            )
        ]
    )
    if data.record is not None:
        if (
            override_method is None
            and data.record.structure.method is not None
            and data.record.structure.method.lower() in const.method_types_ids
        ):
            method = (method * 0) + const.method_types_ids[
                data.record.structure.method.lower()
            ]

    method_feature = from_numpy(method).long()

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"].copy()).float()
    disto_mask = from_numpy(token_data["disto_mask"].copy()).float()

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    bonds_type = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1
        bond_type = token_bond["type"]
        bonds_type[token_1, token_2] = bond_type
        bonds_type[token_2, token_1] = bond_type

    bonds = bonds.unsqueeze(-1)

    # Pocket conditioned feature
    contact_conditioning = (
        np.zeros((len(token_data), len(token_data)))
        + const.contact_conditioning_info["UNSELECTED"]
    )
    contact_threshold = np.zeros((len(token_data), len(token_data)))

    if inference_pocket_constraints is not None:
        for binder, contacts, max_distance, force in inference_pocket_constraints:
            binder_mask = token_data["asym_id"] == binder

            for idx, token in enumerate(token_data):
                if (
                    token["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and (token["asym_id"], token["res_idx"]) in contacts
                ) or (
                    token["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                    and (token["asym_id"], token["atom_idx"]) in contacts
                ):
                    contact_conditioning[binder_mask, idx] = (
                        const.contact_conditioning_info["BINDER>POCKET"]
                    )
                    contact_conditioning[idx, binder_mask] = (
                        const.contact_conditioning_info["POCKET>BINDER"]
                    )
                    contact_threshold[binder_mask, idx] = max_distance
                    contact_threshold[idx, binder_mask] = max_distance

    if inference_contact_constraints is not None:
        for token1, token2, max_distance, force in inference_contact_constraints:
            for idx1, _token1 in enumerate(token_data):
                if (
                    _token1["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and (_token1["asym_id"], _token1["res_idx"]) == token1
                ) or (
                    _token1["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                    and (_token1["asym_id"], _token1["atom_idx"]) == token1
                ):
                    for idx2, _token2 in enumerate(token_data):
                        if (
                            _token2["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                            and (_token2["asym_id"], _token2["res_idx"]) == token2
                        ) or (
                            _token2["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                            and (_token2["asym_id"], _token2["atom_idx"]) == token2
                        ):
                            contact_conditioning[idx1, idx2] = (
                                const.contact_conditioning_info["CONTACT"]
                            )
                            contact_conditioning[idx2, idx1] = (
                                const.contact_conditioning_info["CONTACT"]
                            )
                            contact_threshold[idx1, idx2] = max_distance
                            contact_threshold[idx2, idx1] = max_distance
                            break
                    break

    if binder_pocket_conditioned_prop > 0.0:
        # choose as binder a random ligand in the crop, if there are no ligands select a protein chain
        binder_asym_ids = np.unique(
            token_data["asym_id"][
                token_data["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]
        )

        if len(binder_asym_ids) == 0:
            if not only_ligand_binder_pocket:
                binder_asym_ids = np.unique(token_data["asym_id"])

        while random.random() < binder_pocket_conditioned_prop:
            if len(binder_asym_ids) == 0:
                break

            pocket_asym_id = random.choice(binder_asym_ids)
            binder_asym_ids = binder_asym_ids[binder_asym_ids != pocket_asym_id]

            binder_pocket_cutoff = sample_d(
                min_d=binder_pocket_cutoff_min,
                max_d=binder_pocket_cutoff_max,
                n_samples=1,
                random=random,
            )

            binder_mask = token_data["asym_id"] == pocket_asym_id

            binder_coords = []
            for token in token_data:
                if token["asym_id"] == pocket_asym_id:
                    _coords = data.structure.atoms["coords"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]
                    _is_present = data.structure.atoms["is_present"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]
                    binder_coords.append(_coords[_is_present])
            binder_coords = np.concatenate(binder_coords, axis=0)

            # find the tokens in the pocket
            token_dist = np.zeros(len(token_data)) + 1000
            for i, token in enumerate(token_data):
                if (
                    token["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and token["asym_id"] != pocket_asym_id
                    and token["resolved_mask"] == 1
                ):
                    token_coords = data.structure.atoms["coords"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]
                    token_is_present = data.structure.atoms["is_present"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]
                    token_coords = token_coords[token_is_present]

                    # find chain and apply chain transformation
                    for chain in data.structure.chains:
                        if chain["asym_id"] == token["asym_id"]:
                            break

                    token_dist[i] = np.min(
                        np.linalg.norm(
                            token_coords[:, None, :] - binder_coords[None, :, :],
                            axis=-1,
                        )
                    )

            pocket_mask = token_dist < binder_pocket_cutoff

            if np.sum(pocket_mask) > 0:
                if binder_pocket_sampling_geometric_p > 0.0:
                    # select a subset of the pocket, according
                    # to a geometric distribution with one as minimum
                    pocket_mask = select_subset_from_mask(
                        pocket_mask,
                        binder_pocket_sampling_geometric_p,
                        random,
                    )

                contact_conditioning[np.ix_(binder_mask, pocket_mask)] = (
                    const.contact_conditioning_info["BINDER>POCKET"]
                )
                contact_conditioning[np.ix_(pocket_mask, binder_mask)] = (
                    const.contact_conditioning_info["POCKET>BINDER"]
                )
                contact_threshold[np.ix_(binder_mask, pocket_mask)] = (
                    binder_pocket_cutoff
                )
                contact_threshold[np.ix_(pocket_mask, binder_mask)] = (
                    binder_pocket_cutoff
                )

    # Contact conditioning feature
    if contact_conditioned_prop > 0.0:
        while random.random() < contact_conditioned_prop:
            contact_cutoff = sample_d(
                min_d=binder_pocket_cutoff_min,
                max_d=binder_pocket_cutoff_max,
                n_samples=1,
                random=random,
            )
            if only_pp_contact:
                chain_asym_ids = np.unique(
                    token_data["asym_id"][
                        token_data["mol_type"] == const.chain_type_ids["PROTEIN"]
                    ]
                )
            else:
                chain_asym_ids = np.unique(token_data["asym_id"])

            if len(chain_asym_ids) > 1:
                chain_asym_id = random.choice(chain_asym_ids)

                chain_coords = []
                for token in token_data:
                    if token["asym_id"] == chain_asym_id:
                        _coords = data.structure.atoms["coords"][
                            token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                        ]
                        _is_present = data.structure.atoms["is_present"][
                            token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                        ]
                        chain_coords.append(_coords[_is_present])
                chain_coords = np.concatenate(chain_coords, axis=0)

                # find contacts in other chains
                possible_other_chains = []
                for other_chain_id in chain_asym_ids[chain_asym_ids != chain_asym_id]:
                    for token in token_data:
                        if token["asym_id"] == other_chain_id:
                            _coords = data.structure.atoms["coords"][
                                token["atom_idx"] : token["atom_idx"]
                                + token["atom_num"]
                            ]
                            _is_present = data.structure.atoms["is_present"][
                                token["atom_idx"] : token["atom_idx"]
                                + token["atom_num"]
                            ]
                            if _is_present.sum() == 0:
                                continue
                            token_coords = _coords[_is_present]

                            # check minimum distance
                            if (
                                np.min(cdist(chain_coords, token_coords))
                                < contact_cutoff
                            ):
                                possible_other_chains.append(other_chain_id)
                                break

                if len(possible_other_chains) > 0:
                    other_chain_id = random.choice(possible_other_chains)

                    pairs = []
                    for token_1 in token_data:
                        if token_1["asym_id"] == chain_asym_id:
                            _coords = data.structure.atoms["coords"][
                                token_1["atom_idx"] : token_1["atom_idx"]
                                + token_1["atom_num"]
                            ]
                            _is_present = data.structure.atoms["is_present"][
                                token_1["atom_idx"] : token_1["atom_idx"]
                                + token_1["atom_num"]
                            ]
                            if _is_present.sum() == 0:
                                continue
                            token_1_coords = _coords[_is_present]

                            for token_2 in token_data:
                                if token_2["asym_id"] == other_chain_id:
                                    _coords = data.structure.atoms["coords"][
                                        token_2["atom_idx"] : token_2["atom_idx"]
                                        + token_2["atom_num"]
                                    ]
                                    _is_present = data.structure.atoms["is_present"][
                                        token_2["atom_idx"] : token_2["atom_idx"]
                                        + token_2["atom_num"]
                                    ]
                                    if _is_present.sum() == 0:
                                        continue
                                    token_2_coords = _coords[_is_present]

                                    if (
                                        np.min(cdist(token_1_coords, token_2_coords))
                                        < contact_cutoff
                                    ):
                                        pairs.append(
                                            (token_1["token_idx"], token_2["token_idx"])
                                        )

                    assert len(pairs) > 0

                    pair = random.choice(pairs)
                    token_1_mask = token_data["token_idx"] == pair[0]
                    token_2_mask = token_data["token_idx"] == pair[1]

                    contact_conditioning[np.ix_(token_1_mask, token_2_mask)] = (
                        const.contact_conditioning_info["CONTACT"]
                    )
                    contact_conditioning[np.ix_(token_2_mask, token_1_mask)] = (
                        const.contact_conditioning_info["CONTACT"]
                    )

            elif not only_pp_contact:
                # only one chain, find contacts within the chain with minimum residue distance
                pairs = []
                for token_1 in token_data:
                    _coords = data.structure.atoms["coords"][
                        token_1["atom_idx"] : token_1["atom_idx"] + token_1["atom_num"]
                    ]
                    _is_present = data.structure.atoms["is_present"][
                        token_1["atom_idx"] : token_1["atom_idx"] + token_1["atom_num"]
                    ]
                    if _is_present.sum() == 0:
                        continue
                    token_1_coords = _coords[_is_present]

                    for token_2 in token_data:
                        if np.abs(token_1["res_idx"] - token_2["res_idx"]) <= 8:
                            continue

                        _coords = data.structure.atoms["coords"][
                            token_2["atom_idx"] : token_2["atom_idx"]
                            + token_2["atom_num"]
                        ]
                        _is_present = data.structure.atoms["is_present"][
                            token_2["atom_idx"] : token_2["atom_idx"]
                            + token_2["atom_num"]
                        ]
                        if _is_present.sum() == 0:
                            continue
                        token_2_coords = _coords[_is_present]

                        if (
                            np.min(cdist(token_1_coords, token_2_coords))
                            < contact_cutoff
                        ):
                            pairs.append((token_1["token_idx"], token_2["token_idx"]))

                if len(pairs) > 0:
                    pair = random.choice(pairs)
                    token_1_mask = token_data["token_idx"] == pair[0]
                    token_2_mask = token_data["token_idx"] == pair[1]

                    contact_conditioning[np.ix_(token_1_mask, token_2_mask)] = (
                        const.contact_conditioning_info["CONTACT"]
                    )
                    contact_conditioning[np.ix_(token_2_mask, token_1_mask)] = (
                        const.contact_conditioning_info["CONTACT"]
                    )

    if np.all(contact_conditioning == const.contact_conditioning_info["UNSELECTED"]):
        contact_conditioning = (
            contact_conditioning
            - const.contact_conditioning_info["UNSELECTED"]
            + const.contact_conditioning_info["UNSPECIFIED"]
        )
    contact_conditioning = from_numpy(contact_conditioning).long()
    contact_conditioning = one_hot(
        contact_conditioning, num_classes=len(const.contact_conditioning_info)
    )
    contact_threshold = from_numpy(contact_threshold).float()

    # compute cyclic polymer mask
    cyclic_ids = {}
    for idx_chain, asym_id_iter in enumerate(data.structure.chains["asym_id"]):
        for connection in data.structure.bonds:
            if (
                idx_chain == connection["chain_1"] == connection["chain_2"]
                and data.structure.chains[connection["chain_1"]]["res_num"] > 2
                and connection["res_1"]
                != connection["res_2"]  # Avoid same residue bonds!
            ):
                if (
                    data.structure.chains[connection["chain_1"]]["res_num"]
                    == (connection["res_2"] + 1)
                    and connection["res_1"] == 0
                ) or (
                    data.structure.chains[connection["chain_1"]]["res_num"]
                    == (connection["res_1"] + 1)
                    and connection["res_2"] == 0
                ):
                    cyclic_ids[asym_id_iter] = data.structure.chains[
                        connection["chain_1"]
                    ]["res_num"]
    cyclic = from_numpy(
        np.array(
            [
                (cyclic_ids[asym_id_iter] if asym_id_iter in cyclic_ids else 0)
                for asym_id_iter in token_data["asym_id"]
            ]
        )
    ).float()

    # cyclic period is either computed from the bonds or given as input flag
    cyclic_period = torch.maximum(cyclic, cyclic_period)

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            contact_conditioning = pad_dim(contact_conditioning, 0, pad_len)
            contact_conditioning = pad_dim(contact_conditioning, 1, pad_len)
            contact_threshold = pad_dim(contact_threshold, 0, pad_len)
            contact_threshold = pad_dim(contact_threshold, 1, pad_len)
            method_feature = pad_dim(method_feature, 0, pad_len)
            modified = pad_dim(modified, 0, pad_len)
            cyclic_period = pad_dim(cyclic_period, 0, pad_len)
    else:
        pad_len = 0
    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "type_bonds": bonds_type,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "method_feature": method_feature,
        "modified": modified,
        "cyclic_period": cyclic_period,
    }
    # MODIFICATION: Return pad_len along with features
    return token_features, pad_len  # Changed from: return token_features


def process_atom_features(
    data: Tokenized,
    random: np.random.Generator,
    ensemble_features: dict,
    molecules: dict[str, Mol],
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
    disto_use_ensemble: Optional[bool] = False,
    override_bfactor: bool = False,
    compute_frames: bool = False,
    override_coords: Optional[Tensor] = None,
    bfactor_md_correction: bool = False,
) -> dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Tokenized
        The input to the model.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    dict[str, Tensor]
        The atom features.

    """
    # Filter to tokens' atoms
    atom_data = []
    atom_name = []
    atom_element = []
    atom_charge = []
    atom_conformer = []
    atom_chirality = []
    ref_space_uid = []
    coord_data = []
    if compute_frames:
        frame_data = []
        resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []  # index on cropped atom table
    r_set_to_rep_atom = []
    disto_coords_ensemble = []
    backbone_feat_index = []
    token_to_center_atom = []


    e_offsets = data.structure.ensemble["atom_coord_idx"]
    atom_idx = 0

    # Start atom idx in full atom table for structures chosen. Up to num_ensembles points.
    ensemble_atom_starts = [
        data.structure.ensemble[idx]["atom_coord_idx"]
        for idx in ensemble_features["ensemble_ref_idxs"]
    ]

    # Set unk chirality id
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    chain_res_ids = {}
    res_index_to_conf_id = {}
    for token_id, token in enumerate(data.tokens):
        # Get the chain residue ids
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Get the molecule and conformer
        mol = molecules[token["res_name"]]
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}

        # Sample a random conformer
        if (chain_idx, res_id) not in res_index_to_conf_id:
            conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
            conf_id = int(random.choice(conf_ids))
            res_index_to_conf_id[(chain_idx, res_id)] = conf_id

        conf_id = res_index_to_conf_id[(chain_idx, res_id)]
        conformer = mol.GetConformer(conf_id)

        # Map atoms to token indices
        ref_space_uid.extend([new_idx] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Add atom data
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Add atom ref data
        # element, charge, conformer, chirality
        token_atom_name = np.array([convert_atom_name(a["name"]) for a in token_atoms])
        token_atoms_ref = np.array([atom_name_to_ref[a["name"]] for a in token_atoms])
        token_atoms_element = np.array([a.GetAtomicNum() for a in token_atoms_ref])
        token_atoms_charge = np.array([a.GetFormalCharge() for a in token_atoms_ref])
        token_atoms_conformer = np.array(
            [
                (
                    conformer.GetAtomPosition(a.GetIdx()).x,
                    conformer.GetAtomPosition(a.GetIdx()).y,
                    conformer.GetAtomPosition(a.GetIdx()).z,
                )
                for a in token_atoms_ref
            ]
        )
        token_atoms_chirality = np.array(
            [
                const.chirality_type_ids.get(a.GetChiralTag().name, unk_chirality)
                for a in token_atoms_ref
            ]
        )

        # Map token to representative atom
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)
        token_to_center_atom.append(atom_idx + token["center_idx"] - start)
        if (chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]) and token[
            "resolved_mask"
        ]:
            r_set_to_rep_atom.append(atom_idx + token["center_idx"] - start)

    
        

        if chain["mol_type"] == const.chain_type_ids["PROTEIN"]:
            backbone_index = [
                (
                    const.protein_backbone_atom_index[atom_name] + 1
                    if atom_name in const.protein_backbone_atom_index
                    else 0
                )
                for atom_name in token_atoms["name"]
            ]
        elif (
            chain["mol_type"] == const.chain_type_ids["DNA"]
            or chain["mol_type"] == const.chain_type_ids["RNA"]
        ):
            backbone_index = [
                (
                    const.nucleic_backbone_atom_index[atom_name]
                    + 1
                    + len(const.protein_backbone_atom_index)
                    if atom_name in const.nucleic_backbone_atom_index
                    else 0
                )
                for atom_name in token_atoms["name"]
            ]
        else:
            backbone_index = [0] * token["atom_num"]
        backbone_feat_index.extend(backbone_index)

        # Get token coordinates across sampled ensembles  and apply transforms
        token_coords = np.array(
            [
                data.structure.coords[
                    ensemble_atom_start + start : ensemble_atom_start + end
                ]["coords"]
                for ensemble_atom_start in ensemble_atom_starts
            ]
        )
        coord_data.append(token_coords)

        if compute_frames:
            # Get frame data
            res_type = const.tokens[token["res_type"]]
            res_name = str(token["res_name"])

            if token["atom_num"] < 3 or res_type in ["PAD", "UNK", "-"]:
                idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
                mask_frame = False
            elif (token["mol_type"] == const.chain_type_ids["PROTEIN"]) and (
                res_name in const.ref_atoms
            ):
                idx_frame_a, idx_frame_b, idx_frame_c = (
                    const.ref_atoms[res_name].index("N"),
                    const.ref_atoms[res_name].index("CA"),
                    const.ref_atoms[res_name].index("C"),
                )
                mask_frame = (
                    token_atoms["is_present"][idx_frame_a]
                    and token_atoms["is_present"][idx_frame_b]
                    and token_atoms["is_present"][idx_frame_c]
                )
            elif (
                token["mol_type"] == const.chain_type_ids["DNA"]
                or token["mol_type"] == const.chain_type_ids["RNA"]
            ) and (res_name in const.ref_atoms):
                idx_frame_a, idx_frame_b, idx_frame_c = (
                    const.ref_atoms[res_name].index("C1'"),
                    const.ref_atoms[res_name].index("C3'"),
                    const.ref_atoms[res_name].index("C4'"),
                )
                mask_frame = (
                    token_atoms["is_present"][idx_frame_a]
                    and token_atoms["is_present"][idx_frame_b]
                    and token_atoms["is_present"][idx_frame_c]
                )
            elif token["mol_type"] == const.chain_type_ids["PROTEIN"]:
                # Try to look for the atom nams in the modified residue
                is_ca = token_atoms["name"] == "CA"
                idx_frame_a = is_ca.argmax()
                ca_present = (
                    token_atoms[idx_frame_a]["is_present"] if is_ca.any() else False
                )

                is_n = token_atoms["name"] == "N"
                idx_frame_b = is_n.argmax()
                n_present = (
                    token_atoms[idx_frame_b]["is_present"] if is_n.any() else False
                )

                is_c = token_atoms["name"] == "C"
                idx_frame_c = is_c.argmax()
                c_present = (
                    token_atoms[idx_frame_c]["is_present"] if is_c.any() else False
                )
                mask_frame = ca_present and n_present and c_present

            elif (token["mol_type"] == const.chain_type_ids["DNA"]) or (
                token["mol_type"] == const.chain_type_ids["RNA"]
            ):
                # Try to look for the atom nams in the modified residue
                is_c1 = token_atoms["name"] == "C1'"
                idx_frame_a = is_c1.argmax()
                c1_present = (
                    token_atoms[idx_frame_a]["is_present"] if is_c1.any() else False
                )

                is_c3 = token_atoms["name"] == "C3'"
                idx_frame_b = is_c3.argmax()
                c3_present = (
                    token_atoms[idx_frame_b]["is_present"] if is_c3.any() else False
                )

                is_c4 = token_atoms["name"] == "C4'"
                idx_frame_c = is_c4.argmax()
                c4_present = (
                    token_atoms[idx_frame_c]["is_present"] if is_c4.any() else False
                )
                mask_frame = c1_present and c3_present and c4_present
            else:
                idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
                mask_frame = False
            frame_data.append(
                [
                    idx_frame_a + atom_idx,
                    idx_frame_b + atom_idx,
                    idx_frame_c + atom_idx,
                ]
            )
            resolved_frame_data.append(mask_frame)

        # Get distogram coordinates
        disto_coords_ensemble_tok = data.structure.coords[
            e_offsets + token["disto_idx"]
        ]["coords"]
        disto_coords_ensemble.append(disto_coords_ensemble_tok)

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[
            0
        ]  # atom has a copy of first coords in ensemble
        atom_data.append(token_atoms)
        atom_name.append(token_atom_name)
        atom_element.append(token_atoms_element)
        atom_charge.append(token_atoms_charge)
        atom_conformer.append(token_atoms_conformer)
        atom_chirality.append(token_atoms_chirality)
        atom_idx += len(token_atoms)

    disto_coords_ensemble = np.array(disto_coords_ensemble)  # (N_TOK, N_ENS, 3)

    # Compute ensemble distogram
    L = len(data.tokens)

    if disto_use_ensemble:
        # Use all available structures to create distogram
        idx_list = range(disto_coords_ensemble.shape[1])
    else:
        # Only use a sampled structures to create distogram
        idx_list = ensemble_features["ensemble_ref_idxs"]

    # Create distogram
    disto_target = torch.zeros(L, L, len(idx_list), num_bins)  # TODO1

    # disto_target = torch.zeros(L, L, num_bins)
    for i, e_idx in enumerate(idx_list):
        t_center = torch.Tensor(disto_coords_ensemble[:, e_idx, :])
        t_dists = torch.cdist(t_center, t_center)
        boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
        distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
        # disto_target += one_hot(distogram, num_classes=num_bins)
        disto_target[:, :, i, :] = one_hot(distogram, num_classes=num_bins)  # TODO1

    # Normalize distogram
    # disto_target = disto_target / disto_target.sum(-1)[..., None]  # remove TODO1
    atom_data = np.concatenate(atom_data)
    atom_name = np.concatenate(atom_name)
    atom_element = np.concatenate(atom_element)
    atom_charge = np.concatenate(atom_charge)
    atom_conformer = np.concatenate(atom_conformer)
    atom_chirality = np.concatenate(atom_chirality)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)
    
    num_atoms = len(atom_data)
    
    # MODIFICATION: Added template coordinate extraction (lines 1672-1689)
    # add by huangfuyao - extract template coordinates
    if data.templates and len(data.templates) > 0:
        first_key = next(iter(data.templates.keys()))  
        
        template_atoms = data.templates[first_key].atoms 
        template_coords_np = template_atoms["coords"]
        template_coords = torch.from_numpy(template_coords_np.copy()).unsqueeze(0)
    else:
        # No templates available, create dummy coordinates
        num_atoms = len(atom_data)
        template_coords = torch.zeros(1, num_atoms, 3, dtype=torch.float32)

    # Compute features
    disto_coords_ensemble = from_numpy(disto_coords_ensemble.copy())
    disto_coords_ensemble = disto_coords_ensemble[
        :, ensemble_features["ensemble_ref_idxs"]
    ].permute(1, 0, 2)
    backbone_feat_index = from_numpy(np.asarray(backbone_feat_index)).long()
    ref_atom_name_chars = from_numpy(atom_name.copy()).long()
    ref_element = from_numpy(atom_element.copy()).long()
    ref_charge = from_numpy(atom_charge.copy()).float()
    ref_pos = from_numpy(atom_conformer.copy()).float()
    ref_space_uid = from_numpy(ref_space_uid)
    ref_chirality = from_numpy(atom_chirality.copy()).long()
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"].copy())
    template_resolved_mask = from_numpy(template_atoms['is_present'].copy()) # add by hfy 20250817
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)
    token_to_center_atom = torch.tensor(token_to_center_atom, dtype=torch.long)
    bfactor = from_numpy(atom_data["bfactor"].copy())
    plddt = from_numpy(atom_data["plddt"].copy())
    if override_bfactor:
        bfactor = bfactor * 0.0

    if bfactor_md_correction and data.record.structure.method.lower() == "md":
        # MD bfactor was computed as RMSF
        # Convert to b-factor
        bfactor = 8 * (np.pi**2) * (bfactor**2)

    # We compute frames within ensemble
    if compute_frames:
        frames = []
        frame_resolved_mask = []
        for i in range(coord_data.shape[0]):
            frame_data_, resolved_frame_data_ = compute_frames_nonpolymer(
                data,
                coord_data[i],
                atom_data["is_present"],
                atom_to_token,
                frame_data,
                resolved_frame_data,
            )  # Compute frames for NONPOLYMER tokens
            frames.append(frame_data_.copy())
            frame_resolved_mask.append(resolved_frame_data_.copy())
        frames = from_numpy(np.stack(frames))  # (N_ENS, N_TOK, 3)
        frame_resolved_mask = from_numpy(np.stack(frame_resolved_mask))

    # Convert to one-hot
    backbone_feat_index = one_hot(
        backbone_feat_index,
        num_classes=1
        + len(const.protein_backbone_atom_index)
        + len(const.nucleic_backbone_atom_index),
    )
    ref_atom_name_chars = one_hot(ref_atom_name_chars, num_classes=64)
    ref_element = one_hot(ref_element, num_classes=const.num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1)
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data))
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data))
    token_to_center_atom = one_hot(token_to_center_atom, num_classes=len(atom_data))
    # Center the ground truth coordinates
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]

    # MODIFICATION: Added template center calculation and missing position filling (lines 1756-1768)
    template_center = (template_coords * template_resolved_mask[None, :, None]).sum(dim=1)
    template_center = template_center / template_resolved_mask.sum().clamp(min=1)
    if template_resolved_mask.sum() < len(template_resolved_mask):
        missing_positions = ~template_resolved_mask
        template_coords[0, missing_positions, :] = template_center[0, :]

    if isinstance(override_coords, Tensor):
        coords = override_coords.unsqueeze(0)

    # Apply random roto-translation to the input conformers
    for i in range(torch.max(ref_space_uid)):
        included = ref_space_uid == i
        if torch.sum(included) > 0 and torch.any(resolved_mask[included]):
            ref_pos[included] = center_random_augmentation(
                ref_pos[included][None], resolved_mask[included][None], centering=True
            )[0]

    # Compute padding and apply
    if max_atoms is not None:
        assert max_atoms % atoms_per_window_queries == 0
        pad_len = max_atoms - len(atom_data)
    else:
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        template_resolved_mask = pad_dim(template_resolved_mask, 0, pad_len) # add by hfy 20250817
        

        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_chirality = pad_dim(ref_chirality, 0, pad_len)
        backbone_feat_index = pad_dim(backbone_feat_index, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)  # Pad on dimension 1 to reach specified length
        template_coords = pad_dim(template_coords, 1, pad_len)

        atom_to_token = pad_dim(atom_to_token, 0, pad_len)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len)
        token_to_center_atom = pad_dim(token_to_center_atom, 1, pad_len)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len)
        bfactor = pad_dim(bfactor, 0, pad_len)
        plddt = pad_dim(plddt, 0, pad_len)

    if max_tokens is not None:
        pad_len = max_tokens - token_to_rep_atom.shape[0]
        if pad_len > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len)
            token_to_center_atom = pad_dim(token_to_center_atom, 0, pad_len)
            disto_target = pad_dim(pad_dim(disto_target, 0, pad_len), 1, pad_len)
            disto_coords_ensemble = pad_dim(disto_coords_ensemble, 1, pad_len)

            if compute_frames:
                frames = pad_dim(frames, 1, pad_len)
                frame_resolved_mask = pad_dim(frame_resolved_mask, 1, pad_len)
            

    atom_features = {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "atom_backbone_feat": backbone_feat_index,
        "ref_space_uid": ref_space_uid,
        "coords": coords, 
        "template_coords": template_coords, # add by hfy 20250817
        "template_resolved_mask": template_resolved_mask, # add by hfy 20250817
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "token_to_center_atom": token_to_center_atom,
        "disto_target": disto_target,
        "disto_coords_ensemble": disto_coords_ensemble,
        "bfactor": bfactor,
        "plddt": plddt,
    }
    if compute_frames:
        atom_features["frames_idx"] = frames
        atom_features["frame_resolved_mask"] = frame_resolved_mask

    return atom_features,pad_len



def load_dummy_templates_features(tdim: int, num_tokens: int) -> dict:
    """Load dummy templates for v2."""
    # Allocate features
    res_type = np.zeros((tdim, num_tokens), dtype=np.int64)
    frame_rot = np.zeros((tdim, num_tokens, 3, 3), dtype=np.float32)
    frame_t = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    cb_coords = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    ca_coords = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    frame_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    cb_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    template_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    query_to_template = np.zeros((tdim, num_tokens), dtype=np.int64)
    visibility_ids = np.zeros((tdim, num_tokens), dtype=np.float32)

    # Convert to one-hot
    res_type = torch.from_numpy(res_type)
    res_type = one_hot(res_type, num_classes=const.num_tokens)

    return {
        "template_restype": res_type,
        "template_frame_rot": torch.from_numpy(frame_rot),
        "template_frame_t": torch.from_numpy(frame_t),
        "template_cb": torch.from_numpy(cb_coords),
        "template_ca": torch.from_numpy(ca_coords),
        "template_mask_cb": torch.from_numpy(cb_mask),
        "template_mask_frame": torch.from_numpy(frame_mask),
        "template_mask": torch.from_numpy(template_mask),
        "query_to_template": torch.from_numpy(query_to_template),
        "visibility_ids": torch.from_numpy(visibility_ids),
    }


def compute_template_features(
    query_tokens: Tokenized,
    tmpl_tokens: list[dict],
    num_tokens: int,
) -> dict:
    """Compute the template features."""
    # Allocate features
    res_type = np.zeros((num_tokens,), dtype=np.int64)
    frame_rot = np.zeros((num_tokens, 3, 3), dtype=np.float32)
    frame_t = np.zeros((num_tokens, 3), dtype=np.float32)
    cb_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    ca_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    frame_mask = np.zeros((num_tokens,), dtype=np.float32)
    cb_mask = np.zeros((num_tokens,), dtype=np.float32)
    template_mask = np.zeros((num_tokens,), dtype=np.float32)
    query_to_template = np.zeros((num_tokens,), dtype=np.int64)
    visibility_ids = np.zeros((num_tokens,), dtype=np.float32)
    # Now create features per token
    asym_id_to_pdb_id = {}

    for token_dict in tmpl_tokens:
        idx = token_dict["q_idx"]
        pdb_id = token_dict["pdb_id"]
        token = token_dict["token"]
        query_token = query_tokens.tokens[idx]
        asym_id_to_pdb_id[query_token["asym_id"]] = pdb_id
        res_type[idx] = token["res_type"]
        frame_rot[idx] = token["frame_rot"].reshape(3, 3)
        frame_t[idx] = token["frame_t"]
        cb_coords[idx] = token["disto_coords"]
        ca_coords[idx] = token["center_coords"]
        cb_mask[idx] = token["disto_mask"]
        frame_mask[idx] = token["frame_mask"]
        template_mask[idx] = 1.0

    # Set visibility_id for templated chains
    for asym_id, pdb_id in asym_id_to_pdb_id.items():
        indices = (query_tokens.tokens["asym_id"] == asym_id).nonzero()
        visibility_ids[indices] = pdb_id

    # Set visibility for non templated chain + olygomerics
    for asym_id in np.unique(query_tokens.structure.chains["asym_id"]):
        if asym_id not in asym_id_to_pdb_id:
            # We hack the chain id to be negative to not overlap with the above
            indices = (query_tokens.tokens["asym_id"] == asym_id).nonzero()
            visibility_ids[indices] = -1 - asym_id

    # Convert to one-hot
    res_type = torch.from_numpy(res_type)
    res_type = one_hot(res_type, num_classes=const.num_tokens)

    return {
        "template_restype": res_type,
        "template_frame_rot": torch.from_numpy(frame_rot),
        "template_frame_t": torch.from_numpy(frame_t),
        "template_cb": torch.from_numpy(cb_coords),
        "template_ca": torch.from_numpy(ca_coords),
        "template_mask_cb": torch.from_numpy(cb_mask),
        "template_mask_frame": torch.from_numpy(frame_mask),
        "template_mask": torch.from_numpy(template_mask),
        "query_to_template": torch.from_numpy(query_to_template),
        "visibility_ids": torch.from_numpy(visibility_ids),
    }


def process_template_features(
    data: Tokenized,
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    """Load the given input data.

    Parameters
    ----------
    data : Tokenized
        The input to the model.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, torch.Tensor]
        The loaded template features.

    """
    # Generate full atom mask based on first template's atoms present
    template_atom_present_mask = None
    token_present_mask = None

    if data.templates and len(data.templates) > 0:
        # Get the first template
        first_template_name = list(data.templates.keys())[0]
        first_template_structure = data.templates[first_template_name]
        
        # Generate atom present mask
        num_atoms = len(first_template_structure.atoms)
        template_atom_present_mask = torch.zeros(num_atoms, dtype=torch.bool)
        
        for atom_idx, atom in enumerate(first_template_structure.atoms):
            if atom["is_present"]:
                template_atom_present_mask[atom_idx] = True
        
        # Generate token present mask based on QUERY structure tokens
        num_tokens = max_tokens
        token_present_mask = torch.zeros(num_tokens, dtype=torch.bool)
        
        # Build a mapping from query token to template token
        template_token_map = {}  # (asym_id, res_idx) -> template_token
        for template_token in data.template_tokens[first_template_name]:
            key = (template_token["asym_id"], template_token["res_idx"])
            template_token_map[key] = template_token
        
        # Iterate over query tokens and check if they exist in template
        for query_token_idx, query_token in enumerate(data.tokens):
            if query_token_idx >= num_tokens:
                break
                
            # Check if this token exists in template
            key = (query_token["asym_id"], query_token["res_idx"])
            if key in template_token_map:
                template_token = template_token_map[key]
                # Check if the token's residue is present in template
                # FIXED: numpy.void object uses [] instead of .get()
                if template_token["resolved_mask"]:
                    token_present_mask[query_token_idx] = True


    # Group templates by name
    name_to_templates: dict[str, list[TemplateInfo]] = {}
    for template_info in data.record.templates:
        name_to_templates.setdefault(template_info.name, []).append(template_info)

    # Map chain name to asym_id
    chain_name_to_asym_id = {}
    for chain in data.structure.chains:
        chain_name_to_asym_id[chain["name"]] = chain["asym_id"]

    # Compute the offset
    template_features = []
    for template_id, (template_name, templates) in enumerate(name_to_templates.items()):
        row_tokens = []
        template_structure = data.templates[template_name]
        template_tokens = data.template_tokens[template_name]
        tmpl_chain_name_to_asym_id = {}
        for chain in template_structure.chains:
            tmpl_chain_name_to_asym_id[chain["name"]] = chain["asym_id"]

        for template in templates:
            offset = template.template_st - template.query_st

            # Get query and template tokens to map residues
            query_tokens = data.tokens
            chain_id = chain_name_to_asym_id[template.query_chain]
            q_tokens = query_tokens[query_tokens["asym_id"] == chain_id]
            q_indices = dict(zip(q_tokens["res_idx"], q_tokens["token_idx"]))

            # Get the template tokens at the query residues
            chain_id = tmpl_chain_name_to_asym_id[template.template_chain]
            toks = template_tokens[template_tokens["asym_id"] == chain_id]
            toks = [t for t in toks if t["res_idx"] - offset in q_indices]
            for t in toks:
                q_idx = q_indices[t["res_idx"] - offset]
                row_tokens.append(
                    {
                        "token": t,
                        "pdb_id": template_id,
                        "q_idx": q_idx,
                    }
                )

        # Compute template features for each row
        row_features = compute_template_features(data, row_tokens, max_tokens)
        row_features["template_force"] = torch.tensor(template.force)
        row_features["template_force_threshold"] = torch.tensor(
            template.threshold if template.threshold is not None else float("inf"),
            dtype=torch.float32,
        )
        
        if template_atom_present_mask is not None:
            row_features["template_atom_present_mask"] = template_atom_present_mask
        if token_present_mask is not None:
            row_features["token_present_mask"] = token_present_mask
        
        template_features.append(row_features)
    # Stack each feature
    out = {}
    for k in template_features[0]:
        out[k] = torch.stack([f[k] for f in template_features])  
        if "token_present_mask" in out:
            token_present_mask = out["token_present_mask"]  # shape: [1, 200]
            
            batch_size, num_tokens = token_present_mask.shape
            token_pair_present_mask = torch.zeros(batch_size, num_tokens, num_tokens, dtype=torch.bool)
            
            token_pair_missing_mask = torch.zeros(batch_size, num_tokens, num_tokens, dtype=torch.bool)
            
            for b in range(batch_size):
                # For each batch
                present_tokens = token_present_mask[b]  
                missing_tokens = ~present_tokens  
        
                token_pair_present_mask[b] = present_tokens.unsqueeze(1) & present_tokens.unsqueeze(0)
                
                token_pair_missing_mask[b] = missing_tokens.unsqueeze(1) & missing_tokens.unsqueeze(0)
                
            
            out["token_pair_present_mask"] = token_pair_present_mask 
            out["token_pair_missing_mask"] = token_pair_missing_mask 

    return out
    
def process_ensemble_features(
    data: Tokenized,
    random: np.random.Generator,
    num_ensembles: int,
    ensemble_sample_replacement: bool,
    fix_single_ensemble: bool,
) -> dict[str, Tensor]:
    """Get the ensemble features.

    Parameters
    ----------
    data : Tokenized
        The input to the model.
    random : np.random.Generator
        The random number generator.
    num_ensembles : int
        The maximum number of ensembles to sample.
    ensemble_sample_replacement : bool
        Whether to sample with replacement.

    Returns
    -------
    dict[str, Tensor]
        The ensemble features.

    """
    assert num_ensembles > 0, "Number of conformers sampled must be greater than 0."

    # Number of available conformers in the structure
    # s_ensemble_num = min(len(cropped.structure.ensemble), 24)  # Limit to 24 conformers DEBUG: TODO: remove !
    s_ensemble_num = len(data.structure.ensemble)

    if fix_single_ensemble:
        # Always take the first conformer for train and validation
        assert num_ensembles == 1, (
            "Number of conformers sampled must be 1 with fix_single_ensemble=True."
        )
        ensemble_ref_idxs = np.array([0])
    else:
        if ensemble_sample_replacement:
            # Used in training
            ensemble_ref_idxs = random.integers(0, s_ensemble_num, (num_ensembles,))
        else:
            # Used in validation
            if s_ensemble_num < num_ensembles:
                # Take all available conformers
                ensemble_ref_idxs = np.arange(0, s_ensemble_num)
            else:
                # Sample without replacement
                ensemble_ref_idxs = random.choice(
                    s_ensemble_num, num_ensembles, replace=False
                )

    ensemble_features = {
        "ensemble_ref_idxs": torch.Tensor(ensemble_ref_idxs).long(),
    }

    return ensemble_features



class BoltzFeaturizer:
    """Boltz featurizer."""

    def process(
        self,
        data: Tokenized,
        random: np.random.Generator,
        molecules: dict[str, Mol],
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        num_ensembles: int = 1,
        ensemble_sample_replacement: bool = False,
        disto_use_ensemble: Optional[bool] = False,
        fix_single_ensemble: Optional[bool] = True,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        contact_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff_min: Optional[float] = 4.0,
        binder_pocket_cutoff_max: Optional[float] = 20.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        only_ligand_binder_pocket: Optional[bool] = False,
        only_pp_contact: Optional[bool] = False,
        override_bfactor: float = False,
        override_method: Optional[str] = None,
        compute_frames: bool = False,
        override_coords: Optional[Tensor] = None,
        bfactor_md_correction: bool = False,
        inference_pocket_constraints: Optional[
            list[tuple[int, list[tuple[int, int]], float]]
        ] = None,
        inference_contact_constraints: Optional[
            list[tuple[tuple[int, int], tuple[int, int], float]]
        ] = None,
    ) -> dict[str, Tensor]:
        """Compute features.

        Parameters
        ----------
        data : Tokenized
            The input to the model.
        training : bool
            Whether the model is in training mode.
        max_tokens : int, optional
            The maximum number of tokens.
        max_atoms : int, optional
            The maximum number of atoms

        Returns
        -------
        dict[str, Tensor]
            The features for model training.

        """


        # Compute ensemble features
        ensemble_features = process_ensemble_features(
            data=data,
            random=random,
            num_ensembles=num_ensembles,
            ensemble_sample_replacement=ensemble_sample_replacement,
            fix_single_ensemble=fix_single_ensemble,
        )

        # Compute token features
        token_features , token_pad_len = process_token_features(
            data=data,
            random=random,
            max_tokens=max_tokens,
            binder_pocket_conditioned_prop=binder_pocket_conditioned_prop,
            contact_conditioned_prop=contact_conditioned_prop,
            binder_pocket_cutoff_min=binder_pocket_cutoff_min,
            binder_pocket_cutoff_max=binder_pocket_cutoff_max,
            binder_pocket_sampling_geometric_p=binder_pocket_sampling_geometric_p,
            only_ligand_binder_pocket=only_ligand_binder_pocket,
            only_pp_contact=only_pp_contact,
            override_method=override_method,
            inference_pocket_constraints=inference_pocket_constraints,
            inference_contact_constraints=inference_contact_constraints,
        )
        # Compute atom features
        atom_features, atom_pad_len = process_atom_features(
            data=data,
            random=random,
            molecules=molecules,
            ensemble_features=ensemble_features,
            atoms_per_window_queries=atoms_per_window_queries,
            min_dist=min_dist,
            max_dist=max_dist,
            num_bins=num_bins,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            disto_use_ensemble=disto_use_ensemble,
            override_bfactor=override_bfactor,
            compute_frames=compute_frames,
            override_coords=override_coords,
            bfactor_md_correction=bfactor_md_correction,
        )


        # Compute template features
        num_tokens = data.tokens.shape[0] if max_tokens is None else max_tokens
        if data.templates:
            template_features = process_template_features(
                data=data,
                max_tokens=num_tokens,
            )
        else:
            template_features = load_dummy_templates_features(
                tdim=1,
                num_tokens=num_tokens,
            )
        template_features['template_atom_present_mask'] = pad_dim(template_features['template_atom_present_mask'], 1, atom_pad_len)
        template_features['token_present_mask'] = pad_dim(template_features['token_present_mask'], 1, token_pad_len)
        
   
        
        return {
            **token_features,
            **atom_features,
            **template_features,
            **ensemble_features,
        }
