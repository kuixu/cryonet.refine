"""CPU integration tests using real geometry calculations, not mocked losses."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from CryoNetRefine.data import const
from CryoNetRefine.data.types import (
    AtomV2, BondV2, Chain, ChainInfo, Coords, Ensemble, Interface, Residue, StructureV2,
)
from CryoNetRefine.loss.geometric import GeometricAdapter, GeometricMetricWrapper
from CryoNetRefine.loss.loss import refine_loss


def _synthetic_peptide(tmp_path, cropped):
    # Four complete leucines provide backbone torsions and side-chain rotamers.
    # Deliberately imperfect, non-collinear coordinates exercise nonzero losses.
    local = {
        "N": (0., 0., 0.), "CA": (1.45, .2, .1), "C": (2.1, 1.5, .3),
        "O": (1.6, 2.5, .6), "CB": (1.8, -.7, 1.3),
        "CG": (2.8, -1.6, 1.1), "CD1": (4., -1., 1.7),
        "CD2": (2.4, -2.9, 1.8),
    }
    atoms, residues, positions, token_indices = [], [], [], []
    for i in range(4):
        start = len(atoms)
        for name in const.ref_atoms["LEU"]:
            xyz = np.asarray(local[name]) + np.array([3.2 * i, .4 * i, .3 * (i % 2)])
            atoms.append((name, xyz, True, 20., 0.))
            positions.append(xyz)
            token_indices.append(i)
        residues.append(("LEU", 0, const.token_ids["LEU"], start,
                         len(atoms) - start, i, i, True, True, str(i + 1), "", "LEU"))
    n = len(atoms)
    structure = StructureV2(
        atoms=np.array(atoms, dtype=AtomV2), bonds=np.array([], dtype=BondV2),
        residues=np.array(residues, dtype=Residue),
        chains=np.array([("A", 0, 0, 0, 0, 0, n, 0, 4, 0, "A")], dtype=Chain),
        interfaces=np.array([], dtype=Interface), mask=np.array([True]),
        coords=np.array([(p,) for p in positions], dtype=Coords),
        ensemble=np.array([(0, n)], dtype=Ensemble), pocket=None,
    )
    structure.dump(tmp_path / "synthetic.npz")
    coords = torch.tensor(np.asarray(positions), dtype=torch.float32).unsqueeze(0)
    coords.requires_grad_(True)
    feats = {
        "record": [SimpleNamespace(id="synthetic", chains=[ChainInfo(0, "A", 0, 0, 4)])],
        "atom_pad_mask": torch.ones(1, n, dtype=torch.bool),
        "atom_resolved_mask": torch.ones(1, n, dtype=torch.bool),
        "template_atom_present_mask": torch.ones(1, 1, n, dtype=torch.bool),
        "token_pad_mask": torch.ones(1, 4, dtype=torch.bool),
        "atom_to_token": F.one_hot(torch.tensor(token_indices), 4).unsqueeze(0),
        "res_type": F.one_hot(torch.full((1, 4), const.token_ids["LEU"]), len(const.tokens)),
        "residue_index": torch.arange(4).unsqueeze(0),
        "ref_element": torch.tensor([[7 if a[0] == "N" else 8 if a[0] == "O" else 6 for a in atoms]]),
        "is_cropped": cropped, "crop_type": "molecule_aware",
        "global_atom_indices": torch.arange(n),
        "crop_metadata": {"num_tokens": 4},
    }
    return coords, feats


@pytest.mark.parametrize("cropped", [False, True], ids=["full", "cropped"])
def test_geometric_loss_pipeline(tmp_path, cropped):
    coords, feats = _synthetic_peptide(tmp_path, cropped)
    terms = ("rama", "rotamer", "cbeta", "bond", "angle", "ramaz", "nonbonded", "clash")
    weights = {name: (i + 1) / 10 for i, name in enumerate(terms)}
    weights.update(geometric=1., den=0.)
    args = SimpleNamespace(weight_dict=weights, data_dir=tmp_path,
                           geo_metric_root=None, use_global_clash=False)
    adapter = GeometricAdapter(device="cpu", data_dir=tmp_path)
    wrapper = GeometricMetricWrapper(geom_root=None, pdb_id="synthetic", device=torch.device("cpu"))
    _, total, losses, timings = refine_loss(
        0, coords, None, feats, args, geometric_adapter=adapter, geometric_wrapper=wrapper,
    )
    for name in terms:
        assert name in losses
        assert losses[name].ndim == 0
        assert torch.isfinite(losses[name])
    assert losses["bond"] > 0
    assert losses["angle"] > 0
    # A swallowed CIF/cctbx error must not masquerade as a passing zero-loss test.
    metric = wrapper._crop_cache["0"]
    assert not metric._rmsd_failed_cache_keys
    assert metric._rmsd_cache_key is not None
    torch.testing.assert_close(total, sum(losses[name] for name in terms))
    assert all(np.isfinite(value) and value >= 0 for value in timings.values())
    total.backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0
    assert not list(tmp_path.glob("*_temp.cif"))
