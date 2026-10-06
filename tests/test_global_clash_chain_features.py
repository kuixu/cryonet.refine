"""The memory-light global batch must preserve chain IDs for clash screening."""
from types import SimpleNamespace

import numpy as np
import torch

from main import build_lightweight_global_batch
from CryoNetRefine.loss.loss import probe_style_clash_loss


def test_global_batch_preserves_chain_ids():
    tokens = np.array([(0, 0, 1, 831, 1, "LEU"), (1, 1, 1, 831, 2, "LEU")],
                      dtype=[("token_idx", "i8"), ("atom_idx", "i8"), ("atom_num", "i8"),
                             ("res_idx", "i8"), ("asym_id", "i8"), ("res_name", "U3")])
    structure = SimpleNamespace(
        ensemble=np.array([(0,)], dtype=[("atom_coord_idx", "i8")]),
        coords=np.array([((0., 0., 0.),), ((2.898, 0., 0.),)], dtype=[("coords", "f4", (3,))]),
        atoms=np.array([("CD2", True), ("CD2", True)], dtype=[("name", "U4"), ("is_present", "?")]),
    )
    case = SimpleNamespace(record=SimpleNamespace(id="synthetic"),
                           tokenized=SimpleNamespace(tokens=tokens, structure=structure), molecules={})
    batch = build_lightweight_global_batch(case)
    torch.testing.assert_close(batch["asym_id"], torch.tensor([[1, 2]]))
    coords = batch["template_coords"].squeeze(1).requires_grad_(True)
    _, count = probe_style_clash_loss(coords, batch)
    torch.testing.assert_close(count.reshape(()), torch.sigmoid(torch.tensor(1.02)))
    count.sum().backward()
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0
