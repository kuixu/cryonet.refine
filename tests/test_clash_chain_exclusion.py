"""Chain-aware clash screening, including checkpointed block gradients."""
import pytest
import torch
import torch.nn.functional as F

from CryoNetRefine.data import const
from CryoNetRefine.loss.loss import probe_style_clash_loss


def _pair(chain_ids, residue_ids, mapping, padded=False):
    radius = const.vdw_radii[5]  # carbon, atomic number 6
    distance = 2 * radius - .502
    positions = [[0., 0., 0.], [distance, 0., 0.]]
    tokens = [0, 1]
    pad_mask = [True, True]
    present_mask = [True, True]
    if padded:
        # Interleaved padding and an unresolved atom must both be ignored.
        positions = [[0., 0., 0.], [.1, 0., 0.], [distance, 0., 0.], [.2, 0., 0.]]
        tokens = [0, 0, 1, 1]
        pad_mask = [True, False, True, True]
        present_mask = [True, True, True, False]
    coords = torch.tensor([positions], requires_grad=True)
    feats = {
        "atom_pad_mask": torch.tensor([pad_mask]),
        "template_atom_present_mask": torch.tensor([[present_mask]]),
        "ref_element": torch.full((1, len(tokens)), 6),
        "residue_index": torch.tensor([residue_ids]),
        "asym_id": torch.tensor([chain_ids]),
    }
    if mapping == "dense":
        feats["atom_to_token"] = F.one_hot(torch.tensor([tokens]), 2)
    else:
        feats["atom_token_index"] = torch.tensor([tokens])
    return coords, feats


@pytest.mark.parametrize("mapping", ["dense", "index"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("chains,residues,excluded", [
    ([1, 2], [831, 831], False),
    ([1, 2], [831, 832], False),
    ([1, 1], [831, 831], True),
    ([1, 1], [831, 832], True),
    ([1, 1], [831, 833], False),
])
def test_chain_aware_clash_values_and_gradients(mapping, padded, chains, residues, excluded):
    results = []
    for chunk_size in (5000, 1):
        coords, feats = _pair(chains, residues, mapping, padded)
        score, count = probe_style_clash_loss(coords, feats, chunk_size=chunk_size)
        score, count = score.reshape(()), count.reshape(())
        expected = torch.tensor(0.) if excluded else torch.sigmoid(torch.tensor(1.02))
        torch.testing.assert_close(count, expected)
        torch.testing.assert_close(score, expected * 500.)  # two valid atoms
        score.backward()
        assert torch.isfinite(coords.grad).all()
        if excluded:
            assert coords.grad.abs().sum() == 0
        else:
            assert coords.grad.abs().sum() > 0
        if padded:
            assert coords.grad[0, [1, 3]].abs().sum() == 0
        results.append((score.detach(), count.detach(), coords.grad))
    for full, blocked in zip(*results):
        torch.testing.assert_close(full, blocked)


@pytest.mark.parametrize("mapping", ["dense", "index"])
@pytest.mark.parametrize("chunk_size", [5000, 1])
def test_missing_chain_ids_are_not_silently_assumed(mapping, chunk_size):
    coords, feats = _pair([1, 2], [831, 831], mapping)
    del feats["asym_id"]
    with pytest.raises(ValueError, match="asym_id is required"):
        probe_style_clash_loss(coords, feats, chunk_size=chunk_size)
