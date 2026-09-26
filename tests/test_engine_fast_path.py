from types import SimpleNamespace

import numpy as np
import torch

from CryoNetRefine.model.engine import Engine


class FakeFeaturizer:
    def __init__(self):
        self.calls = 0

    def process_crop(self, **kwargs):
        self.calls += 1
        features = {
            "template_coords": torch.zeros((1, 2, 3)),
            "atom_pad_mask": torch.ones(2, dtype=torch.bool),
        }
        return features, np.array([2]), np.array([1, 3])


def make_engine():
    engine = Engine.__new__(Engine)
    engine.crop_first_tokenized = object()
    engine.crop_first_molecules = object()
    engine.crop_first_record = SimpleNamespace(id="first")
    engine.crop_first_override_method = None
    engine.crop_first_seed = 42
    engine.crop_first_featurizer = FakeFeaturizer()
    engine.max_tokens = 512
    engine.crop_batch_cache = {}
    engine.crop_feature_cache = {"stale": object()}
    engine.crop_atom_types_cache = {"stale": object()}
    return engine


def test_crop_first_batch_is_featurized_once():
    engine = make_engine()
    batch = {"atom_pad_mask": torch.ones((1, 5), dtype=torch.bool)}
    crop = (7, np.array([2]), "PROTEIN", {"num_tokens": 1})

    first, first_mask, _ = engine._get_crop_first_batch(batch, crop)
    second, second_mask, _ = engine._get_crop_first_batch(batch, crop)

    assert engine.crop_first_featurizer.calls == 1
    assert first is not second
    assert first["template_coords"] is second["template_coords"]
    assert torch.equal(first_mask, second_mask)
    assert first_mask.tolist() == [False, True, False, True, False]


def test_new_crop_context_clears_record_local_caches():
    engine = make_engine()

    engine.set_crop_first_context(
        tokenized=object(),
        molecules=object(),
        record=SimpleNamespace(id="second"),
        crop_plans=[],
        featurizer=FakeFeaturizer(),
    )

    assert engine.crop_batch_cache == {}
    assert engine.crop_feature_cache == {}
    assert engine.crop_atom_types_cache == {}
