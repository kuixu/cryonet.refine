from functools import partial
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
    engine.global_template_coords_cache = object()
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
    assert engine.global_template_coords_cache is None
    assert engine.crop_atom_types_cache == {}


def _indexed_values(values, *, indexing_matrix, W=None, H=None):
    return values[indexing_matrix]


def test_static_crop_feature_cpu_round_trip_is_exact():
    engine = make_engine()
    engine.device = torch.device("cpu")
    s = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
    z = torch.tensor([[[3.0, 4.5]]], dtype=torch.float16)
    s_inputs = torch.tensor([[7, 11]], dtype=torch.int64)
    indexing_matrix = torch.tensor([1, 0], dtype=torch.int64)
    conditioning = {
        "q": torch.tensor([[0.125, 0.25]], dtype=torch.float32),
        "to_keys": partial(
            _indexed_values,
            indexing_matrix=indexing_matrix,
            W=2,
            H=1,
        ),
    }

    cached = engine._cache_static_crop_features(s, z, conditioning, s_inputs)

    assert cached["s"].device.type == "cpu"
    assert cached["z"].device.type == "cpu"
    assert cached["s_inputs"].device.type == "cpu"
    assert cached["diffusion_conditioning"]["q"].device.type == "cpu"
    assert (
        cached["diffusion_conditioning"]["to_keys"].keywords["indexing_matrix"].device.type
        == "cpu"
    )

    loaded_s, loaded_z, loaded_conditioning, loaded_s_inputs = (
        engine._load_static_crop_features(cached)
    )

    assert torch.equal(loaded_s, s)
    assert loaded_s.dtype == s.dtype
    assert torch.equal(loaded_z, z)
    assert loaded_z.dtype == z.dtype
    assert torch.equal(loaded_s_inputs, s_inputs)
    assert loaded_s_inputs.dtype == s_inputs.dtype
    assert torch.equal(loaded_conditioning["q"], conditioning["q"])
    assert torch.equal(
        loaded_conditioning["to_keys"].keywords["indexing_matrix"], indexing_matrix
    )
