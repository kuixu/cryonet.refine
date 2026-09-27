from types import SimpleNamespace

import torch

from CryoNetRefine.libs.geometry.GeoMetric import GeoMetric


def test_rama_lookup_reuses_device_table():
    metric = GeoMetric.__new__(GeoMetric)
    metric._rama_tables_cache = {0: torch.arange(16, dtype=torch.float32).reshape(4, 4)}
    metric._rama_limits_cache = {0: (-180.0, 180.0, -180.0, 180.0)}
    metric._rama_table_device_cache = {}
    angles = torch.tensor([[10.0, 20.0], [-30.0, 40.0]])

    first = metric.lookup_rama_scores_cached(angles, 0)
    cached_table = next(iter(metric._rama_table_device_cache.values()))
    second = metric.lookup_rama_scores_cached(angles, 0)

    assert torch.equal(first, second)
    assert next(iter(metric._rama_table_device_cache.values())) is cached_table


def test_rotamer_lookup_reuses_device_tensors():
    metric = GeoMetric.__new__(GeoMetric)
    metric._rotamer_table_device_cache = {}
    table = torch.linspace(0.0, 1.0, 36, dtype=torch.float32)
    ndt = SimpleNamespace(
        minVal=[0.0],
        wBin=[10.0],
        nBins=[36],
        doWrap=[True],
        lookupTable=table,
    )
    angles = torch.tensor([[15.0], [175.0]])

    first = metric._interpolate_rotamer_scores_differentiable(
        angles, ndt, 1, resname="ser"
    )
    cached_tensors = next(iter(metric._rotamer_table_device_cache.values()))
    second = metric._interpolate_rotamer_scores_differentiable(
        angles, ndt, 1, resname="ser"
    )

    assert torch.equal(first, second)
    assert next(iter(metric._rotamer_table_device_cache.values())) is cached_tensors
