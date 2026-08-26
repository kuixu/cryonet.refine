from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import CryoNetRefine.libs.geometry.GeoMetric as geometry_module
from CryoNetRefine.libs.geometry.GeoMetric import GeoMetric


def _bare_metric() -> GeoMetric:
    metric = GeoMetric.__new__(GeoMetric)
    metric._rmsd_grm_cache = None
    metric._rmsd_sites_cart_cache = None
    metric._rmsd_perm_tensor_cache = None
    metric._rmsd_cache_key = None
    metric._rmsd_failed_cache_keys = set()
    return metric


class GeometryFallbackTests(unittest.TestCase):
    def test_rmsd_setup_failure_returns_complete_cached_zero_losses(self):
        calls = 0

        def fail_input(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise KeyError("5*END")

        metric = _bare_metric()
        coords = torch.zeros((3, 3), dtype=torch.float32)
        with patch.object(geometry_module.iotbx.pdb, "input", side_effect=fail_input):
            first = metric.compute_bond_angle_rmsd_from_pdb(
                "broken-rna.cif", coords, cache_key="rna-crop-2"
            )
            second = metric.compute_bond_angle_rmsd_from_pdb(
                "broken-rna.cif", coords, cache_key="rna-crop-2"
            )

        expected_keys = {"bond_rmsd", "angle_rmsd", "nonbonded_loss"}
        self.assertEqual(set(first), expected_keys)
        self.assertEqual(set(second), expected_keys)
        self.assertTrue(all(value.item() == 0.0 for value in first.values()))
        self.assertTrue(all(value.requires_grad for value in first.values()))
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
