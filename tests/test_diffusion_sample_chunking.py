from __future__ import annotations

import pytest
import torch

from CryoNetRefine.model.modules.diffusion import _split_sample_ids


@pytest.mark.parametrize(
    ("multiplicity", "max_parallel_samples", "expected_sizes"),
    [
        (1, 1, [1]),
        (4, 2, [2, 2]),
        (5, 2, [2, 2, 1]),
        (8, 4, [4, 4]),
        (4, 8, [4]),
    ],
)
def test_split_sample_ids_respects_parallel_limit(
    multiplicity: int,
    max_parallel_samples: int,
    expected_sizes: list[int],
) -> None:
    sample_ids = torch.arange(multiplicity)

    chunks = _split_sample_ids(sample_ids, max_parallel_samples)

    assert [chunk.numel() for chunk in chunks] == expected_sizes
    assert torch.equal(torch.cat(chunks), sample_ids)
    assert all(chunk.numel() <= max_parallel_samples for chunk in chunks)


def test_split_sample_ids_none_processes_all_samples_together() -> None:
    sample_ids = torch.arange(4)

    chunks = _split_sample_ids(sample_ids, None)

    assert len(chunks) == 1
    assert torch.equal(chunks[0], sample_ids)


@pytest.mark.parametrize("max_parallel_samples", [0, -1])
def test_split_sample_ids_rejects_non_positive_limit(
    max_parallel_samples: int,
) -> None:
    with pytest.raises(
        ValueError, match="max_parallel_samples must be a positive integer"
    ):
        _split_sample_ids(torch.arange(4), max_parallel_samples)


def test_split_sample_ids_rejects_zero_multiplicity() -> None:
    with pytest.raises(ValueError, match="multiplicity must be a positive integer"):
        _split_sample_ids(torch.arange(0), 1)
