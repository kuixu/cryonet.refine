"""Length-balanced distributed sampler for load balancing across GPUs."""

import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, Dict, List


class LengthBalancedDistributedSampler(Sampler):
    """
    Distributed sampler that balances load based on sequence lengths.
    
    Uses a greedy algorithm to assign samples to ranks, ensuring that each rank
    has approximately the same total sequence length (computational load).
    
    Args:
        dataset: Dataset object, should have a method to get sequence length
        num_replicas: Total number of GPUs
        rank: Current process rank
        shuffle: Whether to shuffle samples within each rank between epochs
        seed: Random seed for shuffling
        drop_last: Whether to drop samples that cannot be evenly divided (usually False)
    """
    
    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Get sequence length information for all samples
        self.sample_lengths = self._get_sample_lengths()
        
        # Use greedy algorithm to assign samples to ranks
        self.rank_indices = self._balance_assignment()
        
        # Number of samples for current rank
        self.num_samples = len(self.rank_indices[self.rank])
        self.total_size = sum(len(indices) for indices in self.rank_indices.values())
        
    def _get_sample_lengths(self) -> np.ndarray:
        """Get sequence lengths for all samples."""
        lengths = []
        for idx in range(len(self.dataset)):
            try:
                # Try to get length from dataset
                if hasattr(self.dataset, 'get_sequence_length'):
                    length = self.dataset.get_sequence_length(idx)
                elif hasattr(self.dataset, 'manifest'):
                    # Get length from manifest
                    # ChainInfo has num_residues attribute, not sequence
                    record = self.dataset.manifest.records[idx]
                    length = sum(chain.num_residues for chain in record.chains)
                else:
                    # Default length of 1 (no load balancing)
                    length = 1
                lengths.append(length)
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️  Warning: Failed to get length for sample {idx}: {e}")
                lengths.append(1)  # Default length
        return np.array(lengths)
    
    def _balance_assignment(self) -> Dict[int, List[int]]:
        """Use greedy algorithm for load-balanced assignment."""
        # Create (index, length) pairs and sort by length in descending order
        indexed_lengths = [(idx, length) for idx, length in enumerate(self.sample_lengths)]
        indexed_lengths.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize sample lists and cumulative lengths for each rank
        rank_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_replicas)}
        rank_total_lengths: Dict[int, int] = {i: 0 for i in range(self.num_replicas)}
        
        # Greedy assignment: assign each sample to the rank with minimum current load
        for idx, length in indexed_lengths:
            # Find rank with minimum current load
            min_rank = min(rank_total_lengths.items(), key=lambda x: x[1])[0]
            
            # Assign sample
            rank_indices[min_rank].append(idx)
            rank_total_lengths[min_rank] += length
        
        # Print load balancing information (only rank 0)
        if self.rank == 0:
            print("\n" + "="*80)
            print("📊 Length-Balanced Sample Assignment:")
            print("="*80)
            for r in range(self.num_replicas):
                num_samples = len(rank_indices[r])
                total_length = rank_total_lengths[r]
                avg_length = total_length / num_samples if num_samples > 0 else 0
                print(f"  Rank {r}: {num_samples:3d} samples, "
                      f"total_length={total_length:6d}, avg_length={avg_length:7.1f}")
            
            # Calculate load balance metric (coefficient of variation)
            lengths_list = list(rank_total_lengths.values())
            std = np.std(lengths_list)
            mean = np.mean(lengths_list)
            cv = (std / mean * 100) if mean > 0 else 0
            min_load = min(lengths_list)
            max_load = max(lengths_list)
            imbalance = ((max_load - min_load) / mean * 100) if mean > 0 else 0
            
            print(f"\n  📈 Load Balance Metrics:")
            print(f"     Coefficient of Variation (CV): {cv:.2f}% (lower is better)")
            print(f"     Load Imbalance: {imbalance:.2f}% (lower is better)")
            print(f"     Min Load: {min_load}, Max Load: {max_load}, Mean: {mean:.1f}")
            print("="*80 + "\n")
        
        return rank_indices
    
    def __iter__(self) -> Iterator[int]:
        """Return iterator over sample indices for current rank."""
        indices = self.rank_indices[self.rank].copy()
        
        # If shuffle is enabled, shuffle samples within current rank while maintaining load balance
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return number of samples for current rank."""
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch (used for shuffling)."""
        self.epoch = epoch

