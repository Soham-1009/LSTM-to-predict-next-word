"""PyTorch Dataset / DataLoader for next-word training samples."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class NextWordDataset(Dataset):
    """Holds (input_sequence, target_token_id) pairs produced from a corpus."""

    def __init__(self, inputs: Sequence[Sequence[int]], targets: Sequence[int]) -> None:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        self.inputs: List[List[int]] = [list(seq) for seq in inputs]
        self.targets: List[int] = [int(t) for t in targets]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.inputs[index], dtype=torch.long)
        y = torch.tensor(self.targets[index], dtype=torch.long)
        return x, y


def make_dataloaders(
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int],
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.05,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split a (inputs, targets) pair into train/val/test DataLoaders.

    The splits are deterministic given a fixed seed.
    """
    n = len(inputs)
    if n == 0:
        raise ValueError("No training samples available")

    # Shuffle indices.
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()

    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples to make a train/val/test split")

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    def _subset(idxs):
        return [inputs[i] for i in idxs], [targets[i] for i in idxs]

    train_ds = NextWordDataset(*_subset(train_idx))
    val_ds = NextWordDataset(*_subset(val_idx))
    test_ds = NextWordDataset(*_subset(test_idx))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
