"""PyTorch Dataset for Causal Language Modeling.

Implements the shifted token input/target pair:
    Input:  w1, w2, w3, ..., wn
    Target: w2, w3, w4, ..., w_{n+1}

Supports document-boundary-aware sequence generation using document offset tables,
guaranteeing that input and target windows never cross document boundaries.
"""

from typing import Optional, Sequence, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class CausalLanguageModelDataset(Dataset):
    """Memory-mapped or in-memory sequence dataset for causal language model training.
    
    When `doc_offsets` is provided, sequences are formed independently within each document,
    guaranteeing that no input/target window crosses document boundaries and no target
    token is sourced from the next document.
    """

    def __init__(
        self,
        token_data: Union[np.ndarray, np.memmap],
        seq_len: int = 35,
        stride: Optional[int] = None,
        doc_offsets: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> None:
        self.data = token_data
        self.seq_len = seq_len
        self.stride = stride if stride is not None else 1
        window_size = self.seq_len + 1

        if doc_offsets is not None:
            offsets = np.asarray(doc_offsets, dtype=np.int64)
            if offsets.ndim != 1:
                raise ValueError("doc_offsets must be a 1D sequence or array.")
            if len(offsets) < 2:
                raise ValueError("doc_offsets must contain at least 2 entries [0, total_tokens].")
            if offsets[0] != 0:
                raise ValueError(f"First document offset must be 0, got {offsets[0]}.")
            if offsets[-1] != len(self.data):
                raise ValueError(
                    f"Final document offset ({offsets[-1]}) must match token data length ({len(self.data)})."
                )
            if np.any(np.diff(offsets) < 0):
                raise ValueError("Document offsets must be monotonically non-decreasing.")

            sample_starts: list[int] = []
            for doc_idx in range(len(offsets) - 1):
                doc_start = int(offsets[doc_idx])
                doc_end = int(offsets[doc_idx + 1])
                doc_len = doc_end - doc_start
                if doc_len >= window_size:
                    for s in range(0, doc_len - window_size + 1, self.stride):
                        sample_starts.append(doc_start + s)

            self.sample_indices = np.array(sample_starts, dtype=np.int64)
            self.num_samples = len(self.sample_indices)
        else:
            self.sample_indices = None
            max_start = len(self.data) - window_size
            if max_start < 0:
                self.num_samples = 0
            else:
                self.num_samples = (max_start // self.stride) + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples.")

        if self.sample_indices is not None:
            start = int(self.sample_indices[idx])
        else:
            start = idx * self.stride

        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

