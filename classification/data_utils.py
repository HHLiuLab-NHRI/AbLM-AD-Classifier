
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    """Dataset for sample-level antibody embeddings.

    Each element in ``samples`` is expected to be a dictionary containing at least:
    - ``'embs'``: array-like of shape ``(N, D)`` (N sequences, D embedding dimension)
    - ``'label'``: scalar (0/1 or float)

    The dataset pads or truncates the sequence dimension to ``max_length`` and
    returns tensors suitable for the CNN1DClassifier.
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(samples) == 0:
            raise ValueError("EmbeddingDataset received an empty list of samples.")

        self.samples = list(samples)

        # Infer maximum length if not provided.
        if max_length is None:
            max_length = max(len(np.asarray(s["embs"])) for s in self.samples)
        self.max_length = int(max_length)

        # Infer embedding dimension from the first sample.
        first_embs = np.asarray(self.samples[0]["embs"], dtype=np.float32)
        if first_embs.ndim != 2:
            raise ValueError("Each 'embs' entry must be 2D: (N, D).")
        self.embedding_dim = int(first_embs.shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        embs = np.asarray(sample["embs"], dtype=np.float32)  # (N, D)
        label = float(sample["label"])

        # Truncate or pad along sequence dimension (axis=0).
        n, d = embs.shape
        if d != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {d}.")

        if n > self.max_length:
            embs = embs[: self.max_length, :]
            n = self.max_length

        if n < self.max_length:
            pad_len = self.max_length - n
            pad = np.zeros((pad_len, self.embedding_dim), dtype=np.float32)
            embs = np.concatenate([embs, pad], axis=0)

        # Now shape is (max_length, D). Transpose to (D, max_length) for Conv1D.
        embs_tensor = torch.from_numpy(embs).transpose(0, 1)  # (D, L)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {"embs": embs_tensor, "label": label_tensor}


def split_train_valid(
    samples: Sequence[Dict[str, Any]],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Randomly split ``samples`` into train and validation sets."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_valid = int(len(indices) * valid_ratio)
    valid_idx = indices[:n_valid]
    train_idx = indices[n_valid:]
    train_samples = [samples[i] for i in train_idx]
    valid_samples = [samples[i] for i in valid_idx]
    return train_samples, valid_samples


def create_dataloaders(
    train_samples: Sequence[Dict[str, Any]],
    valid_samples: Sequence[Dict[str, Any]],
    batch_size: int = 1,
    max_length: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create PyTorch dataloaders for training and validation.

    Returns
    -------
    train_loader, valid_loader, max_length
    """
    # Infer max_length from the union of train + valid if not given.
    if max_length is None:
        combined = list(train_samples) + list(valid_samples)
        if len(combined) == 0:
            raise ValueError("No samples provided to create_dataloaders.")
        max_length = max(len(np.asarray(s["embs"])) for s in combined)

    train_ds = EmbeddingDataset(train_samples, max_length=max_length)
    valid_ds = EmbeddingDataset(valid_samples, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, int(max_length)
