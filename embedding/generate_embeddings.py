
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


DEFAULT_CDR3_WINDOW_STATS = {
    # These are reasonable defaults and can be adjusted.
    # They represent the mean and standard deviation of the desired
    # cropping window length around the CDR3 region.
    "IgA": {"mu": 160.0, "std": 40.0},
    "IgG": {"mu": 160.0, "std": 40.0},
}


def random_crop_cdr3(seq: str, cdr3: str, mu: float, std: float) -> str:
    """Randomly crop a window around the CDR3 region.

    The window length is drawn from a normal distribution N(mu, std).
    The crop is centered on the CDR3 sequence as much as possible,
    and clipped to stay within sequence boundaries.

    If the CDR3 substring cannot be found, the original sequence is returned.
    """
    if not seq or not cdr3:
        return seq

    # Draw a random window length, at least as long as the CDR3 itself.
    length = int(round(np.random.normal(mu, std)))
    length = max(len(cdr3), length)

    start_pos = seq.find(cdr3)
    if start_pos == -1:
        # Fallback: no CDR3 found, return the full sequence.
        return seq

    # Ideal pre/post sizes around CDR3.
    pre = max(0, (length - len(cdr3)) // 2)
    post = length - pre - len(cdr3)

    # Compute initial window bounds.
    left = max(0, start_pos - pre)
    right = min(len(seq), start_pos + len(cdr3) + post)

    # Expand window if it is too short (within sequence boundaries).
    window = seq[left:right]
    if len(window) < length:
        deficit = length - len(window)
        extra_left = min(left, deficit // 2)
        extra_right = min(len(seq) - right, deficit - extra_left)
        left -= extra_left
        right += extra_right
        window = seq[left:right]

    return window


def load_encoder(model_dir: str, device: Optional[torch.device] = None) -> Tuple[RobertaTokenizer, RobertaModel]:
    """Load a pretrained Roberta tokenizer and encoder from ``model_dir``.

    Parameters
    ----------
    model_dir:
        Path or model name that can be resolved by Hugging Face ``from_pretrained``.

    device:
        Torch device to move the model to. If ``None``, CUDA is used if available.

    Returns
    -------
    tokenizer, model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_embeddings_for_sequences(
    model_dir: str,
    sequences: List[str],
    cdr3s: Optional[List[str]] = None,
    target: Optional[str] = None,
    max_length: int = 160,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True,
) -> Tuple[torch.Tensor, List[str]]:
    """Generate embeddings for a list of antibody sequences.

    This function loads a pretrained antibody language model from ``model_dir`` and
    returns one embedding vector per input sequence.

    Parameters
    ----------
    model_dir:
        Directory or Hugging Face model name for the pretrained encoder.

    sequences:
        List of amino acid sequences. Each entry corresponds to one sequence.

    cdr3s:
        Optional list of CDR3 amino acid sequences. If provided and ``target``
        is present in ``DEFAULT_CDR3_WINDOW_STATS``, a random crop around
        CDR3 is performed for each sequence before encoding.

    target:
        Antibody isotype (e.g. "IgA" or "IgG"). Used to pick ``mu`` and ``std``
        for cropping from ``DEFAULT_CDR3_WINDOW_STATS``.

    max_length:
        Maximum tokenized sequence length used for the encoder.

    batch_size:
        Batch size for embedding generation.

    device:
        Torch device. If ``None``, CUDA is used if available.

    use_tqdm:
        Whether to wrap the iteration with ``tqdm`` for progress display.

    Returns
    -------
    embs:
        A tensor of shape ``(N, H)`` where ``N`` is the number of sequences and
        ``H`` is the hidden dimension of the encoder.

    cropped_sequences:
        The list of actual sequences passed to the encoder (after cropping).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_encoder(model_dir, device=device)

    if cdr3s is not None and len(cdr3s) != len(sequences):
        raise ValueError("If cdr3s is provided, it must have the same length as sequences.")

    # Determine cropping statistics if applicable.
    if target is not None and target in DEFAULT_CDR3_WINDOW_STATS:
        mu = DEFAULT_CDR3_WINDOW_STATS[target]["mu"]
        std = DEFAULT_CDR3_WINDOW_STATS[target]["std"]
    else:
        mu = None
        std = None

    all_embs: List[torch.Tensor] = []
    cropped_sequences: List[str] = []

    indices = range(len(sequences))
    iterator = indices
    if use_tqdm:
        iterator = tqdm(indices, desc="Generating embeddings")

    batch_seqs: List[str] = []
    for idx in iterator:
        seq = sequences[idx]
        cdr3 = cdr3s[idx] if (cdr3s is not None) else None

        if mu is not None and std is not None and cdr3:
            seq_to_encode = random_crop_cdr3(seq, cdr3, mu, std)
        else:
            seq_to_encode = seq

        cropped_sequences.append(seq_to_encode)
        batch_seqs.append(seq_to_encode)

        if len(batch_seqs) == batch_size or idx == len(sequences) - 1:
            # Tokenize and encode this batch.
            enc = tokenizer(
                batch_seqs,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                # Average pool over sequence length: (B, L, H) -> (B, H)
                batch_embs = outputs.last_hidden_state.mean(dim=1)

            all_embs.append(batch_embs.cpu())
            batch_seqs = []

    if not all_embs:
        # No sequences.
        return torch.empty(0), []

    embs = torch.cat(all_embs, dim=0)
    return embs, cropped_sequences
