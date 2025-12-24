import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Requires: pip install info-nce-pytorch
from info_nce import InfoNCE

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    enable_contrastive: bool = True,
    checkpoint_dir: str = "checkpoints",  
    model_prefix: str = "cnn1d",          
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train a binary classifier with optional Contrastive Learning regularization
    and checkpoint saving.

    Parameters
    ----------
    model:
        The CNN1DClassifier model.
    train_loader, valid_loader:
        PyTorch dataloaders.
    num_epochs:
        Number of epochs.
    lr:
        Learning rate.
    enable_contrastive:
        If True, applies InfoNCE regularization when batch labels allow it.
    checkpoint_dir:
        Directory to save the best models.
    model_prefix:
        String prefix for the saved model files (e.g., 'IgG_AntiBERTa').
    device:
        Device to train on.

    Returns
    -------
    history:
        Dictionary of metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    loss_func = nn.BCEWithLogitsLoss()
    reg_loss_func = InfoNCE() if enable_contrastive else None

    optimizer = AdamW(model.parameters(), lr=lr)
    act = nn.Sigmoid()

    best_auc = -1.0
    best_f1 = -1.0

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_auc": [],
        "valid_f1": [],
        "train_f1": [],
    }

    for epoch in range(num_epochs):
        print('------------------------')
        print(f'Epoch: {epoch + 1}/{num_epochs}')

        # --- Training ---
        model.train()
        pred_list = []
        y_list = []
        train_losses_accum = 0.0
        num_updates = 0  # <--- key

        _ys = []
        _hs = []
        _losses = []

        for step, batched in enumerate(train_loader):
            batched_y = batched['label'].to(device).float()
            if batched_y.ndim == 1:
                batched_y = batched_y.unsqueeze(1)  # (B, 1)

            batched_x = batched['embs'].to(device)

            outputs = model(batched_x)
            loss = loss_func(outputs, batched_y)

            if enable_contrastive:
                # buffer for 3 steps
                _ys.append(batched_y.cpu())
                _hs.append(model.hidden)
                _losses.append(loss)

                if len(_ys) == 3:
                    optimizer.zero_grad()
                    total_loss = _losses[0] + _losses[1] + _losses[2]

                    labels_are_same = (_ys[0] == _ys[1]).all() and (_ys[0] == _ys[2]).all()

                    if reg_loss_func is not None and not labels_are_same:
                        pos = []
                        neg = []
                        for _y_val, _h_val in zip(_ys, _hs):
                            # assuming batch_size == 1; otherwise use .mean()
                            if _y_val.item() == 1.0:
                                pos.append(_h_val)
                            else:
                                neg.append(_h_val)

                        if pos and neg:
                            if len(pos) > len(neg):
                                two, one = pos, neg
                            else:
                                two, one = neg, pos

                            reg_loss = reg_loss_func(
                                two[0].mean(dim=1),
                                positive_key=two[1].mean(dim=1),
                                negative_keys=one[0].mean(dim=1),
                            )
                            total_loss = total_loss + 0.2 * reg_loss

                    total_loss.backward()
                    optimizer.step()

                    train_losses_accum += total_loss.item()
                    num_updates += 1

                    _ys, _hs, _losses = [], [], []

            else:
                # no contrastive, normal per-batch update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses_accum += loss.item()
                num_updates += 1

            # collect predictions for F1 every batch
            pred_list.append(act(outputs.detach().cpu()))
            y_list.append(batched_y.detach().cpu())

            del batched_x

        # Compute Training Metrics
        if num_updates > 0:
            preds_cat = torch.cat(pred_list, dim=0)
            ys_cat = torch.cat(y_list, dim=0)

            avg_train_loss = train_losses_accum / num_updates

            train_f1 = f1_score(
                ys_cat.numpy().ravel(),
                np.around(preds_cat.numpy()).ravel()
            )
        else:
            avg_train_loss = 0.0
            train_f1 = 0.0

        history["train_loss"].append(avg_train_loss)
        history["train_f1"].append(train_f1)

        print(f'    Train Loss: {avg_train_loss:.4f}')
        print(f'    Train F1: {train_f1:.4f}')

        # --- Validation (unchanged) ---
        model.eval()
        print('    Validation:')

        y_pred_all = torch.tensor([], dtype=torch.float32, device=device)
        y_true_all = torch.tensor([], dtype=torch.float32, device=device)
        val_loss_accum = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_batched in valid_loader:
                val_x = val_batched['embs'].to(device)
                val_y = val_batched['label'].to(device).float()
                if val_y.ndim == 1:
                    val_y = val_y.unsqueeze(1)

                val_out = model(val_x)
                y_pred_all = torch.cat([y_pred_all, val_out], dim=0)
                y_true_all = torch.cat([y_true_all, val_y], dim=0)

                loss = loss_func(val_out, val_y)
                val_loss_accum += loss.item()
                val_steps += 1

                del val_x

        avg_val_loss = val_loss_accum / val_steps if val_steps > 0 else 0.0

        y_pred_act = act(y_pred_all).cpu().numpy()
        y_true_cpu = y_true_all.cpu().numpy()

        try:
            if len(np.unique(y_true_cpu)) > 1:
                val_auc = roc_auc_score(y_true_cpu, y_pred_act)
            else:
                val_auc = 0.0

            val_f1 = f1_score(
                y_true_cpu,
                np.around(y_pred_act),
                average='weighted'
            )
        except Exception:
            val_auc = 0.0
            val_f1 = 0.0

        history["valid_loss"].append(avg_val_loss)
        history["valid_auc"].append(val_auc)
        history["valid_f1"].append(val_f1)

        print(f'        Val Loss: {avg_val_loss:.4f}')
        print(f'        Val AUC: {val_auc:.4f}, Best AUC: {best_auc:.4f}')
        print(f'        Val F1: {val_f1:.4f}')

        if val_auc >= best_auc:
            best_auc = val_auc
            save_path = os.path.join(checkpoint_dir, f'{model_prefix}_best_auc.pt')
            torch.save(model.state_dict(), save_path)
            print('    New best metric (AUC) model was saved!')

        if val_f1 >= best_f1:
            best_f1 = val_f1
            save_path = os.path.join(checkpoint_dir, f'{model_prefix}_best_f1.pt')
            torch.save(model.state_dict(), save_path)
            print('    New best f1 model was saved!')

    return history


def plot_training_history(history: Dict[str, List[float]]):
    """
    Plots Train/Val Loss, Val AUC, and Train/Val F1 from the history dictionary.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(18, 5))

    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot 2: AUC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["valid_auc"], label="Val AUC", color='orange')
    plt.title("Validation AUC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)

    # Plot 3: F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_f1"], label="Train F1")
    plt.plot(epochs, history["valid_f1"], label="Val F1")
    plt.title("F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
