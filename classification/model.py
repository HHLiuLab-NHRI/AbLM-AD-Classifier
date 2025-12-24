from typing import Optional

import torch
from torch import nn
import monai.networks.nets

class CNN1DClassifier(nn.Module):
    """
    1D CNN classifier for antibody sequence embeddings, aligned with
    CNN1D_averaged from CNN1D_IgG-RAA-2025.ipynb.

    It uses a 1D Convolutional block followed by a MONAI DenseNet264.
    It stores the intermediate hidden state for contrastive learning.
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Aligned with Notebook Cell 10 'cnn1'
        # Note: LeakyReLUs were commented out in the source notebook, 
        # so this implies a linear sequence of convolutions before the DenseNet.
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=111, stride=16),
            # nn.LeakyReLU(0.05), # Commented out in source
            nn.Conv1d(512, 256, kernel_size=3, stride=1),
            # nn.LeakyReLU(0.05), # Commented out in source
            nn.Conv1d(256, 128, kernel_size=19, stride=9),
            # nn.LeakyReLU(0.05), # Commented out in source
            nn.Dropout(dropout)
        )

        # Aligned with Notebook Cell 10 'cnn2'
        self.cnn2 = monai.networks.nets.DenseNet264(
            spatial_dims=1,
            in_channels=128,
            out_channels=num_classes
        )
        
        # Placeholder to store intermediate features for Contrastive Loss
        self.hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, C, L)``.

        Returns
        -------
        logits:
            Tensor of shape ``(B,)`` (if squeezed) or ``(B, 1)``.
        """
        # Store intermediate features for InfoNCE loss
        self.hidden = self.cnn1(x)
        
        # Pass through DenseNet
        x = self.cnn2(self.hidden)
        
        # Squeeze the last dimension for binary classification to match typical shape (B,) or (B, 1)
        # The notebook DenseNet output implies (B, 1) usually, typically we want (B, 1) or (B)
        return x
