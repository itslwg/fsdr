"""
Heavily inspired by:
    https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#scrollTo=O9pIrJ674ki0
"""

import torch
import torch.nn as nn
import numpy as np

from skorch import NeuralNet
from skorch import NeuralNetRegressor


class Encoder(nn.Module):
    """Encoder component of autoencoder."""
    def __init__(self, layers, D, act_fn):
        super().__init__()
        self.layers = layers
        self.D = D
        
        seqs = [nn.Sequential(
            nn.Linear(self.D, self.layers[0]),
            act_fn
        )]
        for i in range(len(self.layers)-1):
            nodes = self.layers[i]
            nnodes = self.layers[i + 1]
            seqs.append(nn.Sequential(
                nn.Linear(nodes, nnodes),
                act_fn
            ))

        self.encode = nn.Sequential(*seqs)
        
    def forward(self, X, sparse):
        encoded = self.encode(X)
        return encoded
    

class Decoder(nn.Module):
    """Decoder component of autoencoder."""
    def __init__(self, layers, D, act_fn):
        super().__init__()
        # Reverse input layers for the decoder
        self.D = D
        self.layers = layers
        
        seqs = [nn.Sequential(
            nn.Linear(self.layers[0], self.D)
        )]
        for i in range(len(self.layers)-1):
            nodes = self.layers[i]
            nnodes = self.layers[i + 1]
            seqs.insert(0, nn.Sequential(
                nn.Linear(nnodes, nodes),
                act_fn
            ))
        self.decode = nn.Sequential(*seqs)
        
    def forward(self, X, sparse):
        decoded = self.decode(X)
        return decoded

    
class AutoEncoder(nn.Module):
    """Autoencoder, bundling the encoder and decoder."""
    def __init__(self, layers: list = [100, 40, 10], 
                 D = None, act_fn = nn.ReLU(), 
                 sparse: bool = False):
        super().__init__()
        self.D = D
        self.layers = layers

        self.encoder = Encoder(
            layers=self.layers, 
            D=self.D,
            act_fn=act_fn
        )
        self.decoder = Decoder(
            layers=self.layers,
            D=self.D,
            act_fn=act_fn 
        )
        
    def forward(self, X, sparse):
        encoded = self.encoder(X, sparse=sparse)
        decoded = self.decoder(encoded, sparse=sparse)
        return decoded, encoded
    

class AE(NeuralNetRegressor):
    """Wrapper to enable sparsity option for loss."""
    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        decoded, encoded = y_pred
        sparse = np.unique(X['sparse'].numpy())[0]
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + l1 if sparse else loss_reconstruction