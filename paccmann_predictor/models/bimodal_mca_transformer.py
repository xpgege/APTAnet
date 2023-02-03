from collections import OrderedDict
import torch
import torch.nn as nn
from ..utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from ..utils.layers import (
    dense_layer,
)
import math
from ..utils.utils import get_device



class BimodalMCA(nn.Module):
    """Bimodal Multiscale Convolutional Attentive Encoder.

    This is based on the MCA model as presented in the publication in
    Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
    """

    def __init__(self, params, *args, **kwargs):
        super(BimodalMCA, self).__init__(*args, **kwargs)
        # Model Parameter
        self.device = get_device()
        self.params = params
        self.ligand_padding_length = params['ligand_padding_length']
        self.receptor_padding_length = params['receptor_padding_length']
        self.ligand_hidden_sizes=params['ligand_hidden_sizes']
        self.receptor_hidden_sizes = params['receptor_hidden_sizes']
        self.ligand_attention_size = params.get('ligand_attention_size')
        self.receptor_attention_size = params.get('receptor_attention_size')
        self.attention_head_num=params.get('attention_head_num')
        self.hidden_sizes=[self.ligand_hidden_sizes+self.receptor_hidden_sizes ]+params.get('dense_hidden_sizes')
        self.loss_fn = LOSS_FN_FACTORY[
            params.get('loss_fn', 'binary_cross_entropy')
        ]  # yapf: disable
        # Hyperparameter
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')
        ]  # yapf: disable
        self.dense_dropout = params.get('dense_dropout', 0.5)
        self.attention_dropout = params.get('attention_dropout', 0.5)
        self.use_batch_norm=True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])


        """ Construct model  """
        # Context attention
        self.context_attention_ligand_layers = nn.MultiheadAttention(self.ligand_hidden_sizes,
                                                                     self.attention_head_num,
                                                                     batch_first=True,
                                                                     kdim=self.receptor_hidden_sizes,
                                                                     vdim=self.receptor_hidden_sizes,
                                                                     dropout=self.attention_dropout)
        self.context_attention_receptor_layers = nn.MultiheadAttention(self.receptor_hidden_sizes,
                                                                       self.attention_head_num,
                                                                       batch_first=True,
                                                                       kdim=self.ligand_hidden_sizes,
                                                                       vdim=self.ligand_hidden_sizes,
                                                                       dropout=self.attention_dropout)

        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'dense_{ind}',
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dense_dropout,
                            batch_norm=self.use_batch_norm,
                        ),
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = nn.Linear(self.hidden_sizes[-1], 1)
        if params.get('final_activation', True):
            self.final_dense = nn.Sequential(
                self.final_dense, ACTIVATION_FN_FACTORY['sigmoid']
            )

    def forward(self, ligand, receptors):
        """Forward pass through the biomodal MCA.

        Args:
            ligand (torch.Tensor): of type int and shape
                `[bs, ligand_padding_length]`.
            receptors (torch.Tensor): of type int and shape
                `[bs, receptor_padding_length]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """
        # Embedding
        embedded_ligand = ligand.to(torch.float)
        embedded_receptor = receptors.to(torch.float)

        # Context attention on ligand
        ligand_encodings, ligand_alphas = self.context_attention_ligand_layers(embedded_ligand,embedded_receptor,embedded_receptor)
        # Context attention on receptor
        receptor_encodings, receptor_alphas =self.context_attention_receptor_layers(embedded_receptor,embedded_ligand,embedded_ligand)

        ligand_encodings=torch.mean(ligand_encodings,dim=1)
        receptor_encodings=torch.mean(receptor_encodings,dim=1)
        # Concatenate all encodings
        encodings = torch.cat([ligand_encodings,receptor_encodings],dim=1)
        # Apply batch normalization if specified
        out = self.batch_norm(encodings) if self.use_batch_norm else encodings

        # Stack dense layers
        for dl in self.dense_layers:
            out = dl(out)
        predictions = self.final_dense(out)

        prediction_dict = [ligand_alphas,receptor_alphas]

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
