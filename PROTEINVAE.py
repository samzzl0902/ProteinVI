
import torch
import numpy as np
import torch.nn as nn
from .basemodel import Encoder,Decoder

class ProteinVAE(nn.Module):
    def __init__(self,adata,n_latent = 10,n_hidden = 128,protein_obsm_key=''):
        super().__init__()
        self.n_input = adata.shape[1]
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.protein_obsm_key = protein_obsm_key
        self.num_proteins = len(adata.obsm[self.protein_obsm_key].columns)
        self.protein_names = adata.obsm[self.protein_obsm_key].columns
        self.py_r = torch.nn.Parameter(2 * torch.rand(self.num_proteins))

        self.encoder = Encoder(n_input = self.n_input,n_output = self.n_latent,n_hidden=self.n_hidden)
        self.decoder = Decoder(n_input = self.n_latent,n_hidden=self.n_hidden,n_output_proteins=self.num_proteins)
        self.adata_train = adata[np.random.rand(len(adata))<0.9,:]
        self.adata_test = adata[np.random.rand(len(adata))>=0.9,:]




    def forward(self, x):
        _,_,latent,norm = self.encoder(x)
        py_,_ = self.decoder(latent)
        py_['r'] = self.py_r
        return norm,py_
