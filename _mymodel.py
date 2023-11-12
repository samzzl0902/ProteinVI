
import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from .PROTEINVAE import ProteinVAE
from scvi.distributions import NegativeBinomialMixture




class CustomizedDataset(Dataset):
    def __init__(self,adata):
        self.adata = adata
        self.label = self.adata.obsm['protein_counts'].to_numpy()
    def __getitem__(self,idx):
        cell = torch.tensor(self.adata[idx,:].X.toarray(),dtype = torch.float32)[0]
        return cell,self.label[idx]
    def __len__(self):
        return len(self.label)
    
class testDataset(Dataset):
    def __init__(self,adata):
        self.adata = adata
    def __getitem__(self,idx):
        cell = torch.tensor(self.adata[idx,:].X.toarray(),dtype = torch.float32)[0]
        return cell
    def __len__(self):
        return len(self.label)
    
class VILoss(nn.Module):
    def __init__(self, n_input_proteins=None):
        super(VILoss, self).__init__()
        self.n_input_proteins = n_input_proteins
        self.py_back_alpha_prior = torch.nn.Parameter(
                    torch.randn(self.n_input_proteins))
        self.background_pro_log_beta =torch.nn.Parameter(
                    torch.clamp(torch.randn(self.n_input_proteins), -10, 1))
        self.py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(self.py_back_alpha_prior, self.py_back_beta_prior)

    def forward(self,norm,py_,targets,kl_weight = 1.0):
        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        kl_div_z = kl(norm, Normal(0, 1)).sum(dim=1)
        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        reconst_loss_protein = -py_conditional.log_prob(targets)

        loss = torch.mean(reconst_loss_protein.sum(1)+kl_weight*kl_div_z+kl_weight*kl_div_back_pro_full.sum(1))
        return loss


    
class reconstructionLoss(nn.Module):
    def __init__(self,n_input_proteins):
        super(reconstructionLoss, self).__init__()
        self.n_input_proteins = n_input_proteins
        self.py_back_alpha_prior = torch.nn.Parameter(
                    torch.randn(n_input_proteins))
        self.py_back_beta_prior =torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1))


        self.back_mean_prior = Normal(self.py_back_alpha_prior, self.py_back_beta_prior)

    def forward(self, qz, py_, targets,kl_weight = 1.0):
        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        kl_div_z = kl(qz, Normal(0, 1)).sum(dim=1)
        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )
        reconst_loss_protein = -py_conditional.log_prob(targets)
        
        loss = torch.mean(reconst_loss_protein+kl_weight*kl_div_z+kl_weight*kl_div_back_pro_full)


      
                
        return loss
    


class ProteinVI(nn.Module):
    def __init__(self,adata, n_latent = 10,n_hidden = 128,protein_obsm_key=''):
        super().__init__()
        self.n_input = adata.shape[1]
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.protein_obsm_key = protein_obsm_key
        self.num_proteins = len(adata.obsm[self.protein_obsm_key].columns)
        self.protein_names = adata.obsm[self.protein_obsm_key].columns

        self.adata_train = adata[np.random.rand(len(adata))<0.9,:]
        self.adata_val = adata[np.random.rand(len(adata))>=0.9,:]
        self.model=ProteinVAE(self.adata_train,protein_obsm_key=self.protein_obsm_key)
        self.loss = VILoss(self.num_proteins)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.001) 

        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataloader = DataLoader(CustomizedDataset(self.adata_train),batch_size = 200)
        self.val_dataloader = DataLoader(CustomizedDataset(self.adata_val),batch_size = 200)
        
    def train_step(self):
        self.model.to(self.device)
        self.model.train()

        for i, (cell, label) in enumerate(self.train_dataloader):

            cell = cell.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            norm,py = self.model(cell)
            l = self.loss(norm,py,label)
            l.backward(retain_graph=True)
            self.optimizer.step()
            

            if i % 20 == 0:    # print every 20 iterates
                print(f'iterate {i + 1}: loss={l/label.shape[0]:>7f}')

    def val(self):
        self.model.eval()
        loss_total = 0
        for _, (cell, label) in enumerate(self.val_dataloader):

            cell = cell.to(self.device)
            label = label.to(self.device)
            
            norm,py = self.model(cell)
            loss_total += self.loss(norm,py,label)
            
        print("The validation loss is: %f"%loss_total/len(self.adata_val))
        self.val_loss = loss_total/len(self.adata_val)

    def train(self,epoch):
        min_loss = 1000000
        self.epoch = epoch
        for _ in range(self.epoch):
            self.train_step()
            self.val()
            if self.val_loss<min_loss:
                min_loss = self.val_loss
                pt_path = os.path.join('work_dir', 'best_model')
        torch.save(self.model.state_dict(), pt_path)
        print('save model')

        self.optimizer.step()
    
    def predict(self,adata_test):
        self.adata_test = adata_test
        self.test_dataloader = DataLoader(testDataset(self.adata_test),batch_size = 200)
        self.model.load_state_dict(torch.load(os.path.join('work_dir', 'best_model')))
        self.model.eval()
        for i,cell in self.test_dataloader:
            _,py = self.model(cell)
            py_conditional = NegativeBinomialMixture(
                mu1=py["rate_back"],
                mu2=py["rate_fore"],
                theta1=py["r"],
                mixture_logits=py["mixing"],
            )
            if i==0:
                prediction = py_conditional.sample()
            else:
                prediction = torch.concat([prediction,py_conditional.sample()])


        adata_test.obsm['ProteinVI_Prediction'] = pd.DataFrame(prediction.numpy(),columns = list(self.protein_names),index = list(self.adata_test.index) )









