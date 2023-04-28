import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
#
input_dim = 67
latent_dim = 128
#compress_dims=(128, 128) # size of each hidden layer
#latent_dim=128  # size of the output vector
#decompress_dims=(128, 128)


class TVAE(nn.Module):
    def __init__(self, args, device):  #def __init__(self, input_dim, latent_dim, device):  
        super(TVAE, self).__init__()
        
        #self.input_dim = input_dim
        #self.latent_dim = latent_dim
        self.args = args
        self.device = device
        
        """encoder""" # encoder는 vanilla vae와 동일
        self.encoder = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, args.latent_dim)
        )
        self.fc1 = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(128, latent_dim)
        
        

        """decoder"""  
        self.decoder = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 128)
        )
        #self.sigma = nn.Parameter(torch.ones(args.input_dim)*0.1)    # alpha_bar의 분산 : delta_i ?
        self.fc1_d = nn.Linear(128, args.input_dim)
        self.fc2_d = nn.Linear(128, args.input_dim)
        self.fc3_d = nn.Linear(128, args.input_dim) 
       
        
    def _encode(self, x):
        return self.encoder(x)
    
    def _decode(self, z):
        return self.decoder(z)
    
    def reparameterization(self, mu, var):
        epsilon = torch.randn_like(var).to(device)
        z = mu + var*epsilon
        return z
    
    
    def forward(self, x):
        
        """encoding"""
        feature = self._encode(x)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        
        """generating latent"""
        z = self.reparameterization(mu, torch.exp(0.5*logvar))
        
        """decoding"""
        feature2 = self._decode(z)
        xhat = self.fc1_d(feature2)
        sigma = self.fc2_d(feature2)
        degree_f = self.fc3_d(feature2)
        
        return mu, logvar, z, xhat,sigma, degree_f 
    