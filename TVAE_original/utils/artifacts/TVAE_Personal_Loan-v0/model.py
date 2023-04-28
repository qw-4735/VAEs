#%%
"""
Reference:
[1]: https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/tvae.py
[2]: https://github.com/an-seunghwan/synthesizers/blob/main/tvae/modules/model.py 

tvae.py
"""

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
#
#compress_dims=(128, 128) # size of each hidden layer
#latent_dim=128  # size of the output vector
#decompress_dims=(128, 128)
#
class TVAE(nn.Module):
    def __init__(self, config, device):  #def __init__(self, input_dim, latent_dim, device):  
        super(TVAE, self).__init__()
        
        #self.input_dim = input_dim
        #self.latent_dim = latent_dim
        self.config = config
        self.device = device
        
        
        """encoder""" # encoder는 vanilla vae와 동일
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, config.latent_dim*2)
        )
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(config.input_dim, config.compress_dims[0]),
        #     nn.ReLU(True),
        #     nn.Linear(config.compress_dims[0], config.compress_dims[1]),
        #     nn.ReLU(True),
        #     nn.Linear(config.compress_dims[1], config.latent_dim*2)
        # )
        
        

        """decoder"""  
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, config.input_dim)
        )
        self.sigma = nn.Parameter(torch.ones(config.input_dim)*0.1)    # alpha_bar의 분산 : delta_i ?
        
        # self.sigma = nn.Parameter(torch.ones(input_dim)*0.1) 
        
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
        distributions = self._encode(x)
        mu = distributions[:, :self.config.latent_dim]
        logvar = distributions[:, self.config.latent_dim:]
        #mu = distributions[:, :latent_dim]
        #logvar = distributions[:, latent_dim:]
        
        """generating latent"""
        z = self.reparameterization(mu, torch.exp(0.5*logvar))
        
        """decoding"""
        xhat = self._decode(z)
        
        return mu, logvar, z, xhat 
    
#%%    
# import easydict

# def main():
#     config = easydict.EasyDict({
#         'input_dim':37,
#         'n':64,
#         'latent_dim':128
#     })
#     """TVAE"""
#     model = TVAE(config, 'cpu')
#     for x in model.parameters():
#         print(x.shape)
#     batch = torch.rand(config.n, config.input_dim) 
#     batch.shape
    
#     mean, logvar, latent, xhat = model(batch)
    
#     assert mean.shape == (config.n, config.latent_dim)
#     assert logvar.shape == (config.n, config.latent_dim)
#     assert latent.shape == (config.n, config.latent_dim)
#     assert xhat.shape == (config.n, config.input_dim)
    
#     print("TVAE pass test!")   
#%% 
if __name__ == '__main__':
    main()  
#%%    
# config.input_dim = transformer0.output_dimensions
# encoder = nn.Sequential(
#             nn.Linear(config.input_dim, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 128),
#             nn.ReLU(True),
#             nn.Linear(128, config.latent_dim*2)
#             )
# distribution = encoder(batch)
# mu = distribution[:, :config.latent_dim]
# logvar = distribution[:, config.latent_dim:]
# z = reparameterization(mu, torch.exp(0.5*logvar))
# xhat = decoder(z)
# mu.shape
# logvar.shape
# z.shape
# xhat.shape
        
# def reparameterization(mu, var):
#     epsilon = torch.randn_like(var).to(device)
#     z = mu + var*epsilon
#     return z

# decoder = nn.Sequential(
#         nn.Linear(config.latent_dim, 128),
#         nn.ReLU(True),
#         nn.Linear(128, 128),
#         nn.ReLU(True),
#         nn.Linear(128, config.input_dim)
#     )

# model = TVAE(config, device).to(device)
# mu, logvar, z, xhat = model(x_batch[0])

        
# %%
