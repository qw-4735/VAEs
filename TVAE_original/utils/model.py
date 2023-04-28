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
class TVAE(nn.Module):
    def __init__(self, config, device):  #def __init__(self, input_dim, latent_dim, device):  
        super(TVAE, self).__init__()
        
        
        self.config = config
        self.device = device
        
        
        """encoder""" # encoder는 vanilla vae와 동일
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, config['latent_dim']*2)
        ).to(device)
        
       
        """decoder"""  
        self.decoder = nn.Sequential(
            nn.Linear(config['latent_dim'], 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, config['input_dim'])
        ).to(device)
        self.sigma = nn.Parameter(torch.ones(config['input_dim'])*0.1)    # alpha_bar의 분산 : delta_i 
        
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
        mu = distributions[:, :self.config['latent_dim']]
        logvar = distributions[:, self.config['latent_dim']:]
        
        
        """generating latent"""
        z = self.reparameterization(mu, torch.exp(0.5*logvar))
        
        """decoding"""
        xhat = self._decode(z)
        
        return mu, logvar, z, xhat 
    
    
#%%    

# import easydict

# def main():
#     config = easydict.EasyDict({
#         'input_dim':67,
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
# 
# if __name__ == '__main__':
#     main()  
