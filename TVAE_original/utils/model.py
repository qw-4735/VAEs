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

#%%
class TVAE(nn.Module):
    def __init__(self, config, device):  #def __init__(self, input_dim, latent_dim, device):  
        super(TVAE, self).__init__()
        
        #self.input_dim = input_dim
        #self.latent_dim = latent_dim
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
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(config.input_dim, config.compress_dims[0]),
        #     nn.ReLU(True),
        #     nn.Linear(config.compress_dims[0], config.compress_dims[1]),
        #     nn.ReLU(True),
        #     nn.Linear(config.compress_dims[1], config.latent_dim*2)
        # )
        
        
        """decoder"""  
        self.decoder = nn.Sequential(
            nn.Linear(config['latent_dim'], 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, config['input_dim'])
        ).to(device)
        self.sigma = nn.Parameter(torch.ones(config['input_dim'])*0.1)    # alpha_bar의 분산 : delta_i ?
        
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

# distributions = encoder(batch) 
# distributions.shape   
# mu = distributions[:, :config['latent_dim']]
# logvar = distributions[:, config['latent_dim']:]
# mu.shape
# def reparameterization( mu, var):
#         epsilon = torch.randn_like(var).to(device)
#         z = mu + var*epsilon
#         return z
# z = reparameterization(mu, torch.exp(0.5*logvar))
# z.shape
# epsilon = torch.randn_like(torch.exp(0.5*logvar)).to(device)
# epsilon.mean()
# h = encoder(nn.Flatten()(batch)) # [batch, latent_dim * 2]
# mean, logvar = torch.split(h, config["latent_dim"], dim=1)
# mean.shape
# logvar.shape
# noise = torch.randn(batch.size(0), config["latent_dim"]).to(device) 
# latent = mean + torch.exp(logvar / 2) * noise
# latent.shape
# torch.manual_seed(config['seed'])
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
# #%% 
# if __name__ == '__main__':
#     main()  
#%%    
# 안승환 코드 debugging
#
# import easydict
# config = easydict.EasyDict({
#          'input_dim':67,
#          'n':64,
#          'latent_dim':2,
#          'seed':1
#      })
# batch = torch.rand(config['n'], config['input_dim'])
# batch.shape  # [64, 67]

#
#compress_dims=(128, 128) # size of each hidden layer
#latent_dim=128  # size of the output vector
#decompress_dims=(128, 128)
#

# from utils.datasets import generate_dataset

# _, dataloader, transformer,  _,  _, _, _  = generate_dataset(config, device, random_state=0)
# config['input_dim'] = transformer.output_dimensions


# encoder = nn.Sequential(
#             nn.Linear(config["input_dim"], 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, config["latent_dim"] * 2),
#         ).to(device)

# decoder = nn.Sequential(
#             nn.Linear(config["latent_dim"], 16),
#             nn.ReLU(),
#             nn.Linear(16, 64),
#             nn.ReLU(),
#             nn.Linear(64, config["input_dim"]),
#         ).to(device)

# batch = torch.rand(config['n'], config['input_dim'])
# batch.shape  # [64, 68]

# nn.Flatten()(batch).shape
# h.shape
# def get_posterior(batch):
#         h = encoder(nn.Flatten()(batch)) # [batch, latent_dim * 2] : [64, 4]
#         mean, logvar = torch.split(h, config["latent_dim"], dim=1)
#         return mean, logvar
    
# def encode(batch, deterministic=False):
#     mean, logvar = get_posterior(batch)
    
#     """Latent Generating Process"""
#     if deterministic:
#         latent = mean
#     else:
#         noise = torch.randn(batch.size(0), config["latent_dim"]).to(device)  # [64, 2] 
#         latent = mean + torch.exp(logvar / 2) * noise  #[64, 2]
    
#     return mean, logvar, latent  

# def forward(self, input, deterministic=False):
#         """encoding"""
#         mean, logvar, latent = encode(batch, deterministic=False)
        
#         """decoding"""
#         xhat = decoder(latent)
#         xhat.shape
#         return mean, logvar, latent, xhat  
#%%

# nn.ReLU()  vs. nn.ReLU(True)  debugging
# import easydict
# config = easydict.EasyDict({
#          'input_dim':67,
#          'n':2,
#          'latent_dim':2,
#          'seed':1
#      })
# batch = torch.rand(config['n'], config['input_dim'])   # 0~1 사이의 숫자를 균등하게 생성
# batch.shape  # [64, 67]
# batch
# #%%
# """encoder"""
# l1 = nn.Linear(config['input_dim'], 16)
# out = l1(batch)   # (2, 16)  # out값이 변하지 않음
# out.shape
# acti1 = nn.ReLU()
# out2 = acti1(out)
# out2.shape  # (2, 16)

# l1_1 = nn.Linear(config['input_dim'], 16)
# out_1 =l1_1(batch)   # out_1 값이 달라짐!!!  out_1 = out2_1
# acti1_1 = nn.ReLU(True)
# out2_1 = acti1_1(out_1)
# out2_1.shape

# l2 = nn.Linear(16,8)
# out3 =l2(out2)
# out3.shape  # [2, 8]
# acti2 = nn.ReLU()
# out4 = acti2(out3)

# l2_1 = nn.Linear(16,8)
# out3_1 =l2(out2_1)
# acti2_1 = nn.ReLU(True)
# out4_1 = acti2_1(out3_1)

# l3 = nn.Linear(8, config['latent_dim']*2)
# out5 = l3(out4)
# out5_1 = l3(out4_1)

# distributions = l3(out4)   # out5
# dist2 = l3(out4_1)
# mu = distributions[:, :config['latent_dim']]
# logvar = distributions[: ,config['latent_dim']:]



# "latent sampling"
# var = torch.exp(0.5*logvar)
# epsilon = torch.randn_like(var).to(device)
# z = mu + var*epsilon

# """decoder"""
# l4 = nn.Linear(config['latent_dim'], 16)
# out6 = l4(z)
# acti6 = nn.ReLU()
# out7 = acti6(out6)

# out6_1 = l4(z)
# acti6_1 =nn.ReLU(True)
# out7_1 = acti6_1(out6_1)

# acti6_1 = nn.ReLU(True)
# out7_1 = acti6_1(out6)

# l5 = nn.Linear(16, 64)
# out8 = l5(out7)
# acti7 = nn.ReLU(True)
# out9 = acti7(out8)

# l6 = nn.Linear(64, config['input_dim'])
# xhat =l6(out9)
# xhat.shape

# sigma = nn.Parameter(torch.ones(config['input_dim'])*0.1)
# #
# (0.1*0.9)*0.09*0.09*0.09
