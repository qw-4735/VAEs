#%%
"""
Reference:
[1]: https://github.com/1Konny/Beta-VAE/blob/master/model.py

model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


#

#%%
def reparametrize(mu, logvar):  
    std = logvar.div(2).exp()   # div() : tensor 요소 간의 나누기  / exp(logvar/2)  = torch.exp(0.5 * log_var)
    eps = std.data.new(std.size()).normal_().requires_grad_()  # Variable(std.data.new(std.size()).normal_())  대체
    return mu + std*eps
# new().normal_() : Tensor/변수와 동일한 size 및 데이터 유형인 가우시안 노이즈 텐서가 생성
# Tensor.normal_(mean=0, std=1) -> Tensor :Fills self tensor with elements samples from the normal distribution parameterized by mean and std.    
#%%
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
#%%
class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""
    
    def __init__(self, z_dim=10, nc=3, beta=4):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        # self.beta = beta
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0) : in_channels(입력채널 수, 흑백이미지일 경우 1, RGB값을 가진 이미지일 경우 3) 
        # output : (B = batch size, out_channel=32, Height, Weight)  # H,W = (64-4+2*1)/2 + 1 = 32
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),    # B, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),    # B, 32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),   # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),    # B, 64, 8, 8
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),      # B, 64, 4, 4
            nn.ReLU(True),
            View((-1, 256*1*1)),           # B, 256   # (batch size, 256*1*1)  #view(-1, 1) : 텐서의 차원을 변경  -> (?, 1)
            nn.Linear(256, z_dim*2)               # B, z_dim*2
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )
    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())
        return x_recon, mu,logvar
    
    def _encode(self, x):
        return self.encoder(x)
        
    def _decode(self, z):
        return self.decoder(z)
   
    
if __name__ =='__main__':
    pass

#%%
# nn.Conv2d() 입력 파라미터의 변화?..... out_channels은 내 맘대로 결정
# 이 github 코드는 너무 high level 이다...?
# # 현재 수준에서는 pytorch tutorial vae base 참조해 좀 더 low level인 구현이 필요함!