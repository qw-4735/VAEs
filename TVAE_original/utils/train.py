"""
Reference:
[1]: https://github.com/an-seunghwan/synthesizers/blob/main/tvae/modules/train.py

tvae.py
"""

# import warnings
# warnings.filterwarnings('ignore')

# import os
# os.getcwd()
# os.chdir('D:\VAE\TVAE')
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.functional import cross_entropy

from utils.datasets import generate_dataset
from utils.model import TVAE



#%%
def train(output_info_list, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [],
        'recon':[],
        'KLD':[]
    }
    
   
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc='inner loop'):
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        
        mu, logvar, z, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        start = 0
        recon = 0
        
        for column_info in output_info_list: 
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':  # tanh
                    end = start + span_info.dim  
                    std = model.sigma[start]  # root_delta_{i}  # 0.1
                    residual = x_batch[:, start] - torch.tanh(xhat[:, start])
                    recon += (residual ** 2 / 2 / (std ** 2)).mean()  
                    recon += torch.log(std) 
                    start  = end
                else: 
                    end =  start + span_info.dim 
                    recon += cross_entropy(xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim =-1), reduction='mean')  
                    start = end
        loss_.append(('recon', recon))
        
        
        KLD = torch.pow(mu,2).sum(axis=1)  # 256개의 mu1+ mu2 계산   #[256]
        KLD += torch.exp(logvar).sum(axis=1)  # 256개의 sigma1 + sigma2 계산  # [256]
        KLD -= config['latent_dim']  # k = 2
        KLD -= logvar.sum(axis=1)   # 256개의 log(sigma1) + log(sigma2) 계산   # [256]
        KLD *= 0.5
        KLD = KLD.mean()  # 256개에 대하여 평균냄
       
        loss_.append(('KLD', KLD))
        
        """loss"""
        loss = recon + KLD
        loss_.append(('loss',loss))     
        
        loss.backward()
        optimizer.step()
        torch.clamp(model.sigma.data, min=config['sigma_range'][0], max=config['sigma_range'][1])
        #torch.clamp(model.sigma.data, min=0.01, max=1.0) # sigma의 범위를 [min, max]로 고정
        # model.sigma.data.clamp_(0.01, 1.0) 
        
        """accumulate losses"""
        for x, y in loss_:  # loss_ = [('loss', loss), ('recon', recon), ('KLD', KLD)]
            logs[x] = logs.get(x) + [y.item()]  # logs['loss'] = logs.get('loss') + [loss.item()]
            
    return logs

