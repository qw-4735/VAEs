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

#for debugging

# import easydict

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config = easydict.EasyDict({
#     'input_dim':68,
#     'batch_size':256,
#     'latent_dim':2,
#     'lr' : 0.005,
#     'weight_decay': 1e-5,
#     'seed':1
# })
# _, dataloader, transformer,  _,  _, _, _  = generate_dataset(config, device, random_state=0)
# config['input_dim'] = transformer.output_dimensions

# model = TVAE(config, device).to(device)
# optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr = config['lr'],
#         weight_decay = config['weight_decay']
#     )
#

#%%
def train(output_info_list, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [],
        'recon':[],
        'KLD':[]
    }
    
    # x_batch= next(iter(dataloader)) # shape: 256 rows x 68 columns
    # x_batch = x_batch[0]
    # x_batch.shape
    # len(x_batch[0])
    # len(x_batch[0][0])
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc='inner loop'):
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        
        mu, logvar, z, xhat = model(x_batch)
        
        loss_ = []
        # xhat[:, 1:10].shape
        # x_batch[:, 1:10].shape
        # F.softmax(x_batch[:, 1:10], dim=-1)[2]
        # F.cross_entropy(xhat[:, 1:10], torch.argmax(x_batch[:, 1:10], dim =-1), reduction='mean')
        # torch.argmax(x_batch[:,1:10], dim=-1)[0:9]
        # transformer.output_info_list
        """reconstruction"""
        start = 0
        recon = 0
        # output_info_list =  [[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')], [SpanInfo(num_categories, 'softmax)]
        # transformer.output_info_list[0]  # column_info
        # transformer.output_info_list[0][0]  # span_info
        
        for column_info in output_info_list: 
            for span_info in column_info:
                if span_info.activation_fn != 'softmax':  # tanh
                    end = start + span_info.dim  # end = 0 + 1 (alpha의 차원 : 1차원 scalar)
                    std = model.sigma[start]  # root_delta_{i}  # 0.1
                    residual = x_batch[:, start] - torch.tanh(xhat[:, start]) # (alpha_{i,j} - alpha_bar{i,j})
                    recon += (residual ** 2 / 2 / (std ** 2)).mean()  # alpha_hat에 대한 loss(recon_error)
                    recon += torch.log(std) # alpha_hat에 대한 loss(recon_error)
                    start  = end
                else: 
                    end =  start + span_info.dim # span_info.dim = num_componets or num_categories
                    # input(predicted prob) : xhat[:, start:end] , target(ground truth class indices) : torch.argmax(x_batch[:, start:end], dim =-1)  (여기서는 ground truth class probs가 아니라 index를 뽑으려고 argmax를 사용)
                    recon += cross_entropy(xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim =-1), reduction='mean')   # cross_entropy 함수에 예측값(input)에 softmax를 취하는 과정이 포함됨
                    start = end
        loss_.append(('recon', recon))
        
        # start = 1
        # end = 9
        # cross_entropy(xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim =-1), reduction='mean')
        
        # z2 = xhat[:,start:end]
        # z2.shape
        # y2 = torch.argmax(x_batch[:, start:end], dim =-1)
        # F.nll_loss(F.log_softmax(z2, dim=1),y2)
        # F.nll_loss(torch.log(F.softmax(z2, dim=1)), y2)
        """KL-Divergence"""      
        #-config['latent_dim'] - logvar.sum(axis=1) + torch.pow(mu,2).sum(axis=1) + torch.exp(logvar).sum(axis=1)                                                                                                                                  
        
        # torch.pow(mu,2).shape  # [256, 2]  (256 : batch_size)
        KLD = torch.pow(mu,2).sum(axis=1)  # 256개의 mu1+ mu2 계산   #[256]
        KLD += torch.exp(logvar).sum(axis=1)  # 256개의 sigma1 + sigma2 계산  # [256]
        KLD -= config['latent_dim']  # k = 2
        KLD -= logvar.sum(axis=1)   # 256개의 log(sigma1) + log(sigma2) 계산   # [256]
        KLD *= 0.5
        KLD = KLD.mean()  # 256개에 대하여 평균냄
        #KLD =  0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())  
        #KLD = KLD / x_batch.size()[0]
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

# logs = train(transformer1.output_info_list, dataloader1, model, optimizer, device)

# _, _, dataloader0, dataloader1, transformer0, transformer1, _, _,_, _, _ = generate_dataset(args, device, random_state=1)
# args.input_dim = transformer0.output_dimensions
# x_batch1= next(iter(dataloader1)) # shape: 64 rows x 67 columns
#     len(x_batch1[0])
# x_batch0= next(iter(dataloader0)) # shape: 64 rows x 67 columns
# len(x_batch0[0])
#%%
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss1 = cross_entropy(input, target)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# torch.manual_seed(1)

# # softmax
# z = torch.FloatTensor([1,2,3])
# h = F.softmax(z, dim=0)  # dim=0 : row들끼리의 합
# # dim을 따르는 모든 슬라이스의 합은 1이 됨
# # dim = 0 : 첫번째 차원 (e.g 행렬에서 첫번째 차원은 '행')을 제거한다. 
# torch.exp(torch.FloatTensor([3]))/(torch.exp(torch.FloatTensor([1]))+torch.exp(torch.FloatTensor([2]))+torch.exp(torch.FloatTensor([3])))
# h[0]
# # cross entropy loss
# z = torch.rand(3, 5, requires_grad=True)
# h2 = F.softmax(z, dim=1)
# y= torch.randint(5, (3,))
# y.size()
# y_one_hot = torch.zeros_like(h2)
# y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# cost = (y_one_hot * -torch.log(h2)).sum(dim=1).mean()

# # cross entropy loss with torch.nn.functional
# # low level
# torch.log(F.softmax(z, dim=1))  # torch.log(h2)
# # hig level
# F.log_softmax(z, dim=1)  # torch에는 log_softmax함수가 존재

# # low level
# cost_low_level = (y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
# # high level 
# cost_high_level = F.nll_loss(F.log_softmax(z, dim=1), y) # NLL : Negative Log Likelihood

# # F.cross_entropy  = F.log_softmax() + F.nll_loss()
# F.cross_entropy(z,y, dim=-1)

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss = F.cross_entropy(input, target)
# loss.backward()

# input2 = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# loss = F.cross_entropy(input2, target)

