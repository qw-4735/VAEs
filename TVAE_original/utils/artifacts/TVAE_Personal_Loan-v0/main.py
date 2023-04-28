"""
Reference:
[1]: https://github.com/an-seunghwan/synthesizers/blob/main/tvae/modules/train.py

tvae.py
"""

import os
os.getcwd()
os.chdir('D:\VAE\TVAE')

#import sys
#sys.path.append('/utils')

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy
from utils.simulation import set_random_seed
from utils.datasets import generate_dataset
from utils.model import TVAE
from utils.train import train

#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')
    
    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--latent_dim', default=128, type=int, help='the dimension of the latent variable z')
    parser.add_argument('--input_dim', default=37, type=int, help='the dimension of the input variable x')
    #parser.add_argument('--compress_dims', default=(128,128), type=int, help='size of each hidden layer')
    #parser.add_argument('--decompress_dims', default=(128,128), type=int, help='size of each hidden layer')
    
    parser.add_argument('--num_epochs', type = int, default=200, help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay parameter')
    
    
    if debug:
        return parser.parse_known_args(args=[])
    else:
        return parser.parse_known_args()

#args,_= get_args(debug=True)
#args.lr

#%%
#import wandb

#wandb.init(project='TVAE', entity='qw4735') #wandb web 서버와 연결시켜주는 기능


    
#%%
#train
def main():
    args,_= get_args(debug=False)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #wandb.config.update(args)
    
    
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    """dataset"""
    _, dataloader, transformer,  _,  _, _, _  = generate_dataset(args, device, random_state=0)
    args.input_dim = transformer.output_dimensions
        
    
    """model"""
    model = TVAE(args, device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay = args.weight_decay
    )
    model.train()
    
    """train"""
    for epoch in range(args.num_epochs):
        logs = train(transformer.output_info_list, dataloader, model, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([',{}:{:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        #wandb.log({x : np.mean(y) for x, y in  logs.items()})  # metric을 추적
        
    """model save"""
    torch.save(model.state_dict(), './assets/TVAE_{}.pth'.format('Personal_Loan_train_200epoch'))    
    #wandb.run.finish()

    
#%%
if __name__ == '__main__':
     main()
     
#%%
#train1

# def main():
#     args,_= get_args(debug=True)
#     args.cuda = torch.cuda.is_available()
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     #wandb.config.update(args)
    
    
#     set_random_seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
    
#     """dataset"""
#     _, _, _, dataloader1, _, transformer1, _,_, _,_, _, _ = generate_dataset(args, device, random_state=1)
#     args.input_dim = transformer1.output_dimensions
        
    
#     """model"""
#     model = TVAE(args, device).to(device)
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay = args.weight_decay
#     )
#     model.train()
    
#     """train"""
#     for epoch in range(args.num_epochs):
#         logs = train(transformer1.output_info_list, dataloader1, model, optimizer, device)
        
#         print_input = "[epoch {:03d}]".format(epoch + 1)
#         print_input += ''.join([',{}:{:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
#         print(print_input)
        
#         """update log"""
#         #wandb.log({x : np.mean(y) for x, y in  logs.items()})  # metric을 추적
        
#     """model save"""
#     torch.save(model.state_dict(), './assets/TVAE_{}.pth'.format('Personal_Loan_train1_100epoch'))    
#     #wandb.run.finish()

    
# #%%
# if __name__ == '__main__':
#      main()
     
# #%%
# # train0

# def main():
#     args,_= get_args(debug=True)
#     args.cuda = torch.cuda.is_available()
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     #wandb.config.update(args)
    
    
#     set_random_seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
    
#     """dataset"""
#     _, _, dataloader0, _, transformer0, _, _, _, _,_, _, _ = generate_dataset(args, device, random_state=1)
#     args.input_dim = transformer0.output_dimensions
        
    
#     """model"""
#     model = TVAE(args, device).to(device)
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay = args.weight_decay
#     )
#     model.train()
    
#     """train"""
#     for epoch in range(args.num_epochs):
#         logs = train(transformer0.output_info_list, dataloader0, model, optimizer, device)
        
#         print_input = "[epoch {:03d}]".format(epoch + 1)
#         print_input += ''.join([',{}:{:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
#         print(print_input)
        
#         """update log"""
#         #wandb.log({x : np.mean(y) for x, y in  logs.items()})  # metric을 추적
        
#     """model save"""
#     torch.save(model.state_dict(), './assets/TVAE_{}.pth'.format('Personal_Loan_train0_100epoch'))    
#     #wandb.run.finish()

    
# #%%
# if __name__ == '__main__':
#      main()     
     
#%%         
# _, _, dataloader1, _, transformer1, _, _, _,_, _, _ = generate_dataset(args, device, random_state=0)
# args.input_dim = transformer1.output_dimensions
    
# SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
# ColumnTransformInfo = namedtuple(
#     'ColumnTransformInfo', [
#         'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
#     ]
# )

# def train(output_info_list, dataloader, model, optimizer, device):
#     logs = {
#         'loss': [],
#         'recon':[],
#         'KLD':[]
#     }
    
#     x_batch= next(iter(dataloader)) # shape: 64 rows x 67 columns
#     len(x_batch[0])
#     len(x_batch[0][0])
    
#     x_batch= next(iter(dataloader1)) # shape: 64 rows x 63 columns
#     len(x_batch[0])
#     len(x_batch[0][0])
#     for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc='inner loop'):
#         x_batch = x_batch.to(device)
#         optimizer.zero_grad()
        
#         mu, logvar, z, xhat = model(x_batch[0])
        
#         loss_ = []
        
#         """reconstruction"""
#         start = 0
#         recon = 0
#         output_info_list =  [[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')], [SpanInfo(num_categories, 'softmax')]]
#         for column_info in output_info_list: 
#             for span_info in column_info:
#                 if span_info.activation_fn != 'softmax':  # tanh
#                     end = start + span_info.dim  # end = 0 + 1 (alpha의 차원 : 1차원 scalar)
#                     std = model.sigma[start]  # root_delta_{i}
#                     residual = x_batch[:, start] - torch.tanh(xhat[:, start]) # (alpha_{i,j} - alpha_bar{i,j})
#                     recon += (residual ** 2 / 2 / (std ** 2)).mean()  # alpha_hat에 대한 loss(recon_error)
#                     recon += torch.log(std) # alpha_hat에 대한 loss(recon_error)
#                     start  = end
#                 else: 
#                     end =  start + span_info.dim # span_info.dim = num_componets or num_categories
#                     recon += cross_entropy(xhat[:, start:end], torch.argmax(x_batch[:, start:end], dim =-1), reduction='mean')
#                     start = end
#         loss_.append(('recon', recon))
        
#         """KL-Divergence"""
#         KLD =  -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())  
#         loss_.append(('KLD', KLD))
        
#         """loss"""
#         loss = recon + KLD
#         loss_.append(('loss',loss))     
        
#         loss.backward()
#         optimizer.step()
#         torch.clamp(model.sigma.data, min=0.01, max=1.0) # sigma의 범위를 [min, max]로 고정
#         # model.sigma.data.clamp_(0.01, 1.0) 
        
#         """accumulate losses"""
#         for x, y in loss_:  # loss_ = [('loss', loss), ('recon', recon), ('KLD', KLD)]
#             logs[x] = logs.get(x) + [y.item()]  # logs['loss'] = logs.get('loss') + [loss.item()]
            
#     return logs

# #logs = train(transformer.output_info_list, dataloader, model, optimizer, device)


# #%%
# import tqdm
# import os
# import numpy as np
# import pandas as pd

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data import Dataset

# from sklearn.mixture import BayesianGaussianMixture
# from utils.data_transformer import DataTransformer

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu') 


# def generate_dataset1(args, device, random_state=0):
    
#     df = pd.read_csv('D:\VAE\TVAE\Bank_Personal_Loan_Modelling.csv')
#     df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # 데이터프레임으로부터 random sampling  # frac : 표본추출비율(0~1)
#     df = df.drop(columns=['ID'])
#     df.info()
#     continuous = [
#         'CCAvg', 
#         'Mortgage', 
#         'Income',
#         'Experience',
#         'Age'
#         ] 
#     discrete = [
#         'Family', 
#         'Education', 
#         'Personal Loan',
#         'Securities Account',
#         'CD Account',
#         'Online',
#         'CreditCard'
#         ]

#     df = df[continuous + discrete]
#     # df = df.dropna(axis=0)

#     train= df.iloc[:4000]
#     test = df.iloc[4000:]
    
#     #train0 = train[train['Personal Loan'] == 0]  # (3622, 12)
#     #train1 = train[train['Personal Loan'] == 1]  # (378, 12)
    
    
#     transformer = DataTransformer()
#     transformer.fit(train, discrete_columns=discrete, random_state = random_state)
#     train_data = transformer.transform(train)
    
    
#     dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(device))
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    
#     return dataset, dataloader,  transformer, train, test, continuous, discrete
  
# transformer.output_dimensions
    
    
    
    
   
      