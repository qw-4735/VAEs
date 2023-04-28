
#%%
"""
Reference:
[1]: https://github.com/an-seunghwan/synthetic/blob/main/tvae/modules/datasets.py

tvae.py
"""
import os
os.getcwd()
os.chdir('D:\VAE\TVAE')


import tqdm
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from sklearn.mixture import BayesianGaussianMixture
from utils.data_transformer import DataTransformer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    

#%%

def generate_dataset(config, device, random_state=0):
    
    df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # 데이터프레임으로부터 random sampling  # frac : 표본추출비율(0~1)
    df = df.drop(columns=['ID'])
  
    continuous = [
        'CCAvg', 
        'Mortgage', 
        'Income',
        'Experience',
        'Age'
        ] 
    discrete = [
        'Family', 
        'Education', 
        'Personal Loan',
        'Securities Account',
        'CD Account',
        'Online',
        'CreditCard'
        ]

    df = df[continuous + discrete]
    # df = df.dropna(axis=0)
    
    train= df.iloc[:4000]
    test = df.iloc[4000:]
    
    transformer = DataTransformer()
    transformer.fit(train, discrete_columns=discrete, random_state = random_state)
    train_data = transformer.transform(train)
    
    dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(device))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    
    
    return dataset, dataloader, transformer,  train,  test, continuous, discrete


#_, dataloader, transformer,  _,  _, _, _  = generate_dataset(config, device, random_state=0)
#%%
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.distplot(train['CCAvg'])
#sns.distplot(train['Income'])
#_, _, _, _, transformer0, _, train, train0, _, test, continuous, discrete = generate_dataset(config, device, random_state=1)
#_, _, _, _, transformer0, _, train, train0, _, test, continuous, discrete = generate_dataset(config, device, random_state=1)
# %%
# def generate_dataset(config, device, random_state=1):
    
#     df = pd.read_csv('D:\VAE\TVAE\Bank_Personal_Loan_Modelling.csv')
#     df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # 데이터프레임으로부터 random sampling  # frac : 표본추출비율(0~1)
#     df = df.drop(columns=['ID'])
    
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
    
#     train0 = train[train['Personal Loan'] == 0]  # (3622, 12)
#     train1 = train[train['Personal Loan'] == 1]  # (378, 12)
#     raw_data = train0
#     column_name='CCAvg'
#     raw_data[[column_name]].columns[0]
#     transformer0 = DataTransformer()
#     transformer0.fit(train0, discrete_columns=discrete, random_state = random_state)
#     train_data0 = transformer0.transform(train0)
    
#     transformer1 = DataTransformer()
#     transformer1.fit(train1, discrete_columns=discrete, random_state = random_state)
#     train_data1 = transformer1.transform(train1)
#     #transformer0.output_dimensions
    
#     dataset0 = TensorDataset(torch.from_numpy(train_data0.astype('float32')).to(device))
#     dataloader0 = DataLoader(dataset0, batch_size=config.batch_size, shuffle=True, drop_last=False)
    
#     dataset1 = TensorDataset(torch.from_numpy(train_data1.astype('float32')).to(device))
#     dataloader1 = DataLoader(dataset1, batch_size=config.batch_size, shuffle=True, drop_last=False)
    
#     return dataset0, dataset1, dataloader0, dataloader1, transformer0, transformer1, train, train0, train1, test, continuous, discrete
