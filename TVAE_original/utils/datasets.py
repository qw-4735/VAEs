
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


