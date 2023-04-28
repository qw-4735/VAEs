#%%
"""
solver.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm  

import torch  
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import wandb
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils.model import BetaVAE_H 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')
    
    parser.add_argument('--epochs', type = int, default=10, help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    #parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
args = vars(get_args(debug=True))
#%%
trans = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder('./causal_data/pendulum_1/train', transform=trans)  # train폴더 안에 하위폴더 하나 더 만들것
test_dataset = torchvision.datasets.ImageFolder('./causal_data/pendulum_1/test', transform = trans)
# train_dataset = torchvision.datasets.ImageFolder(r'D:\GitHub\vae\betavae\causal_data\pendulum_1\train', transform=trans)  # train폴더 안에 하위폴더 하나 더 만들것
# test_dataset = torchvision.datasets.ImageFolder(r'D:\GitHub\vae\betavae\causal_data\pendulum_1\test', transform = trans)

train_loader = DataLoader(train_dataset, batch_size = args["batch_size"], shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)

#%%
wandb.init(project="beta vae", entity="qw4735")
wandb.watch(model) 
wandb.config.update(args)
#%%    
# x =torch.randn(10, 3, 64, 64) 
#x_recon, mu,logvar = model(x)  # 튜플

model = BetaVAE_H() 

optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
#%%
def train(train_loader, model, optimizer, args):
    model.train()
    logs = {
        'loss': [],
        'recon_loss' : [],
        'KLD' : []
    }
    for x,_ in tqdm.tqdm(train_loader):
        x=x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        
        loss_ = []
        
        """reconstruction"""
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=[1, 2, 3]).mean()
        loss_.append(('recon_loss', recon_loss))
        
        """KLD"""
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()
        loss_.append(('KLD',KLD))
        
        """loss"""
        loss = recon_loss + args["beta"] * KLD
        loss /= x.size(0)
        loss_.append(('loss', loss))

        loss.backward()
        optimizer.step()
        
        for x,y in loss_:
            logs[x] = logs.get(x) + [y.item()]   
    
    return logs, x_recon        

for epoch in tqdm(range(args["epochs"])):
    logs, x_recon = train(train_loader, model, optimizer, args)
    
    print_input = "[epoch {:02d}]".format(epoch+1)
    print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x,y in logs.items()])
    print(print_input)
    
    wandb.log({x : np.mean(y) for x,y in logs.items()}) 
#%%    
    
    
    
    
    
    


