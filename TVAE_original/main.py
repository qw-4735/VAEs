"""
Reference:
[1]: https://github.com/an-seunghwan/synthesizers/blob/main/tvae/modules/train.py

tvae.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.getcwd()
#os.chdir('D:\VAE\TVAE')

#import sys
#sys.path.append('/utils')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb
    
run = wandb.init(
    project='TVAE',
    entity='qw4735',
    tags=['TVAE-2']
)    

#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')
    
    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--latent_dim', default=2, type=int, help='the dimension of the latent variable z')
    #parser.add_argument('--input_dim', default=67, type=int, help='the dimension of the input variable x')
    #parser.add_argument('--compress_dims', default=(128,128), type=int, help='size of each hidden layer')
    #parser.add_argument('--decompress_dims', default=(128,128), type=int, help='size of each hidden layer')
    
    parser.add_argument('--num_epochs', type = int, default=200, help='maximum iteration')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay parameter')
    parser.add_argument('--sigma_range', default=[0.1,1], type=arg_as_list, help='range of observational noise')
    
    if debug:
        #return parser.parse_known_args(args=[])
        return parser.parse_args(args=[])
    else:
        #return parser.parse_known_args()
        return parser.parse_args()


#
# import easydict
# config = easydict.EasyDict({
#          'input_dim':67,
#          'n':2,
#          'latent_dim':2,
#          'seed':1,
#          'num_epochs':200,
#          'batch_size':256,
#          'lr':0.005,
#          'weight_decay':1e-5,
#          'sigma_range':[0.1,1]
#      })

#%%
def main():
    config= vars(get_args(debug=False))
    config['cuda'] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config) #한번에 관리
   

    set_random_seed(config['seed'])  # model이 바뀜
    torch.manual_seed(config['seed'])
    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])
    
    """dataset"""
    _, dataloader, transformer,  _,  _, _, _  = generate_dataset(config, device, random_state=0)
    config['input_dim'] = transformer.output_dimensions
        
    
    """model"""
    model = TVAE(config, device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config['lr'],
        weight_decay = config['weight_decay']
    )
    model.train()
  
    """train"""
    for epoch in range(config['num_epochs']):
        logs = train(transformer.output_info_list, dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([',{}:{:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in  logs.items()})  # metric을 추적
        
    """model save"""
    torch.save(model.state_dict(), './assets/TVAE_{}.pth'.format('Personal_Loan'))    
    artifact = wandb.Artifact('TVAE_Personal_Loan', type='model', metadata=config) # 우리가 원하는 것을 저장하는 디렉토리  # Artifact의 큰 장점은 수정이 되더라도 수정된 부분만 저장한다는 것. 즉, 저장 용량을 최적화하며 버져닝이 가능하다. 
    artifact.add_file('./assets/TVAE_{}.pth'.format('Personal_Loan'))
    artifact.add_file('./main.py')
    artifact.add_file('./utils/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#wandb.agent(sweep_id, main, count = 10)
#wandb.run.finish()
   
#%%   
#%%
if __name__ == '__main__':
     main()
    
#for param_tensor in model.state_dict():
#   print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#%%     

