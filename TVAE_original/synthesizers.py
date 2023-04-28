"""
Reference:
[1]: https://github.com/an-seunghwan/synthesizers/blob/main/tvae/modules/synthesizers.py

tvae.py
"""
import os
#os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.chdir('D:\VAE\TVAE')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from utils.simulation import set_random_seed
from utils.model import TVAE
from utils.datasets import generate_dataset
from utils.activation_fn import apply_activate

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


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
    
#device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

run = wandb.init(
    project='TVAE',
    entity='qw4735',
    tags=['TVAE-2, Synthetic']
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
    
    parser.add_argument('--num', type=int, default=0, help='model version')
    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--latent_dim', default=2, type=int, help='the dimension of the latent variable z')
    # parser.add_argument('--input_dim', default=67, type=int, help='the dimension of the input variable x')
    # #parser.add_argument('--compress_dims', default=(128,128), type=int, help='size of each hidden layer')
    # #parser.add_argument('--decompress_dims', default=(128,128), type=int, help='size of each hidden layer')
    
    parser.add_argument('--num_epochs', type = int, default=200, help='maximum iteration')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay parameter')
    parser.add_argument('--sigma_range', default=[0.1,1], type=arg_as_list, help='range of observational noise')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
# import easydict
# config = easydict.EasyDict({
#          'input_dim':68,
#          'n':2,
#          'latent_dim':2,
#          'seed':4,
#          'num_epochs':200,
#          'batch_size':256,
#          'lr':0.005,
#          'weight_decay':1e-5,
#          'sigma_range':[0.1,1]
#      })


#%%
def main():
    config = vars(get_args(debug=False))
    device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config['cuda'] = torch.cuda.is_available()
    
    """dataset"""
    _, _, transformer, train, test, continuous, discrete  = generate_dataset(config, device, random_state= 0 )

    config['input_dim'] = transformer.output_dimensions 
    
    
    """model load"""
    artifact = wandb.use_artifact('qw4735/TVAE/TVAE_{}:v{}'.format('Personal_Loan', config['num']), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    

    wandb.config.update(config)
    
    set_random_seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])
    
    
    """model"""
    model = TVAE(config, device).to(device)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(torch.load(model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(torch.load(model_dir + '/' + model_name, map_location=torch.device('cpu')))
        
    #model.load_state_dict(torch.load('./assets/TVAE_Personal_Loan.pth')) 
    model.eval()  
    
    """preprocess"""
    # 연속형 변수만 표준화
    std = train[continuous].std(axis=0)  # 열별 std
    mean = train[continuous].mean(axis=0) # 열별 mean
    train[continuous] = (train[continuous] - mean)/std 
    test[continuous] = (test[continuous]- mean)/std  
    
 
    # df = pd.concat([train, test], axis=0)  
    
    # df_dummy = []
    # for d in discrete:
    #     df_dummy.append(pd.get_dummies(df[d], prefix=d))
   
    # df= pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)  # 5000 rows x 22 columns(이산형 컬럼 더미변수 추가)
    
    # df_train = df.iloc[:4000]
    # df_test = df.iloc[4000:]
    
    """generating synthetic dataset"""
    torch.manual_seed(config['seed'])
    steps = len(train) // config['batch_size'] + 1  # 63 
    #steps = len(train) // 64+1  
    #
    
    data = []
    with torch.no_grad():
        for _ in range(steps):
            mean = torch.zeros(config['batch_size'], config['latent_dim'])
            #mean = torch.zeros(256, 2)
            std = mean + 1  # std = torch.ones(256, 2)
            noise = torch.normal(mean=mean, std=std).to(device) # [256, 2]
            fake = model._decode(noise)  # xhat  # [batch_size,  input_dim] = [256, 68]
            fake_after_activation = apply_activate(fake, transformer, F.gumbel_softmax)
            #fake = torch.tanh(fake)
            data.append(fake_after_activation.numpy())
    data_concat = np.concatenate(data, axis=0) # shape : (4096, 68)
    data = data_concat[:len(train)] # len(df_train) # shape : (4000, 68)
    synthetic_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy()) # 재현 데이터 생성 : 4000 rows x 12 columns  # categorical variables : argmax
    #synthetic_df.shape # (4000, 12)
    #synthetic_df.head()
    #data[0]

    # df_dummy = []
    # for d in discrete:
    #     df_dummy.append(pd.get_dummies(synthetic_df[d], prefix=d))
   
    # synthetic_df = pd.concat([synthetic_df.drop(columns=discrete)] + df_dummy, axis=1)  # 5000 rows x 22 columns(이산형 컬럼 더미변수 추가)
    
    
    """preprocess synthetic dataset"""
    # 생성한 재현데이터 전처리(표준화, 더미변수 설정)
    std = synthetic_df[continuous].std(axis=0)
    mean = synthetic_df[continuous].mean(axis=0)
    synthetic_df[continuous] = (synthetic_df[continuous] - mean) / std
    
    
    """model fitting"""
    covariates = [x for x in  train.columns if x != 'Personal Loan' ]
    target_ = ['Personal Loan']
    #train_target = train[target_].idxmax(axis=1)  
    #test_target = test[target_].idxmax(axis=1).to_numpy()
    
    
    # baseline
    base_rf_clf = RandomForestClassifier()
    base_rf_clf.fit(train[covariates], train[target_])
    base_pred = base_rf_clf.predict(test[covariates])
    
    base_f1 = f1_score(test[target_], base_pred, average='micro')
    print(confusion_matrix(test[target_], base_pred))
    wandb.log({'F1 (Baseline)' : base_f1})
    
    # feature_importance = pd.DataFrame({
    #     'feature': train[covariates].columns,
    #     'importance': base_rf_clf.feature_importances_
    # })
    #feature_importance
    
    
    # TVAE
    tvae_rf_clf = RandomForestClassifier()
    tvae_rf_clf.fit(synthetic_df[covariates], synthetic_df[target_])
    tvae_pred = tvae_rf_clf.predict(test[covariates])
    
    tvae_f1 = f1_score(test[target_], tvae_pred, average='micro')
    print(confusion_matrix(test[target_], tvae_pred))
    wandb.log({'F1 (TVAE)' : tvae_f1})
    wandb.sklearn.plot_confusion_matrix(test[target_], tvae_pred)
    # feature_importance = pd.DataFrame({
    #     'feature': synthetic_df[covariates].columns,
    #     'importance': tvae_rf_clf.feature_importances_
    # })
    #feature_importance

#%% 
if __name__ == '__main__':
    main()
    
 #%%
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.distplot(train['CCAvg'])
# sns.distplot(train['Income']) 
# sns.distplot(train['Mortgage'])  


# plt.hist(train['Income'])
# plt.title('Income(original)')
# plt.hist(synthetic_df['Income'])
# plt.title('Income(synthetic)')


# # fig,ax= plt.subplots()
# # ax.boxplot([train['Mortgage'], synthetic_df['Mortgage']])
# # plt.title('Mortgage')
# # plt.xticks([1,2],['original', 'synthetic'])
# # plt.show()



