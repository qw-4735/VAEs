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
#from utils.evaluation import (
#    regression_eval,
#    classification_eval,
#    goodness_of_fit,
#    privacy_metrics
#)

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
    
    # 여기는 왜필요한지 모르겠음.... 뒤에서 쓰는 것 같지도 않은데....
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
    #train_target = train[target_].idxmax(axis=1)  # 행별로 최대인 값의 인덱스 반환
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



# # train['CD Account'].value_counts().plot.bar()

# import pandas as pd

# df_a = train['Online'].value_counts().rename_axis('index').to_frame('original')
# df_b = synthetic_df['Online'].value_counts().rename_axis('index').to_frame('synthetic')
# df_total = pd.concat([df_a, df_b], axis=1)
# df_total.plot.bar(color=['lightblue','lightsalmon'],rot=0, title='Online')  # color=['cornflowerblue','salmon']

# plt.hist((train['CreditCard'], synthetic_df['CreditCard']), histtype='bar')
# plt.title('CD Account')


# df = pd.read_csv('D:\VAE\TVAE\Bank_Personal_Loan_Modelling.csv')
# df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # 데이터프레임으로부터 random sampling  # frac : 표본추출비율(0~1)
# df = df.drop(columns=['ID'])
# X = df.drop(['Personal Loan'], axis=1)
# y = df['Personal Loan']
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# clf = RandomForestClassifier()
# clf.fit(x_train, y_train)
# y_pred= clf.predict(x_test)
# accuracy_score(y_pred, y_test)

#%%
#Onehotencoder
# def _reverse_transform(self, data):
#         """Convert float values back to the original categorical values.
#         Args:
#             data (pd.Series or numpy.ndarray):
#                 Data to revert.
#         Returns:
#             pandas.Series
#         """
#         if not isinstance(data, np.ndarray):
#             data = data.to_numpy()  # (4000,4)

#         if data.ndim == 1:
#             data = data.reshape(-1, 1)
        
#         indices = np.argmax(data, axis=1)
#         indices.shape   # (4000,) 
        
#         # 
#         z = np.random.gumbel(loc=0, scale=1, size=data.shape)  # (4000, 4)
#         indices = (data +  z).argmax(axis=1) 
#         indices.shape
#         #
#         return pd.Series(indices).map(self.dummies.__getitem__)

# #BASE
# def reverse_transform(self, data):
#         """Revert the transformations to the original values.
#         Args:
#             data (pandas.DataFrame):
#                 The entire table.
#         Returns:
#             pandas.DataFrame:
#                 The entire table, containing the reverted data.
#         """
#         # if `data` doesn't have the columns that were transformed, don't reverse_transform
#         if any(column not in data.columns for column in self.output_columns):
#             return data

#         data_family = data.copy()
#         columns_data = self._get_columns_data(data, self.output_columns)
#         reversed_data = self._reverse_transform(columns_data)
#         data = data.drop(self.output_columns, axis=1)
#         data = self._add_columns_to_data(data, reversed_data, self.columns)

#         return data
 
# self.output_columns = self.get_output_columns()  
# def get_output_columns(self):
#         """Return list of column names created in ``transform``.
#         Returns:
#             list:
#                 Names of columns created during ``transform``.
#         """
#         return list(self._get_output_to_property('sdtype'))
    
# def _get_output_to_property(self, property_):
#         output = {}
#         for output_column, properties in self.output_properties.items():
#             # if 'sdtype' is not in the dict, ignore the column
#             if property_ not in properties:
#                 continue
#             if output_column is None:
#                 output[f'{self.column_prefix}'] = properties[property_]
#             else:
#                 output[f'{self.column_prefix}.{output_column}'] = properties[property_]

#         return output  
    
# def __init__(self):
#         self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}    
        

#  ##################
# def _store_columns(columns, data):
#         if isinstance(columns, tuple) and columns not in data:
#             columns = list(columns)
#         elif not isinstance(columns, list):
#             columns = [columns]

#         missing = set(columns) - set(data.columns)
#         if missing:
#             raise KeyError(f'Columns {missing} were not present in the data.')

# column_prefix = '#'.join(columns)    
# output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}  

# def _get_output_to_property( property_):
#         output = {}
#         for output_column, properties in output_properties.items():
#             # if 'sdtype' is not in the dict, ignore the column
#             if property_ not in properties:
#                 continue
#             if output_column is None:
#                 output[f'{column_prefix}'] = properties[property_]
#             else:
#                 output[f'{column_prefix}.{output_column}'] = properties[property_]

#         return output  
    
# def get_output_columns():
#         """Return list of column names created in ``transform``.
#         Returns:
#             list:
#                 Names of columns created during ``transform``.
#         """
#         return list(_get_output_to_property('sdtype'))  
      
# output_columns = get_output_columns()   

# from utils.transformer_base import BaseTransformer
# BaseTransformer.get_output_columns()
# data_family = data.copy()
# columns_data = self._get_columns_data(data, output_columns)   



# ####### 
 
# _, dataloader, transformer,  _,  _, _, _  = generate_dataset(config, device, random_state=0)
# column_transform_info_list = transformer._column_transform_info_list


# # data_transformer
# def inverse_transform_continuous(column_transform_info, column_data, sigmas, st):
#         gm = column_transform_info.transform
#         data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
#         data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
#         if sigmas is not None:
#             selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
#             data.iloc[:, 0] = selected_normalized_value

#         return gm.reverse_transform(data)

# def inverse_transform_discrete(column_transform_info, column_data):
#     ohe = column_transform_info.transform
#     data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
#     return ohe.reverse_transform(data)[column_transform_info.column_name]

# def inverse_transform(data, sigmas=None):
#     """Take matrix data and output raw data.
#     Output uses the same type as input to the transform function.
#     Either np array or pd dataframe.
#     """
#     st = 0
#     recovered_column_data_list = []
#     column_names = []
#     for column_transform_info in column_transform_info_list:
#         dim = column_transform_info.output_dimensions
#         column_data = data[:, st:st + dim]
#         if column_transform_info.column_type == 'continuous':
#             recovered_column_data = inverse_transform_continuous(
#                 column_transform_info, column_data, sigmas, st)
#         else:
#             recovered_column_data = inverse_transform_discrete(
#                 column_transform_info, column_data)

#         recovered_column_data_list.append(recovered_column_data)
#         column_names.append(column_transform_info.column_name)
#         st += dim

#     recovered_data = np.column_stack(recovered_column_data_list)
#     recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
#                         .astype(self._column_raw_dtypes))
#     if not self.dataframe:
#         recovered_data = recovered_data.to_numpy()

#     return recovered_data