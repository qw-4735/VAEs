#%%
import torch
from utils.datasets import generate_dataset

#%%
def apply_activate(data, transformer, gumbel_softmax):
    data_t = []
    st = 0 
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn == 'tanh':
                ed = st + span_info.dim
                data_t.append(torch.tanh(data[:, st:ed]))   
                st = ed 
            elif span_info.activation_fn == 'softmax':
                ed = st + span_info.dim
                transformed = gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            else:
                raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
    return torch.cat(data_t, dim=1)  
#%%