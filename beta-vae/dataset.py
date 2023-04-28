"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0

class CustomImageFolder(ImageFolder):   # ImageFolder를 상속받는 파이썬 클래스
    def __init__(self, root, transform=None):   # 모델의 구조와 동작을 정의하는 생성자를 정의
        super(CustomImageFolder, self).__init__(root, transform)   # super()함수를 부르면 여기서 만든 클래스는 ImageFolder 클래스의 속성들을 가지고 초기화 됨
    
    def __gensim__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img 
    
class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):   # 필요한 변수 선언, 데이터셋의 전처리를 해주는 함수
        self.data_tensor = data_tensor
        
    def __getitem__(self, data_tensor):  # 데이터셋에서 특정 데이터를 가져오는 함수(index번째 데이터를 반환하는 함수)
        return self.data_tensor[index]
    
    def __len__(self):                   # 데이터셋의 길이(즉, 총 샘플의 수를 가져오는 함수)
        return self.data_tensor.size(0)

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'
    
    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])    
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])    
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
        
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir,'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finish')
        data = np.load(root, encoding = 'bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()  # (64, 64) -> (64, 1, 64) ??  # data['imgs](64x64)를 넘파이 배열로 받아 Tensor로 바꾸어줌  # torch.unsqueeze(input, dim) : dim을 늘려주고, 그 값을 1로 만듬         
        # torch.unsqueeze(input, dim) : dim을 늘려주고, 그 값을 1로 만듬 
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    
    else:
        raise NotImplementedError
    
    train_data = dset(**train_kwargs)    # Keyword aarguments are also possible  
    train_loader = DataLoader(train_data, 
                              batch_size = batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    data_loader = train_loader
    
    return data_loader    


if __name__ == '__main__':  
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()])
    
    dset = CustomImageFolder('data/CelebA', transform)
    loader =  DataLoader(dset,
                         bath_size=32,
                         shuffle=True,
                         num_workers=1,
                         pin_memory=False,
                         drop_last=True)
    images1 = iter(loader).next()
    


                    
#%%
