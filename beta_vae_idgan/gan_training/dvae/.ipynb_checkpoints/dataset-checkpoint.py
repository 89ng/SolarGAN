"""dataset.py"""

import os
import numpy as np
from tqdm import tqdm

import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import cv2


def Gray_Label_To_Torch_Tensor(path):
    example =  cv2.imread(path)
    
    gt = np.round(np.mean(example,axis=2)/64)
    gt = torch.LongTensor(gt)

    def get_one_hot(label, N):
        size = list(label.size())
        label = label.view(-1)   #reshape to a long vector
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)   #turn to one hot
        size.append(N)  #reshape to h*w*channel classes
        return ones.view(*size)


    gt_one_hot = get_one_hot(gt, 5)
    #print(gt_one_hot)
    #print(gt_one_hot.shape)
    #print(gt_one_hot.argmax(-1) == gt)  # check if one hot converting correct or not (1:correct)

    #gt_remove_edge = gt_one_hot[:,:,1:].permute(2,0,1)
    gt_reserve_edge = gt_one_hot.permute(2,0,1)
    img_tensor = gt_reserve_edge
    
    return img_tensor

def load_simple_im(path):
    example =  cv2.imread(path)
    
    gt = np.mean(example,axis=2)/256

    img_tensor = torch.from_numpy(gt).unsqueeze(0).float()
    
    return img_tensor


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

    
class solargan_trainset(Dataset):
    def __init__(self,images, loader):
        
        self.images = images #image path
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        one_hot_tensor = self.loader(fn)
        return one_hot_tensor

    def __len__(self):
        return len(self.images)
    
class solargan_im_trainset(Dataset):
    def __init__(self,images, loader):
        
        self.images = images #image path
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        tensor = self.loader(fn)
        return tensor

    def __len__(self):
        return len(self.images)

def return_data(args):
    dsetname = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = 128 #modifided from 64 to 128
    
    if dsetname.lower() == 'grayfisheye':
        grayimg_list = glob.glob(dset_dir+"/grayfisheye/*.PNG")
        train_kwargs = {'images':grayimg_list,'loader':Gray_Label_To_Torch_Tensor}
        dset = solargan_trainset
        
    elif dsetname.lower() == 'graycube':
        grayimg_list = glob.glob(dset_dir+"/graycube/*.PNG")
        train_kwargs = {'images':grayimg_list,'loader':Gray_Label_To_Torch_Tensor}
        dset = solargan_trainset
        
    elif dsetname.lower() == 'grayfisheye_im':
        grayimg_list = glob.glob(dset_dir+"/grayfisheye/*.PNG")
        train_kwargs = {'images':grayimg_list,'loader':load_simple_im}
        dset = solargan_im_trainset
        
    elif dsetname.lower() == 'graycube_im':
        grayimg_list = glob.glob(dset_dir+"/graycube/*.PNG")
        train_kwargs = {'images':grayimg_list,'loader':load_simple_im}
        dset = solargan_im_trainset

    elif dsetname.lower() == 'chairs':
        root = os.path.join(dset_dir, 'Chairs_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'cars':
        root = os.path.join(dset_dir, 'Cars_64')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif dsetname.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader
