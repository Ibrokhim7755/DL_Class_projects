import torch, os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision import transforms as T
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from matplotlib import pyplot as plt


"""
This class function is used to pull the dataset from the directory and defines parameter,
transformation, albumentation.
we have three method in this code to call the dataset:
def __init__():
initialize the root(path) and transformation

def __len__():
defines a number of images in the dataset

def __getitim__(): 
defines the items and label 

Parameters :
   A = albumentation
  im = Image
  gt = Ground Truth
"""
class CustomDataset(Dataset):
    def __init__(self, root, transformation = None):
        self.transformation = transformation
        
        self.im_paths = glob(os.path.join(root,'D:/Data/Datasets/CARS_DATA/*/*.jpg'))
        
        self.cls_names = []
        for idx, im_path in enumerate(self.im_paths):
            dirname = os.path.dirname(im_path)
            cls_name = dirname.split('/')[-1]
            
            if cls_name not in self.cls_names:
                self.cls_names.append(cls_name)
                
        self.classes = {idx: cls_name for idx, cls_name in enumerate(self.cls_names)}
        
    def __len__(self):
        return len(self.im_paths)
    
    def get_classes(self):
        return self.classes
    
    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        
        for idx, (cls_num, cls_name) in enumerate(self.classes.items()):
            if cls_name in im_path:
                gt = cls_num;break
                
        im = Image.open(im_path).convert('RGB')
        
        if self.transformation is not None:
            im = self.transformation(im)
            
        return im, gt
    
tfs = T.Compose([
    T.Resize(224),
    T.ToTensor()])
    
        
ds = CustomDataset(root='CARS_DATA',transformation=tfs)
classes = ds.get_classes()

image, label = ds[0]
print(image.shape)
len(classes)                