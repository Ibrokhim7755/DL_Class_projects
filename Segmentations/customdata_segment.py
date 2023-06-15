import torch, os, cv2, numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import albumentations as A
from glob import glob



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

im_path_jpeg = Image path for Jpeg extensions
im_path_png  = Image path for png extensions
im_mask_jpeg = Mask path for jpeg
im_mask_png = Mask path for png
    
"""

class segmentations(Dataset):
    def __init__(self, root, transformations = None):
        self.transformations = transformations
        self.tensorize = T.Compose([T.ToTensor()])
        self.im_path_jpeg = sorted(glob(os.path.join(root, 'D:/Data/Datasets/Clothing/IMAGES/*.jpeg')))
        self.im_path_png = sorted(glob(os.path.join(root, 'D:/Data/Datasets/Clothing/Images_png/*.png')))
        self.im_mask_jpeg = sorted(glob(os.path.join(root, 'D:/Data/Datasets/Clothing/MASKS/*.jpeg')))
        self.im_mask_png = sorted(glob(os.path.join(root, 'D:/Data/Datasets/Clothing/png_masks/*.png')))
        
        self.ims_paths = self.im_path_jpeg + self.im_path_png
        self.gts_paths = self.im_mask_jpeg + self.im_mask_png
        self.total_ims = len(self.ims_paths)
        self.total_gts = len(self.gts_paths)
        
        assert self.total_ims == self.total_gts
        print(f'There are {self.total_ims} images and {self.total_gts} masks in the dataset')  
        
        
    def __len__(self):
        return len(self.total_ims)
    
    
    def __getitem__(self, idx):
        im = cv2.cvtColor(cv2.imread(self.ims_paths[idx]), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(self.gts_paths[idx]), cv2.COLOR_BGR2GRAY)
        
        if self.transformations is not None:
            transformed = self.transformations(image = im, mask = gt)
            im, gt = transformed['image'], transformed['mask']
        
        return self.tensorize(im), torch.tensor(gt).long()
    
tfs = A.Compose([A.Resize(224,224), 
                 A.HorizontalFlip(),
                 A.GridDistortion(p = 0.2)])
     
ds = segmentations(root='data', transformations=tfs)