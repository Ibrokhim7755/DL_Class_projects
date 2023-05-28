import torch
import timm
from torchvision.datasets import ImageFolder
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import albumentations as A
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
This code is extract the feature based on the inception model from timm
and visualize it using matplotlib
    
"""


path = 'D:\portfolio_projects\Assignments\Blue_Bugatti_Car_HD_Image.jpg'

im = Image.open(path)
#plt.imshow(im)


device = "cpu"
m = timm.create_model("inception_v3", pretrained = True)
m.to(device)
m.eval()
# inp = torch.rand(1,3,224,224).to(device)
ts = tfs.Compose([tfs.Resize((256,256)), tfs.ToTensor(), tfs.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
im = ts(Image.open(path)).unsqueeze(0).to(device)
# im.shape
print(m(im).shape)  # output ->torch.Size([1, 1000])


# checking input image shape
print(im.shape)  #outpu> torch.Size([1, 3, 256, 256])

#extracting the children layers from the model
child = list(m.named_children())

for name,_ in child:
    print(name)

# Getting these layers as a dictionary to access easyly
l1 = m.Conv2d_1a_3x3
l2 = m.Conv2d_2a_3x3
l3 = m.Conv2d_2b_3x3
l4 = m.Pool1
l5 = m.Conv2d_3b_1x1
l6 = m.Conv2d_4a_3x3
l7 = m.Pool2
l8 = m.Mixed_5b
l9 = m.Mixed_5c
l10 = m.Mixed_5d
l11 = m.Mixed_6a
l12 = m.Mixed_6b
l13 = m.Mixed_6c
l14 = m.Mixed_6d
l15 = m.Mixed_6e
l16 = m.Mixed_7a
l17 = m.Mixed_7b
l18 = m.Mixed_7c

layers = {'Conv2d_1a_3x3':l1,'Conv2d_2a_3x3':l2,'Conv2d_2b_3x3':l3,'Pool1':l4,'Conv2d_3b_1x1':l5,'Conv2d_4a_3x3':l6,
         'Pool2':l7,'Mixed_5b':l8,'Mixed_5c':l9,'Mixed_5d':l10,'Mixed_6a':l11,'Mixed_6b':l12,'Mixed_6c':l13,'Mixed_6d':l14,
         'Mixed_6e':l15,'Mixed_7a':l16,'Mixed_7b':l17,'Mixed_7c':l18}



# getting the items and checking the value shape
fms= {}
for key,layer in layers.items():
    im = layer(im)
    fms[key]= im
    
for fm in fms.values():
    print(fm.shape)
    
"""
Outputs:

torch.Size([1, 32, 127, 127])
torch.Size([1, 32, 125, 125])
torch.Size([1, 64, 125, 125])
torch.Size([1, 64, 62, 62])
torch.Size([1, 80, 62, 62])
torch.Size([1, 192, 60, 60])
torch.Size([1, 192, 29, 29])
torch.Size([1, 256, 29, 29])
torch.Size([1, 288, 29, 29])
torch.Size([1, 288, 29, 29])
torch.Size([1, 768, 14, 14])
torch.Size([1, 768, 14, 14])
torch.Size([1, 768, 14, 14])
torch.Size([1, 768, 14, 14])
torch.Size([1, 768, 14, 14])
torch.Size([1, 1280, 6, 6])
torch.Size([1, 2048, 6, 6])
torch.Size([1, 2048, 6, 6])

"""


        

