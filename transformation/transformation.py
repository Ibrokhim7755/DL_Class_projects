import torch
from torchvision import transforms as tfs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


"""
  This function creates Data Augmentation in image dataset using transformation
  
  Arguments:
    Resize
    ToTensor
    Normalize
    RandomHorizontalFlip
    CenterCrop
    RandomRotation
    ColorJitter
    RandomGrayscale
    RandomResizedCrop
    
"""


def transform():
    # define transforms
    
    trform = tfs.Compose([tfs.Resize((255,255)),
                          #tfs.ToTensor(),
                          #tfs.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                          tfs.RandomHorizontalFlip(),
                          tfs.CenterCrop(250),
                          tfs.RandomRotation(90),
                          tfs.ColorJitter(brightness=1, contrast=0.3, saturation=0.3, hue=0.3),
                          tfs.RandomGrayscale(155),
                          tfs.RandomResizedCrop(150)])
    
# Define Image Path

    path = 'D:\portfolio_projects\Assignments\husky_dog.jpg'
    
# Open Image
    img = Image.open(path)
    
# Apply Transformation
    transformed = trform(img)
    
    img = np.array(img) # Orginal Image for visualization
    
# Visualizing original and transformed image
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].imshow(img)
    ax[0].set_title('Orignal Image')
    ax[1].imshow(transformed)
    ax[1].set_title('Transformed Image')
    plt.show()
    
