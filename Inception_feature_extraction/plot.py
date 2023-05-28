import matplotlib.pyplot as plt 
import numpy as np

"""
This function gets tensor, detach to cpu and permutate the shape of input
directing to numpy and change the type

"""

def tensor2im(t): return ((t)*256).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)



"""
This function gets fms, rows, cols and returns output of feature maps
  
  parameters:
  
  fms = Feature maps
  rows = rows
  cols = columns
    
"""


def plot_fms(fms, rows, cols):
    for idx, (key, fm) in enumerate(fms.items()):
        plt.figure(figsize = (20, 20))
        print(f"\nStart plotting features maps of {key}\n")
        for i in range(rows * cols): 
            plt.subplot(rows, cols, i + 1) 
            plt.imshow(tensor2im(fm[0])[:, :, i + 1])
            plt.title(f"{key} feature map#{i + 1}")
            plt.axis('off')
        plt.show()
        
