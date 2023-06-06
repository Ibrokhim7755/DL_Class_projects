import cv2
import matplotlib.pyplot as plt 


'''
This function visualizes the segmentiation dataset with its mask to check whether it is in a correct manner 

Parameters: 
 ds      = custom dataset class function
 img_num = number of images that we want to visualize 
 row     = number of row we want to add
 
'''

def showing(ds, img_num, row):
    for idx, (im, gt) in enumerate(ds):
        count = 1
        plt.figure(figsize=(18,18))
        if idx == img_num: break
        im = (im * 255).detach().cpu().permute(1,2,0).numpy().astype('uint8')# integer 8
        gt = (gt * 255).detach().cpu().numpy().astype('uint8')
        plt.subplot(row, img_num // row, count)
        plt.imshow(im)
        plt.axis('off')
        count += 1
        plt.subplot(row, img_num // row, count)
        plt.imshow(gt)
        plt.axis('off')
        count += 1
        plt.show()
        
        
#showing(ds, 10, 2)