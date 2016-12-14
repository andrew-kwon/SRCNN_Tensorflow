
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import h5py
import math

def modcrop(imgs, modulo):    
    if np.size(imgs.shape)==3:
        (sheight, swidth,_) = image.shape
        sheight = sheight - np.mod(sheight,modulo)
        swidth = swidth - np.mod(swidth,modulo)
        imgs = imgs[0:sheight,0:swidth,:]
    else:
        (sheight, swidth) = image.shape
        sheight = sheight - np.mod(sheight,modulo)
        swidth = swidth - np.mod(swidth,modulo)
        imgs = imgs[0:sheight,0:swidth]        
    return imgs
resultFolder = 'result'
folder = 'Test/Set5'
savepath = 'test_py.h5'
scale = 3
filepath = 'butterfly_GT.bmp'
        
image = cv2.imread(filepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
subname="result/"+"original.jpg" 
plt.savefig(subname)
image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
im_label = modcrop(image, scale)
(hei, wid,_) = im_label.shape

print(im_label.shape)

im_input = cv2.resize(im_label, (0,0), fx=1.0/scale, fy=1.0/scale, interpolation = cv2.INTER_CUBIC)
im_input = cv2.resize(im_input, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC) 

im_input = im_input.astype('float32')/255.0
im_label = im_label.astype('float32')/255.0


temp_image = (im_input* 255).astype('uint8')
temp_image = cv2.cvtColor(temp_image, cv2.COLOR_YCrCb2RGB)
plt.imshow(temp_image)
subname="result/"+"input.jpg" 
plt.savefig(subname)
residual_input=temp_image

residual_origin = (im_label* 255).astype('uint8')
residual_origin=cv2.cvtColor(residual_origin, cv2.COLOR_YCrCb2RGB)
residual_image= residual_origin- residual_input 
plt.imshow(residual_image)
subname="result/"+"residual.jpg" 
plt.savefig(subname)

capture_image= residual_input + residual_image
plt.imshow(capture_image)
subname="result/"+"sumImage.jpg" 
plt.savefig(subname)

im_label = im_label.reshape(1,hei,wid,3)
im_input = im_input.reshape(1,hei,wid,3)

print(im_label.shape)


with h5py.File(savepath,'w') as hf:
    hf.create_dataset('test_input', data=im_input)
    hf.create_dataset('test_label', data=im_label)




