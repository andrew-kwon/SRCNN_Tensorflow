
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import h5py
import math

def search(dirname):
    train_name =[]
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.bmp':
            train_name.append(full_filename)            
    return train_name

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



folder = 'Test/Set5'
savepath = 'test_py.h5'
size_input = 33
size_label = 21
scale = 3
stride = 14

data = []
label = []

input_images = []
label_images = []

padding = abs(size_input - size_label)/2;

for (root, dir, files) in os.walk(folder):
    for file in files:   
        filepath = root+'/'+file
        
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image = image[:,:]
        image = image.astype('float') / 255.0          # int to float
        
        im_label = modcrop(image, scale)
        (hei, wid) = im_label.shape
        
        im_input = cv2.resize(im_label, (0,0), fx=1.0/scale, fy=1.0/scale, interpolation = cv2.INTER_CUBIC)     
        im_input = cv2.resize(im_input, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC) # return original size
    
        input_images.append(im_input)        
        label_images.append(im_label)
        
        for x in range(0, hei - size_input + 1, stride) :                     #add stride in for
            for y in range(0, wid - size_input + 1, stride) : 
               
               subim_input = im_input[x:x+size_input, y: y+size_input]
               subim_label = im_label[x:x+size_input, y: y+size_input]
               
               subim_input = subim_input.reshape([size_input,size_input,1])
               subim_label = subim_label.reshape([size_input,size_input,1])
                          
               data.append(subim_input)
               label.append(subim_label)               

data = np.asarray(data)
label = np.asarray(label)

#order = np.random.permutation(data.shape[0])
#data = data[order,:,:,:]
#label = label[order,:,:,:]

with h5py.File(savepath,'w') as hf:
    hf.create_dataset('dataset_1',data=data)
    hf.create_dataset('dataset_2',data=label)

plt.imshow(input_images[0], cmap='Greys_r')
plt.show()

plt.imshow(label_images[0], cmap='Greys_r')
plt.show()

plt.imshow(label[0,:,:,0], cmap='Greys_r')
plt.show()

plt.imshow(data[0,:,:,0], cmap='Greys_r')
plt.show()

print(data[0,:,:,0].shape)





