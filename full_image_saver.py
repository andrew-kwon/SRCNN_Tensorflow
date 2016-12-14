import tensorflow as tf
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.stats.mstats as mstats
import sys
import os
import math


learning_rate = 0.000000001

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


with h5py.File('test_py.h5','r') as hf:
    hf_test_data = hf.get('test_input')
    test_data = np.array(hf_test_data)
    hf_test_label = hf.get('test_label')
    test_label = np.array(hf_test_label)

X = tf.placeholder("float32", [None, 33, 33, 3])
Y = tf.placeholder("float32", [None, 21, 21, 3])

W1 = init_weights([9, 9, 3, 64])       # 9x9x3 conv, 64 outputs
W2 = init_weights([1, 1, 64, 32])     # 1x1x64 conv, 32 outputs
W3 = init_weights([5, 5, 32, 3])      # 5x5x32 conv, 3 output

B1 = tf.Variable(tf.zeros([64]), name="Bias1")
B2 = tf.Variable(tf.zeros([32]), name="Bias2")
B3 = tf.Variable(tf.zeros([3]), name="Bias3")

L1 = tf.nn.relu(tf.nn.conv2d(X, W1,                       # l1 shape=(?, 25, 25, 64)
                        strides=[1, 1, 1, 1], padding='VALID') + B1)
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2,                     # l2 shape=(?, 25, 25, 32)
                        strides=[1, 1, 1, 1], padding='VALID') + B2)

hypothesis = tf.nn.conv2d(L2, W3,                     # l3 shape=(?, 21, 21, 3)
                        strides=[1, 1, 1, 1], padding='VALID') + B3

subim_input = X[:,6:27, 6:27, 0:3]
   
cost = tf.reduce_mean(tf.reduce_sum(tf.square((Y-subim_input)-hypothesis), reduction_indices=1))

var_list = [W1,W2,W3,B1,B2,B3]
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=var_list)
step=0

checkpoint_dir = "cps/"

with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    print('start tf.Session()')
    if ckpt and ckpt.model_checkpoint_path:
        print ('load learning')
        saver.restore(sess, ckpt.model_checkpoint_path)

    test1 = tf.nn.relu(tf.nn.conv2d(test_data, W1,                       # l1 shape=(?, 33, 33, 64)
                        strides=[1, 1, 1, 1], padding='SAME') + B1)
    test2 = tf.nn.relu(tf.nn.conv2d(test1, W2,                     # l2 shape=(?, 33, 33, 32)
                        strides=[1, 1, 1, 1], padding='SAME') + B2)

    test_hypothesis = tf.nn.conv2d(test2, W3,                     # l3 shape=(?, 33, 33, 3)
                        strides=[1, 1, 1, 1], padding='SAME') + B3
   

    print(test_hypothesis)
    output_image=sess.run(test_hypothesis)[0,:,:,0:3]

    output_image += test_data[0,:,:,0:3]
    for k in range(0,test_data.shape[1]):
        for j in range(0,test_data.shape[2]):
             for c in range(0,3):
                  if(output_image[k,j,c]>1.0) : output_image[k,j,c]=1;
                  elif(output_image[k,j,c]<0) : output_image[k,j,c]=0;

    tmp_image = (output_image * 255).astype('uint8')
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_YCrCb2RGB)
    plt.imshow(tmp_image)            
    subname="result/"+"result.jpg" 
    plt.savefig(subname)






























