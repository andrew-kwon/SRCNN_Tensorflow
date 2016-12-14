import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
import string
import cv2

file = open('train_num_epoch')
epoch_number =int(file.readline())

# f 1 = 9, f 2 = 1, f 3 = 5, n 1 = 64, and n 2 = 32.
learning_rate = 0.000001
train_num=1000000
epoch_cost_string=""

def data_iterator(data, label, batch_size):
    num_examples = data.shape[0]    
    num_batch = num_examples // batch_size
    num_total = num_batch * batch_size

    while(True):
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        shuf_data = data[perm]
        shuf_label = label[perm]
        for i in range(0, num_total, batch_size):
            batch_data = shuf_data[i:i+batch_size]
            batch_label = shuf_label[i:i+batch_size]
            yield batch_data, batch_label

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

with h5py.File('train_py.h5','r') as hf:
    hf_data = hf.get('input')
    data = np.array(hf_data)
    hf_label = hf.get('label')
    label = np.array(hf_label)

with h5py.File('test_py.h5','r') as hf:
    hf_test_data = hf.get('test_input')
    test_data = np.array(hf_test_data)
    hf_test_label = hf.get('test_label')
    test_label = np.array(hf_test_label)


print(data.shape)

batch = data_iterator(data, label, 128)


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
               
cost = tf.reduce_mean(tf.reduce_sum(tf.square((Y-subim_input)-hypothesis), reduction_indices=0))

var_list = [W1,W2,W3,B1,B2,B3]
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=var_list)
step=0

checkpoint_dir = "cps/"
epoch_file = open('train_epoch_cost','a')


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    print('start tf.Session()')
    if ckpt and ckpt.model_checkpoint_path:
        print ('load learning')
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(train_num):
        batch_data, batch_label = batch.__next__()
        #sess.run(optimizer, feed_dict={X: data, Y: label})
        sess.run(optimizer, feed_dict={X: batch_data, Y: batch_label})      
        step+=1
        if (epoch_number+step)%1000 == 0 :

            print_step = (epoch_number+step)
            epoch_cost_string="[epoch] : "+(str)(print_step)+" [cost] : "
            current_cost_sum=0.0
            mean_batch_size = (int)((data.shape[0]/128))
            for j in range(0,mean_batch_size):
                current_cost_sum+=sess.run(cost, feed_dict={X:data[j].reshape(1,33,33,3), Y: label[j].reshape(1,21,21,3)})
            epoch_cost_string+=str(float(current_cost_sum/mean_batch_size))
            #epoch_cost_string+=" [learning_rate] : "+(str)(learning_rate) 
            epoch_cost_string+="\n"
            print(epoch_cost_string)

        if (epoch_number+step)%1000 == 0 :
            test_L1 = tf.nn.relu(tf.nn.conv2d(test_data, W1,                       # l1 shape=(?, 33, 33, 64)
                        strides=[1, 1, 1, 1], padding='SAME') + B1)
            test_L2 = tf.nn.relu(tf.nn.conv2d(test_L1, W2,                     # l2 shape=(?, 33, 33, 32)
                        strides=[1, 1, 1, 1], padding='SAME') + B2)

            test_hypothesis = tf.nn.conv2d(test_L2, W3,                     # l3 shape=(?, 33, 33, 3)
                        strides=[1, 1, 1, 1], padding='SAME') + B3
   

            output_image=sess.run(test_hypothesis)[0,:,:,0:3]

            output_image += test_data[0,:,:,0:3]
            for k in range(0,test_data.shape[1]):
                for j in range(0,test_data.shape[2]):
                    for c in range(0,3):
                        if(output_image[k,j,c]>1.0) : output_image[k,j,c]=1;
                        elif(output_image[k,j,c]<0) : output_image[k,j,c]=0;
                
            temp_image = (output_image * 255).astype('uint8')
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_YCrCb2RGB)
            
            plt.imshow(temp_image)
           
            subname="shot/"+str(epoch_number+step)+".jpg" 
            plt.savefig(subname)

            saver.save(sess, checkpoint_dir + 'model.ckpt', print_step)
            train_file_num = epoch_number+step
            train_file = open('train_num_epoch', 'w')
            train_file.write('%d' % train_file_num)
            epoch_file.write(epoch_cost_string)













