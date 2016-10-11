import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt

# f 1 = 9, f 2 = 1, f 3 = 5, n 1 = 64, and n 2 = 32.

learning_rate = 0.0001
train_num=10000;

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

with h5py.File('train_py.h5','r') as hf:
    hf_data = hf.get('dataset_1')
    data = np.array(hf_data)
    hf_label = hf.get('dataset_2')
    label = np.array(hf_label)
    print(data.shape, label.shape)

with h5py.File('test_py.h5','r') as hf:
    hf_test_data = hf.get('dataset_1')
    test_data = np.array(hf_test_data)
    hf_test_label = hf.get('dataset_2')
    test_label = np.array(hf_test_label)
    print(test_data.shape, test_label.shape)


X = tf.placeholder("float", [None, 33, 33, 1])
Y = tf.placeholder("float", [None, 33, 33, 1])

W1 = init_weights([9, 9, 1, 64])       # 9x9x1 conv, 64 outputs
W2 = init_weights([1, 1, 64, 32])     # 1x1x64 conv, 32 outputs
W3 = init_weights([5, 5, 32, 1])      # 5x5x32 conv, 1 output

B1 = tf.Variable(tf.zeros([64]), name="Bias1")
B2 = tf.Variable(tf.zeros([32]), name="Bias2")
B3 = tf.Variable(tf.zeros([1]), name="Bias3")

L1 = tf.nn.relu(tf.nn.conv2d(X, W1,                       # l1 shape=(?, 33, 33, 64)
                        strides=[1, 1, 1, 1], padding='SAME') + B1)
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2,                     # l2 shape=(?, 33, 33, 32)
                        strides=[1, 1, 1, 1], padding='SAME') + B2)

hypothesis = tf.nn.conv2d(L2, W3,                     # l3 shape=(?, 33, 33, 1)
                        strides=[1, 1, 1, 1], padding='SAME') + B3
   
cost = tf.reduce_mean(tf.reduce_sum(tf.square(hypothesis - Y), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#plt.imshow(test_label[0,:,:,0], cmap='Greys_r')
#plt.savefig("original_data")
step=0

with tf.Session() as sess:

    tf.initialize_all_variables().run()

    for i in range(train_num):
        sess.run(optimizer, feed_dict={X: data, Y: label})
	   
        step+=1
        if step%100==0 :
		print (step," : ",sess.run(cost, feed_dict={X:data, Y: label }))
		plt.imshow(sess.run(hypothesis, {X: test_data})[0,:,:,0], cmap='Greys_r') # after data
    		subname="shot/"+str(step)
    		plt.savefig(subname)




    print(hypothesis)

#    plt.imshow(test_data[0,:,:,0], cmap='Greys_r')  # origin test_data
#    plt.show()


"""
for i in range(0,len(data)):
    data[i] = data[i].reshape(33, 33, 1)  # 33x33x1 input img
    label[i] = label[i].reshape (33, 33, 1)  # 33x33x1 input img
"""





