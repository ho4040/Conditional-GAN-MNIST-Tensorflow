import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse), tf.device('/device:GPU:0'):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, 784, kernel_initializer=w_init)
        g = tf.nn.tanh(dense2)

        return g

    
# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse), tf.device('/device:GPU:0'):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        lrelu1 = tf.nn.leaky_relu(dense1)

        dense2 = tf.layers.dense(lrelu1, 1, kernel_initializer=w_init)
        d = tf.nn.sigmoid(dense2)

        return d, dense2

onehot = np.eye(10) # make I matrix
# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1~1 from 0~1
train_label = mnist.train.labels
# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 100
# variables : input
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))
z = tf.placeholder(tf.float32, shape=(None, 100))
#z = tf.random_uniform([batch_size, 100], minval=-1.0, maxval=1.0)
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y, isTrain)
# networks : discriminator
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)
with tf.device('/device:GPU:0'):
    # loss for each network
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))

    #D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_real, labels=tf.ones([batch_size, 1])))
    #D_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros([batch_size, 1])))

    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)    

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./cGAN_log', sess.graph)
merged = tf.summary.merge_all()

np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

for epoch in range(train_epoch):
    
    # update discriminator
    for iter in range(len(train_set) // batch_size):
        
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
        y_ = train_label[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 100))

        _ = sess.run([D_optim], {x: x_, y: y_, z: z_, isTrain: True})

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        y_ = np.random.randint(0, 9, (batch_size, 1))
        y_ = onehot[y_.astype(np.int32)].squeeze()
        _ = sess.run([G_optim], {z: z_, x: x_, y: y_, isTrain: True})
        
    _summ = sess.run([merged], {z: z_, x: x_, y: y_, isTrain: True})
    writer.add_summary(_summ[0], epoch)
    if epoch % 10 == 0:
        print("epoch {}".format(epoch))

# TEST witht label

test_num = 9
for i in range(10):    
    z_ = np.random.normal( 0, 1, (test_num, 100) )
    y_ = [onehot[i]] * test_num
    
    _fakes = sess.run(fake_images, {z:z_, y:y_, isTrain:False})
    _fakes = np.reshape(_fakes, (test_num,28,28))
    
    plt.figure(figsize=(10,1))
    for k in range(test_num):
        plt.subplot(''.join(str(x) for x in [1,test_num,k+1]))
        plt.axis('off')
        plt.imshow(_fakes[k], cmap="gray")
    plt.show()
        
