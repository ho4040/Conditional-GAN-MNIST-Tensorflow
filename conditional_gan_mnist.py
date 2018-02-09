import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
import time

def generator(z_inputs, y_inputs, reuse=False, training=True, color_channel=3):
	
	with tf.device('/device:GPU:0'):
		s_size = 4
		initializer = tf.random_normal_initializer(mean=0, stddev=0.02)

		with tf.variable_scope('g', reuse=reuse):

			z = tf.convert_to_tensor(inputs)

			# z vector convert into tensor
			#outputs = tf.layers.dense(z, s_size * s_size * 1024)
			#outputs = tf.nn.relu(outputs) # shape (batch_size, 4, 4, 1024)
			
			# Both z and y are mapped to hidden layers with Rectified Linear Unit (ReLu) activation [4, 11]
			# shape of z = (None, 100)
			z_layer = tf.layers.dense(z, 200, activation=tf.nn.relu)
			z_layer = tf.layers.dense(z_layer, 200, activation=tf.nn.relu)
			z_layer = tf.layers.dense(z_layer, 200, activation=tf.nn.relu)
			z_layer = tf.layers.dense(z_layer, 200, activation=tf.nn.relu)
			
			
			y_layer = tf.layers.dense(y_inputs, 1000)
			
			
			# tanh output
			g = tf.nn.tanh(tconv4, name='generator') # output shape = (batch_size, 64, 64, 3)
			#g = tf.nn.sigmoid(tconv4)
			return g

def discriminator(inputs, reuse=False, training=True):
	
	with tf.device('/device:GPU:0'):

		initializer = tf.random_normal_initializer(mean=0, stddev=0.02)

		with tf.variable_scope('d', reuse=reuse):
			d_inputs = tf.convert_to_tensor(inputs)

			# conv 1
			conv1 = tf.layers.conv2d(d_inputs, 64, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv1 = tf.layers.batch_normalization(conv1, training=training)
			conv1 = tf.nn.leaky_relu(conv1) # shape (batch_size, 32, 32, 64)

			# conv 2
			conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv2 = tf.layers.batch_normalization(conv2, training=training)
			conv2 = tf.nn.leaky_relu(conv2) # shape (batch_size, 16, 16, 128)

			# conv 3
			conv3 = tf.layers.conv2d(conv2, 256, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv3 = tf.layers.batch_normalization(conv3, training=training)
			conv3 = tf.nn.leaky_relu(conv3) # shape (batch_size, 4, 4, 256)

			# conv 4
			conv4 = tf.layers.conv2d(conv3, 512, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv4 = tf.layers.batch_normalization(conv4, training=training)
			conv4 = tf.nn.leaky_relu(conv4) # shape (batch_size, 2, 2, 512)

			batch_size = conv4.get_shape()[0].value

			reshape = tf.reshape(conv4, [batch_size, -1])
			d = tf.layers.dense(reshape, 1, name='d')
			d = tf.nn.sigmoid(reshape)
			return d


def build_graph(z_dim = 100, batch_size = 128, test_result_num=(8,2), input_image_shape=(28,28,1)):
	
	img_w = input_image_shape[0]
	img_h = input_image_shape[1]
	color_channel = input_image_shape[2]
	
	with tf.device('/device:GPU:0'):
		
		
		# discriminator value from fake image
		z = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0)
		fake_data = generator(z, reuse=False, training=True, color_channel=color_channel)
		d_f = discriminator(fake_data, reuse=False, training=True)
		
		# discriminator value from real image
		real_data_input = tf.placeholder(shape=[batch_size, None], dtype=tf.float32)		
		real_images = tf.reshape(real_data_input, (batch_size, img_w, img_h, color_channel)) # convert raw data to valid image data
		real_images = tf.image.resize_images(real_images, (64,64)) # change size as D input
		d_r = discriminator(real_images, reuse=True, training=True)
		
		# loss of D and G
		d_loss = -tf.reduce_mean( tf.log(d_r) + tf.log(1-d_f))  # Minus sign for Adam optimizer that only can minimize.
		g_loss = tf.reduce_mean(tf.log(1-d_f))  
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # turn on batchorm
		with tf.control_dependencies(update_ops):
			opt_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
			d_train = opt_d.minimize(d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d'))
			opt_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
			g_train = opt_g.minimize(g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g'))
		
	row = test_result_num[1]
	col = test_result_num[0]
	
	gen_size = row*col
	gen_inputs = tf.random_uniform([gen_size, z_dim], minval=-1.0, maxval=1.0)
	gen_images = generator(gen_inputs, reuse=True, training=False, color_channel=color_channel)
	gen_images = tf.image.convert_image_dtype(tf.div(tf.add(gen_images, 1.0), 2.0), tf.uint8)
	gen_images = [img for img in tf.split(gen_images, gen_size, axis=0)]
	rows = []
	for i in range(0, row, 1):
		rows.append(tf.concat(gen_images[col * i + 0:col * i + col], 2))
	gen_fakes = tf.concat(rows, 1)
	
	
	return g_train, d_train, d_loss, g_loss, real_data_input, gen_fakes, real_images

def main():
	# download mnist data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	batch_size = 128

	g_train, d_train, d_loss, g_loss, real_data_input, gen_fakes, real_images = build_graph( z_dim = 100,  batch_size = batch_size, test_result_num=(8,2), input_image_shape=(28,28,1))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	start_time = time.time()

	batches = mnist.train.next_batch(batch_size)[0]
	_fakes, _real = sess.run([gen_fakes, real_images], feed_dict={real_data_input:batches})

	#plt.axis('off')
	# plt.figure(figsize=(8,1))	
	# plt.subplot(121)
	# plt.imshow(np.squeeze(np.squeeze(_real[0]*255)).astype(np.uint8), cmap='gray')
	# plt.subplot(122)
	# plt.imshow(np.squeeze(np.squeeze(_fakes[0])).astype(np.uint8), cmap='gray')
	# plt.show()


	k = 1
	for i in range(50000):
		
		batches = mnist.train.next_batch(batch_size)[0]
		sess.run([d_train], feed_dict={real_data_input:batches})
		sess.run([g_train], feed_dict={real_data_input:batches})

		if i % 100 == 0 :
			print("--- step %d,  %s seconds ---" % (i, time.time() - start_time))
			_fakes = sess.run([gen_fakes], feed_dict={real_data_input:batches})
			plt.figure(figsize=(8,4))	
			plt.axis('off')
			plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
			plt.imshow(np.squeeze(_fakes[0][0]).astype(np.uint8), cmap='gray')
			plt.show()


main()