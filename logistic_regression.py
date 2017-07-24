""" Simple logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
Extra comments plus change in test aggregation code by Matt Goldrick
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01 #roughly, size of change in weights along gradient
batch_size = 128 #number of examples to include in each batch of training
n_epochs = 30 # number of times to draw a batch from training set

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
# "one hot" = vector consists of a single 1, remainder are 0
mnist = input_data.read_data_sets('/data/mnist', one_hot=True) 

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# each label is a one-hot 1x10 tensor.
# note that placeholders are set to size of entire batch of training data 
# allows batch calculation of gradient
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y to insure that Y = tf.matmul(X, w) (see Step 4)
# alternatively, think of each of the 784 input units connected to each of 10 output units
# shape of b depends on Y (a bias on each output unit)
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(X, w) + b 

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
# cross entropy is the number of bits required to specify deviation 
# of current probability distribution (network output) from the true distribution (labels)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	#include time so we can see how awesome batch processing is    
	start_time = time.time()
    
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size) #divide training data into batches
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			# draw batch from training set and train            
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	
	preds = tf.nn.softmax(logits) # prediction is exponential of logit 
	# get list of booleans: is highest probability digit e
	#qual to the actual training label?
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1)) 
	# convert boolean to float and get mean = percentage correct
	accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))   
    
	#divide test set into batches
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += accuracy_batch
	
	print('Accuracy {0}'.format(total_correct_preds/n_batches))

	writer.close()

# To view graph
#tensorboard --logdir='./graphs'