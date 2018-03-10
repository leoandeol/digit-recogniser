# -*- coding: utf-8 -*-

import tensorflow as tf

# cleaning
tf.reset_default_graph()

# construction
#n_inputs = 28*28
n_hidden1 = 500
n_hidden2 = 200
n_hidden3 = 50
n_outputs = 10


X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
y = tf.placeholder(tf.int64, shape=[None], name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):# Input Layer
    input_layer = tf.reshape(X, [-1, 28, 28, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=training == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    # conv1 = tf.layers.conv2d(X, filters=2, kernel_size=5, strides=[2,2], padding="SAME")
    # hidden1 = tf.layers.dense(conv1, n_hidden1, name="hidden1", activation=tf.nn.relu)
    # hidden1drop = tf.layers.dropout(hidden1, rate=0.5, training=training)
    # hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    # hidden2drop = tf.layers.dropout(hidden2, rate=0.5, training=training)
    # hidden3 = tf.layers.dense(hidden1, n_hidden3, name="hidden3", activation=tf.nn.relu)
    # hidden3drop = tf.layers.dropout(hidden3, rate=0.5, training=training)
    # logits = tf.layers.dense(hidden3drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()