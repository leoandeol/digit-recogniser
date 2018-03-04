# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Algo génétique todo
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("C:\\Users\\leoan\\Downloads")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

# cleaning

tf.reset_default_graph()

# construction

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
y = tf.placeholder(tf.int64, shape=[None], name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    
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

# execution

n_epochs = 40
batch_size = 50
n_batches_per_epoch = mnist.train.num_examples // batch_size
timer = time.clock()
beginning = timer
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches_per_epoch):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        diff = time.clock()-timer
        timer = time.clock()
        print(epoch,"Time elapsed","{0:.2f}".format(diff), "Train accuracy", acc_train, "Test accuracy", acc_test)
        
    save_path = saver.save(sess, "./model.ckpt")
end = time.clock()
total = end - beginning
print("Total duration",total,"s")
# =============================================================================
# 
# cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# dnn_clf_tflearn = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], 
#                                                  n_classes=10, feature_columns=cols)
# dnn_clf = tf.contrib.learn.SKCompat(dnn_clf_tflearn)
# dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
# 
# from sklearn.metrics import accuracy_score
# 
# y_pred = dnn_clf.predict(X_test)
# 
# print("\n\n\nResultats : ")
# print(accuracy_score(y_test, y_pred['classes']))
# =============================================================================
