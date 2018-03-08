# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Algo génétique todo
generaliser phase construction
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from build import *

# import data

mnist = input_data.read_data_sets("C:\\Users\\leoan\\Downloads")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")


# execution
n_epochs = 100
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