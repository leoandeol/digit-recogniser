# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:35:18 2018

@author: leoan
"""

import numpy as np
import cv2
import tensorflow as tf
import scipy

# window variables

running = True
drawing = False
last_x, last_y = -1, -1

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


def draw(event, x, y, flags , param):
    global last_x, last_y, drawing, img3
    # dessin
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (last_x, last_y), (x, y), (255,255,255), 5)
            last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (last_x, last_y), (x, y), (255,255,255), 5)
    #prédiction
    if event == cv2.EVENT_LBUTTONUP:
        resized = scipy.misc.imresize(np.resize(img, [280, 280]), 10, mode='L')
        redimensioned = np.reshape(resized, [1, 784])
        img3 = resized
        img3 = np.reshape(img3, [28, 28, 1])
        final = redimensioned/255
        with tf.Session() as sess:
            saver.restore(sess=sess, save_path="model.ckpt")
            Z = logits.eval(feed_dict={X: final})
            y = np.argmax(Z, axis=1)
            cv2.putText(img2,str(y[0]), (5,250), cv2.FONT_HERSHEY_SIMPLEX, 10, 255)
        
img = np.zeros((280, 280, 1), np.uint8)
img2 = np.zeros((280, 280, 1), np.uint8)
img3 = np.zeros((28, 28, 1), np.uint8)
cv2.namedWindow('image')
cv2.namedWindow('predictions')
cv2.namedWindow('transformed')
cv2.setMouseCallback('image',draw)

while(running):
    cv2.imshow('image', img)
    cv2.imshow('predictions', img2)
    cv2.imshow('transformed', img3)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        running = False
    elif k == ord('r'):
        img = np.zeros((280, 280, 1), np.uint8)
        img2 = np.zeros((280, 280, 1), np.uint8)

cv2.destroyAllWindows()