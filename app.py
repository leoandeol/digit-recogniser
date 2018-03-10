# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:35:18 2018

@author: leoan
"""

import numpy as np
import cv2
import tensorflow as tf
import scipy

from build import *

# window variables

running = True
drawing = False
last_x, last_y = -1, -1

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
        img3 = resized
        img3 = np.reshape(img3, [28, 28, 1])
        final = np.reshape(resized, [1,28,28,1])/255
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