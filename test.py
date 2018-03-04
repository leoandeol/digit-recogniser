# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:10:03 2018

@author: leoan
"""

import numpy as np
from PIL import Image

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

liens = []

im = Image.open("homemade_test/eight.png")
im_array = np.reshape(np.array(im.convert('L')), [1, 784])/255
liens.append([im_array, 8])
im = Image.open("homemade_test/four.png")
im_array2 = np.reshape(np.array(im.convert('L')), [1, 784])/255
im_array2 = 1 - im_array2
liens.append([im_array2, 4])
im = Image.open("homemade_test/nine.png")
im_array3 = np.reshape(np.array(im.convert('L')), [1, 784])/255
im_array3 = 1-im_array3
liens.append([im_array3, 9])
im = Image.open("homemade_test/seven.png")
im_array4 = np.reshape(np.array(im.convert('L')), [1, 784])/255
im_array4 = 1 - im_array4
liens.append([im_array4, 7])
im = Image.open("homemade_test/two.png")
im_array5 = np.reshape(np.array(im.convert('L')), [1, 784])/255
im_array5 = 1 - im_array5
liens.append([im_array5, 2])

with tf.Session() as sess:
    saver.restore(sess=sess, save_path="model.ckpt")
    for couple in liens:
        Z = logits.eval(feed_dict={X: couple[0]})
        y = np.argmax(Z, axis=1)
        print("Valeur : ",couple[1],"& prédit : ",y[0])