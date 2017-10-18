#!/usr/bin/python
#coding=utf-8
''' tf mnist
if can not download from http url, u can download yourself from here:
donwload data http://yann.lecun.com/exdb/mnist/ put in current directory
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
'''
# pylint: disable=invalid-name

import logging as log
import matplotlib.pyplot as plt
import common
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    ''' main '''
    imagesize = 28*28
    mnist = input_data.read_data_sets('./', one_hot=True)

    x_data = tf.placeholder(tf.float32, [None, imagesize])
    theta = tf.Variable(tf.zeros([imagesize, 10]))
    bias = tf.Variable(tf.zeros([10]))
    #comput y
    y = tf.matmul(x_data, theta) + bias

    y_data = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(\
                              tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y))

    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    batch_xdata, batch_ydata = mnist.test.next_batch(2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            batch_xdata, batch_ydata = mnist.train.next_batch(100)
            sess.run(train, feed_dict={x_data:batch_xdata, y_data:batch_ydata})

        # test from my draw

        test_all = 1
        if test_all == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
            accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            log.debug(sess.run(accuray,
                               feed_dict={x_data:mnist.test.images, y_data:mnist.test.labels}))
        elif test_all == 1:
            batch_xdata, batch_ydata = mnist.test.next_batch(2)
            log.debug(batch_ydata)
            log.debug(sess.run(y, feed_dict={x_data:batch_xdata, y_data:batch_ydata}))
            log.debug(sess.run(tf.argmax(y, 1), feed_dict={x_data:batch_xdata, y_data:batch_ydata}))

            batch_xdata = batch_xdata.reshape(2, 28, 28)
            plt.subplot(1, 2, 1)
            plt.imshow(batch_xdata[0])
            plt.subplot(1, 2, 2)
            plt.imshow(batch_xdata[1])
            common.blockplt()
        elif test_all == 2:
            myimg = common.getimgdata('./image/number3.jpg')
            ndimg = myimg.reshape(1, 28*28)
            log.debug(sess.run([y, tf.argmax(y, 1)], feed_dict={x_data:ndimg}))
            plt.imshow(myimg)
            common.blockplt()
if __name__ == '__main__':
    main()
    #myimg = getimgdata('./mydrawnum3.bmp')
