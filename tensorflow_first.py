#!/usr/bin/python
#coding=utf-8
''' tf first'''

import numpy as np
import tensorflow as tf

def main():
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.3

    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(0, 201):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run([W, b]))

if __name__ == '__main__':
    main()
