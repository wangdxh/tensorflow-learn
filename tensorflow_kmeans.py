#!/usr/bin/python
#coding=utf-8
''' kmeans
plt scatter
http://blog.csdn.net/u013634684/article/details/49646311
http://blog.csdn.net/freedom098/article/details/56021013
http://download.csdn.net/download/u011433684/9709997

'''
# pylint: disable=invalid-name

import os
import time
import random
import logging as log
import matplotlib.pyplot as plt
import common
import numpy as np
from numpy.linalg import cholesky
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def gettestdata(sampleNo):
    '''二维正态分布'''
    mu = np.array([[1, 5]]) # 0-aix mean is1， 1-aix mean is 5
    sigma = np.array([[0, 2], [1, 3]]) # sigma
    #sigma = np.array([[2, 0], [3, 1]]) # sigma
    return np.dot(np.random.randn(sampleNo, 2), sigma) + mu

K = 4 # 类别数目
MAX_ITERS = 100 # 最大迭代次数
M = 300 # 样本点数目

def clusterMean(data, nid, num):
    ''' cluster mean'''
    total = tf.unsorted_segment_sum(data, nid, num) # 第一个参数是tensor，第二个参数是簇标签，第三个是簇数目
    count = tf.unsorted_segment_sum(tf.ones_like(data), nid, num)
    return total/count

def dothejob():
    ''' 构建graph'''
    data = gettestdata(M)

    points = tf.Variable(data)
    cluster = tf.Variable(tf.zeros([M], dtype=tf.int64))
    centers = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))# 将原始数据前k个点当做初始中心

    repCenters = tf.reshape(tf.tile(centers, [M, 1]), [M, K, 2]) # 复制操作，便于矩阵批量计算距离
    repPoints = tf.reshape(tf.tile(points, [1, K]), [M, K, 2])
    sumSqure = tf.reduce_sum(tf.square(repCenters-repPoints), reduction_indices=2) # 计算距离 m*k
    bestCenter = tf.argmin(sumSqure, axis=1)  # 寻找最近的簇中心 牛逼啊真是人才

    change = tf.reduce_any(tf.not_equal(bestCenter, cluster)) # 检测簇中心是否还在变化
    means = clusterMean(points, bestCenter, K)  # 计算簇内均值 great job
    # 将簇内均值变成新的簇中心，同时分类结果也要更新
    with tf.control_dependencies([change]):
        update = tf.group(centers.assign(means), cluster.assign(bestCenter)) # 复制函数

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        changed = True
        iterNum = 0
        while changed and iterNum < MAX_ITERS:
            iterNum += 1

            [changed, _] = sess.run([change, update])
            [centersArr, clusterArr] = sess.run([centers, cluster])

            common.showkmeansresult(data, centersArr, clusterArr, str(iterNum))
    common.blockplt()

def testtile():
    ''' test tile'''
    temp = tf.tile([1, 2, 3], [2])
    temp2 = tf.tile([[1, 2], [3, 4], [5, 6]], [2, 3])
    temp3 = tf.square(temp2)
    temp4 = tf.reshape(temp2, [6, 3, 2])
    temp5 = tf.reduce_sum(temp4, reduction_indices=2)# the index demension will be and to one number
    temp6 = tf.argmin(temp5, axis=1)
    total = tf.unsorted_segment_sum([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], [0, 1, 2, 0, 1], 3)

    data = tf.Variable([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    count = tf.unsorted_segment_sum(tf.ones_like(data.initialized_value()), [0, 1, 2, 0, 1], 3)
    with tf.Session() as sess:
        log.debug(sess.run(temp))
        log.debug(sess.run(temp2))
        log.debug(sess.run(temp3))
        log.debug(sess.run(temp4))
        log.debug(sess.run(temp5))
        log.debug('%s %s', sess.run(temp6), type(sess.run(temp6)))
        log.debug(sess.run(total))
        log.debug(sess.run(count))
        log.debug(sess.run(total/count))

if __name__ == '__main__':
    dothejob()
    #testtile()
