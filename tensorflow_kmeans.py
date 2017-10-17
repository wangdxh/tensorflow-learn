#!/usr/bin/python
#coding=utf-8
''' kmeans
plt scatter
http://blog.csdn.net/u013634684/article/details/49646311
http://blog.csdn.net/freedom098/article/details/56021013
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

def gettestdata():
    '''二维正态分布'''
    sampleNo = 100
    mu = np.array([[1, 5]]) # 0-aix mean is1， 1-aix mean is 5
    sigma = np.array([[0, 2], [1, 3]]) # sigma
    #sigma = np.array([[2, 0], [3, 1]]) # sigma
    return np.dot(np.random.randn(sampleNo, 2), sigma) + mu

def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    #Find out the dimensionality
    dim = len(vectors[0])

    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    random.shuffle(vector_indices)

    #GRAPH OF COMPUTATION
    #We initialize a new graph and set it as the default during each run
    #of this algorithm. This ensures that as this function is called
    #multiple times, the default graph doesn't keep getting crowded with
    #unused ops and Variables from previous function calls.

    graph = tf.Graph()

    with graph.as_default():

        #SESSION OF COMPUTATION

        sess = tf.Session()

        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        ##Now lets construct the node that will compute the mean
        #The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)

        ##Node for computing Euclidean distances
        #Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
        

        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        #Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        ##INITIALIZING STATE VARIABLES

        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.global_variables_initializer()

        #Initialize all variables
        sess.run(init_op)

        ##CLUSTERING ITERATIONS

        #Now perform the Expectation-Maximization steps of K-Means clustering
        #iterations. To keep things simple, we will only do a set number of
        #iterations, instead of using a Stopping Criterion.
        noofiterations = 5
        for iteration_n in range(noofiterations):

            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            #Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                #Compute Euclidean distance between this vector and each
                #centroid. Remember that this list cannot be named
                #'centroid_distances', since that is the input to the
                #cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            ##MAXIMIZATION STEP
            #Based on the expected state computed from the Expectation Step,
            #compute the locations of the centroids so as to maximize the
            #overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: np.array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

def main(_):
    '''main '''
    pass

if __name__ == '__main__':
    #main(0)
    srcdata = gettestdata()
    center, result = TFKMeansCluster(srcdata, 4)
    print(center)
    print(result)
    '''result = np.array(result)
    shape = result.shape
    z1 = result.reshape(shape[0], 1)
    color = 50 * np.column_stack((z1, z1[:, 0], z1[:, 0], z1[:, 0]))'''
    '''color = []
    for item in result:
        if item == 0:
            color.append('b')
        elif item == 1:
            color.append('g')
        elif item == 2:
            color.append('b')
        elif item == 3:
            color.append('r')'''
    fig = plt.Figure()
    corlist = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    color = [corlist[item] for item in result]
    plt.scatter(srcdata[:, 0], srcdata[:, 1], marker='o', c=color, s=30)
    x = [item[0] for item in center]
    y = [item[1] for item in center]
    plt.scatter(x, y, marker='x', c='m', s = 60)
    plt.ioff()
    plt.show()
    print('xxxxxxxxxxxxxxxxxxxx')
    fig.clear()
    
    #plt.scatter(srcdata[:, 0], srcdata[:, 1], marker='o', c=color, s=30)
    plt.scatter(x, y, marker='x', c='k', s = 10)
    time.sleep(1000)