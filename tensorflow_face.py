#!/usr/bin/python
#coding=utf-8
''' face detect
https://github.com/seathiefwang/FaceRecognition-tensorflow
http://tumumu.cn/2017/05/02/deep-learning-face/

python opencv whl
http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
pip install xxxx.whl

other people's faces
图片集下载:http://vis-www.cs.umass.edu/lfw/lfw.tgz
'''
# pylint: disable=invalid-name
import os
import logging as log
import matplotlib.pyplot as plt
import common
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import tensorflow_face_conv as myconv

def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64


def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img

def relight(imgsrc, alpha=1, bias=0):
    '''relight'''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getface(imgpath, outdir):
    ''' get face from path file'''
    filename = os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath)
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    n = 0
    for f_x, f_y, f_w, f_h in faces:
        n += 1
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]
        # may be do not need resize now
        #face = cv2.resize(face, (64, 64))
        face = dealwithimage(face, IMGSIZE, IMGSIZE)
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            facetemp = relight(face, alpha, bias)
            cv2.imwrite(os.path.join(outdir, '%s_%d_%d.jpg' % (filename, n, inx)), facetemp)

def getfilesinpath(filedir):
    ''' get all file from file directory'''
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))

def generateface(pairdirs):
    ''' generate face '''
    for inputdir, outputdir in pairdirs:
        for name in os.listdir(inputdir):
            inputname, outputname = os.path.join(inputdir, name), os.path.join(outputdir, name)
            if os.path.isdir(inputname):
                createdir(outputname)
                for fileitem in getfilesinpath(inputname):
                    getface(fileitem, outputname)

def readimage(pairpathlabel):
    '''read image to list'''
    imgs = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in getfilesinpath(filepath):
            img = cv2.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)

def onehot(numlist):
    ''' get one hot return host matrix is len * max+1 demensions'''
    b = np.zeros([len(numlist), max(numlist)+1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

def getfileandlabel(filedir):
    ''' get path and host paire and class index to name'''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
                    #for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])

    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))

    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))

def main(_):
    ''' main '''
    savepath = './checkpoint/face.ckpt'
    isneedtrain = False
    if os.path.exists(savepath+'.meta') is False:
        isneedtrain = True
    if isneedtrain:
        #first generate all face
        log.debug('generateface')
        generateface([['./image/trainimages', './image/trainfaces']])
        pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')

        train_x, train_y = readimage(pathlabelpair)
        train_x = train_x.astype(np.float32) / 255.0
        log.debug('len of train_x : %s', train_x.shape)
        myconv.train(train_x, train_y, savepath)
        log.debug('training is over, please run again')
    else:
        testfromcamera(savepath)
        
        #print(np.column_stack((out, argmax)))

def testfromcamera(chkpoint):
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')
    output = myconv.cnnLayer(len(pathlabelpair))
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, chkpoint)
        

        n = 1
        while 1:
            if (n <= 20000):
                print('It`s processing %s image.' % n)
                # 读帧
                success, img = camera.read()

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for f_x, f_y, f_w, f_h in faces:
                    face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                    #could deal with face to train
                    test_x = np.array([face])
                    test_x = test_x.astype(np.float32) / 255.0
                    
                    res = sess.run([predict, tf.argmax(output, 1)],\
                                   feed_dict={myconv.x_data: test_x,\
                                   myconv.keep_prob_5:1.0, myconv.keep_prob_75: 1.0})
                    print(res)

                    cv2.putText(img, indextoname[res[1][0]], (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    n+=1
                cv2.imshow('img', img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    break
            else:
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # first generate all face
    main(0)
    #onehot([1, 3, 9])
    #print(getfileandlabel('./image/trainimages'))
    #generateface([['./image/trainimages', './image/trainfaces']])

