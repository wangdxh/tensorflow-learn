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

DIR_MY_FACE = './image/my_faces'
DIR_PEOPLE_FACE = './image/people_faces'

DIR_TEST_MY_FACE = './image/test_my_faces'
DIR_TEST_PEOPLE_FACE = './image/test_people_faces'
IMGSIZE = 64

createdir(DIR_MY_FACE, DIR_PEOPLE_FACE, DIR_TEST_PEOPLE_FACE)

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
        for fileitem in getfilesinpath(inputdir):
            getface(fileitem, outputdir)

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


def main(_):
    ''' main '''
    savepath = './checkpoint/face.ckpt'
    isneedtrain = False
    if os.path.exists(savepath+'.meta') is False:
        isneedtrain = True
    if isneedtrain:
        #first generate all face
        log.debug('generateface')
        generateface([['./image/people_images', DIR_PEOPLE_FACE],
                      ['./image/my_images', DIR_MY_FACE]])
        # then read file and tarin

        train_x, train_y = readimage([[DIR_MY_FACE, [1, 0]], [DIR_PEOPLE_FACE, [0, 1]]])
        train_x = train_x.astype(np.float32) / 255.0
        log.debug('len of train_x : %s', train_x.shape)
        myconv.train(train_x, train_y, savepath)
        log.debug('training is over, please run again')
    else:
        generateface([['./image/test_people_images', DIR_TEST_PEOPLE_FACE]])
        test_x, test_y = readimage([[DIR_TEST_PEOPLE_FACE, [0, 1]]])
        test_x = test_x.astype(np.float32) / 255.0
        log.debug('len of test_x : %s', test_x.shape)
        log.debug('y is %s', test_y)
        log.debug(myconv.validate(test_x, test_y, savepath))

if __name__ == '__main__':
    # first generate all face
    main(0)



