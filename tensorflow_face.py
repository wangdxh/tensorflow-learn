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

def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

DIR_MYFACE = './image/my_faces'
DIR_PEOPLEFACE = './image/people_faces'

createdir(DIR_MYFACE, DIR_PEOPLEFACE)

def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage(imgpath, h=64, w=64):
    ''' dealwithimage '''
    img = cv2.imread(imgpath)
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
        face = cv2.resize(face, (64, 64))
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

def main(_):
    ''' main '''
    savepath = './checkpoint/face.ckpt'
    isneedtrain = False
    if os.path.exists(savepath+'.meta') is False:
        isneedtrain = True
    if isneedtrain:
        #first generate all face
        log.debug('generateface')
        generateface([['./image/people_images', './image/people_faces'],
                      ['./image/my_images', './image/my_faces']])
        # then read file and tarin

    #restore session and judge file

if __name__ == '__main__':
    # first generate all face
    main(0)
    


