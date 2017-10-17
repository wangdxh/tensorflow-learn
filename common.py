#!/usr/bin/python
#coding=utf-8
''' common 
get image 
http://blog.csdn.net/sparta_117/article/details/66965760
'''
# pylint: disable=invalid-name

import logging as log
from PIL import Image
import numpy as np
log.basicConfig(level=log.DEBUG)

def getimgdata(filepath, size=(28, 28)):
    ''' must be 28*28, rgb24bits ./mydrawnum3.bmp '''
    imgfile = Image.open(filepath)
    if imgfile.size != size:
        imgfile = imgfile.resize(size, Image.BICUBIC)
    imgdata = np.asarray(imgfile.convert('L'))

    log.debug('%s : %s', imgdata.shape, imgdata.dtype)
    imgdata = (255.0 - imgdata) / 255.0

    threshold = 0.0
    if threshold != 0.0:
        imgdata[imgdata > threshold] = 1
        imgdata[imgdata <= threshold] = 0

    return imgdata

if __name__ == '__main__':
    log.debug(getimgdata('number9.jpg'))