# tensorflow-learn

Building Machine Learning Projects with TensorFlow
book pdf:

链接:http://pan.baidu.com/s/1hsQqZLM  密码:nasm

https://github.com/wangdxh/Building-Machine-Learning-Projects-with-TensorFlow/


vgg data download  

http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

face recognization

使用face_camera先进行不同的人脸采集，每次采集先根据提示输入不同人的名称，然后采集一会，会根据采集的人数，生成输出的种类。

第一次运行face，会进行train，也是根据人数的目录去进行face train，会将训练结果保存到checkpoint下

第二次运行face，进行test，会运行摄像头采集图像，前面采集过的人，会将其名称标注在头像上

### 问题

* weight 和 bias 的初始化好像有些问题，随机初始化会造成在某些情况下cost很大，梯度下不去，导致train结果很差.

* 输出的种类数目是根据采集的人数去动态变化，但是没有给陌生人预留class，所以结果肯定在某个采集的人中，区别不出陌生人来
