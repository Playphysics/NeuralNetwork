# NeuralNetwork
简单的手写数字识别，未使用第三方库；准确率：训练集98.97%，测试集97.27%

## 数据集
http://yann.lecun.com/exdb/mnist/

使用数据集时，需要修改mainFunc.cpp文件void PredictTest()函数的四个路径

## 算法流程
参考：https://www.bilibili.com/video/BV1G441177oz @大野喵渣

此代码为上述视频的Cpp版本，主要公式: Y = softmax(A2 tanh(A1 X + B1) + B2)

具体实现均为朴素算法

## 公式推导

见 公式推导.pdf 矩阵函数的章节

细节原理见知乎：  
矩阵求导术（上）https://zhuanlan.zhihu.com/p/24709748  
矩阵求导术（下）https://zhuanlan.zhihu.com/p/24863977


