# 项目名称:PytorchCifar10
## 1.项目介绍
使用Cifar10数据集制作图像识别卷积神经网络小模型，作为深度学习新手入门，参考了https://github.com/tensor-yu/PyTorch_Tutorial/tree/master/Data

## 2.项目文件概要
- envCheck.py主要实现以下功能:相关的硬件及环境的检验函数
- dataDownload.py主要实现以下功能呢个:将cifar-10数据集从pickle压缩后的batch格式转化为png格式 **(注意:准备工作要从cifar-10官网上将cifar-10-batches-py.tar.gz下载下来解压后放入./Data路径下)**
- dataSplit.py主要实现raw_test中10000张图片划分，将10000张图片打乱并且按训练集:验证集:测试集=8:1:1分割成三部分，用于后续模型训练 **(没有使用raw_train中的图片是因为图片数量扁多，训练时间长，可以将raw_test换成raw_train以达到更好的效果。)**
- dataForm.py主要实现将数据集中的图片制作出对应的数据-标签表单
- userBuiltDtatset.py主要实现自建数据集并写出对应的类
- netModel.py主要实现建立一个对3*32*32图片处理的卷积神经网络类
- main.py主要实现卷积神经网络对数据集的分类和建立模型并且保存，查看loss曲线和accuracy曲线使用**tensorboard --logdir=./Result**命令后打开浏览器的**http://localhost:6006**
- result.py主要实现测试集的检验，检验模型的准确度