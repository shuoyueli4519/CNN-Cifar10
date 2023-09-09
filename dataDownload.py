"""
    基础介绍:cifar-10数据集包含10个类别的32x32三通道图像,每个类别6000个图像,总计60000张图片,
    数据集分为5个训练批次和1个测试批次,每个批次10000张图像。

    库介绍:imageio库是一个用于读写图像数据的库,可以读取和写入多种格式的图像数据,包括numpy数组。
    pickle库是一个用于序列化和反序列化数据的库,可以将数据序列化为二进制数据,也可以将二进制数据反序列化为数据。
"""
from imageio import imwrite
import numpy as np
import os
import pickle

"""
    @param base_dir:基础路径,即linux下的当前路径
    @param data_dir:数据集路径,即cifar-10-batches-py文件夹路径,其中的数据使用了pickle序列化
    @param train_dir:训练集路径,即反序列化cifar-10-batches-py文件夹后得到的训练集路径
    @param test_dir:测试集路径,即反序列化cifar-10-batches-py文件夹后得到的测试集路径
"""
base_dir = './'
data_dir = os.path.join(base_dir, 'Data', 'cifar-10-batches-py')
train_dir = os.path.join(base_dir, 'Data', 'cifar-10-png', 'raw_train')
test_dir = os.path.join(base_dir, 'Data', 'cifar-10-png', 'raw_test')

"""
    @params Train:反序列化训练集或者测试集,如果为True,则反序列化训练集,否则反序列化测试集
"""
Train = False

"""
    反序列化函数,将pickle序列化的cifar-10-batches数据集反序列化为python数据并且保存为字典
"""
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

"""
    如果没有文件路径,添加相关文件路径
"""
def my_mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    if Train:
        for j in range(1, 6):
            #反序列化data_batch_1~data_batch_5
            data_path = os.path.join(data_dir, 'data_batch_' + str(j))
            train_data = unpickle(data_path)
            print(data_path + ' is loading...')

            for i in range(0, 10000):
                #将序列化的数据转换为图片,由于序列化的数据是一维的,所以需要将其转换为三维的
                #train_data[b'data'][i]表示第i张图片的数据,train_data[b'labels'][i]表示第i张图片的标签
                #而且由于反序列化所以需要加上b,表示bytes类型
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)

                label_num = str(train_data[b'labels'][i])
                o_dir = os.path.join(train_dir, label_num)
                my_mkdir(o_dir)

                img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'
                img_path = os.path.join(o_dir, img_name)
                imwrite(img_path, img)
            print(data_path + " loaded.")   
    
    else:
        print("test_batch is loading...")

        #反序列化test_batch
        test_data_path = os.path.join(data_dir, "test_batch")
        test_data = unpickle(test_data_path)
        for i in range(0, 10000):
            #同训练集操作将序列化的数据转换为图片
            img = np.reshape(test_data[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)

            label_num = str(test_data[b'labels'][i])
            o_dir = os.path.join(test_dir, label_num)
            my_mkdir(o_dir)

            img_name = label_num + '_' + str(i) + '.png'
            img_path = os.path.join(o_dir, img_name)
            imwrite(img_path, img)

        print("test_batch loaded.")