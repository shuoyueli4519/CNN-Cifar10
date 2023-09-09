"""
    基础介绍:将raw_test文件夹中的图片按照8:1:1的比例分为训练集,验证集和测试集,目地是减少训练数据集的大小,加快训练速度
    可以对raw_train文件夹中的图片进行同样的操作,但是由于raw_train文件夹中的图片数量较多,所以不进行操作
"""
import os
import glob
import random
import shutil
from dataDownload import my_mkdir

"""
    @param dataset_dir:原始数据集路径,即raw_test文件夹路径
    @param train_dir:导入的训练集路径
    @param valid_dir:导入的验证集路径
    @param test_dir:导入的测试集路径
"""
dataset_dir = os.path.join('.', 'Data', 'cifar-10-png', 'raw_test')
train_dir = os.path.join('.', 'Data', 'train')
valid_dir = os.path.join('.', 'Data', 'valid')
test_dir = os.path.join('.', 'Data', 'test')

"""
    @param train_percent:训练集所占比例
    @param valid_percent:验证集所占比例
    @param test_percent:测试集所占比例
"""
train_percent = 0.8
valid_percent = 0.1
test_percent = 0.1

if __name__ == '__main__':
    #root:当前目录 
    #dirs:当前目录下的文件夹名称列表 
    #files:当前目录下的文件名称列表
    for root, dirs, files in os.walk(dataset_dir):

        #sdirs:当前目录下的文件夹名称,即0-9
        for sdirs in dirs:
            #img_list:0-9文件夹中的图片名称列表
            img_list = glob.glob(os.path.join(root, sdirs, '*.png'))

            #打乱img_list中的图片顺序
            random.seed(666)
            random.shuffle(img_list)

            #imgs_num:0-9文件夹中的图片数量,即1000
            imgs_num = len(img_list)
            train_point = int(imgs_num * train_percent)
            valid_point = int(imgs_num * (train_percent + valid_percent))

            #将0-799张图片放入训练集文件夹中
            #将800-899张图片放入验证集文件夹中
            #将900-999张图片放入测试集文件夹中
            for i in range(imgs_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sdirs)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sdirs)
                else:
                    out_dir = os.path.join(test_dir, sdirs)
                my_mkdir(out_dir)

                #os.path.split(img_list[i])[-1]:获取图片名称
                out_path = os.path.join(out_dir, os.path.split(img_list[i])[-1])

                #图片拷贝函数
                shutil.copy(img_list[i], out_path)