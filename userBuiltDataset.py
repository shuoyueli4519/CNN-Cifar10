"""
    基础介绍:制作自己的数据集
"""
from PIL import Image
from torch.utils.data import Dataset
import csv

"""
    从Dataset中继承来的类,需要重写__getitem__和__len__方法
"""
class MyDataset(Dataset):
    #初始化函数,得到数据,返回一个含有图片位置和图片标签的元组
    def __init__(self, csv_path, transform=None, target_transform=None):
        with open(csv_path, 'r', newline='') as csvfile:
            imgs = []
            for csv_line in csv.reader(csvfile):
                imgs.append((csv_line[0], int(csv_line[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        csvfile.close()

    #重写__getitem__方法,得到图片和标签,self.transform是对图片的处理,可以自己定义
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    #重写__len__方法,返回数据集长度
    def __len__(self):
        return len(self.imgs)
        