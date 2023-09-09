import pickle
import torch
from netModel import Net
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
sum = 0

for folder_name in range(0, 10):
    folder_name = str(folder_name)
    folder_path = './Data/cifar-10-png/raw_test/' + folder_name
    file_names = os.listdir(folder_path)
    image_names = []
    for file_name in file_names:
        if glob.fnmatch.fnmatch(file_name, '*.png'):
            image_names.append(file_name)

    correct = 0
    total = 0

    for i in range(0, len(image_names)):
        """
            @model:网络模型
            @img:测试图片
            @normMean:归一化均值
            @normStd:归一化方差
            @normTransform:总归一化
            @transform:测试集归一化,不包括RamdomCrop,因为测试集不需要数据增强,保证稳定性
        """
        model = torch.load('./Result/07-11_20-56-03/net_params.pkl')
        img = Image.open('./Data/cifar-10-png/raw_test/' + folder_name + '/' + image_names[i]).convert('RGB')
        normMean = [0.4948052, 0.48568845, 0.44682974]
        normStd = [0.24580306, 0.24236229, 0.2603115]
        normTransform = transforms.Normalize(normMean, normStd)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        """
            @img_tensor:测试图片的tensor
            @batch_size:批处理大小
            @img_tensor:扩展为4维张量
            @tensor:预测结果
            @max:最大概率
        """
        img_tensor = transform(img)
        batch_size = 1
        img_tensor = img_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
        tensor = model(img_tensor)
        index = torch.max(tensor, 1)[1].squeeze().numpy()
        if index == int(folder_name):
            correct += 1
        total += 1
    sum += correct / total
    print("{}数据集的准确率为:{:.6f}".format(classes_name[int(folder_name)], correct / total))
print("平均准确率为:{}".format(sum / len(classes_name)))