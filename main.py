import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from datetime import datetime
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np

from userBuiltDataset import MyDataset
from netModel import Net
from netModel import ResNet

"""
    建立tensorboardX的SummaryWriter对象,用于写入数据
    并且建立文件夹,用于存放数据
"""
result_dir = os.path.join('.', 'Result')
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

"""
    @max_epoch:最大迭代次数
    @classes_name:类别名
    @normMean:归一化均值
    @normStd:归一化方差
    @normTransform:总归一化
    @trainTransform:训练集归一化,包括RamdomCrop的随机裁减并并在图像边缘填充4个像素
    @validTransform:验证集归一化,不包括RamdomCrop,因为验证集不需要数据增强,保证稳定性
    @train_data:训练集
    @valid_data:验证集
    @train_loader:训练集加载器
    @valid_loader:验证集加载器
    @net:网络模型
    @criterion:损失函数
    @optimizer:优化器
"""
max_epoch = 50
classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

train_data = MyDataset(csv_path='./Data/train.csv', transform=trainTransform)
valid_data = MyDataset(csv_path='./Data/valid.csv', transform=validTransform)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=16)

net = ResNet()
net.initialize_weights()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

"""
    开始训练
"""
for epoch in range(max_epoch):
    """
        @loss_sigma:损失总和
        @correct:正确的个数
        @total:总个数
    """
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0

    """
        @i:当前遍历的 batch 的索引
        @data:当前batch的输入数据和对应的标签
    """
    for i, data in enumerate(train_loader):
        """
            @inputs:输入,即图像张量
            @labels:标签
            @outputs:经过网络前向传播函数的输出
            @loss:损失,由criterion计算
            @optimizer.zero_grad():梯度清零,防止梯度累加,即上次的梯度影响了这次的训练
            @loss.backward():反向传播,计算梯度
            @optimizer.step():更新参数,即loss.backward()计算的梯度更新参数
        """
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """
            @predicted:预测的类别,torch.max()返回最大值的索引
            @total:总个数,labels.size(0)通常将第0个维度用于表示 batch 的大小
            @correct:正确的个数,即预测正确的个数
            @loss_sigma:损失总和
        """
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()
        print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, i + 1, len(train_loader), loss_sigma, correct / total))

        """
            @writer.add_scalars():写入数据到tensorboardX中
        """
        writer.add_scalars('loss_group', {'train_loss': loss_sigma}, epoch)
        writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

        loss_sigma = 0.0
    """
        将网络的参数和梯度信息写入tensorboardX的日志文件中
        遍历了网络 net 中所有的参数,并将参数的梯度和参数值分别写入到日志文件中
    """
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    """
        每两个epoch验证一次,并计算验证集的损失和准确率
    """
    if epoch % 2 == 0:
        """
            @loss_sigma:损失总和
            @cls_num:类别的个数
            @conf_mat:混淆矩阵
            @net.eval():将网络设置为评估模式,即关闭dropout和batch normalization等层的随机性
                        以便于获得模型在测试数据上的稳定结果
        """
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])
        net.eval()

        for i, data in enumerate(valid_loader):
            images, labels = data
            outputs = net(images)
            """
                @detach_():将Variable类型的数据与计算图分离,不再进行反向传播,以避免对其进行梯度更新
                @loss.item():返回loss的标量值
                @predicted:预测的类别,torch.max()返回最大值的索引
            """
            outputs.detach_()
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            """
                @cate_i:真实类别
                @pre_i:预测类别
                @conf_mat:混淆矩阵
            """
            for j in range(len(labels)):
                cate_i = labels[j].numpy()
                pre_i = predicted[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))

        """
            @writer.add_scalars():写入数据到tensorboardX中
        """
        writer.add_scalars('loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)

print('Finished Training')

"""
    @net_save_path:网络模型保存路径
    @torch.save():保存网络模型
"""
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net, net_save_path)