import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #卷积层定义,计算方式是(32-5+2*0)/1+1=28,所以输出的是28*28*6,32是图片的长宽,
        #5是卷积核的大小,0是填充的大小,1是步长
        self.conv1 = nn.Conv2d(3, 6, 5)

        #池化层定义,计算方式是(28-2+2*0)/2+1=14,所以输出的是14*14*6,28是图片的长宽,
        #2是池化核的大小,0是填充的大小,2是步长
        self.pool1 = nn.MaxPool2d(2, 2)

        #卷积层定义,计算方式是(14-5+2*0)/1+1=10,所以输出的是10*10*16,14是图片的长宽,
        #5是卷积核的大小,0是填充的大小,1是步长
        self.conv2 = nn.Conv2d(6, 16, 5)

        #池化层定义,计算方式是(10-2+2*0)/2+1=5,所以输出的是5*5*16,10是图片的长宽,
        #2是池化核的大小,0是填充的大小,2是步长
        self.pool2 = nn.MaxPool2d(2, 2)

        #全连接层定义,输入是16*5*5,输出是120
        #16是上一层的输出通道数,5*5是上一层的输出的长宽
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    #前向传播函数,输入是x,输出是x,这里的x是图片,所以是四维的,第一维是batch_size,第二维是通道数,
    #第三维是图片的长,第四维是图片的宽
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    #初始化权重,这里的权重初始化是按照论文LeNet-5中的方法初始化的
    #xavier_normal_是按照高斯分布初始化权重,目的是让每一层的输出方差尽量相等
    #zero_是将偏置初始化为0
    #m.weight.data.fill_(1)是将权重初始化为1
    #m.bias.data.zero_()是将偏置初始化为0
    #normal_是按照高斯分布初始化权重,目的是让每一层的输出方差尽量相等,0是均值,0.01是方差
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        #第一层卷积层,输入是3通道,输出是64通道,卷积核大小是3*3,步长是1,填充是1
        #卷积层定义,计算方式是(32-3+2*1)/1+1=32,所以输出的是32*32*64,32是图片的长宽,
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #批归一化，对每个通道的数据进行归一化，使得每个通道的数据均值为0，方差为1
        self.bn1 = nn.BatchNorm2d(64)
        #激活函数,这里使用的是relu函数,relu函数的公式是f(x)=max(0,x),所以输出的是32*32*64
        self.relu = nn.ReLU(inplace=True)

        #第一层残差块
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        #第二层残差块
        self.layer2 = nn.Sequential(
            #这里的stride=2,所以计算方式是(32-3+2*1)/2+1=16,所以输出的是16*16*128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        #第三层残差块
        self.layer3 = nn.Sequential(
            #这里的stride=2,所以计算方式是(16-3+2*1)/2+1=8,所以输出的是8*8*256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        #第四层残差块
        self.layer4 = nn.Sequential(
            #这里的stride=2,所以计算方式是(8-3+2*1)/2+1=4,所以输出的是4*4*512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        #平均池化层,这里的输出是1*1*512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #全连接层,这里的输入是512,输出是1000
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        residual = x
        x = self.layer1(x)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.layer2(x)
        residual = nn.Conv2d(
            64, 128, kernel_size=1, stride=2, bias=False
        )(residual)
        residual = nn.BatchNorm2d(128)(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.layer3(x)
        residual = nn.Conv2d(
            128, 256, kernel_size=1, stride=2, bias=False
        )(residual)
        residual = nn.BatchNorm2d(256)(residual)
        x += residual
        x = self.relu(x)

        residual = x
        x = self.layer4(x)
        residual = nn.Conv2d(
            256, 512, kernel_size=1, stride=2, bias=False
        )(residual)
        residual = nn.BatchNorm2d(512)(residual)
        x += residual
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    #初始化权重,这里的权重初始化是按照论文LeNet-5中的方法初始化的
    #xavier_normal_是按照高斯分布初始化权重,目的是让每一层的输出方差尽量相等
    #zero_是将偏置初始化为0
    #m.weight.data.fill_(1)是将权重初始化为1
    #m.bias.data.zero_()是将偏置初始化为0
    #normal_是按照高斯分布初始化权重,目的是让每一层的输出方差尽量相等,0是均值,0.01是方差
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()