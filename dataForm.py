"""
    基础介绍:将数据集转换为csv格式储存图片对应的标签制作成表单
"""
import csv
import os

"""
    @param train_txt_path:训练集表单路径
    @param valid_txt_path:验证集表单路径
    @param train_dir:训练集路径
    @param valid_dir:验证集路径
"""
train_txt_path = os.path.join('.', 'Data', 'train.csv')
valid_txt_path = os.path.join('.', 'Data', 'valid.csv')
train_dir = os.path.join('.', 'Data', 'train')
valid_dir = os.path.join('.', 'Data', 'valid')

"""
    生成表单函数
    @param txt_path:表单路径
    @param img_dir:图片路径
"""
def generateTxt(txt_path, img_dir):
    with open(txt_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for root, dirs, files in os.walk(img_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                for img_name in img_names:
                    img_path = os.path.join(sub_dir, img_name)
                    label = sub_dir
                    writer.writerow([img_dir + '/' + img_path, label])
    csvfile.close()

if __name__ == '__main__':
    generateTxt(train_txt_path, train_dir)
    generateTxt(valid_txt_path, valid_dir)
