import os
import shutil
import pandas as pd

# 定义路径
test_csv_path = r"E:\机器学习\final\miniImageNet--ravi\test.csv"
train_csv_path = r"E:\机器学习\final\miniImageNet--ravi\train.csv"
images_dir = r"E:\机器学习\final\miniImageNet--ravi\images"
train_dir = r"E:\机器学习\final\miniImageNet--ravi\train_images"
test_dir = r"E:\机器学习\final\miniImageNet--ravi\test_images"

# 读取 CSV 文件
train_data = pd.read_csv(train_csv_path)  # 训练数据csv文件
test_data = pd.read_csv(test_csv_path)  # 测试数据csv文件

# 创建类别文件夹并移动图片
for _, row in train_data.iterrows():
    image_name, label = row['filename'], row['label']
    label_dir = os.path.join(train_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    src = os.path.join(images_dir, image_name)
    dst = os.path.join(label_dir, image_name)
    shutil.move(src, dst)

for _, row in test_data.iterrows():
    image_name, label = row['filename'], row['label']
    label_dir = os.path.join(test_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    src = os.path.join(images_dir, image_name)
    dst = os.path.join(label_dir, image_name)
    shutil.move(src, dst)

print("数据整理完成！")
