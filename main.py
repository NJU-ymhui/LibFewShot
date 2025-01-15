import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from abc import abstractmethod

from core.model import FORT
from core.model.abstract_model import AbstractModel
from core.utils import ModelType

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(84),  # mini-ImageNet 图像大小
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 mini-ImageNet 数据集
train_dataset = datasets.ImageFolder(root='E:\\机器学习\\final\\miniImageNet--ravi\\train_images', transform=transform)
test_dataset = datasets.ImageFolder(root='E:\\机器学习\\final\\miniImageNet--ravi\\test_images', transform=transform)

# 分割数据集为训练集和测试集
# train_size = int(0.8 * len(dataset))  # 80% 用于训练
# test_size = len(dataset) - train_size  # 20% 用于测试
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def train(model, train_loader, epochs=10, alpha=0.5):
    optimizer = Adam(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            loss = model.set_forward_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')


def validate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')


# 初始化模型
model = FORT(init_type="normal", num_classes=100).cuda()  # mini-ImageNet 有 100 个类
model.alpha = 0.5  # 设置注意力增强损失的权重

# 训练模型
train(model, train_loader, epochs=10, alpha=0.5)

# 保存整个模型
torch.save(model, "fort_model_full.pth")
print("Entire model saved successfully!")

# 验证模型
validate(model, test_loader)
