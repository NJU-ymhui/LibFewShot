from core.model.init import *
from .metric_model import MetricModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


class FORT(MetricModel):
    def __init__(self, init_type="normal", num_classes=200, **kwargs):
        super(FORT, self).__init__(init_type, **kwargs)
        self.num_classes = num_classes
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.head = nn.Linear(self.vit.head.in_features, num_classes)
        self.feature_transform = torch.nn.Linear(1000, 768)

        self._init_network()

        # 注册钩子
        def save_attention_maps(module, input, output):
            self.attention_maps = output  # 保存注意力映射

        self.vit.blocks[-1].attn.register_forward_hook(save_attention_maps)

    def _init_network(self):
        init_weights(self, self.init_type)

    def forward(self, x):
        if self.training:
            return self.set_forward_loss(x)
        else:
            return self.set_forward(x)

    def set_forward(self, batch):
        images, _ = batch
        images = images.cuda()
        features = self.vit(images)
        logits = self.head(features)
        return logits

    def set_forward_loss(self, batch):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vit(images)
        features = features.to('cuda')  # 确保 features 在 GPU 上
        features = self.feature_transform(features)

        logits = self.head(features)
        classification_loss = F.cross_entropy(logits, labels)

        position_prompts = self.generate_position_prompts(images, labels)
        attention_loss = self.compute_attention_loss(features, position_prompts)

        total_loss = classification_loss - self.alpha * attention_loss
        return total_loss

    def generate_position_prompts(self, images, labels, lambda_val=1.0):
        # self.eval()  # 训练时不要开eval
        self.train()
        images = images.requires_grad_()
        # with torch.no_grad():  # 不计算梯度，减少内存消耗
            # with torch.no_grad():
        outputs = self.vit(images)
            # outputs.requires_grad = True
        gradients = torch.autograd.grad(outputs.sum(), images, retain_graph=True)[0]
        gradients = gradients.mean(dim=1).view(-1, 224, 224)
        gradients = torch.linalg.svd(gradients)[2][:, :, 0]
        return gradients

    def compute_attention_loss(self, features, position_prompts, tau=1.0):
        if self.attention_maps is None:
            raise ValueError("Attention maps have not been captured. Check your hook setup.")
        # attention_logits = self.attention_maps  # 使用钩子捕获的注意力映射
        attention_logits = F.softmax(self.attention_maps / tau, dim=1)
        attention_loss = 0
        for i, prompt in enumerate(position_prompts):
            prompt = prompt.long()  # 确保 `prompt` 是整数类型
            # print(torch.sum(attention_logits[i, :, prompt]))
            attention_loss += -torch.log(torch.sum(attention_logits[i, :, prompt]))
        attention_loss /= len(position_prompts)
        return attention_loss


