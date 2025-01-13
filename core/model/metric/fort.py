from core.model.init import *
from .metric_model import MetricModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class FORT(MetricModel):
    def __init__(self, init_type="normal", num_classes=200, **kwargs):
        super(FORT, self).__init__(init_type, **kwargs)
        self.num_classes = num_classes
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.head = nn.Linear(self.vit.head.in_features, num_classes)
        self._init_network()

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
        # transform to 224 * 224, if not will raise error: Input height (84) doesn't match model (224).(i dkn data sz)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vit(images)
        logits = self.head(features)
        classification_loss = F.cross_entropy(logits, labels)

        # 生成位置提示
        position_prompts = self.generate_position_prompts(images, labels)

        # 计算注意力增强损失
        attention_loss = self.compute_attention_loss(features, position_prompts)

        total_loss = classification_loss - self.alpha * attention_loss
        return total_loss

    def generate_position_prompts(self, images, labels, lambda_val=1.0):
        self.eval()
        with torch.no_grad():
            outputs = self.vit(images)
            attention_maps = self.vit.blocks[-1].attn.attention_weights
            gradients = torch.autograd.grad(outputs, images, grad_outputs=labels, retain_graph=True)[0]
            gradients = gradients.view(-1, 224, 224)
            gradients = torch.svd(gradients)[2][:, :, 0]
            attention_maps = attention_maps + lambda_val * gradients.unsqueeze(1)
            importance_scores = torch.mean(attention_maps, dim=1)
            topk_indices = torch.topk(importance_scores, k=14, dim=1).indices
        self.train()
        return topk_indices

    def compute_attention_loss(self, features, position_prompts, tau=1.0):
        attention_logits = self.vit.blocks[-1].attn.attention_weights
        attention_loss = 0
        for i, prompt in enumerate(position_prompts):
            attention_loss += -torch.log(torch.sum(attention_logits[i, :, prompt] / tau))
        attention_loss /= len(position_prompts)
        return attention_loss

    def train(self, mode=True):
        super(FORT, self).train(mode)
        if hasattr(self, "distill_layer"):
            self.distill_layer.train(False)

    def eval(self):
        return super(FORT, self).eval()

