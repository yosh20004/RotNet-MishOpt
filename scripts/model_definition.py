import torch
import torch.nn as nn
from torchvision import models

class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=False):
        super(RotationPredictionModel, self).__init__()
        # 使用预训练权重来利用迁移学习
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')

        # 如果选择冻结，则编码器的权重在训练中不更新
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        num_features = self.encoder.fc.in_features
        # 替换掉原始的分类头
        self.encoder.fc = nn.Identity()

        # 添加我们自己的分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs
