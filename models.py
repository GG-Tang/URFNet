import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights



class URFNet(nn.Module):
    def __init__(self, num_classes=13):
        super(URFNet, self).__init__()

        # ResNet18 用于提取正面图像和侧面图像的特征
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # 移除ResNet分类头，保留特征提取部分

        # Cross-Attention层
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # 用于将特征融合后进行分类的全连接层
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x1, x2):
        """
        :param x1: 正面图像
        :param x2: 侧面图像
        """
        # 正面图像通过ResNet18提取特征
        features1 = self.resnet(x1)
        
        # 侧面图像通过ResNet18提取特征
        features2 = self.resnet(x2)

        # 将特征拼接起来并输入Cross-Attention层
        features = torch.stack([features1, features2], dim=1)  # Shape: [batch_size, 2, feature_dim]
        
        # Cross-attention机制：正面特征作为查询，侧面特征作为键和值
        attn_output, _ = self.cross_attention(features, features, features)  # Cross-modal attention

        # 合并注意力输出的特征
        x = attn_output.mean(dim=1)  # 通过平均池化合并，获取融合后的特征

        # 全连接层
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        # 分类
        output = self.classifier(x)
        return output



