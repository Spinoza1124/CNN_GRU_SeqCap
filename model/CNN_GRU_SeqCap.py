import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(vectors, dim=-1):
    """
    非线性激活函数，将向量的模长压缩到0-1之间，同时保持方向不变。
    """
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    # 加上一个很小的 epsilon 防止除以零
    return scale * vectors / (torch.sqrt(squared_norm) + 1e-8)

class CapsuleLayer(nn.Module):
    """
    实现动态路由的胶囊层。
    """
    def __init__(self, in_caps, out_caps, in_dim, out_dim, routing_iters=3):
        super(CapsuleLayer, self).__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.routing_iters = routing_iters
        
        self.W = nn.Parameter(torch.randn(in_caps, out_caps, out_dim, in_dim))

    def forward(self, u):
        batch_size = u.size(0)
        u_expanded = u.unsqueeze(2).unsqueeze(-1)
        W_tiled = self.W.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W_tiled, u_expanded).squeeze(-1)

        b = torch.zeros(batch_size, self.in_caps, self.out_caps, device=u.device)

        for i in range(self.routing_iters):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)
            agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
            b = b + agreement
        
        return v

class CapsuleBranch(nn.Module):
    """
    专门为CNN-GRU-SeqCap设计的胶囊分支，接收来自CNN主干的特征图。
    """
    def __init__(self, cnn_output_channels, cnn_output_height, cnn_output_width, num_classes=4):
        super(CapsuleBranch, self).__init__()

        # 将CNN特征图 [C, H] 转换为初级胶囊
        # 例如，每个时间步的 [16, 8] 特征图可以看作 16 个 8D 胶囊
        self.primary_caps_dim = cnn_output_height
        self.num_primary_caps = cnn_output_channels

        # 动态路由层，聚合时间序列上的所有初级胶囊
        self.digit_caps = CapsuleLayer(
            in_caps=cnn_output_width,  # 输入胶囊数 = 时间步数
            out_caps=num_classes,
            in_dim=self.num_primary_caps * self.primary_caps_dim, # 将 C 和 H 展平作为输入维度
            out_dim=16 # 最终的输出胶囊维度
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16, out_features=num_classes)
        )
        
    def forward(self, x):
        # x 是来自CNN的特征图, 尺寸: [Batch, Channels, Height, Width/Time]
        B, C, H, W = x.shape

        # 1. 重塑为初级胶囊序列
        # 将 C 和 H 展平，并将 W 作为序列长度
        # 尺寸变为: [B, W, C*H]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, W, C * H)
        
        # 为了与CapsuleLayer的输入匹配 [B, in_caps, in_dim], in_caps是序列长度
        # 这里需要将 x 的维度调整为 [B, W, C*H] -> [B, 序列长度, 特征维度]
        # x 已经是正确的形状了
        
        # 2. 动态路由
        # 注意：这里我们将整个时间序列的特征作为一个大的向量输入给每个时间步的胶囊
        # 这是一个简化的实现，更复杂的会保持胶囊的结构
        # 为了更符合胶囊思想，我们将 C*H 维度视为初级胶囊
        # [B, W, C*H] -> [B, W, 16*8]
        # 让我们把C看作胶囊数，H看作胶囊维度
        x = x.view(B, W, self.num_primary_caps * self.primary_caps_dim) # [B, 50, 128]
        
        # 路由层期望输入 [B, in_caps, in_dim], in_caps 是序列长度
        caps_output = self.digit_caps(x) # [B, num_classes, 16]

        # 3. 计算输出概率
        # 使用模长作为存在的概率
        lengths = torch.sqrt((caps_output ** 2).sum(dim=-1)) # [B, num_classes]
        
        return lengths


class BiGRUBranch(nn.Module):
    """
    实现图中所示的 BiGRU -> Classifier 分支。
    """
    def __init__(self, cnn_output_channels, cnn_output_height, num_classes=4, gru_hidden_size=64, dropout_rate=0.5):
        super(BiGRUBranch, self).__init__()
        gru_input_size = cnn_output_channels * cnn_output_height
        self.bigru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=gru_hidden_size * 2, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        _, h_n = self.bigru(x)
        gru_output_combined = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1)
        logits = self.classifier(gru_output_combined)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

class CNN_GRU_SeqCap(nn.Module):
    """
    论文 "Speech Emotion Recognition Using Capsule Networks" 提出的最终模型
    """
    def __init__(self, num_classes=4, lambda_val=0.6):
        super(CNN_GRU_SeqCap, self).__init__()
        self.lambda_val = lambda_val

        # --- 1. 共享的CNN主干 ---
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 8), padding='same')
        self.conv1b = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(8, 2), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=(4,1))
        )

        # --- 2. 定义两个并行的分支 ---
        # 经过CNN主干后，我们预期的特征图尺寸为 [B, 16, 8, 50] (C=16, H=8, W=50)
        # 这个尺寸是基于一个典型的输入尺寸计算得出的，如果输入尺寸变化，这里也需要调整
        cnn_output_channels = 16
        cnn_output_height = 4
        cnn_output_width = 50 # 假设时间维度是50

        # 实例化胶囊分支
        self.capsule_branch = CapsuleBranch(
            cnn_output_channels=cnn_output_channels,
            cnn_output_height=cnn_output_height,
            cnn_output_width=cnn_output_width,
            num_classes=num_classes
        )

        # 实例化GRU分支
        self.gru_branch = BiGRUBranch(
            cnn_output_channels=cnn_output_channels,
            cnn_output_height=cnn_output_height,
            num_classes=num_classes
        )

    def forward(self, x):
        # --- 1. 通过共享的CNN主干 ---
        x_a = self.conv1a(x)
        x_b = self.conv1b(x)
        x_concat = torch.cat((x_a, x_b), dim=1)
        
        # 注意：论文图示和描述有些许模糊，这里我们采用最合理的结构
        # Concat -> Pool1 -> Conv2 -> Conv3
        x_pooled = self.pool1(x_concat)
        x_conv2_out = self.conv2(x_pooled)
        cnn_features = self.conv3(x_conv2_out) # [B, C, H, W]

        # --- 2. 将特征图送入两个分支 ---
        # 胶囊分支输出
        caps_output_probs = self.capsule_branch(cnn_features)

        # GRU分支输出
        gru_output_probs = self.gru_branch(cnn_features)

        # --- 3. 融合两个分支的输出 ---
        # 根据论文，在测试阶段，使用加权求和
        final_probs = self.lambda_val * caps_output_probs + (1 - self.lambda_val) * gru_output_probs
        
        return final_probs


# # --- 如何使用 ---
# if __name__ == '__main__':
#     # 假设输入语谱图尺寸：batch=16, 通道=1, 频率=128, 时间=200
#     dummy_input = torch.randn(16, 1, 128, 200)

#     # 实例化最终模型
#     model = CNN_GRU_SeqCap(num_classes=4)

#     # 前向传播
#     final_output = model(dummy_input)

#     # 打印输出尺寸
#     print("输入尺寸:", dummy_input.shape)
#     print("最终模型输出尺寸:", final_output.shape) # 应该是 [16, 4]
#     print("第一个样本的输出概率:", final_output[0])
#     print("概率和 (应该约等于1):", torch.sum(final_output[0]))