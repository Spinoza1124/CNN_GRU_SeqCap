import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(vectors, dim=-1):
    """
    非线性激活函数 Squash
    :param vectors: 输入张量，例如 [batch_size, num_caps, caps_dim]
    :param dim: 在哪个维度上进行计算
    :return: 经过 squash 操作后的张量
    """
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * vectors / (torch.sqrt(squared_norm) + 1e-8) # 1e-8 防止除以零

class CapsuleLayer(nn.Module):
    """
    实现动态路由的胶囊层
    """
    def __init__(self, in_caps, out_caps, in_dim, out_dim, routing_iters=3):
        """
        :param in_caps: 输入胶囊的数量
        :param out_caps: 输出胶囊的数量
        :param in_dim: 输入胶囊的维度
        :param out_dim: 输出胶囊的维度
        :param routing_iters: 路由迭代次数
        """
        super(CapsuleLayer, self).__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iters = routing_iters

        # 权重矩阵 W_ij，对应公式 (3) 中的 W_ij
        # 这是一个可训练的参数，用于将低层胶囊的输出 u_i 转换为预测向量 û_j|i
        # 尺寸: [in_caps, out_caps, out_dim, in_dim]
        # 这样设计是为了方便后续使用 torch.matmul
        self.W = nn.Parameter(torch.randn(in_caps, out_caps, out_dim, in_dim))

    def forward(self, u):
        """
        :param u: 来自低层胶囊的输入，尺寸 [batch_size, in_caps, in_dim]
        :return: 当前层的胶囊输出，尺寸 [batch_size, out_caps, out_dim]
        """
        batch_size = u.size(0)

        # 步骤 1: 计算预测向量 û_j|i = W_ij * u_i
        # u 的原始尺寸: [batch_size, in_caps, in_dim]
        # 为了与 W 相乘，我们需要调整 u 的维度
        # u_expanded 尺寸: [batch_size, in_caps, 1, 1, in_dim]
        u_expanded = u.unsqueeze(2).unsqueeze(3)
        u_expanded = u_expanded.transpose(-2, -1)

        # W_tiled 尺寸: [batch_size, in_caps, out_caps, out_dim, in_dim]
        W_tiled = self.W.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # u_hat 尺寸: [batch_size, in_caps, out_caps, out_dim]
        # 这是通过矩阵乘法得到的预测向量
        u_hat = torch.matmul(W_tiled, u_expanded).squeeze(-1)

        # 步骤 2: 动态路由
        # 初始化对数先验概率 b_ij 为 0
        b = torch.zeros(batch_size, self.in_caps, self.out_caps, device=u.device)

        for i in range(self.routing_iters):
            # c_ij = softmax(b_ij)，对应公式 (4)
            # 在 out_caps 维度上进行 softmax
            c = F.softmax(b, dim=2)

            # s_j = sum(c_ij * û_j|i)，对应公式 (2)
            # s_j 是所有输入对输出胶囊 j 的加权和
            # c 尺寸 [batch, in_caps, out_caps, 1], u_hat 尺寸 [batch, in_caps, out_caps, out_dim]
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)

            # v_j = squash(s_j)，对应公式 (1)
            v = squash(s)

            # 更新 b_ij: b_ij = b_ij + û_j|i · v_j
            # agreement 尺寸: [batch_size, in_caps, out_caps]
            agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
            b = b + agreement

        return v

class CNN_SeqCap(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_SeqCap, self).__init__()
        self.num_classes = num_classes

        # 1. CNN Backbone (参照论文图2的结构)
        # 论文中使用了两个并行的初始卷积，这里为了简化，我们使用一个标准的序列
        # 1x128x200
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 论文中使用了多个卷积和池化
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 2. Primary Capsules
        # 这是一个卷积层，其输出将作为胶囊网络的输入
        # 我们希望输出 32 组 8D 胶囊, 所以 out_channels = 32 * 8
        self.primary_caps_conv = nn.Conv2d(in_channels=32, out_channels=32 * 8, kernel_size=3, stride=2)
        self.primary_caps_dim = 8
        self.primary_num_caps = 32

        # 4. Window Emo-Caps
        # 这个胶囊层处理每个窗口的 Primary Capsules 输出
        # in_caps参数将在forward中动态计算
        # 输出是窗口级的情感胶囊，比如4个情感类别，每个是16D
        self.window_emo_caps = None  # 将在forward中初始化

        # 5. Utterance Caps
        # 这个胶囊层处理所有窗口的输出序列
        # 输入胶囊维度是 window_emo_caps 的 out_dim (16)
        # 输入胶囊数量是窗口的数量
        # 输出是最终的句子级情感胶囊，4个类别，每个16D
        # 注意：这里的 in_caps 是窗口数量，是动态的，所以不能在这里写死。
        # 我们需要在 forward 中动态构建这个层，或者使用一个足够大的固定值。
        # 更好的方法是使用注意力或RNN/GRU来聚合窗口信息。
        # 为了严格复现论文，我们用另一个胶囊层。
        # 假设最大窗口数是 50
        self.utterance_caps = CapsuleLayer(
            in_caps=50, # 假设最大窗口数
            out_caps=num_classes,
            in_dim=16, # 和 window_emo_caps 的 out_dim 一致
            out_dim=16,
            routing_iters=3
        )
        
        # 应用Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化方法初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对CNN层使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, CapsuleLayer):
                # 对胶囊层使用Xavier初始化
                nn.init.xavier_uniform_(m.W)
        print("已应用Xavier初始化到CNN层和胶囊层")

    def forward(self, x):
        # x 初始尺寸: [batch, 1, freq_bins, time_steps]
        # 比如 [16, 1, 128, 200]

        # 1. CNN 特征提取
        features = self.cnn_backbone(x)
        # features 尺寸: [16, 32, 32, 50] (示例)

        # 2. Primary Capsules
        primary_caps_out = self.primary_caps_conv(features)
        # primary_caps_out 尺寸: [16, 256, 15, 24] (示例)
        
        # 3. Windowing (核心步骤)
        # 论文中窗口大小40，步长20
        window_size = 40
        window_stride = 20
        
        # 使用 unfold 在时间维度(dim=3)上创建滑动窗口
        # features 的时间维度是最后一个维度
        # 我们先对 Primary Capsules 的输出进行窗口化
        # 尺寸: [B, C, H, T]
        B, C, H, T = primary_caps_out.shape
        
        # 暂时将通道和高度维度合并，方便窗口化和后续处理
        primary_caps_out = primary_caps_out.view(B, C*H, T) # [B, C*H, T]
        
        # unfold 在时间 T 维度上操作
        windows = primary_caps_out.unfold(dimension=2, size=10, step=5) # 这里的size/step需要根据输入调整
        # windows 尺寸: [B, C*H, num_windows, window_size]
        
        # 调整维度以应用 Window Emo-Caps
        windows = windows.permute(0, 2, 1, 3).contiguous()
        # windows 尺寸: [B, num_windows, C*H, window_size]
        
        num_windows = windows.size(1)
     
        # 将窗口展平，为每个窗口准备胶囊
        # reshape 成 [B * num_windows, num_primary_caps, primary_caps_dim]
        # 首先，将每个窗口内的特征 reshape 成胶囊
        # [B, num_windows, C*H*window_size] -> [B, num_windows, num_caps, caps_dim]
        windows_flat = windows.view(B, num_windows, -1)

        # 重新计算 in_caps
        # 这里需要确保 C*H*window_size / primary_caps_dim 是一个整数
        # 这是模型设计时最需要小心匹配的地方
        in_caps_per_window = windows_flat.size(2) // self.primary_caps_dim

        # 动态初始化 window_emo_caps
        if self.window_emo_caps is None:
            self.window_emo_caps = CapsuleLayer(
                in_caps=in_caps_per_window,
                out_caps=self.num_classes,
                in_dim=self.primary_caps_dim,
                out_dim=16,
                routing_iters=3
            ).to(x.device)
        
        window_caps_in = windows_flat.view(B, num_windows, in_caps_per_window, self.primary_caps_dim)
        
        # 为了共享权重，我们将 batch 和 num_windows 合并
        window_caps_in_reshaped = window_caps_in.view(B * num_windows, in_caps_per_window, self.primary_caps_dim)

        # 4. 应用 Window Emo-Caps
        window_caps_out = self.window_emo_caps(window_caps_in_reshaped)

        # window_caps_out 尺寸: [B * num_windows, num_classes, 16]
        
        # 恢复 batch 和 num_windows 维度
        window_caps_out = window_caps_out.view(B, num_windows, self.num_classes, 16)
        
        # 论文中没有明确说明如何聚合一个窗口内的4个情感胶囊，
        # 通常是直接将它们作为输入序列
        # 我们选择将4个类别胶囊拼接或平均，这里我们直接用它（相当于每个时间步有4个胶囊）
        # 为了简单，我们取第一个类别胶囊作为代表（或者平均）
        utterance_caps_in = window_caps_out.mean(dim=2) # [B, num_windows, 16]

        # 5. 应用 Utterance Caps
        # 这里需要处理变长的窗口序列，可以用 padding
        if num_windows < self.utterance_caps.in_caps:
            pad_size = self.utterance_caps.in_caps - num_windows
            padding = torch.zeros(B, pad_size, 16, device=x.device)
            utterance_caps_in = torch.cat([utterance_caps_in, padding], dim=1)
        elif num_windows > self.utterance_caps.in_caps:
            utterance_caps_in = utterance_caps_in[:, :self.utterance_caps.in_caps, :]

        final_caps = self.utterance_caps(utterance_caps_in)
        # final_caps 尺寸: [B, num_classes, 16]
        
        # 6. 计算最终输出
        # 输出是最终胶囊的模长，代表每个类别的概率
        lengths = torch.sqrt((final_caps ** 2).sum(dim=-1)) # [B, num_classes]
        
        return lengths


# === 如何使用 ===
if __name__ == '__main__':
    # 假设 batch_size=16, 语谱图尺寸为 1x128x200 (通道x频率x时间)
    dummy_input = torch.randn(16, 1, 128, 200)

    # 创建模型
    # 注意：模型内部的维度需要根据你的CNN输出精确计算
    # 这里的实现是一个示例，你需要根据你的数据预处理和CNN架构调整维度
    # 尤其是 window_emo_caps 和 utterance_caps 的 in_caps
    model = CNN_SeqCap(num_classes=4)
    
    # 前向传播
    try:
        output = model(dummy_input)
        print("模型输出尺寸:", output.shape) # 应该是 [16, 4]
    except Exception as e:
        print("模型运行出错，很可能是维度不匹配问题:", e)
        print("这是正常的，因为胶囊网络的维度需要精确设计。请根据你的CNN输出调整CapsuleLayer的参数。")