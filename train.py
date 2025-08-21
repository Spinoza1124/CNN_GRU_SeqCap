import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import argparse
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import sys
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，将跳过训练曲线绘制")
from collections import defaultdict

from model.CNN_GRU_SeqCap import CNN_SeqCap

class DynamicLearningRateScheduler:
    """基于训练损失的动态学习率调度器"""
    
    def __init__(self, optimizer, initial_lr=0.001, window_size=100):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.window_size = window_size
        self.loss_history = []
        self.lr_schedule = [0.001, 0.0005, 0.0002, 0.0001]
        self.current_lr_index = 0
        self.last_avg_loss = float('inf')
        
    def step(self, loss):
        """更新学习率"""
        self.loss_history.append(loss)
        
        # 只有当损失历史达到窗口大小时才开始调整
        if len(self.loss_history) >= self.window_size:
            # 计算最近100步的平均损失
            recent_losses = self.loss_history[-self.window_size:]
            current_avg_loss = sum(recent_losses) / len(recent_losses)
            
            # 检查是否需要降低学习率（损失下降10倍）
            if self.last_avg_loss / current_avg_loss >= 10.0 and self.current_lr_index < len(self.lr_schedule) - 1:
                self.current_lr_index += 1
                new_lr = self.lr_schedule[self.current_lr_index]
                
                # 更新优化器的学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                print(f"\n🔄 学习率调整: {self.get_current_lr():.6f} -> {new_lr:.6f}")
                print(f"   触发条件: 平均损失从 {self.last_avg_loss:.6f} 降至 {current_avg_loss:.6f}")
                
                self.last_avg_loss = current_avg_loss
            elif len(self.loss_history) == self.window_size:
                # 第一次计算平均损失
                self.last_avg_loss = current_avg_loss
    
    def get_current_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_info(self):
        """获取学习率调度信息"""
        return {
            'current_lr': self.get_current_lr(),
            'lr_index': self.current_lr_index,
            'loss_history_length': len(self.loss_history),
            'last_avg_loss': self.last_avg_loss
        }

class IEMOCAPDataset(Dataset):
    """IEMOCAP数据集加载器"""
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict: 包含seg_spec, seg_label等的字典
            transform: 数据变换函数
        """
        self.seg_spec = data_dict['seg_spec']  # (N, 1, 200, 300)
        self.seg_label = data_dict['seg_label']  # (N,)
        self.transform = transform
        
        # 数据预处理：调整维度以匹配模型输入 [batch, 1, freq, time]
        # 原始数据是 (N, 1, 200, 300)，需要转换为 (N, 1, 200, 300)
        # 模型期望输入是 [batch, 1, freq_bins, time_steps]
        print(f"原始数据形状: {self.seg_spec.shape}")
        
    def __len__(self):
        return len(self.seg_spec)
    
    def __getitem__(self, idx):
        spec = self.seg_spec[idx]  # (1, 200, 300)
        label = self.seg_label[idx]
        
        # 转换为torch tensor
        spec = torch.FloatTensor(spec)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            spec = self.transform(spec)
            
        return spec, label

def load_data_by_session(data_path):
    """按Session加载IEMOCAP数据，保持Session结构"""
    print(f"正在加载数据: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    session_data = {}
    total_samples = 0
    
    # 检查数据格式并处理
    if isinstance(list(data.values())[0], dict) and 'features' in list(list(data.values())[0].values())[0]:
        # 测试数据格式：Session1 -> {Session1_M: {features: [], labels: []}, Session1_F: {...}}
        for session_key in sorted(data.keys()):
            session_info = data[session_key]
            
            # 合并该Session下所有说话人的数据
            all_features = []
            all_labels = []
            
            for speaker_key in session_info.keys():
                speaker_data = session_info[speaker_key]
                all_features.extend(speaker_data['features'])
                all_labels.extend(speaker_data['labels'])
            
            # 转换为numpy数组
            session_data[session_key] = {
                'seg_spec': np.array(all_features),
                'seg_label': np.array(all_labels)
            }
            
            num_samples = len(all_features)
            total_samples += num_samples
            print(f"Session {session_key}: {num_samples} 个样本")
    else:
        # 真实IEMOCAP数据格式：1F, 1M, 2F, 2M, ...
        # 需要将同一Session的不同说话人数据合并
        session_groups = {}
        
        for session_key in sorted(data.keys()):
            # 提取Session编号（如"1F" -> "Session1", "2M" -> "Session2"）
            session_num = session_key[0]  # 取第一个字符作为session编号
            session_id = f"Session{session_num}"
            
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(session_key)
            
            num_samples = data[session_key]['seg_spec'].shape[0]
            total_samples += num_samples
            print(f"{session_key}: {num_samples} 个样本")
        
        # 合并同一Session的数据
        for session_id, session_keys in session_groups.items():
            all_features = []
            all_labels = []
            
            for session_key in session_keys:
                session_info = data[session_key]
                all_features.append(session_info['seg_spec'])
                all_labels.append(session_info['seg_label'])
            
            # 合并数据
            session_data[session_id] = {
                'seg_spec': np.concatenate(all_features, axis=0),
                'seg_label': np.concatenate(all_labels, axis=0)
            }
    
    print(f"总样本数: {total_samples}")
    print(f"合并后的Session数: {len(session_data)}")
    for session_id in sorted(session_data.keys()):
        print(f"{session_id}: {session_data[session_id]['seg_spec'].shape[0]} 个样本")
    
    return session_data

def create_five_fold_splits(session_data):
    """创建五折交叉验证的数据划分
    
    Args:
        session_data: 按Session组织的数据字典
        
    Returns:
        List of 5 folds, each containing (train_data, val_data, test_data)
    """
    sessions = list(session_data.keys())
    if len(sessions) != 5:
        raise ValueError(f"IEMOCAP应该有5个Session，但找到了{len(sessions)}个")
    
    folds = []
    
    for i, test_session in enumerate(sessions):
        print(f"\n=== 创建第{i+1}折交叉验证 ===")
        print(f"测试Session: {test_session}")
        
        # 训练集：其他4个Session的数据
        train_sessions = [s for s in sessions if s != test_session]
        print(f"训练Sessions: {train_sessions}")
        
        # 合并训练数据
        train_specs = []
        train_labels = []
        for session in train_sessions:
            train_specs.append(session_data[session]['seg_spec'])
            train_labels.append(session_data[session]['seg_label'])
        
        train_data = {
            'seg_spec': np.concatenate(train_specs, axis=0),
            'seg_label': np.concatenate(train_labels, axis=0)
        }
        
        # 测试Session的数据需要分为验证集和测试集
        test_session_spec = session_data[test_session]['seg_spec']
        test_session_label = session_data[test_session]['seg_label']
        
        # 按说话人划分验证集和测试集（假设前一半是一个说话人，后一半是另一个说话人）
        n_samples = len(test_session_spec)
        mid_point = n_samples // 2
        
        val_data = {
            'seg_spec': test_session_spec[:mid_point],
            'seg_label': test_session_label[:mid_point]
        }
        
        test_data = {
            'seg_spec': test_session_spec[mid_point:],
            'seg_label': test_session_label[mid_point:]
        }
        
        print(f"训练集: {len(train_data['seg_spec'])} 样本")
        print(f"验证集: {len(val_data['seg_spec'])} 样本")
        print(f"测试集: {len(test_data['seg_spec'])} 样本")
        
        folds.append((train_data, val_data, test_data))
    
    return folds

def normalize_data(train_data, val_data, test_data):
    """对数据进行标准化处理（零均值单位方差归一化）
    
    Args:
        train_data, val_data, test_data: 数据字典
        
    Returns:
        标准化后的数据和标准化参数
    """
    # 计算训练集的均值和标准差
    train_spec = train_data['seg_spec']
    mean = np.mean(train_spec, axis=(0, 2, 3), keepdims=True)  # 保持维度用于广播
    std = np.std(train_spec, axis=(0, 2, 3), keepdims=True)
    
    # 避免除零
    std = np.where(std == 0, 1, std)
    
    print(f"数据标准化参数: mean={mean.flatten()}, std={std.flatten()}")
    
    # 标准化所有数据集
    normalized_train = {
        'seg_spec': (train_data['seg_spec'] - mean) / std,
        'seg_label': train_data['seg_label'].copy()
    }
    
    normalized_val = {
        'seg_spec': (val_data['seg_spec'] - mean) / std,
        'seg_label': val_data['seg_label'].copy()
    }
    
    normalized_test = {
        'seg_spec': (test_data['seg_spec'] - mean) / std,
        'seg_label': test_data['seg_label'].copy()
    }
    
    normalization_params = {'mean': mean, 'std': std}
    
    return normalized_train, normalized_val, normalized_test, normalization_params

def split_data(data_dict, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """划分训练、验证、测试集"""
    np.random.seed(random_seed)
    
    total_samples = len(data_dict['seg_spec'])
    indices = np.random.permutation(total_samples)
    
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    def create_subset(indices):
        return {
            'seg_spec': data_dict['seg_spec'][indices],
            'seg_label': data_dict['seg_label'][indices]
        }
    
    train_data = create_subset(train_indices)
    val_data = create_subset(val_indices)
    test_data = create_subset(test_indices)
    
    print(f"数据划分: 训练集={len(train_indices)}, 验证集={len(val_indices)}, 测试集={len(test_indices)}")
    
    return train_data, val_data, test_data

def calculate_metrics(y_true, y_pred, num_classes=4):
    """计算评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    # 未加权准确率 (UA) - 每个类别准确率的平均
    class_accuracies = []
    for i in range(num_classes):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    ua = np.mean(class_accuracies)
    
    # 加权准确率 (WA) - 总体准确率
    wa = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'UA': ua,
        'WA': wa,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist()  # 转换为list以便JSON序列化
    }

def train_epoch(model, dataloader, criterion, optimizer, device, lr_scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 创建进度条
    pbar = tqdm(dataloader, desc="训练中", file=sys.stdout)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新学习率调度器
        if lr_scheduler is not None:
            lr_scheduler.step(loss.item())
        
        total_loss += loss.item()
        
        # 预测结果
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # 更新进度条显示
        current_avg_loss = total_loss / (batch_idx + 1)
        current_lr = lr_scheduler.get_current_lr() if lr_scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg_Loss': f'{current_avg_loss:.6f}',
            'LR': f'{current_lr:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 创建进度条
    pbar = tqdm(dataloader, desc="验证中", file=sys.stdout)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # 更新进度条显示
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{current_avg_loss:.6f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def save_experiment_log(log_data, save_dir):
    """保存实验日志"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存JSON格式的日志
    log_file = os.path.join(save_dir, 'experiment_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"实验日志已保存到: {log_file}")

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """绘制训练曲线"""
    if not MATPLOTLIB_AVAILABLE:
        print("跳过训练曲线绘制 (matplotlib未安装)")
        return
        
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # UA曲线
    plt.subplot(1, 3, 2)
    train_ua = [m['UA'] for m in train_metrics]
    val_ua = [m['UA'] for m in val_metrics]
    plt.plot(epochs, train_ua, 'b-', label='训练UA')
    plt.plot(epochs, val_ua, 'r-', label='验证UA')
    plt.title('未加权准确率 (UA)')
    plt.xlabel('Epoch')
    plt.ylabel('UA')
    plt.legend()
    plt.grid(True)
    
    # WA曲线
    plt.subplot(1, 3, 3)
    train_wa = [m['WA'] for m in train_metrics]
    val_wa = [m['WA'] for m in val_metrics]
    plt.plot(epochs, train_wa, 'b-', label='训练WA')
    plt.plot(epochs, val_wa, 'r-', label='验证WA')
    plt.title('加权准确率 (WA)')
    plt.xlabel('Epoch')
    plt.ylabel('WA')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {os.path.join(save_dir, 'training_curves.png')}")

def train_single_fold(fold_idx, train_data, val_data, test_data, config, experiment_dir):
    """训练单个折"""
    print(f"\n{'='*80}")
    print(f"🔄 开始第 {fold_idx + 1} 折训练")
    print(f"{'='*80}")
    
    # 强制使用CPU以避免CUDA兼容性问题
    device = torch.device('cpu')
    
    # 数据标准化
    train_data, val_data, test_data, normalization_params = normalize_data(train_data, val_data, test_data)
    
    # 创建数据加载器
    train_dataset = IEMOCAPDataset(train_data)
    val_dataset = IEMOCAPDataset(val_data)
    test_dataset = IEMOCAPDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"📊 数据统计:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    model = CNN_SeqCap(num_classes=config['num_classes']).to(device)
    
    # 定义损失函数和优化器（按照规范配置）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          betas=(0.9, 0.999), eps=1e-8)
    
    # 创建动态学习率调度器
    lr_scheduler = DynamicLearningRateScheduler(optimizer, initial_lr=config['learning_rate'])
    
    # 训练记录
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_wa = 0.0  # 使用WA作为模型选择标准
    best_model_state = None
    
    print(f"\n🚀 开始训练 (共 {config['epochs']} 个epoch)")
    start_time = time.time()
    
    # 创建epoch级别的进度条
    epoch_pbar = tqdm(range(config['epochs']), desc=f"Fold {fold_idx + 1}", position=0, leave=True)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device, lr_scheduler)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        
        # 验证
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        # 保存最佳模型（基于验证集WA）
        if val_metric['WA'] > best_val_wa:
            best_val_wa = val_metric['WA']
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ua': val_metric['UA'],
                'val_wa': val_metric['WA'],
                'fold': fold_idx
            }
        
        # 更新进度条
        lr_info = lr_scheduler.get_lr_info()
        epoch_pbar.set_postfix({
            'T_Loss': f'{train_loss:.4f}',
            'V_WA': f'{val_metric["WA"]:.4f}',
            'Best_WA': f'{best_val_wa:.4f}',
            'LR': f'{lr_info["current_lr"]:.6f}'
        })
        
        # 每5个epoch打印详细信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Epoch {epoch+1}/{config['epochs']} 结果:")
            print(f"   训练: Loss={train_loss:.6f}, UA={train_metric['UA']:.4f}, WA={train_metric['WA']:.4f}")
            print(f"   验证: Loss={val_loss:.6f}, UA={val_metric['UA']:.4f}, WA={val_metric['WA']:.4f}")
            print(f"   学习率: {lr_info['current_lr']:.6f}, 耗时: {epoch_time:.2f}s")
    
    epoch_pbar.close()
    
    # 加载最佳模型进行测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # 测试集评估
    test_loss, test_metric = evaluate(model, test_loader, criterion, device)
    
    training_time = time.time() - start_time
    
    print(f"\n✅ 第 {fold_idx + 1} 折训练完成")
    print(f"   训练时间: {training_time:.2f}秒")
    print(f"   最佳验证WA: {best_val_wa:.4f}")
    print(f"   测试结果: UA={test_metric['UA']:.4f}, WA={test_metric['WA']:.4f}")
    
    # 保存折的结果
    fold_results = {
        'fold_idx': fold_idx,
        'training_time': training_time,
        'best_val_wa': best_val_wa,
        'test_results': {
            'loss': test_loss,
            'UA': test_metric['UA'],
            'WA': test_metric['WA'],
            'class_accuracies': test_metric['class_accuracies']
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ua': [m['UA'] for m in train_metrics],
            'val_ua': [m['UA'] for m in val_metrics],
            'train_wa': [m['WA'] for m in train_metrics],
            'val_wa': [m['WA'] for m in val_metrics]
        }
    }
    
    # 保存折的模型和结果
    fold_dir = os.path.join(experiment_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(fold_dir, 'best_model.pth'))
    
    with open(os.path.join(fold_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, fold_dir)
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description='CNN-GRU-SeqCap五折交叉验证训练脚本')
    parser.add_argument('--data_path', type=str, default='data/IEMOCAP_multi.pkl',
                        help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小（固定为16）')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（固定为20）')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率（固定为0.001）')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='分类数量')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='实验结果保存目录')
    
    args = parser.parse_args()
    
    # 强制使用CPU以避免CUDA兼容性问题
    device = torch.device('cpu')
    print(f"🖥️  使用设备: {device}")
    if torch.cuda.is_available():
        print("⚠️  注意: 检测到CUDA但由于兼容性问题使用CPU运行")
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"five_fold_cv_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存超参数配置
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'num_classes': args.num_classes,
        'device': str(device),
        'data_path': args.data_path,
        'timestamp': timestamp,
        'experiment_type': 'five_fold_cross_validation',
        'optimizer': 'Adam(β1=0.9, β2=0.999, ε=1e-8)',
        'initialization': 'Xavier',
        'lr_schedule': 'Dynamic(window=100, ratios=[0.001, 0.0005, 0.0002, 0.0001])',
        'model_selection': 'Best validation WA'
    }
    
    print("\n📋 五折交叉验证配置")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # 加载数据并创建五折划分
    print("\n📂 数据加载与划分")
    print("="*50)
    session_data = load_data_by_session(args.data_path)
    fold_splits = create_five_fold_splits(session_data)
    
    print(f"✅ 成功创建 {len(fold_splits)} 折数据划分")
    for i, (train_sessions, val_session, test_session) in enumerate(fold_splits):
        print(f"   Fold {i+1}: 训练={train_sessions}, 验证={val_session}, 测试={test_session}")
    
    # 执行五折交叉验证
    all_fold_results = []
    total_start_time = time.time()
    
    print("\n🚀 开始五折交叉验证")
    print("="*80)
    
    for fold_idx, (train_data, val_data, test_data) in enumerate(fold_splits):
        # 数据已经在create_five_fold_splits中准备好了
        
        # 训练当前折
        fold_result = train_single_fold(fold_idx, train_data, val_data, test_data, config, experiment_dir)
        all_fold_results.append(fold_result)
    
    total_time = time.time() - total_start_time
    
    # 计算五折交叉验证的总体结果
    print("\n📊 五折交叉验证总体结果")
    print("="*80)
    
    test_uas = [result['test_results']['UA'] for result in all_fold_results]
    test_was = [result['test_results']['WA'] for result in all_fold_results]
    val_was = [result['best_val_wa'] for result in all_fold_results]
    
    mean_test_ua = np.mean(test_uas)
    std_test_ua = np.std(test_uas)
    mean_test_wa = np.mean(test_was)
    std_test_wa = np.std(test_was)
    mean_val_wa = np.mean(val_was)
    std_val_wa = np.std(val_was)
    
    print(f"🎯 测试集 UA: {mean_test_ua:.4f} ± {std_test_ua:.4f}")
    print(f"🎯 测试集 WA: {mean_test_wa:.4f} ± {std_test_wa:.4f}")
    print(f"🎯 验证集 WA: {mean_val_wa:.4f} ± {std_val_wa:.4f}")
    print(f"⏱️  总训练时间: {total_time:.2f}秒")
    
    print("\n📋 各折详细结果:")
    for i, result in enumerate(all_fold_results):
        print(f"   Fold {i+1}: 测试UA={result['test_results']['UA']:.4f}, "
              f"测试WA={result['test_results']['WA']:.4f}, "
              f"验证WA={result['best_val_wa']:.4f}")
    
    # 保存总体实验结果
    final_results = {
        'config': config,
        'total_time': total_time,
        'cross_validation_results': {
            'mean_test_ua': mean_test_ua,
            'std_test_ua': std_test_ua,
            'mean_test_wa': mean_test_wa,
            'std_test_wa': std_test_wa,
            'mean_val_wa': mean_val_wa,
            'std_val_wa': std_val_wa,
            'individual_folds': all_fold_results
        }
    }
    
    with open(os.path.join(experiment_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 实验结果已保存到: {experiment_dir}")
    print(f"🏆 最终结果: 测试UA={mean_test_ua:.4f}±{std_test_ua:.4f}, 测试WA={mean_test_wa:.4f}±{std_test_wa:.4f}")

if __name__ == "__main__":
    main()