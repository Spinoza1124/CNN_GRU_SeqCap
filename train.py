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
    print("Warning: matplotlib not installed, skipping training curve plotting")
from collections import defaultdict

from model.CNN_GRU_SeqCap import CNN_GRU_SeqCap
import gc

class GPUManager:
    """
    GPU检测和管理类
    提供GPU型号检测、选择、验证和错误处理功能
    """
    
    def __init__(self):
        self.available_gpus = []
        self.gpu_info = {}
        self.detect_gpus()
    
    def detect_gpus(self):
        """检测系统中可用的GPU设备"""
        self.available_gpus = []
        self.gpu_info = {}
        
        # 添加CPU选项
        self.available_gpus.append("cpu")
        self.gpu_info["cpu"] = {
            "name": "CPU (Central Processing Unit)",
            "memory": "System RAM",
            "device_id": "cpu"
        }
        
        # 检测CUDA GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    
                    device_key = f"cuda:{i}"
                    self.available_gpus.append(device_key)
                    self.gpu_info[device_key] = {
                        "name": gpu_name,
                        "memory": f"{gpu_memory:.1f} GB",
                        "device_id": device_key,
                        "gpu_id": i
                    }
                except Exception as e:
                    print(f"⚠️  Error detecting GPU {i}: {e}")
        
        return self.available_gpus
    
    def list_available_gpus(self):
        """列出所有可用的GPU设备信息"""
        print("\n" + "="*60)
        print("🔍 Available GPU Devices:")
        print("="*60)
        
        for i, gpu_key in enumerate(self.available_gpus):
            if gpu_key in self.gpu_info:
                info = self.gpu_info[gpu_key]
                status = "✅ Available"
                
                # 验证GPU可用性
                if gpu_key != "cpu":
                    is_available, _ = self.validate_gpu_availability(gpu_key)
                    status = "✅ Available" if is_available else "❌ Unavailable"
                
                print(f"  [{i}] {gpu_key}:")
                print(f"      Name: {info['name']}")
                print(f"      Memory: {info['memory']}")
                print(f"      Status: {status}")
                print()
        
        print("="*60)
    
    def validate_gpu_availability(self, gpu_key):
        """验证指定GPU是否可用"""
        if gpu_key == "cpu":
            return True, "CPU is always available"
        
        if gpu_key.startswith("cuda:"):
            try:
                gpu_id = int(gpu_key.split(":")[1])
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    # 测试GPU访问性
                    device = torch.device(gpu_key)
                    test_tensor = torch.tensor([1.0]).to(device)
                    del test_tensor  # 清理测试张量
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    return True, f"GPU {gpu_id} is available and accessible"
                else:
                    return False, f"GPU {gpu_id} is not available or CUDA is not accessible"
            except Exception as e:
                return False, f"Error accessing GPU {gpu_key}: {str(e)}"
        
        return False, f"Unknown device: {gpu_key}"
    
    def get_device_by_index(self, index):
        """根据索引获取设备"""
        if 0 <= index < len(self.available_gpus):
            return self.available_gpus[index]
        return None
    
    def get_device_info(self, gpu_key):
        """获取设备详细信息"""
        return self.gpu_info.get(gpu_key, None)
    
    def select_best_gpu(self):
        """自动选择最佳GPU（内存最大的可用GPU）"""
        best_gpu = "cpu"
        max_memory = 0
        
        for gpu_key in self.available_gpus:
            if gpu_key == "cpu":
                continue
                
            is_available, _ = self.validate_gpu_availability(gpu_key)
            if is_available and gpu_key in self.gpu_info:
                try:
                    gpu_id = int(gpu_key.split(":")[1])
                    memory = torch.cuda.get_device_properties(gpu_id).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = gpu_key
                except:
                    continue
        
        return best_gpu

def setup_device_with_gpu_selection(args, gpu_manager):
    """
    根据用户选择和参数设置计算设备
    支持交互式GPU选择和自动选择
    """
    device = None
    
    # 如果用户指定了特定GPU
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        if args.gpu_id == -1:  # -1 表示CPU
            device = torch.device('cpu')
            print(f"🖥️  Using device: {device} (user specified)")
        else:
            gpu_key = f"cuda:{args.gpu_id}"
            is_available, message = gpu_manager.validate_gpu_availability(gpu_key)
            if is_available:
                device = torch.device(gpu_key)
                print(f"🖥️  Using device: {device} - GPU #{args.gpu_id} (user specified)")
            else:
                print(f"❌ Error: {message}")
                print("🔄 Falling back to automatic device selection...")
                device = None
    
    # 如果用户强制使用CPU
    elif args.cpu_only:
        device = torch.device('cpu')
        print(f"🖥️  Using device: {device} (CPU-only mode specified)")
    
    # 交互式GPU选择模式
    elif hasattr(args, 'interactive_gpu') and args.interactive_gpu:
        gpu_manager.list_available_gpus()
        
        while True:
            try:
                choice = input(f"\n🎯 Please select a device [0-{len(gpu_manager.available_gpus)-1}] or 'auto' for automatic selection: ").strip().lower()
                
                if choice == 'auto':
                    selected_gpu = gpu_manager.select_best_gpu()
                    device = torch.device(selected_gpu)
                    gpu_id = selected_gpu.split(':')[1] if ':' in selected_gpu else 'N/A'
                    print(f"🤖 Auto-selected device: {device} - GPU #{gpu_id}")
                    break
                else:
                    index = int(choice)
                    selected_gpu = gpu_manager.get_device_by_index(index)
                    if selected_gpu:
                        is_available, message = gpu_manager.validate_gpu_availability(selected_gpu)
                        if is_available:
                            device = torch.device(selected_gpu)
                            gpu_id = selected_gpu.split(':')[1] if ':' in selected_gpu else 'N/A'
                            print(f"✅ Selected device: {device} - GPU #{gpu_id}")
                            break
                        else:
                            print(f"❌ Error: {message}")
                            print("Please select another device.")
                    else:
                        print("❌ Invalid selection. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input. Please enter a number or 'auto'.")
    
    # 自动设备选择（默认行为）
    if device is None:
        if torch.cuda.is_available():
            # 自动选择最佳GPU
            best_gpu = gpu_manager.select_best_gpu()
            device = torch.device(best_gpu)
            gpu_id = best_gpu.split(':')[1] if ':' in best_gpu else 'N/A'
            print(f"🤖 Auto-selected device: {device} - GPU #{gpu_id}")
        else:
            device = torch.device('cpu')
            print(f"🖥️  Using device: {device} (CUDA not available)")
    
    # 显示设备详细信息
    if device.type == 'cuda':
        gpu_info = gpu_manager.get_device_info(str(device))
        if gpu_info:
            print(f"📊 GPU Info: {gpu_info['name']} ({gpu_info['memory']})")
        
        print(f"✅ CUDA detected, using GPU accelerated training")
        print(f"   GPU device count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.1f} GB")
    elif torch.cuda.is_available():
        print("⚠️  CUDA detected but using CPU")
    else:
        print("⚠️  CUDA not detected, running on CPU")
    
    return device

def xavier_init_weights(model):
    """Apply Xavier initialization to CNN and Capsule layers"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

def validate_model_input(model, sample_input, device):
    """Validate model architecture and input compatibility"""
    print(f"\n🔍 Validating model architecture and input compatibility...")
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Input device: {sample_input.device}")
    print(f"   Model device: {next(model.parameters()).device}")
    
    try:
        model.eval()
        with torch.no_grad():
            # Test forward pass with sample input
            output = model(sample_input)
            print(f"   ✅ Forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output device: {output.device}")
            
            # Check output dimensions
            expected_batch_size = sample_input.shape[0]
            if output.shape[0] != expected_batch_size:
                raise ValueError(f"Batch size mismatch: expected {expected_batch_size}, got {output.shape[0]}")
            
            # Check if output is probability distribution
            if len(output.shape) == 2:
                prob_sum = torch.sum(output, dim=1)
                print(f"   Probability sum range: [{prob_sum.min():.4f}, {prob_sum.max():.4f}]")
                if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
                    print(f"   ⚠️  Warning: Output may not be properly normalized probabilities")
            
            print(f"   ✅ Model validation passed")
            return True
            
    except Exception as e:
        print(f"   ❌ Model validation failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False
    finally:
        model.train()  # Reset to training mode

class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class DynamicLearningRateScheduler:
    """Dynamic learning rate scheduler based on recent 100 training steps average loss"""
    
    def __init__(self, optimizer, initial_lr=0.001):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.train_losses = []  # Store recent training losses
        self.window_size = 100  # Window size for averaging
        self.epoch_count = 0
        self.lr_stages = [0.001, 0.0005, 0.0002, 0.0001]  # Learning rate stages
        self.current_stage = 0
        self.previous_avg_loss = float('inf')
        
    def add_train_loss(self, loss):
        """Add training loss to the window"""
        self.train_losses.append(loss)
        if len(self.train_losses) > self.window_size:
            self.train_losses.pop(0)  # Keep only recent losses
            
    def step_epoch(self):
        """Called at the end of each epoch"""
        self.epoch_count += 1
        
        # For first 3 epochs, keep initial learning rate
        if self.epoch_count <= 3:
            return
            
        # Check if we have enough losses to calculate average
        if len(self.train_losses) >= self.window_size:
            current_avg_loss = sum(self.train_losses) / len(self.train_losses)
            
            # Check if average loss decreased by 10x
            if self.previous_avg_loss != float('inf') and current_avg_loss <= self.previous_avg_loss / 10:
                if self.current_stage < len(self.lr_stages) - 1:
                    self.current_stage += 1
                    new_lr = self.lr_stages[self.current_stage]
                    self.current_lr = new_lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"\n🔄 Learning rate reduced to {new_lr:.6f} (avg loss decreased 10x: {self.previous_avg_loss:.6f} -> {current_avg_loss:.6f})")
                    
            self.previous_avg_loss = current_avg_loss
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_info(self):
        """Get learning rate scheduling information"""
        avg_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0
        return {
            'current_lr': self.get_current_lr(),
            'epoch_count': self.epoch_count,
            'current_stage': self.current_stage,
            'avg_train_loss': avg_loss,
            'window_size': len(self.train_losses)
        }

class MemoryManager:
    """Memory monitoring and dynamic batch size adjustment"""
    
    def __init__(self, initial_batch_size=16, min_batch_size=1):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.oom_count = 0
        
    def get_gpu_memory_info(self, device):
        """Get current GPU memory usage"""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            return {
                'allocated': allocated,
                'cached': cached,
                'total': total,
                'free': total - allocated
            }
        return None
    
    def clear_cache(self, device):
        """Clear GPU cache"""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def handle_oom(self, device):
        """Handle out of memory error by reducing batch size"""
        self.oom_count += 1
        self.clear_cache(device)
        
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            print(f"⚠️  CUDA OOM detected! Reducing batch size to {self.current_batch_size}")
            return True
        else:
            print(f"❌ Cannot reduce batch size further (current: {self.current_batch_size})")
            return False
    
    def get_current_batch_size(self):
        return self.current_batch_size
    
    def monitor_memory_usage(self, device, threshold_percent=85):
        """Monitor GPU memory usage and issue warnings"""
        if device.type == 'cuda':
            mem_info = self.get_gpu_memory_info(device)
            if mem_info:
                usage_percent = (mem_info['allocated'] / mem_info['total']) * 100
                
                if usage_percent > threshold_percent:
                    print(f"⚠️  High GPU memory usage: {usage_percent:.1f}% ({mem_info['allocated']:.1f}GB/{mem_info['total']:.1f}GB)")
                    return True
                    
        return False
    
    def get_memory_stats(self, device):
        """Get detailed memory statistics"""
        stats = {
            'oom_count': self.oom_count,
            'current_batch_size': self.current_batch_size,
            'initial_batch_size': self.initial_batch_size
        }
        
        if device.type == 'cuda':
            mem_info = self.get_gpu_memory_info(device)
            if mem_info:
                stats.update({
                    'gpu_memory_allocated_gb': mem_info['allocated'],
                    'gpu_memory_cached_gb': mem_info['cached'],
                    'gpu_memory_total_gb': mem_info['total'],
                    'gpu_memory_free_gb': mem_info['free'],
                    'gpu_memory_usage_percent': (mem_info['allocated'] / mem_info['total']) * 100
                })
        
        return stats

class DataAugmentation:
    """Data augmentation for spectrograms"""
    
    def __init__(self, noise_factor=0.01, time_mask_param=10, freq_mask_param=5):
        self.noise_factor = noise_factor
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def add_noise(self, spec):
        """Add Gaussian noise"""
        noise = torch.randn_like(spec) * self.noise_factor
        return spec + noise
    
    def time_mask(self, spec):
        """Time masking"""
        if len(spec.shape) == 3:
            _, freq_bins, time_steps = spec.shape
            mask_size = np.random.randint(0, min(self.time_mask_param, time_steps))
            mask_start = np.random.randint(0, time_steps - mask_size + 1)
            spec_masked = spec.clone()
            spec_masked[:, :, mask_start:mask_start + mask_size] = 0
        else:
            _, _, freq_bins, time_steps = spec.shape
            mask_size = np.random.randint(0, min(self.time_mask_param, time_steps))
            mask_start = np.random.randint(0, time_steps - mask_size + 1)
            spec_masked = spec.clone()
            spec_masked[:, :, :, mask_start:mask_start + mask_size] = 0
        return spec_masked
    
    def freq_mask(self, spec):
        """Frequency masking"""
        if len(spec.shape) == 3:
            _, freq_bins, time_steps = spec.shape
            mask_size = np.random.randint(0, min(self.freq_mask_param, freq_bins))
            mask_start = np.random.randint(0, freq_bins - mask_size + 1)
            spec_masked = spec.clone()
            spec_masked[:, mask_start:mask_start + mask_size, :] = 0
        else:
            _, _, freq_bins, time_steps = spec.shape
            mask_size = np.random.randint(0, min(self.freq_mask_param, freq_bins))
            mask_start = np.random.randint(0, freq_bins - mask_size + 1)
            spec_masked = spec.clone()
            spec_masked[:, :, mask_start:mask_start + mask_size, :] = 0
        return spec_masked
    
    def __call__(self, spec):
        """Randomly apply data augmentation"""
        if np.random.random() < 0.5:
            spec = self.add_noise(spec)
        if np.random.random() < 0.3:
            spec = self.time_mask(spec)
        if np.random.random() < 0.3:
            spec = self.freq_mask(spec)
        return spec

class IEMOCAPDataset(Dataset):
    """IEMOCAP dataset loader with enhanced data preprocessing"""
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict: Dictionary containing seg_spec, seg_label, etc.
            transform: Data transformation function
        """
        self.seg_spec = data_dict['seg_spec']  # Expected: (N, 1, freq, time)
        self.seg_label = data_dict['seg_label']  # (N,)
        self.transform = transform
        
        # Validate and preprocess data dimensions
        print(f"Original data shape: {self.seg_spec.shape}")
        
        # Ensure data is in correct format [N, 1, freq, time]
        if len(self.seg_spec.shape) == 3:
            # If shape is (N, freq, time), add channel dimension
            self.seg_spec = np.expand_dims(self.seg_spec, axis=1)
            print(f"Added channel dimension, new shape: {self.seg_spec.shape}")
        elif len(self.seg_spec.shape) == 4:
            # Shape is already (N, C, freq, time)
            print(f"Data already has correct 4D shape: {self.seg_spec.shape}")
        else:
            raise ValueError(f"Unexpected data shape: {self.seg_spec.shape}")
        
        # Validate label format
        if len(self.seg_label.shape) > 1:
            self.seg_label = self.seg_label.flatten()
            print(f"Flattened labels, new shape: {self.seg_label.shape}")
        
        print(f"Final data shape: {self.seg_spec.shape}")
        print(f"Label shape: {self.seg_label.shape}")
        print(f"Label range: [{np.min(self.seg_label)}, {np.max(self.seg_label)}]")
        
    def __len__(self):
        return len(self.seg_spec)
    
    def __getitem__(self, idx):
        spec = self.seg_spec[idx]  # (1, freq, time) or (C, freq, time)
        label = self.seg_label[idx]
        
        # Convert to torch tensor
        spec = torch.FloatTensor(spec)
        label = torch.LongTensor([label])[0]
        
        # Ensure spec has correct shape [1, freq, time]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # Add channel dimension
        
        # Apply data augmentation if provided
        if self.transform:
            spec = self.transform(spec)
            
        return spec, label

def load_data_by_speaker(data_path):
    """Load IEMOCAP data by speaker for leave-one-speaker-out cross-validation"""
    print(f"Loading data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    speaker_data = {}
    total_samples = 0
    
    # Check data format and process
    first_value = list(data.values())[0]
    if isinstance(first_value, dict):
        # Check if it's the nested format with 'features' key
        first_nested_value = list(first_value.values())[0]
        has_features_key = isinstance(first_nested_value, dict) and 'features' in first_nested_value
    else:
        has_features_key = False
        
    if isinstance(first_value, dict) and has_features_key:
        # Test data format: Session1 -> {Session1_M: {features: [], labels: []}, Session1_F: {...}}
        for session_key in sorted(data.keys()):
            session_info = data[session_key]
            
            for speaker_key in session_info.keys():
                speaker_data_info = session_info[speaker_key]
                
                # Convert to numpy arrays
                speaker_data[speaker_key] = {
                    'seg_spec': np.array(speaker_data_info['features']),
                    'seg_label': np.array(speaker_data_info['labels'])
                }
                
                num_samples = len(speaker_data_info['features'])
                total_samples += num_samples
                print(f"Speaker {speaker_key}: {num_samples} samples")
    else:
        # Real IEMOCAP data format: 1F, 1M, 2F, 2M, ...
        # Keep data separated by speaker for leave-one-speaker-out
        for speaker_key in sorted(data.keys()):
            speaker_info = data[speaker_key]
            
            speaker_data[speaker_key] = {
                'seg_spec': speaker_info['seg_spec'],
                'seg_label': speaker_info['seg_label']
            }
            
            num_samples = speaker_info['seg_spec'].shape[0]
            total_samples += num_samples
            print(f"Speaker {speaker_key}: {num_samples} samples")
    
    print(f"Total samples: {total_samples}")
    print(f"Number of speakers: {len(speaker_data)}")
    
    return speaker_data

def create_ten_fold_speaker_splits(speaker_data):
    """Create 10-fold leave-one-speaker-out cross-validation data splits
    
    Args:
        speaker_data: Data dictionary organized by speaker
        
    Returns:
        List of 10 folds, each containing (train_data, val_data, test_data)
    """
    speakers = list(speaker_data.keys())
    num_speakers = len(speakers)
    
    if num_speakers < 3:
        raise ValueError(f"Need at least 3 speakers for cross-validation, but found {num_speakers}")
    
    # For testing with fewer speakers, adjust the number of folds
    num_folds = min(num_speakers, 10)  # Use actual number of speakers if less than 10
    
    if num_speakers != 10:
        print(f"⚠️  Warning: Expected 10 speakers for IEMOCAP, but found {num_speakers}")
        print(f"   Using {num_folds}-fold cross-validation instead")
    
    folds = []
    
    for i in range(num_folds):
        test_speaker = speakers[i % num_speakers]  # Circular selection for test speaker
        print(f"\n=== Creating fold {i+1}/{num_folds} cross-validation (Leave-One-Speaker-Out) ===")
        print(f"Test speaker: {test_speaker}")
        
        # Get remaining speakers for training and validation
        remaining_speakers = [s for s in speakers if s != test_speaker]
        
        if len(remaining_speakers) >= 2:
            # Use one speaker for validation, rest for training
            val_speaker = remaining_speakers[i % len(remaining_speakers)]
            train_speakers = [s for s in remaining_speakers if s != val_speaker]
        else:
            # If only one remaining speaker, use it for both training and validation
            val_speaker = remaining_speakers[0]
            train_speakers = remaining_speakers
            print(f"   ⚠️  Warning: Using same speaker for training and validation due to limited data")
        
        print(f"Training speakers: {train_speakers}")
        print(f"Validation speaker: {val_speaker}")
        
        # Merge training data from 8 speakers
        train_specs = []
        train_labels = []
        for speaker in train_speakers:
            train_specs.append(speaker_data[speaker]['seg_spec'])
            train_labels.append(speaker_data[speaker]['seg_label'])
        
        train_data = {
            'seg_spec': np.concatenate(train_specs, axis=0),
            'seg_label': np.concatenate(train_labels, axis=0)
        }
        
        # Validation data from 1 speaker
        val_data = {
            'seg_spec': speaker_data[val_speaker]['seg_spec'],
            'seg_label': speaker_data[val_speaker]['seg_label']
        }
        
        # Test data from the left-out speaker
        test_data = {
            'seg_spec': speaker_data[test_speaker]['seg_spec'],
            'seg_label': speaker_data[test_speaker]['seg_label']
        }
        
        print(f"Training set: {len(train_data['seg_spec'])} samples")
        print(f"Validation set: {len(val_data['seg_spec'])} samples")
        print(f"Test set: {len(test_data['seg_spec'])} samples")
        
        folds.append((train_data, val_data, test_data))
    
    return folds

def normalize_data(train_data, val_data, test_data):
    """Normalize data (zero mean unit variance normalization)
    
    Args:
        train_data, val_data, test_data: Data dictionaries
        
    Returns:
        Normalized data and normalization parameters
    """
    # Calculate mean and standard deviation of training set
    train_spec = train_data['seg_spec']
    
    # Check data dimensions and calculate statistics accordingly
    if train_spec.ndim == 3:  # (samples, freq, time)
        mean = np.mean(train_spec, axis=0, keepdims=True)  # Shape: (1, freq, time)
        std = np.std(train_spec, axis=0, keepdims=True)
    elif train_spec.ndim == 4:  # (samples, channels, freq, time)
        mean = np.mean(train_spec, axis=(0, 2, 3), keepdims=True)  # Keep dimensions for broadcasting
        std = np.std(train_spec, axis=(0, 2, 3), keepdims=True)
    else:
        raise ValueError(f"Unexpected data dimensions: {train_spec.shape}. Expected 3D or 4D array.")
    
    print(f"Data shape: {train_spec.shape}, dimensions: {train_spec.ndim}D")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    print(f"Data normalization parameters: mean={mean.flatten()}, std={std.flatten()}")
    
    # Normalize all datasets
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
    """Split training, validation, and test sets"""
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
    
    print(f"Data split: training={len(train_indices)}, validation={len(val_indices)}, test={len(test_indices)}")
    
    return train_data, val_data, test_data

def calculate_metrics(y_true, y_pred, num_classes=4):
    """Calculate evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    # Unweighted accuracy (UA) - average of per-class accuracies
    class_accuracies = []
    for i in range(num_classes):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    ua = np.mean(class_accuracies)
    
    # Weighted accuracy (WA) - overall accuracy
    wa = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'UA': ua,
        'WA': wa,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
    }

def train_epoch(model, dataloader, criterion, optimizer, device, lr_scheduler=None, memory_manager=None):
    """Train one epoch with memory monitoring"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training", file=sys.stdout)
    
    for batch_idx, (data, target) in enumerate(pbar):
        try:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Add loss to learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.add_train_loss(loss.item())
            
            # Prediction results
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar display
            current_avg_loss = total_loss / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]['lr']
            postfix = {
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{current_avg_loss:.6f}',
                'LR': f'{current_lr:.6f}'
            }
            
            # Add memory info if available
            if memory_manager and device.type == 'cuda':
                mem_info = memory_manager.get_gpu_memory_info(device)
                if mem_info:
                    postfix['GPU_Mem'] = f'{mem_info["allocated"]:.1f}GB'
            
            pbar.set_postfix(postfix)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and memory_manager:
                print(f"\n💥 CUDA OOM Error at batch {batch_idx}: {str(e)}")
                
                # Handle OOM by clearing cache and potentially reducing batch size
                if memory_manager.handle_oom(device):
                    print("🔄 Retrying with reduced memory usage...")
                    continue
                else:
                    print("❌ Cannot recover from OOM error")
                    raise e
            else:
                raise e
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds) if all_labels else {}
    
    return avg_loss, metrics

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Validating", file=sys.stdout)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar display
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{current_avg_loss:.6f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def save_experiment_log(log_data, save_dir):
    """Save experiment log"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save log in JSON format
    log_file = os.path.join(save_dir, 'experiment_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment log saved to: {log_file}")

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """Plot training curves"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping training curve plotting (matplotlib not installed)")
        return
        
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # UA curves
    plt.subplot(1, 3, 2)
    train_ua = [m['UA'] for m in train_metrics]
    val_ua = [m['UA'] for m in val_metrics]
    plt.plot(epochs, train_ua, 'b-', label='Training UA')
    plt.plot(epochs, val_ua, 'r-', label='Validation UA')
    plt.title('Unweighted Accuracy (UA)')
    plt.xlabel('Epoch')
    plt.ylabel('UA')
    plt.legend()
    plt.grid(True)
    
    # WA curves
    plt.subplot(1, 3, 3)
    train_wa = [m['WA'] for m in train_metrics]
    val_wa = [m['WA'] for m in val_metrics]
    plt.plot(epochs, train_wa, 'b-', label='Training WA')
    plt.plot(epochs, val_wa, 'r-', label='Validation WA')
    plt.title('Weighted Accuracy (WA)')
    plt.xlabel('Epoch')
    plt.ylabel('WA')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")

def train_single_fold(fold_idx, train_data, val_data, test_data, config, experiment_dir):
    """Train single fold"""
    print(f"\n{'='*80}")
    print(f"🔄 Starting fold {fold_idx + 1} training")
    print(f"{'='*80}")
    
    # Automatically select device: prioritize GPU, use CPU if unavailable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize memory manager for dynamic batch size adjustment
    memory_manager = MemoryManager(initial_batch_size=config['batch_size'], min_batch_size=1)
    print(f"🧠 Memory manager initialized with batch size: {memory_manager.get_current_batch_size()}")
    
    # Display initial GPU memory info if using CUDA
    if device.type == 'cuda':
        mem_info = memory_manager.get_gpu_memory_info(device)
        if mem_info:
            print(f"📊 Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free (Total: {mem_info['total']:.1f}GB)")
    
    # Data normalization
    train_data, val_data, test_data, normalization_params = normalize_data(train_data, val_data, test_data)
    
    # Create data augmentation
    train_augmentation = DataAugmentation(noise_factor=0.01, time_mask_param=10, freq_mask_param=5)
    
    # Create data loaders (use data augmentation only for training set)
    train_dataset = IEMOCAPDataset(train_data, transform=train_augmentation)
    val_dataset = IEMOCAPDataset(val_data)
    test_dataset = IEMOCAPDataset(test_data)
    
    # Set num_workers: use multi-threading for GPU training, single-threading for CPU training
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    print(f"📊 Data Statistics:")
    print(f"   Training set: {len(train_dataset)} samples")
    print(f"   Validation set: {len(val_dataset)} samples")
    print(f"   Test set: {len(test_dataset)} samples")
    
    # GPU memory management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🔧 GPU cache cleared")
    
    # Create model (CNN-GRU-SeqCap architecture)
    # Note: The model only accepts num_classes and lambda_val parameters
    model = CNN_GRU_SeqCap(
        num_classes=config['num_classes'],
        lambda_val=config.get('lambda_val', 0.6)  # Fusion weight between capsule and GRU branches
    ).to(device)
    
    # Apply Xavier initialization
    xavier_init_weights(model)
    print(f"✅ Applied Xavier initialization to CNN and Capsule layers")
    
    # Validate model with sample input
    try:
        # Create a sample batch for validation
        sample_batch = next(iter(train_loader))
        sample_input, sample_target = sample_batch
        sample_input = sample_input.to(device)
        
        # Validate model architecture and input compatibility
        validation_success = validate_model_input(model, sample_input, device)
        if not validation_success:
            raise RuntimeError("Model validation failed. Please check model architecture and input data format.")
            
    except Exception as e:
        print(f"❌ Error during model validation: {str(e)}")
        raise e
    
    # Define loss function and optimizer (according to user specifications)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          betas=(0.9, 0.999), eps=1e-8)  # β1=0.9, β2=0.999, ε=1e-8
    
    # Create dynamic learning rate scheduler (based on recent 100 training steps)
    lr_scheduler = DynamicLearningRateScheduler(optimizer, initial_lr=config['learning_rate'])
    
    # Create early stopping mechanism
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training records
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_val_wa = 0.0  # Use WA as model selection criterion
    best_model_state = None
    
    print(f"\n🚀 Starting training ({config['epochs']} epochs total)")
    start_time = time.time()
    
    # Create epoch-level progress bar
    epoch_pbar = tqdm(range(config['epochs']), desc=f"Fold {fold_idx + 1}", position=0, leave=True)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Training with memory monitoring
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device, lr_scheduler, memory_manager)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        
        # Validation
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # Update learning rate scheduler (based on recent training losses)
        lr_scheduler.step_epoch()
        
        # Check early stopping
        if early_stopping(val_loss):
            print(f"\n⏹️  Early stopping triggered! Stopped training at epoch {epoch+1}")
            print(f"   Validation loss has not improved for {early_stopping.patience} consecutive epochs")
            break
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Save best model (based on validation WA)
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
        
        # Update progress bar
        lr_info = lr_scheduler.get_lr_info()
        epoch_pbar.set_postfix({
            'T_Loss': f'{train_loss:.4f}',
            'V_WA': f'{val_metric["WA"]:.4f}',
            'Best_WA': f'{best_val_wa:.4f}',
            'LR': f'{lr_info["current_lr"]:.6f}'
        })
        
        # Print detailed information every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Epoch {epoch+1}/{config['epochs']} Results:")
            print(f"   Training: Loss={train_loss:.6f}, UA={train_metric['UA']:.4f}, WA={train_metric['WA']:.4f}")
            print(f"   Validation: Loss={val_loss:.6f}, UA={val_metric['UA']:.4f}, WA={val_metric['WA']:.4f}")
            print(f"   Learning rate: {lr_info['current_lr']:.6f}, Time: {epoch_time:.2f}s")
    
    epoch_pbar.close()
    
    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # Test set evaluation
    test_loss, test_metric = evaluate(model, test_loader, criterion, device)
    
    training_time = time.time() - start_time
    
    # Get final memory statistics
    final_memory_stats = memory_manager.get_memory_stats(device)
    
    # GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\n✅ Fold {fold_idx + 1} training completed")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best validation WA: {best_val_wa:.4f}")
    print(f"   Test results: UA={test_metric['UA']:.4f}, WA={test_metric['WA']:.4f}")
    
    # Display memory usage summary
    if final_memory_stats['oom_count'] > 0:
        print(f"   ⚠️  OOM events: {final_memory_stats['oom_count']}")
        print(f"   📉 Final batch size: {final_memory_stats['current_batch_size']} (initial: {final_memory_stats['initial_batch_size']})")
    
    if device.type == 'cuda' and 'gpu_memory_usage_percent' in final_memory_stats:
        print(f"   🧠 Final GPU memory usage: {final_memory_stats['gpu_memory_usage_percent']:.1f}%")
    
    # Save fold results
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
        },
        'memory_stats': final_memory_stats
    }
    
    # Save fold model and results
    fold_dir = os.path.join(experiment_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(fold_dir, 'best_model.pth'))
    
    with open(os.path.join(fold_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, fold_dir)
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description='CNN-GRU-SeqCap Ten-Fold Leave-One-Speaker-Out Cross-Validation Training Script')
    parser.add_argument('--data_path', type=str, default='data/IEMOCAP_multi.pkl',
                        help='Data file path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes (default: 4)')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='Experiment results save directory')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU training (even if GPU is available)')
    
    # GPU selection parameters
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Specific GPU ID to use (0, 1, 2, ...), -1 for CPU, None for auto-selection')
    parser.add_argument('--interactive_gpu', action='store_true',
                        help='Enable interactive GPU selection mode')
    parser.add_argument('--list_gpus', action='store_true',
                        help='List available GPU devices and exit')
    
    # CNN-GRU-SeqCap model parameters
    parser.add_argument('--lambda_val', type=float, default=0.6,
                        help='Fusion weight between capsule and GRU branches (default: 0.6)')
    
    args = parser.parse_args()
    
    # Handle GPU listing request
    if args.list_gpus:
        gpu_manager = GPUManager()
        gpu_manager.list_available_gpus()
        return
    
    # Setup device with GPU selection
    gpu_manager = GPUManager()
    device = setup_device_with_gpu_selection(args, gpu_manager)

    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"ten_fold_loso_cv_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save hyperparameter configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'num_classes': args.num_classes,
        'lambda_val': getattr(args, 'lambda_val', 0.6),  # Fusion weight between capsule and GRU branches
        'device': str(device),
        'data_path': args.data_path,
        'timestamp': timestamp,
        'experiment_type': 'ten_fold_leave_one_speaker_out_cross_validation',
        'optimizer': 'Adam(β1=0.9, β2=0.999, ε=1e-8)',
        'initialization': 'Xavier (CNN and Capsule layers)',
        'lr_schedule': 'Dynamic (first 3 epochs: 0.001, then 0.0005→0.0002→0.0001 based on 100-step avg loss)',
        'early_stopping': 'patience=10, min_delta=0.001',
        'regularization': 'Cross-entropy loss',
        'model_selection': 'Best validation WA (Weighted Accuracy)',
        'validation_strategy': '10-fold leave-one-speaker-out cross-validation',
        'model_architecture': 'CNN-GRU-SeqCap (dual-branch with CNN backbone + Capsule and BiGRU branches)',
        'data_augmentation': 'Gaussian noise, time masking, frequency masking'
    }
    
    print("\n📋 Ten-Fold Leave-One-Speaker-Out Cross-Validation Configuration")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Load data and create ten-fold speaker splits
    print("\n📂 Data Loading and Splitting")
    print("="*50)
    speaker_data = load_data_by_speaker(args.data_path)
    fold_splits = create_ten_fold_speaker_splits(speaker_data)
    
    print(f"✅ Successfully created {len(fold_splits)} fold data splits")
    
    # Execute ten-fold leave-one-speaker-out cross-validation
    all_fold_results = []
    total_start_time = time.time()
    
    print("\n🚀 Starting Ten-Fold Leave-One-Speaker-Out Cross-Validation")
    print("="*80)
    
    for fold_idx, (train_data, val_data, test_data) in enumerate(fold_splits):
        # Data is already prepared in create_five_fold_splits
        
        # Train current fold
        fold_result = train_single_fold(fold_idx, train_data, val_data, test_data, config, experiment_dir)
        all_fold_results.append(fold_result)
    
    total_time = time.time() - total_start_time
    
    # Calculate overall results of ten-fold cross-validation
    print("\n📊 Ten-Fold Leave-One-Speaker-Out Cross-Validation Overall Results")
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
    
    print(f"🎯 Test UA: {mean_test_ua:.4f} ± {std_test_ua:.4f}")
    print(f"🎯 Test WA: {mean_test_wa:.4f} ± {std_test_wa:.4f}")
    print(f"🎯 Validation WA: {mean_val_wa:.4f} ± {std_val_wa:.4f}")
    print(f"⏱️  Total training time: {total_time:.2f} seconds")
    
    print("\n📋 Detailed results for each fold:")
    for i, result in enumerate(all_fold_results):
        print(f"   Fold {i+1}: Test UA={result['test_results']['UA']:.4f}, "
              f"Test WA={result['test_results']['WA']:.4f}, "
              f"Val WA={result['best_val_wa']:.4f}")
    
    # Save overall experiment results
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
    
    print(f"\n💾 Experiment results saved to: {experiment_dir}")
    print(f"🏆 Final results: Test UA={mean_test_ua:.4f}±{std_test_ua:.4f}, Test WA={mean_test_wa:.4f}±{std_test_wa:.4f}")

if __name__ == "__main__":
    main()