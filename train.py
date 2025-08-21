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

class DataAugmentation:
    """Data augmentation class"""
    
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
    """IEMOCAP dataset loader"""
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict: Dictionary containing seg_spec, seg_label, etc.
            transform: Data transformation function
        """
        self.seg_spec = data_dict['seg_spec']  # (N, 1, 200, 300)
        self.seg_label = data_dict['seg_label']  # (N,)
        self.transform = transform
        
        # Data preprocessing: adjust dimensions to match model input [batch, 1, freq, time]
        # Original data is (N, 1, 200, 300), needs to be converted to (N, 1, 200, 300)
        # Model expects input as [batch, 1, freq_bins, time_steps]
        print(f"Original data shape: {self.seg_spec.shape}")
        
    def __len__(self):
        return len(self.seg_spec)
    
    def __getitem__(self, idx):
        spec = self.seg_spec[idx]  # (1, 200, 300)
        label = self.seg_label[idx]
        
        # Convert to torch tensor
        spec = torch.FloatTensor(spec)
        label = torch.LongTensor([label])[0]
        
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
    if isinstance(list(data.values())[0], dict) and 'features' in list(list(data.values())[0].values())[0]:
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
    if len(speakers) != 10:
        raise ValueError(f"IEMOCAP should have 10 speakers, but found {len(speakers)}")
    
    folds = []
    
    for i, test_speaker in enumerate(speakers):
        print(f"\n=== Creating fold {i+1} cross-validation (Leave-One-Speaker-Out) ===")
        print(f"Test speaker: {test_speaker}")
        
        # Get remaining 9 speakers for training and validation
        remaining_speakers = [s for s in speakers if s != test_speaker]
        
        # Use 8 speakers for training and 1 speaker for validation
        # Choose validation speaker as the next speaker in the list (circular)
        val_speaker = remaining_speakers[i % len(remaining_speakers)]
        train_speakers = [s for s in remaining_speakers if s != val_speaker]
        
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
    mean = np.mean(train_spec, axis=(0, 2, 3), keepdims=True)  # Keep dimensions for broadcasting
    std = np.std(train_spec, axis=(0, 2, 3), keepdims=True)
    
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

def train_epoch(model, dataloader, criterion, optimizer, device, lr_scheduler=None):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training", file=sys.stdout)
    
    for batch_idx, (data, target) in enumerate(pbar):
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
        pbar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg_Loss': f'{current_avg_loss:.6f}',
            'LR': f'{current_lr:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    
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
    
    # Create model (CNN-GRU-SeqCap architecture aligned with Wu et al. paper)
    model = CNN_GRU_SeqCap(
        num_classes=config['num_classes'],
        window_size=config.get('window_size', 40),  # Wu et al. paper: 40 input steps
        window_stride=config.get('window_stride', 20),  # Wu et al. paper: 20 step stride
        gru_hidden_size=config.get('gru_hidden_size', 128),  # GRU hidden dimension
        gru_num_layers=config.get('gru_num_layers', 2),  # Number of GRU layers
        dropout_rate=config.get('dropout_rate', 0.5)  # Dropout rate
    ).to(device)
    
    # Apply Xavier initialization
    xavier_init_weights(model)
    print(f"✅ Applied Xavier initialization to CNN and Capsule layers")
    
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
        
        # Training
        train_loss, train_metric = train_epoch(model, train_loader, criterion, optimizer, device, lr_scheduler)
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
    
    # GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\n✅ Fold {fold_idx + 1} training completed")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best validation WA: {best_val_wa:.4f}")
    print(f"   Test results: UA={test_metric['UA']:.4f}, WA={test_metric['WA']:.4f}")
    
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
        }
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
    
    # CNN-GRU-SeqCap model parameters (aligned with Wu et al. paper)
    parser.add_argument('--window_size', type=int, default=40,
                        help='Window size for temporal segmentation (Wu et al. paper: 40)')
    parser.add_argument('--window_stride', type=int, default=20,
                        help='Window stride for temporal segmentation (Wu et al. paper: 20)')
    parser.add_argument('--gru_hidden_size', type=int, default=128,
                        help='GRU hidden dimension (default: 128)')
    parser.add_argument('--gru_num_layers', type=int, default=2,
                        help='Number of GRU layers (default: 2)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    args = parser.parse_args()
    
    # Select device based on parameters and availability
    if args.cpu_only:
        device = torch.device('cpu')
        print(f"🖥️  Using device: {device} (user specified)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"✅ CUDA detected, using GPU accelerated training")
        print(f"   GPU device count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.cuda.is_available() and device.type == 'cpu':
        print("⚠️  CUDA detected but user chose to run on CPU")
    else:
        print("⚠️  CUDA not detected, running on CPU")
    
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
        'device': str(device),
        'data_path': args.data_path,
        'timestamp': timestamp,
        'experiment_type': 'ten_fold_leave_one_speaker_out_cross_validation',
        'optimizer': 'Adam(β1=0.9, β2=0.999, ε=1e-8)',
        'initialization': 'Xavier (CNN and Capsule layers)',
        'lr_schedule': 'Dynamic (first 3 epochs: 0.001, then 0.0005→0.0002→0.0001 based on 100-step avg loss)',
        'early_stopping': 'patience=7, min_delta=0.001',
        'regularization': 'Cross-entropy loss',
        'model_selection': 'Best validation WA (Weighted Accuracy)',
        'validation_strategy': '10-fold leave-one-speaker-out cross-validation',
        # CNN-GRU-SeqCap model parameters (aligned with Wu et al. paper)
        'window_size': 40,  # Wu et al. paper: 40 input steps
        'window_stride': 20,  # Wu et al. paper: 20 step stride
        'gru_hidden_size': 128,  # GRU hidden dimension
        'gru_num_layers': 2,  # Number of GRU layers
        'dropout_rate': 0.5,  # Dropout rate
        'model_architecture': 'CNN-GRU-SeqCap (dual-branch with CNN backbone + GRU temporal modeling)'
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