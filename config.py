#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for CNN-GRU-SeqCap model
Based on Wu et al. "Speech Emotion Recognition Using Capsule Networks"

This file contains all hyperparameters and configuration settings
for reproducible experiments and easy parameter tuning.
"""

import os

class Config:
    """
    Configuration class containing all model and training parameters
    """
    
    # ==================== Data Configuration ====================
    DATA_DIR = 'data'
    X_TRAIN_FILE = os.path.join(DATA_DIR, 'X_train.npy')
    Y_TRAIN_FILE = os.path.join(DATA_DIR, 'y_train.npy')
    
    # Data preprocessing
    INPUT_CHANNELS = 1
    FREQ_BINS = 128  # Frequency dimension of spectrogram
    MAX_TIME_STEPS = 200  # Maximum time steps in spectrogram
    NUM_CLASSES = 4  # Number of emotion classes
    
    # ==================== Model Configuration ====================
    # Based on Wu et al. paper specifications
    
    # CNN Branch (following Sabour et al. configuration)
    CNN_FILTERS_1 = 256  # First conv layer: 256 filters, 9x9, stride 1
    CNN_KERNEL_SIZE_1 = 9
    CNN_STRIDE_1 = 1
    CNN_PADDING_1 = 4
    
    CNN_FILTERS_2 = 256  # Second conv layer
    CNN_KERNEL_SIZE_2 = 3
    CNN_STRIDE_2 = 1
    CNN_PADDING_2 = 1
    
    # Primary Capsules (following Sabour et al.)
    PRIMARY_CAPS_FILTERS = 32 * 8  # 32 8-dimensional capsules
    PRIMARY_CAPS_KERNEL_SIZE = 9
    PRIMARY_CAPS_STRIDE = 2
    PRIMARY_CAPS_DIM = 8
    NUM_PRIMARY_CAPS = 32
    
    # GRU Branch
    GRU_HIDDEN_DIM = 128
    GRU_NUM_LAYERS = 2
    GRU_BIDIRECTIONAL = True
    
    # Feature Fusion
    FUSION_DIM = 256
    
    # Windowing (based on Wu et al. paper)
    WINDOW_SIZE = 40  # 40 input steps
    WINDOW_STRIDE = 20  # 20 step shift
    
    # Capsule Network
    WINDOW_EMO_CAPS_OUT_DIM = 16
    UTTERANCE_CAPS_OUT_DIM = 16
    ROUTING_ITERATIONS = 3
    
    # Regularization
    DROPOUT_RATE = 0.2
    WEIGHT_DECAY = 1e-4
    
    # ==================== Training Configuration ====================
    
    # Optimization
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'
    
    # Learning rate scheduling
    LR_SCHEDULER = 'reduce_on_plateau'
    LR_FACTOR = 0.5
    LR_PATIENCE = 5
    LR_MIN = 1e-6
    
    # Training procedure
    EPOCHS = 50
    BATCH_SIZE = 16
    EARLY_STOPPING_PATIENCE = 10
    
    # Cross-validation
    K_FOLDS = 5
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Gradient clipping
    GRAD_CLIP_NORM = 1.0
    
    # ==================== Loss Configuration ====================
    
    # Margin Loss (for capsule networks)
    MARGIN_LOSS_M_POS = 0.9
    MARGIN_LOSS_M_NEG = 0.1
    MARGIN_LOSS_LAMBDA = 0.5
    
    # ==================== Hardware Configuration ====================
    
    # Device
    USE_CUDA = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ==================== Logging and Saving ====================
    
    # Output directories
    OUTPUT_DIR = 'outputs'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    
    # Results
    SAVE_PREDICTIONS = True
    SAVE_CONFUSION_MATRIX = True
    SAVE_TRAINING_CURVES = True
    
    # ==================== Evaluation Configuration ====================
    
    # Metrics
    METRICS = ['accuracy', 'weighted_accuracy', 'unweighted_accuracy', 'f1_score']
    
    # Class names (IEMOCAP emotions)
    CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad']
    
    @classmethod
    def create_directories(cls):
        """
        Create necessary directories for outputs
        """
        directories = [
            cls.OUTPUT_DIR,
            cls.MODEL_SAVE_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
            cls.DATA_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_params(cls):
        """
        Get model parameters as a dictionary
        """
        return {
            'num_classes': cls.NUM_CLASSES,
            'input_dim': cls.FREQ_BINS,
            'max_seq_len': cls.MAX_TIME_STEPS,
            'window_size': cls.WINDOW_SIZE,
            'window_stride': cls.WINDOW_STRIDE,
            'dropout': cls.DROPOUT_RATE
        }
    
    @classmethod
    def get_training_params(cls):
        """
        Get training parameters as a dictionary
        """
        return {
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'patience': cls.EARLY_STOPPING_PATIENCE,
            'batch_size': cls.BATCH_SIZE
        }
    
    @classmethod
    def get_loss_params(cls):
        """
        Get loss function parameters as a dictionary
        """
        return {
            'm_pos': cls.MARGIN_LOSS_M_POS,
            'm_neg': cls.MARGIN_LOSS_M_NEG,
            'lambda_val': cls.MARGIN_LOSS_LAMBDA
        }
    
    @classmethod
    def print_config(cls):
        """
        Print current configuration
        """
        print("\n" + "="*60)
        print("CNN-GRU-SeqCap Model Configuration")
        print("="*60)
        
        print("\nData Configuration:")
        print(f"  Input shape: ({cls.INPUT_CHANNELS}, {cls.FREQ_BINS}, {cls.MAX_TIME_STEPS})")
        print(f"  Number of classes: {cls.NUM_CLASSES}")
        print(f"  Class names: {cls.CLASS_NAMES}")
        
        print("\nModel Architecture:")
        print(f"  CNN filters: {cls.CNN_FILTERS_1}, {cls.CNN_FILTERS_2}")
        print(f"  Primary capsules: {cls.NUM_PRIMARY_CAPS} x {cls.PRIMARY_CAPS_DIM}D")
        print(f"  GRU hidden dim: {cls.GRU_HIDDEN_DIM} x {cls.GRU_NUM_LAYERS} layers")
        print(f"  Window size/stride: {cls.WINDOW_SIZE}/{cls.WINDOW_STRIDE}")
        print(f"  Routing iterations: {cls.ROUTING_ITERATIONS}")
        
        print("\nTraining Configuration:")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Dropout: {cls.DROPOUT_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print(f"  Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")
        
        print("\nCross-validation:")
        print(f"  K-folds: {cls.K_FOLDS}")
        print(f"  Validation split: {cls.VALIDATION_SPLIT}")
        print(f"  Random seed: {cls.RANDOM_SEED}")
        
        print("\nLoss Function (Margin Loss):")
        print(f"  m_pos: {cls.MARGIN_LOSS_M_POS}")
        print(f"  m_neg: {cls.MARGIN_LOSS_M_NEG}")
        print(f"  lambda: {cls.MARGIN_LOSS_LAMBDA}")
        
        print("\n" + "="*60)


# Alternative configurations for experimentation
class ConfigExperimental(Config):
    """
    Experimental configuration with different hyperparameters
    """
    
    # Modified parameters for experimentation
    WINDOW_SIZE = 50
    WINDOW_STRIDE = 25
    GRU_HIDDEN_DIM = 256
    LEARNING_RATE = 0.0005
    DROPOUT_RATE = 0.3
    

class ConfigFast(Config):
    """
    Fast configuration for quick testing
    """
    
    # Reduced parameters for faster training
    EPOCHS = 10
    BATCH_SIZE = 32
    K_FOLDS = 3
    EARLY_STOPPING_PATIENCE = 5
    GRU_HIDDEN_DIM = 64
    CNN_FILTERS_1 = 128
    CNN_FILTERS_2 = 128


# Export default configuration
default_config = Config