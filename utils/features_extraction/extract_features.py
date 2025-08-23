"""
Audio Feature Extraction Script for Speech Emotion Recognition

This script extracts audio features from emotion speech databases and saves
them for training machine learning models.
"""

import argparse
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml

from database import SER_DATABASES
from features_util import extract_features

# Constants
DEFAULT_CONFIG_PATH = "configs/feature_config.yaml"
DEFAULT_RANDOM_SEED = 111
FILE_EXTENSION = ".pkl"

# Emotion mappings for different datasets
EMOTION_MAPPINGS = {
    'IEMOCAP_4CLASS': {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3},
    'IEMOCAP_5CLASS': {'ang': 0, 'sad': 1, 'hap': 2, 'exc': 2, 'neu': 3}
}


class FeatureExtractionConfig:
    """
    Configuration class for feature extraction parameters.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self.dataset = config_dict['dataset']
        self.features = config_dict['features']
        self.dataset_dir = config_dict['dataset_dir']
        self.save_dir = config_dict.get('save_dir')
        self.save_label = config_dict.get('save_label', 'features')
        self.mixnoise = config_dict.get('mixnoise', False)
        
        # Feature extraction parameters
        self.params = {
            'window': config_dict['window'],
            'win_length': config_dict['win_length'],
            'hop_length': config_dict['hop_length'],
            'ndft': config_dict['ndft'],
            'nfreq': config_dict['nfreq'],
            'segment_size': config_dict['segment_size'],
            'mixnoise': self.mixnoise,
        }
        
        # Add mel parameters if present
        if 'nmel' in config_dict:
            self.params['nmel'] = config_dict['nmel']
    
    def get_output_filename(self) -> str:
        """
        Generate output filename for features.
        
        Returns:
            Output filename or 'None' if save_dir is not specified
        """
        if self.save_dir is None:
            return 'None'
        
        # Use pathlib for cross-platform path handling
        save_path = Path(self.save_dir)
        filename = f"{self.dataset}_{self.save_label}_features{FILE_EXTENSION}"
        return str(save_path / filename)
    
    def print_configuration(self) -> None:
        """
        Print configuration summary.
        """
        print('\n' + '*' * 50)
        print('\nFEATURE EXTRACTION CONFIGURATION')
        print(f'\t{"Dataset":>20}: {self.dataset}')
        print(f'\t{"Features":>20}: {self.features}')
        print(f'\t{"Dataset dir.":>20}: {self.dataset_dir}')
        print(f'\t{"Output file":>20}: {self.get_output_filename()}')
        print(f'\t{"Mix noise":>20}: {self.mixnoise}')
        print(f"\nFEATURE PARAMETERS:")
        for key, value in self.params.items():
            print(f'\t{key:>20}: {value}')
        print('\n')


def setup_reproducibility(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """
    Set up reproducible random number generation.
    
    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Note: PyTorch seeding would be added here if using PyTorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def initialize_database(config: FeatureExtractionConfig):
    """
    Initialize the appropriate database based on configuration.
    
    Args:
        config: Feature extraction configuration
        
    Returns:
        Initialized database instance
        
    Raises:
        ValueError: If dataset is not supported
    """
    if config.dataset == 'IEMOCAP':
        # Use 5-class emotion mapping (happy + excited combined)
        emotion_map = EMOTION_MAPPINGS['IEMOCAP_5CLASS']
        include_scripted = True
        
        return SER_DATABASES[config.dataset](
            config.dataset_dir,
            emotion_map=emotion_map,
            include_scripted=include_scripted
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")


def save_features(features_data: Dict, output_filename: str) -> None:
    """
    Save extracted features to file.
    
    Args:
        features_data: Dictionary containing extracted features
        output_filename: Path to output file
    """
    if output_filename == 'None':
        print("No save directory specified, skipping save.")
        return
    
    try:
        # Ensure output directory exists
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_filename, "wb") as file_out:
            pickle.dump(features_data, file_out)
        
        print(f"Features saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error saving features: {e}")


def generate_statistics_report(features_data: Dict, 
                             database, 
                             config: FeatureExtractionConfig) -> None:
    """
    Generate and print statistics report for extracted features.
    
    Args:
        features_data: Dictionary containing extracted features
        database: Database instance
        config: Feature extraction configuration
    """
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    
    # Get emotion classes
    emotion_classes = database.get_classes()
    if emotion_classes is None:
        print("Warning: Could not retrieve emotion classes from database")
        return
    
    num_speakers = len(features_data)
    num_classes = len(emotion_classes)
    
    # Initialize class distribution matrix
    class_distribution = np.zeros((num_speakers, num_classes), dtype=np.int64)
    speaker_ids = []
    data_shapes = []
    
    # Collect statistics for each speaker
    for i, (speaker_id, speaker_data) in enumerate(features_data.items()):
        # Count segments per emotion class
        segment_counts = Counter(speaker_data["seg_label"])
        
        for emotion_class, count in segment_counts.items():
            if emotion_class < num_classes:
                class_distribution[i, emotion_class] = count
        
        speaker_ids.append(speaker_id)
        
        # Get data shape information
        if config.mixnoise:
            shape_info = str(speaker_data["seg_spec"][0].shape)
        else:
            shape_info = str(speaker_data["seg_spec"].shape)
        data_shapes.append(shape_info)
    
    # Create DataFrame for display
    report_data = {
        "Speaker ID": speaker_ids,
        "Shape (N,C,F,T)": data_shapes
    }
    
    # Add emotion class columns
    for class_idx in range(num_classes):
        if class_idx in emotion_classes:
            class_name = emotion_classes[class_idx]
            report_data[class_name] = class_distribution[:, class_idx]
    
    # Create and display DataFrame
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))
    
    # Print summary statistics
    total_segments = np.sum(class_distribution)
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total speakers: {num_speakers}")
    print(f"Total segments: {total_segments}")
    print(f"Average segments per speaker: {total_segments / num_speakers:.1f}")
    
    # Print class distribution
    class_totals = np.sum(class_distribution, axis=0)
    print(f"\nCLASS DISTRIBUTION:")
    for class_idx, total in enumerate(class_totals):
        if class_idx in emotion_classes:
            class_name = emotion_classes[class_idx]
            percentage = (total / total_segments) * 100
            print(f"  {class_name}: {total} ({percentage:.1f}%)")


def main(config: FeatureExtractionConfig) -> None:
    """
    Main feature extraction pipeline.
    
    Args:
        config: Feature extraction configuration
    """
    # Print configuration
    config.print_configuration()
    
    # Set up reproducibility
    setup_reproducibility()
    
    # Initialize database
    try:
        database = initialize_database(config)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get file paths and labels
    print("Loading file paths from database...")
    speaker_files = database.get_files()
    
    if not speaker_files:
        print("Error: No files found in database")
        return
    
    print(f"Found {len(speaker_files)} speakers")
    
    # Extract features
    print("Extracting features...")
    features_data = extract_features(speaker_files, config.features, config.params)
    
    if not features_data:
        print("Error: No features extracted")
        return
    
    print(f"Successfully extracted features for {len(features_data)} speakers")
    
    # Save features
    output_filename = config.get_output_filename()
    save_features(features_data, output_filename)
    
    # Generate statistics report
    generate_statistics_report(features_data, database, config)
    
    print('\n' + '*' * 50 + '\n')


def load_configuration(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary or None if failed
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract audio features for speech emotion recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to feature extraction configuration file"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config_dict = load_configuration(args.config)
    if config_dict is None:
        sys.exit(1)
    
    # Create configuration object
    try:
        config = FeatureExtractionConfig(config_dict)
    except KeyError as e:
        print(f"Missing required configuration parameter: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Override random seed if provided
    if hasattr(args, 'seed'):
        DEFAULT_RANDOM_SEED = args.seed
    
    # Run main pipeline
    try:
        main(config)
    except KeyboardInterrupt:
        print("\nFeature extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        sys.exit(1)