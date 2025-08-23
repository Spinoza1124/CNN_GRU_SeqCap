"""
Audio Feature Extraction Utilities for Speech Emotion Recognition

This module provides functions for extracting various audio features including
log spectrograms, mel spectrograms, and delta features from audio signals.
"""

import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor

# Constants
DEFAULT_WAV2VEC_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_HOP_LENGTH = 160
DEFAULT_MFCC_FEATURES = 40
DEFAULT_SAMPLING_RATE = 16000
SEGMENT_SIZE_MULTIPLIER = 160

# Global processor instance to avoid repeated initialization
_wav2vec_processor = None


def get_wav2vec_processor(model_path: Optional[str] = None) -> Wav2Vec2Processor:
    """
    Get or initialize Wav2Vec2 processor.
    
    Args:
        model_path: Path to the Wav2Vec2 model. If None, uses default model.
        
    Returns:
        Wav2Vec2Processor instance
    """
    global _wav2vec_processor
    if _wav2vec_processor is None:
        model_name = model_path or DEFAULT_WAV2VEC_MODEL
        try:
            _wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load Wav2Vec2 model {model_name}: {e}")
            _wav2vec_processor = None
    return _wav2vec_processor


def validate_audio_file(wav_path: str) -> bool:
    """
    Validate if audio file exists and is not empty.
    
    Args:
        wav_path: Path to the audio file
        
    Returns:
        True if file is valid, False otherwise
    """
    # Normalize path to handle double slashes
    wav_path = os.path.normpath(wav_path)
    
    if not os.path.exists(wav_path):
        print(f"Warning: File not found: {wav_path}")
        return False
        
    if os.path.getsize(wav_path) == 0:
        print(f"Warning: Empty file: {wav_path}")
        return False
        
    return True


def load_and_preprocess_audio(wav_path: str, 
                             apply_preemphasis: bool = True) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load and preprocess audio file.
    
    Args:
        wav_path: Path to the audio file
        apply_preemphasis: Whether to apply pre-emphasis filter
        
    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) if failed
    """
    try:
        audio_data, sample_rate = librosa.load(wav_path, sr=None)
        
        if len(audio_data) == 0:
            print(f"Warning: Empty audio data: {wav_path}")
            return None, None
            
        if apply_preemphasis:
            audio_data = librosa.effects.preemphasis(audio_data, zi=[0.0])
            
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file {wav_path}: {str(e)}")
        return None, None


def extract_features(speaker_files: Dict[str, List[Tuple[str, int]]], 
                    feature_type: str, 
                    params: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract audio features for all speakers.
    
    Args:
        speaker_files: Dictionary mapping speaker IDs to list of (file_path, emotion_label) tuples
        feature_type: Type of features to extract ('logspec', 'logmelspec', 'logdeltaspec')
        params: Feature extraction parameters
        
    Returns:
        Dictionary mapping speaker IDs to their extracted features
    """
    if feature_type not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unsupported feature type: {feature_type}")
        
    processor = get_wav2vec_processor()
    speaker_features = defaultdict(dict)
    
    for speaker_id in tqdm(speaker_files.keys(), desc="Processing speakers"):
        features_data = _extract_speaker_features(
            speaker_files[speaker_id], feature_type, params, processor
        )
        
        if features_data:
            speaker_features[speaker_id] = features_data
            _print_speaker_stats(speaker_id, features_data)
    
    assert len(speaker_features) == len(speaker_files), \
        "Mismatch between input and output speaker count"
    
    return dict(speaker_features)


def _extract_speaker_features(file_list: List[Tuple[str, int]], 
                             feature_type: str, 
                             params: Dict,
                             processor: Optional[Wav2Vec2Processor]) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract features for a single speaker.
    
    Args:
        file_list: List of (file_path, emotion_label) tuples for the speaker
        feature_type: Type of features to extract
        params: Feature extraction parameters
        processor: Wav2Vec2 processor instance
        
    Returns:
        Dictionary containing extracted features or None if no valid data
    """
    data_collections = {
        'spectrograms': [],
        'utterance_labels': [],
        'segment_labels': [],
        'segment_counts': [],
        'mfcc_features': [],
        'audio_features': []
    }
    
    for wav_path, emotion_label in file_list:
        if not validate_audio_file(wav_path):
            continue
            
        audio_data, sample_rate = load_and_preprocess_audio(wav_path)
        if audio_data is None:
            continue
            
        try:
            # Extract spectral features
            spectral_features = FEATURE_EXTRACTORS[feature_type](audio_data, sample_rate, params)
            
            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=DEFAULT_MFCC_FEATURES, 
                hop_length=DEFAULT_HOP_LENGTH, 
                htk=True
            ).T
            
            # Segment features
            segmented_data = segment_features(
                audio_data, mfcc_features, spectral_features, 
                emotion_label, params['segment_size'], processor
            )
            
            # Collect segmented data
            _collect_segmented_data(data_collections, segmented_data)
            
        except Exception as e:
            print(f"Error processing file {wav_path}: {str(e)}")
            continue
    
    return _finalize_speaker_features(data_collections)


def _collect_segmented_data(collections: Dict[str, List], 
                           segmented_data: Tuple) -> None:
    """
    Collect segmented data into collections.
    
    Args:
        collections: Dictionary of data collections
        segmented_data: Tuple containing segmented features
    """
    num_segments, spectrograms, segment_labels, utterance_label, mfcc_data, audio_data = segmented_data
    
    collections['spectrograms'].append(spectrograms)
    collections['utterance_labels'].append(utterance_label)
    collections['segment_labels'].extend(segment_labels)
    collections['segment_counts'].append(num_segments)
    collections['mfcc_features'].append(mfcc_data)
    collections['audio_features'].append(audio_data)


def _finalize_speaker_features(collections: Dict[str, List]) -> Optional[Dict[str, np.ndarray]]:
    """
    Finalize and validate speaker features.
    
    Args:
        collections: Dictionary of collected data
        
    Returns:
        Dictionary of finalized features or None if no valid data
    """
    if len(collections['spectrograms']) == 0:
        return None
    
    # Stack and convert data types
    try:
        spectrograms = np.vstack(collections['spectrograms']).astype(np.float32)
        mfcc_features = np.vstack(collections['mfcc_features']).astype(np.float32)
        audio_features = np.vstack(collections['audio_features']).astype(np.float32)
        utterance_labels = np.asarray(collections['utterance_labels'], dtype=np.int64)
        segment_labels = np.asarray(collections['segment_labels'], dtype=np.int64)
        segment_counts = np.asarray(collections['segment_counts'], dtype=np.int64)
        
        # Validate data consistency
        assert len(utterance_labels) == len(segment_counts), \
            "Mismatch between utterance labels and segment counts"
        assert spectrograms.shape[0] == segment_labels.shape[0] == sum(segment_counts), \
            "Mismatch between spectrograms and segment labels"
        
        return {
            "seg_spec": spectrograms,
            "utter_label": utterance_labels,
            "seg_label": segment_labels,
            "seg_num": segment_counts,
            "seg_mfcc": mfcc_features,
            "seg_audio": audio_features
        }
        
    except Exception as e:
        print(f"Error finalizing speaker features: {e}")
        return None


def _print_speaker_stats(speaker_id: str, features_data: Dict[str, np.ndarray]) -> None:
    """
    Print statistics for speaker features.
    
    Args:
        speaker_id: Speaker identifier
        features_data: Dictionary containing speaker features
    """
    print(f"Speaker {speaker_id}:")
    print(f"  Spectrograms shape: {features_data['seg_spec'].shape}")
    print(f"  Segments shape: {features_data['seg_label'].shape}")
    print(f"  Audio shape: {features_data['seg_audio'].shape}")
    print(f"  Utterance labels shape: {features_data['utter_label'].shape}")
    print(f"  Segment counts shape: {features_data['seg_num'].shape}")


def apply_padding(feature: np.ndarray, 
                 max_length: int, 
                 padding_mode: str = 'zeros', 
                 padding_location: str = 'back') -> np.ndarray:
    """
    Apply padding to feature array.
    
    Args:
        feature: Input feature array
        max_length: Maximum length after padding
        padding_mode: 'zeros' or 'normal' distribution padding
        padding_location: 'front' or 'back' padding location
        
    Returns:
        Padded feature array
    """
    length = feature.shape[0]
    if length >= max_length:
        return feature[:max_length, :]
    
    pad_length = max_length - length
    feature_dim = feature.shape[-1]
    
    if padding_mode == "zeros":
        pad = np.zeros([pad_length, feature_dim])
    elif padding_mode == "normal":
        mean, std = feature.mean(), feature.std()
        pad = np.random.normal(mean, std, (pad_length, feature_dim))
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")
    
    if padding_location == "front":
        return np.concatenate([pad, feature], axis=0)
    else:
        return np.concatenate([feature, pad], axis=0)


def pad_sequences(sequences: List[np.ndarray]) -> np.ndarray:
    """
    Pad sequences to uniform length.
    
    Args:
        sequences: List of feature sequences
        
    Returns:
        Padded sequences array
    """
    if len(sequences) == 0:
        return np.array(sequences)
    
    feature_dim = sequences[0].shape[-1]
    lengths = [s.shape[0] for s in sequences]
    
    # Calculate final length using mean + 3*std
    final_length = int(np.mean(lengths) + 3 * np.std(lengths))
    
    # Pad sequences to final length
    padded_sequences = np.zeros([len(sequences), final_length, feature_dim])
    for i, sequence in enumerate(sequences):
        padded_sequences[i] = apply_padding(sequence, final_length)
    
    return padded_sequences


def extract_log_spectrogram(audio: np.ndarray, 
                           sample_rate: int, 
                           params: Dict) -> np.ndarray:
    """
    Extract log spectrogram features.
    
    Args:
        audio: Audio signal
        sample_rate: Sampling rate
        params: Feature extraction parameters
        
    Returns:
        Log spectrogram with shape (C, F, T) where C=1
    """
    # Extract parameters
    window = params['window']
    win_length = int((params['win_length'] / 1000) * sample_rate)
    hop_length = int((params['hop_length'] / 1000) * sample_rate)
    n_fft = params['ndft']
    n_freq = params['nfreq']
    
    # Calculate STFT
    spectrogram = np.abs(librosa.stft(
        audio, 
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    ))
    
    # Convert to log scale
    log_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Extract required frequency bins
    log_spec = log_spec[:n_freq]
    
    # Shape to (C, F, T) with C=1
    return np.expand_dims(log_spec, 0)


def extract_log_mel_spectrogram(audio: np.ndarray, 
                               sample_rate: int, 
                               params: Dict) -> np.ndarray:
    """
    Extract log mel spectrogram features.
    
    Args:
        audio: Audio signal
        sample_rate: Sampling rate
        params: Feature extraction parameters
        
    Returns:
        Log mel spectrogram with shape (C, F, T) where C=1
    """
    # Extract parameters
    window = params['window']
    win_length = int((params['win_length'] / 1000) * sample_rate)
    hop_length = int((params['hop_length'] / 1000) * sample_rate)
    n_fft = params['ndft']
    n_mels = params['nmel']
    
    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Shape to (C, F, T) with C=1
    return np.expand_dims(log_mel_spec, 0)


def extract_log_delta_spectrogram(audio: np.ndarray, 
                                 sample_rate: int, 
                                 params: Dict) -> np.ndarray:
    """
    Extract log delta spectrogram features (original + delta + delta-delta).
    
    Args:
        audio: Audio signal
        sample_rate: Sampling rate
        params: Feature extraction parameters
        
    Returns:
        Log delta spectrogram with shape (C, F, T) where C=3
    """
    # Get base log spectrogram
    log_spec = extract_log_spectrogram(audio, sample_rate, params)  # (1, F, T)
    
    # Calculate delta and delta-delta features
    base_spec = log_spec.squeeze(0)  # (F, T)
    delta_spec = librosa.feature.delta(base_spec)
    delta2_spec = librosa.feature.delta(base_spec, order=2)
    
    # Combine into (C, F, T) with C=3
    delta_spec = np.expand_dims(delta_spec, axis=0)
    delta2_spec = np.expand_dims(delta2_spec, axis=0)
    
    return np.concatenate([log_spec, delta_spec, delta2_spec], axis=0)


def segment_features(audio_data: np.ndarray,
                    mfcc_features: np.ndarray,
                    spectral_features: np.ndarray,
                    emotion_label: int,
                    segment_size: int,
                    processor: Optional[Wav2Vec2Processor] = None) -> Tuple:
    """
    Segment features into fixed-size frames with padding.
    
    Args:
        audio_data: Raw audio signal
        mfcc_features: MFCC features with shape (T, F)
        spectral_features: Spectral features with shape (C, F, T)
        emotion_label: Emotion label for the utterance
        segment_size: Size of each segment in frames
        processor: Wav2Vec2 processor for audio processing
        
    Returns:
        Tuple of (num_segments, segmented_spectral, segment_labels, 
                 utterance_label, segmented_mfcc, segmented_audio)
    """
    segment_size_audio = segment_size * SEGMENT_SIZE_MULTIPLIER
    
    # Transpose spectral features to (C, T, F)
    spectral_features = spectral_features.transpose(0, 2, 1)
    
    time_frames = spectral_features.shape[1]
    time_audio = audio_data.shape[0]
    num_channels = spectral_features.shape[0]
    
    # Calculate number of segments
    num_segments = math.ceil(time_frames / segment_size)
    
    # Initialize collections
    segmented_mfcc = []
    segmented_audio = []
    segmented_spectral = []
    
    # Process each segment
    for i in range(num_segments):
        # Calculate segment boundaries
        start_frame = i * segment_size
        end_frame = min(start_frame + segment_size, time_frames)
        
        start_audio = i * segment_size_audio
        end_audio = min(start_audio + segment_size_audio, time_audio)
        
        # Handle last segment
        if end_frame == time_frames and end_frame - start_frame < segment_size:
            start_frame = max(0, end_frame - segment_size)
        if end_audio == time_audio and end_audio - start_audio < segment_size_audio:
            start_audio = max(0, end_audio - segment_size_audio)
        
        # Pad MFCC features
        mfcc_segment = mfcc_features[start_frame:end_frame]
        mfcc_padded = np.pad(
            mfcc_segment, 
            ((0, segment_size - (end_frame - start_frame)), (0, 0)), 
            mode="constant"
        )
        segmented_mfcc.append(mfcc_padded)
        
        # Pad audio data
        audio_segment = audio_data[start_audio:end_audio]
        audio_padded = np.pad(
            audio_segment,
            (segment_size_audio - (end_audio - start_audio), 0),
            mode="constant"
        )
        
        # Process with Wav2Vec2 if available
        if processor is not None:
            try:
                audio_processed = processor(
                    audio_padded, 
                    sampling_rate=DEFAULT_SAMPLING_RATE, 
                    return_tensors="pt"
                ).input_values
                audio_processed = audio_processed.view(-1).cpu().detach().numpy()
                segmented_audio.append(audio_processed)
            except Exception as e:
                print(f"Warning: Wav2Vec2 processing failed: {e}")
                segmented_audio.append(audio_padded)
        else:
            segmented_audio.append(audio_padded)
        
        # Pad spectral features
        spectral_segment = []
        for c in range(num_channels):
            channel_data = spectral_features[c, start_frame:end_frame]
            channel_padded = np.pad(
                channel_data,
                ((0, segment_size - (end_frame - start_frame)), (0, 0)),
                mode="constant"
            )
            spectral_segment.append(channel_padded)
        
        segmented_spectral.append(np.array(spectral_segment))
    
    # Stack all segments
    segmented_mfcc = np.stack(segmented_mfcc)
    segmented_spectral = np.stack(segmented_spectral)
    segmented_audio = np.stack(segmented_audio)
    
    # Create segment labels
    segment_labels = [emotion_label] * num_segments
    
    # Transpose spectral output to (N, C, F, T)
    segmented_spectral = segmented_spectral.transpose(0, 1, 3, 2)
    
    return (num_segments, segmented_spectral, segment_labels, 
            emotion_label, segmented_mfcc, segmented_audio)


# Feature extraction function mapping
FEATURE_EXTRACTORS = {
    'logspec': extract_log_spectrogram,
    'logmelspec': extract_log_mel_spectrogram,
    'logdeltaspec': extract_log_delta_spectrogram
}


