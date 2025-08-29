import numpy as np
import torch
from typing import Union, Tuple, List, Optional

import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft
from scipy.stats import skew, kurtosis

def compute_ecdf_features(window_data: Union[np.ndarray, torch.Tensor], n_points: int = 25) -> np.ndarray:
    """
    Feature extraction based on ECDF (Empirical Cumulative Distribution Function)
    
    Args:
        window_data: Input time series data [window_size, channels]  (168, 9)
        n_points: Number of points to extract from each time series (default: 25)
    
    Returns:
        ECDF feature vector of shape [3, (n_points + 1) * 3] with dimensions [3, 78]
        Each axis (x, y, z) contains 3 sensors with n_points (ECDF points + 1 mean value) each
    """
    if isinstance(window_data, torch.Tensor):
        window_data = window_data.detach().cpu().numpy()
    
    channels = window_data.shape[1]
    if channels != 9 and channels != 18:
        raise ValueError(f"Unsupported number of channels: {channels}. Must be 9 (acc only) or 18 (acc+gyro).")
    
    if channels == 18:
        acc_indices = list(range(0, 18, 2))
        window_data = window_data[:, acc_indices]
    
    ecdf_features = np.zeros((3, (n_points + 1) * 3))
    
    axes_indices = [
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]
    ]
    
    for axis_idx, axis_channels in enumerate(axes_indices):
        for i, channel_idx in enumerate(axis_channels):
            channel_data = window_data[:, channel_idx]
            mean_value = np.mean(channel_data)
            sorted_data = np.sort(channel_data)
            indices = np.around(np.linspace(0, len(sorted_data) - 1, num=n_points)).astype(int)
            ecdf_points = sorted_data[indices]
            start_idx = i * (n_points + 1)
            ecdf_features[axis_idx, start_idx:start_idx + n_points] = ecdf_points
            ecdf_features[axis_idx, start_idx + n_points] = mean_value
            
    return ecdf_features.astype(np.float32)

def compute_batch_ecdf_features(batch_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Calculate ECDF features for batch data
    """
    if isinstance(batch_data, torch.Tensor):
        batch_data = batch_data.detach().cpu().numpy()
        
    batch_size = batch_data.shape[0]
    features = np.zeros((batch_size, 3, 78))
    
    for i in range(batch_size):
        features[i] = compute_ecdf_features(batch_data[i])
    
    return features

def get_ecdf_dimension() -> tuple:
    return (3, 78)

class Normalizer(object):
    def __init__(self, norm_type):
        self.norm_type = norm_type
        
    def fit(self, df):
        if self.norm_type == "standardization":
            self.mean = df.mean(0)
            self.std = df.std(0)
        elif self.norm_type == "minmax":
            self.max_val = df.max()
            self.min_val = df.min()
        else:
            pass

    def normalize(self, df):
        if self.norm_type == "standardization":
            return (df - self.mean) / (self.std + np.finfo(float).eps)
        elif self.norm_type == "minmax":
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        else:
            return df

def components_selection_one_signal(t_signal,freq1,freq2,sampling_freq):
    t_signal=np.array(t_signal)
    t_signal_length=len(t_signal)
    f_signal=fft(t_signal)
    freqs=np.array(sp.fftpack.fftfreq(t_signal_length, d=1/float(sampling_freq)))
    f_DC_signal=[]
    f_body_signal=[]
    f_noise_signal=[]
    for i in range(len(freqs)):
        freq=freqs[i]
        value= f_signal[i]
        if abs(freq)>freq1:
            f_DC_signal.append(float(0))
        else:
            f_DC_signal.append(value)
        if (abs(freq)<=freq2):
            f_noise_signal.append(float(0))
        else:
            f_noise_signal.append(value)
        if (abs(freq)<=freq1 or abs(freq)>freq2):
            f_body_signal.append(float(0))
        else:
            f_body_signal.append(value)
    t_DC_component= ifft(np.array(f_DC_signal)).real
    t_body_component= ifft(np.array(f_body_signal)).real
    return (t_DC_component,t_body_component)

def compute_extended_features(window_data: Union[np.ndarray, torch.Tensor], n_points: int = 25) -> np.ndarray:
    """
    ECDF, 평균, 표준편차, 왜도, 첨도를 포함한 확장된 특징을 추출합니다.
    """
    if isinstance(window_data, torch.Tensor):
        window_data = window_data.detach().cpu().numpy()
    
    channels = window_data.shape[1]
    if channels != 9:
        raise ValueError(f"지원하지 않는 채널 수: {channels}. 9개여야 합니다.")
    
    n_stats = 4
    n_features_per_channel = n_points + n_stats
    total_features_per_axis = n_features_per_channel * 3

    extended_features = np.zeros((3, total_features_per_axis))
    
    axes_indices = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    
    for axis_idx, axis_channels in enumerate(axes_indices):
        for i, channel_idx in enumerate(axis_channels):
            channel_data = window_data[:, channel_idx]
            
            mean_value = np.mean(channel_data)
            std_value = np.std(channel_data)
            skew_value = skew(channel_data)
            kurt_value = kurtosis(channel_data)
            
            sorted_data = np.sort(channel_data)
            indices = np.around(np.linspace(0, len(sorted_data) - 1, num=n_points)).astype(int)
            ecdf_points = sorted_data[indices]
            
            start_idx = i * n_features_per_channel
            end_idx_ecdf = start_idx + n_points
            
            extended_features[axis_idx, start_idx:end_idx_ecdf] = ecdf_points
            extended_features[axis_idx, end_idx_ecdf] = mean_value
            extended_features[axis_idx, end_idx_ecdf + 1] = std_value
            extended_features[axis_idx, end_idx_ecdf + 2] = skew_value
            extended_features[axis_idx, end_idx_ecdf + 3] = kurt_value
            
    return extended_features.astype(np.float32)

def compute_batch_extended_features(batch_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    배치 데이터에 대해 확장된 특징을 계산합니다.
    """
    if isinstance(batch_data, torch.Tensor):
        batch_data = batch_data.detach().cpu().numpy()
        
    batch_size = batch_data.shape[0]
    n_features = (25 + 4) * 3 
    features = np.zeros((batch_size, 3, n_features))
    
    for i in range(batch_size):
        features[i] = compute_extended_features(batch_data[i])
    
    return features

class TargetNormalizer:
    """
    Normalizes target features of shape (n_samples, 3, n_features_per_axis).
    The mean and std are computed for each feature across all samples,
    independently for each of the 3 main axes.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """
        Computes mean and std for normalization.
        
        Args:
            data (np.ndarray): A numpy array of shape (n_samples, 3, n_features_per_axis).
        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3-dimensional, but got shape {data.shape}")
        # Calculate mean and std along the samples axis (axis 0).
        # This results in shape (3, n_features_per_axis).
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies normalization to the data.
        
        Args:
            data (np.ndarray): A numpy array of shape (n_samples, 3, n_features_per_axis)
                               or (3, n_features_per_axis) for a single sample.
        
        Returns:
            np.ndarray: Normalized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")
        
        is_single_sample = data.ndim == 2
        if is_single_sample:
            if data.shape != self.mean.shape:
                 raise ValueError(f"Single sample shape {data.shape} is incompatible with normalizer shape {self.mean.shape}")
            data = np.expand_dims(data, axis=0)

        if data.ndim != 3 or data.shape[1:] != self.mean.shape:
            raise ValueError(f"Data shape {data.shape} is incompatible with normalizer shape {self.mean.shape}")

        normalized_data = (data - self.mean) / (self.std + np.finfo(float).eps)
        
        return np.squeeze(normalized_data) if is_single_sample else normalized_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies inverse normalization to the data.
        
        Args:
            data (np.ndarray): Normalized numpy array.
        
        Returns:
            np.ndarray: Denormalized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")
        
        is_single_sample = data.ndim == 2
        if is_single_sample:
            if data.shape != self.mean.shape:
                 raise ValueError(f"Single sample shape {data.shape} is incompatible with normalizer shape {self.mean.shape}")
            data = np.expand_dims(data, axis=0)
            
        if data.ndim != 3 or data.shape[1:] != self.mean.shape:
            raise ValueError(f"Data shape {data.shape} is incompatible with normalizer shape {self.mean.shape}")

        denormalized_data = (data * (self.std + np.finfo(float).eps)) + self.mean
        
        return np.squeeze(denormalized_data) if is_single_sample else denormalized_data