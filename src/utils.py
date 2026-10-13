"""
Utility functions for analysis and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_3d_attractor(trajectory, title="Lorenz Attractor", save_path=None):
    """
    Plot 3D Lorenz attractor.
    
    Args:
        trajectory: Array of shape (n_steps, 3)
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_trajectories_comparison(trajectories_dict, horizon=500, save_path=None):
    """
    Compare multiple trajectories.
    
    Args:
        trajectories_dict: Dict of {name: trajectory}
        horizon: Number of steps to plot
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    time_steps = np.arange(horizon)
    
    for name, traj in trajectories_dict.items():
        for i, ax in enumerate(axes):
            ax.plot(time_steps, traj[:horizon, i], label=name, marker='o', 
                   markersize=2, alpha=0.7)
            ax.set_ylabel(f'{"XYZ"[i]}')
            ax.legend()
    
    axes[-1].set_xlabel('Time step')
    fig.suptitle('Trajectory Comparison')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_error_analysis(errors, metric_name="RMSE", save_path=None):
    """
    Plot error evolution over time.
    
    Args:
        errors: Dict of {model_name: error_array}
        metric_name: Name of error metric
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    time_steps = np.arange(len(list(errors.values())[0]))
    
    # Linear scale
    for name, error in errors.items():
        ax1.plot(time_steps, error, label=name, marker='o', markersize=2)
    
    ax1.set_xlabel('Time step')
    ax1.set_ylabel(metric_name)
    ax1.set_title(f'{metric_name} vs Time (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    for name, error in errors.items():
        ax2.semilogy(time_steps, error, label=name, marker='o', markersize=2)
    
    ax2.set_xlabel('Time step')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'{metric_name} vs Time (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_performance_comparison(timings, labels=None, save_path=None):
    """
    Plot execution time comparison.
    
    Args:
        timings: Dict or array of execution times
        labels: Optional labels
        save_path: Path to save figure
    """
    if isinstance(timings, dict):
        names = list(timings.keys())
        times = list(timings.values())
    else:
        times = timings
        names = labels if labels else [f"Method {i}" for i in range(len(times))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f} ms',
               ha='center', va='bottom')
    
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_model_comparison(results_dict, metric='rmse', save_path=None):
    """
    Compare multiple models.
    
    Args:
        results_dict: Dict of {model_name: {metric: value}}
        metric: Metric to compare
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results_dict.keys())
    values = [results_dict[name].get(metric, 0) for name in names]
    
    bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom')
    
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison ({metric.upper()})')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_phase_space(trajectory, dimensions=(0, 1), save_path=None, title="Phase Space"):
    """
    Plot 2D phase space projection.
    
    Args:
        trajectory: Array of shape (n_steps, 3)
        dimensions: Tuple of dimensions to plot (default: (0, 1))
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    dim1, dim2 = dimensions
    ax.plot(trajectory[:, dim1], trajectory[:, dim2], linewidth=0.5, alpha=0.7)
    ax.scatter(trajectory[0, dim1], trajectory[0, dim2], c='green', s=100, 
              marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, dim1], trajectory[-1, dim2], c='red', s=100, 
              marker='X', label='End', zorder=5)
    
    ax.set_xlabel(['X', 'Y', 'Z'][dim1])
    ax.set_ylabel(['X', 'Y', 'Z'][dim2])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def compute_rmse(predictions, ground_truth):
    """
    Compute RMSE between predictions and ground truth.
    
    Args:
        predictions: Predicted trajectories
        ground_truth: Ground truth trajectories
        
    Returns:
        RMSE values
    """
    mse = np.mean((predictions - ground_truth) ** 2, axis=-1)
    rmse = np.sqrt(mse)
    return rmse


def compute_mae(predictions, ground_truth):
    """
    Compute MAE between predictions and ground truth.
    
    Args:
        predictions: Predicted trajectories
        ground_truth: Ground truth trajectories
        
    Returns:
        MAE values
    """
    return np.mean(np.abs(predictions - ground_truth), axis=-1)


def ensure_directory(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
