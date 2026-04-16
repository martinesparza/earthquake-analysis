"""
TimeSeriesPlotter: Visualize aligned kinematic data for perturbation trials.
Phase 1, Step 2b of the behavior analysis platform.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from tools.behavior.behavior_dataset import BehaviorDataset
from tools.params import Params


class TimeSeriesPlotter:
    """
    Visualizes kinematic data for a single session, direction, and trial type.
    
    Supports:
    - 18-subplot grid (6 body parts × 3 coordinates) for time series
    - 6-subplot grid (2D trajectories: X-Z plane) for spatial movement
    - Statistics table with mean/SEM across trials
    """
    
    def __init__(self, dataset: BehaviorDataset, figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize plotter with a behavior dataset.
        
        Parameters
        ----------
        dataset : BehaviorDataset
            Loaded session data to visualize
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.dataset = dataset
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_kinematics_grid(self, 
                            direction: int, 
                            trial_type: str = 'trial',
                            pre_ms: int = 200, 
                            post_ms: int = 1500,
                            show_sem: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot 18-subplot grid: 6 body parts × 3 coordinates (X, Y, Z).
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        show_sem : bool
            Show shaded SEM bands
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        axes : np.ndarray
            Array of axes (6, 3)
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(6)])
        
        # Coordinate labels
        coord_labels = ['X (forward/back)', 'Y (vertical)', 'Z (left/right)']
        
        # Get kinematics for this direction
        kinematics_dict = {}
        for i, body_part in enumerate(self.dataset.BODY_PARTS):
            kinematics_concat = self.dataset.get_kinematics(body_part, direction, trial_type)
            aligned, perturb_idx = self.dataset.align_to_perturbation(
                kinematics_concat, direction, pre_ms, post_ms
            )
            kinematics_dict[body_part] = aligned
        
        # Time axis in milliseconds relative to perturbation
        n_frames = list(kinematics_dict.values())[0].shape[1]
        time_ms = np.arange(n_frames) * Params.BIN_SIZE * 1000 - pre_ms
        
        # Plot each body part
        for i, body_part in enumerate(self.dataset.BODY_PARTS):
            aligned = kinematics_dict[body_part]
            mean_traj = aligned.mean(axis=0)  # (n_frames, 3)
            sem_traj = np.apply_along_axis(lambda x: np.std(x) / np.sqrt(len(x)), 0, aligned)
            
            # Plot each coordinate
            for j in range(3):
                ax = axes[i, j]
                
                # Mean line
                ax.plot(time_ms, mean_traj[:, j], linewidth=2, color=self.colors[0])
                
                # SEM band
                if show_sem:
                    ax.fill_between(time_ms, 
                                   mean_traj[:, j] - sem_traj[:, j],
                                   mean_traj[:, j] + sem_traj[:, j],
                                   alpha=0.3, color=self.colors[0])
                
                # Perturbation onset line
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perturbation')
                
                # Labels
                if j == 0:
                    ax.set_ylabel(body_part.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                if i == 0:
                    ax.set_title(coord_labels[j], fontsize=11, fontweight='bold')
                if i == 5:
                    ax.set_xlabel('Time (ms)', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                ax.set_xlim([time_ms[0], time_ms[-1]])
        
        # Overall title
        n_trials = self.dataset.n_trials_per_direction.get(direction, 0)
        fig.suptitle(f"{self.dataset.session_name} | Direction {direction} ({n_trials} trials) | {trial_type}", 
                     fontsize=14, fontweight='bold', y=0.995)
        
        return fig, axes
    
    def plot_trajectories_2d(self,
                            direction: int,
                            trial_type: str = 'trial',
                            pre_ms: int = 200,
                            post_ms: int = 1500,
                            show_perturbation: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot all 6 body parts on single 2D trajectory (X-Z plane).
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        show_perturbation : bool
            Mark perturbation onset on trajectory
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        ax : plt.Axes
            Single axes with all body parts plotted
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get kinematics and align
        kinematics_dict = {}
        perturb_idx = None
        perturb_positions = {}  # Store perturbation onset positions for each body part
        
        for body_part in self.dataset.BODY_PARTS:
            kinematics_concat = self.dataset.get_kinematics(body_part, direction, trial_type)
            aligned, perturb_idx = self.dataset.align_to_perturbation(
                kinematics_concat, direction, pre_ms, post_ms
            )
            kinematics_dict[body_part] = aligned
        
        # Plot each body part on the same axes
        for idx, body_part in enumerate(self.dataset.BODY_PARTS):
            aligned = kinematics_dict[body_part]  # (n_trials, n_frames, 3)
            color = self.colors[idx % len(self.colors)]
            
            # Mean trajectory
            mean_x = aligned[:, :, 0].mean(axis=0)  # (n_frames,)
            mean_z = aligned[:, :, 2].mean(axis=0)
            
            # Plot individual trials (faint background)
            for trial_idx in range(aligned.shape[0]):
                ax.plot(aligned[trial_idx, :, 0], aligned[trial_idx, :, 2], 
                       alpha=0.03, color=color, linewidth=0.3)
            
            # Plot mean trajectory with body part label
            ax.plot(mean_x, mean_z, linewidth=2.5, color=color, 
                   label=body_part.replace('_', ' ').title(), zorder=3)
            
            # Store perturbation onset position
            if show_perturbation and perturb_idx is not None:
                perturb_positions[body_part] = (mean_x[perturb_idx], mean_z[perturb_idx])
                # Mark perturbation onset with small circle
                ax.plot(mean_x[perturb_idx], mean_z[perturb_idx], 'o', 
                       color=color, markersize=8, zorder=4, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add a larger star at origin (average of all body part onset positions)
        if show_perturbation and perturb_positions:
            onset_x = np.mean([pos[0] for pos in perturb_positions.values()])
            onset_z = np.mean([pos[1] for pos in perturb_positions.values()])
            ax.plot(onset_x, onset_z, 'r*', markersize=20, 
                   label='Perturbation onset', zorder=5, markeredgecolor='darkred', markeredgewidth=0.5)
        
        # Formatting
        ax.set_xlabel('X (forward/back) [cm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (left/right) [cm]', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Overall title
        n_trials = self.dataset.n_trials_per_direction.get(direction, 0)
        fig.suptitle(f"{self.dataset.session_name} | 2D Trajectories (X-Z plane) | Direction {direction} ({n_trials} trials)", 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def get_statistics_table(self,
                            direction: int,
                            trial_type: str = 'trial',
                            pre_ms: int = 200,
                            post_ms: int = 1500,
                            body_parts: Optional[list] = None) -> pd.DataFrame:
        """
        Generate statistics table for mean/SEM across trials.
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        body_parts : list, optional
            Subset of body parts to include. If None, includes all.
        
        Returns
        -------
        df : pd.DataFrame
            Statistics table with columns: body_part, x_mean, x_sem, y_mean, y_sem, z_mean, z_sem
        """
        if body_parts is None:
            body_parts = self.dataset.BODY_PARTS
        
        rows = []
        
        for body_part in body_parts:
            kinematics_concat = self.dataset.get_kinematics(body_part, direction, trial_type)
            aligned, _ = self.dataset.align_to_perturbation(kinematics_concat, direction, pre_ms, post_ms)
            
            # Compute statistics at perturbation onset
            perturb_idx = int(pre_ms / (Params.BIN_SIZE * 1000))
            
            mean_vals = aligned[:, perturb_idx, :]  # (n_trials, 3)
            mean = mean_vals.mean(axis=0)
            sem_vals = np.std(mean_vals, axis=0) / np.sqrt(mean_vals.shape[0])
            
            rows.append({
                'Body Part': body_part.replace('_', ' ').title(),
                'X (mean)': f"{mean[0]:.2f}",
                'X (SEM)': f"{sem_vals[0]:.2f}",
                'Y (mean)': f"{mean[1]:.2f}",
                'Y (SEM)': f"{sem_vals[1]:.2f}",
                'Z (mean)': f"{mean[2]:.2f}",
                'Z (SEM)': f"{sem_vals[2]:.2f}",
            })
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_statistics_snapshot(self,
                                direction: int,
                                trial_type: str = 'trial',
                                pre_ms: int = 200,
                                post_ms: int = 1500,
                                body_parts: Optional[list] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot statistics table as a formatted matplotlib table.
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        body_parts : list, optional
            Subset of body parts to include
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        ax : plt.Axes
            Axes with table
        """
        df = self.get_statistics_table(direction, trial_type, pre_ms, post_ms, body_parts)
        
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        n_trials = self.dataset.n_trials_per_direction.get(direction, 0)
        title = f"{self.dataset.session_name} | Statistics at Perturbation Onset\nDirection {direction} ({n_trials} trials) | {trial_type}"
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        return fig, ax
    
    def plot_trajectories_animation(self,
                                   direction: int,
                                   trial_type: str = 'trial',
                                   pre_ms: int = 200,
                                   post_ms: int = 1500,
                                   fps: int = 20) -> Tuple[plt.Figure, FuncAnimation]:
        """
        Animate 2D trajectories showing body parts moving frame-by-frame (mean across trials).
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        fps : int
            Frames per second for animation
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        anim : FuncAnimation
            Animation object (assign to variable to keep it alive)
        """
        # Get kinematics and align
        kinematics_dict = {}
        perturb_idx = None
        
        for body_part in self.dataset.BODY_PARTS:
            kinematics_concat = self.dataset.get_kinematics(body_part, direction, trial_type)
            aligned, perturb_idx = self.dataset.align_to_perturbation(
                kinematics_concat, direction, pre_ms, post_ms
            )
            # Use mean across trials
            kinematics_dict[body_part] = aligned.mean(axis=0)  # (n_frames, 3)
        
        n_frames = list(kinematics_dict.values())[0].shape[0]
        
        # Create figure with dark background for better visibility
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # Set up plot limits (use all data to establish bounds)
        all_x = np.concatenate([kin[:, 0] for kin in kinematics_dict.values()])
        all_z = np.concatenate([kin[:, 2] for kin in kinematics_dict.values()])
        x_min, x_max = all_x.min() - 2, all_x.max() + 2
        z_min, z_max = all_z.min() - 2, all_z.max() + 2
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([z_min, z_max])
        ax.set_xlabel('X (forward/back) [cm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (left/right) [cm]', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        
        # Initialize scatter plots and line objects
        scatter_dict = {}
        line_dict = {}
        
        for idx, body_part in enumerate(self.dataset.BODY_PARTS):
            color = self.colors[idx % len(self.colors)]
            # Scatter for current position
            scatter = ax.scatter([], [], s=150, c=[color], zorder=5, edgecolor='black', linewidth=1.5)
            scatter_dict[body_part] = scatter
            
            # Line for full trail (all past positions)
            line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.8, zorder=3,
                          label=body_part.replace('_', ' ').title())
            line_dict[body_part] = line
        
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        
        # Time/frame indicator
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Solenoid ON indicator (show during first 500ms)
        solenoid_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=11,
                               verticalalignment='top', fontweight='bold', color='white',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Perturbation onset marker (cross at the perturbation position)
        perturb_x = kinematics_dict[self.dataset.BODY_PARTS[0]][perturb_idx, 0]
        perturb_z = kinematics_dict[self.dataset.BODY_PARTS[0]][perturb_idx, 2]
        perturb_marker = ax.plot([], [], 'x', color='orange', markersize=20, zorder=1, markeredgewidth=3)[0]
        
        def init():
            """Initialize animation."""
            for scatter in scatter_dict.values():
                scatter.set_offsets(np.empty((0, 2)))
            for line in line_dict.values():
                line.set_data([], [])
            perturb_marker.set_data([perturb_x], [perturb_z])
            solenoid_text.set_text('')
            return list(scatter_dict.values()) + list(line_dict.values()) + [time_text, solenoid_text, perturb_marker]
        
        def animate(frame_idx):
            """Animate frame."""
            # Time in ms relative to perturbation
            time_ms = (frame_idx - perturb_idx) * Params.BIN_SIZE * 1000
            
            # Trail length: show last 30 frames (300ms at 100Hz)
            trail_start = max(0, frame_idx - 30)
            
            for body_part in self.dataset.BODY_PARTS:
                kin = kinematics_dict[body_part]
                
                # Current position
                curr_x, curr_z = kin[frame_idx, 0], kin[frame_idx, 2]
                scatter_dict[body_part].set_offsets([[curr_x, curr_z]])
                
                # Trail from 30 frames back to current frame
                trail_x = kin[trail_start:frame_idx+1, 0]
                trail_z = kin[trail_start:frame_idx+1, 2]
                line_dict[body_part].set_data(trail_x, trail_z)
            
            # Update time text
            time_text.set_text(f'Time: {time_ms:+.0f} ms')
            
            # Update solenoid indicator (ON for first 500ms after perturbation)
            if 0 <= time_ms <= 500:
                solenoid_text.set_text('SOLENOID ON')
            else:
                solenoid_text.set_text('')
            
            # Update perturbation marker position
            perturb_marker.set_data([perturb_x], [perturb_z])
            
            return list(scatter_dict.values()) + list(line_dict.values()) + [time_text, solenoid_text, perturb_marker]
        
        # Create animation (blit=False to handle text objects properly)
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                           interval=1000/fps, blit=False, repeat=True)
        
        # Title
        n_trials = self.dataset.n_trials_per_direction.get(direction, 0)
        fig.suptitle(f"{self.dataset.session_name} | 2D Animation (Mean) | Direction {direction} ({n_trials} trials)",
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig, anim
    
    def plot_trajectories_animation_single_trial(self,
                                                direction: int,
                                                trial_idx: int = 0,
                                                trial_type: str = 'trial',
                                                pre_ms: int = 200,
                                                post_ms: int = 1500,
                                                fps: int = 20) -> Tuple[plt.Figure, FuncAnimation]:
        """
        Animate 2D trajectories for a single trial.
        
        Parameters
        ----------
        direction : int
            Perturbation direction (0-11)
        trial_idx : int
            Which trial to animate (0-indexed)
        trial_type : str
            Trial type ('trial', 'free0', 'free1', 'intertrial')
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        fps : int
            Frames per second for animation
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        anim : FuncAnimation
            Animation object (assign to variable to keep it alive)
        """
        # Get kinematics and align
        kinematics_dict = {}
        perturb_idx = None
        
        for body_part in self.dataset.BODY_PARTS:
            kinematics_concat = self.dataset.get_kinematics(body_part, direction, trial_type)
            aligned, perturb_idx = self.dataset.align_to_perturbation(
                kinematics_concat, direction, pre_ms, post_ms
            )
            # Use single trial
            if trial_idx >= aligned.shape[0]:
                raise ValueError(f"Trial {trial_idx} out of range (0-{aligned.shape[0]-1})")
            kinematics_dict[body_part] = aligned[trial_idx]  # (n_frames, 3)
        
        n_frames = list(kinematics_dict.values())[0].shape[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # Set up plot limits
        all_x = np.concatenate([kin[:, 0] for kin in kinematics_dict.values()])
        all_z = np.concatenate([kin[:, 2] for kin in kinematics_dict.values()])
        x_min, x_max = all_x.min() - 2, all_x.max() + 2
        z_min, z_max = all_z.min() - 2, all_z.max() + 2
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([z_min, z_max])
        ax.set_xlabel('X (forward/back) [cm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (left/right) [cm]', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        
        # Initialize scatter plots and line objects
        scatter_dict = {}
        line_dict = {}
        
        for idx, body_part in enumerate(self.dataset.BODY_PARTS):
            color = self.colors[idx % len(self.colors)]
            scatter = ax.scatter([], [], s=150, c=[color], zorder=5, edgecolor='black', linewidth=1.5)
            scatter_dict[body_part] = scatter
            
            line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.8, zorder=3,
                          label=body_part.replace('_', ' ').title())
            line_dict[body_part] = line
        
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        
        # Time/frame indicator
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
                # Solenoid ON indicator (show during first 500ms)
        solenoid_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=11,
                               verticalalignment='top', fontweight='bold', color='white',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                # Perturbation onset marker (cross at the perturbation position)
        perturb_x = kinematics_dict[self.dataset.BODY_PARTS[0]][perturb_idx, 0]
        perturb_z = kinematics_dict[self.dataset.BODY_PARTS[0]][perturb_idx, 2]
        perturb_marker = ax.plot([], [], 'x', color='orange', markersize=20, zorder=1, markeredgewidth=3)[0]
        
        def init():
            """Initialize animation."""
            for scatter in scatter_dict.values():
                scatter.set_offsets(np.empty((0, 2)))
            for line in line_dict.values():
                line.set_data([], [])
            perturb_marker.set_data([perturb_x], [perturb_z])
            solenoid_text.set_text('')
            return list(scatter_dict.values()) + list(line_dict.values()) + [time_text, solenoid_text, perturb_marker]
        
        def animate(frame_idx):
            """Animate frame."""
            time_ms = (frame_idx - perturb_idx) * Params.BIN_SIZE * 1000
            
            # Trail length: show last 30 frames (300ms at 100Hz)
            trail_start = max(0, frame_idx - 30)
            
            for body_part in self.dataset.BODY_PARTS:
                kin = kinematics_dict[body_part]
                
                # Current position
                curr_x, curr_z = kin[frame_idx, 0], kin[frame_idx, 2]
                scatter_dict[body_part].set_offsets([[curr_x, curr_z]])
                
                # Trail from 30 frames back to current frame
                trail_x = kin[trail_start:frame_idx+1, 0]
                trail_z = kin[trail_start:frame_idx+1, 2]
                line_dict[body_part].set_data(trail_x, trail_z)
            
            # Update time text
            time_text.set_text(f'Time: {time_ms:+.0f} ms')
            
            # Update solenoid indicator (ON for first 500ms after perturbation)
            if 0 <= time_ms <= 500:
                solenoid_text.set_text('SOLENOID ON')
            else:
                solenoid_text.set_text('')
            
            # Update perturbation marker position
            perturb_marker.set_data([perturb_x], [perturb_z])
            
            return list(scatter_dict.values()) + list(line_dict.values()) + [time_text, solenoid_text, perturb_marker]
        
        # Create animation (blit=False to handle text objects properly)
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                           interval=1000/fps, blit=False, repeat=True)
        
        # Title
        n_trials = self.dataset.n_trials_per_direction.get(direction, 0)
        fig.suptitle(f"{self.dataset.session_name} | 2D Animation (Trial {trial_idx+1}/{n_trials}) | Direction {direction}",
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig, anim
