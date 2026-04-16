"""
BehaviorDataset: Core data handling for a single session's kinematics.
Phase 1, Step 2 of the behavior analysis platform.
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import pyaldata as pyal
from scipy.stats import sem

from tools.behavior.session_metadata import SessionRegistry
from tools.dataTools import load_sessions, get_n_time
from tools.params import Params


class BehaviorDataset:
    """
    Loads and provides query interface to a single session's kinematic data.
    
    Supports:
    - Querying trials by direction and trial type
    - Extracting and aligning kinematics data
    - Computing statistics (mean, SEM)
    - Spatial centering for perturbation trials
    - Extracting windowed continuous data
    """
    
    BODY_PARTS = ["left_foot", "right_foot", "hip_center", "shoulder_center", "left_paw", "right_paw"]
    TRIAL_TYPES = ["trial", "free0", "free1", "intertrial"]
    COORDINATES = ["x", "y", "z"]
    
    def __init__(self, session_name: str, df: Optional[pd.DataFrame] = None, 
                 registry: Optional[SessionRegistry] = None):
        """
        Initialize BehaviorDataset for a session.
        
        Parameters
        ----------
        session_name : str
            Session name, e.g., "M103_2026_02_18_15_30"
        df : pd.DataFrame, optional
            Preprocessed dataframe. If None, loads from disk using load_sessions()
        registry : SessionRegistry, optional
            Session metadata registry. If None, creates new instance.
        """
        self.session_name = session_name
        
        # Load metadata
        if registry is None:
            registry = SessionRegistry()
        self.registry = registry
        self.metadata = registry.get(session_name)
        
        # Load or use provided dataframe
        if df is None:
            loaded_dfs = load_sessions([session_name], prep=True, only_trials=False)
            self.df = loaded_dfs[0]
        else:
            self.df = df
        
        # Pre-compute useful indices
        self._compute_trial_indices()
        self._compute_session_properties()
    
    def _compute_trial_indices(self):
        """Pre-compute which trials belong to each direction and trial type."""
        # Extract scalar direction values (each row has array, take first element)
        directions_raw = []
        for val in self.df['values_Sol_direction'].dropna():
            try:
                if isinstance(val, np.ndarray):
                    if len(val) > 0:
                        directions_raw.append(int(val[0]))
                else:
                    directions_raw.append(int(val))
            except (IndexError, TypeError, ValueError):
                # Skip values that can't be converted
                continue
        
        self.directions = sorted(set(directions_raw)) if directions_raw else []
        self.available_trial_types = sorted(self.df['trial_name'].unique())
        
        # Index trials by direction and trial type
        self._trials_by_direction = {}
        for direction in self.directions:
            mask = []
            for i, val in enumerate(self.df['values_Sol_direction']):
                if pd.notna(val):
                    try:
                        if isinstance(val, np.ndarray):
                            if len(val) > 0:
                                mask.append(int(val[0]) == direction)
                            else:
                                mask.append(False)
                        else:
                            mask.append(int(val) == direction)
                    except (IndexError, TypeError, ValueError):
                        mask.append(False)
                else:
                    mask.append(False)
            self._trials_by_direction[direction] = np.where(mask)[0].tolist()
        
        self._trials_by_type = {}
        for trial_type in self.available_trial_types:
            mask = self.df['trial_name'] == trial_type
            self._trials_by_type[trial_type] = np.where(mask)[0].tolist()
    
    def _compute_session_properties(self):
        """Compute session-level properties."""
        # Get n_time for each free period
        try:
            self.free0_n_frames = get_n_time(self.df, trial_name='free0', field=self.BODY_PARTS[0])
            self.free0_duration_sec = self.free0_n_frames * Params.BIN_SIZE
        except (ValueError, KeyError):
            self.free0_n_frames = 0
            self.free0_duration_sec = 0.0
        
        try:
            self.free1_n_frames = get_n_time(self.df, trial_name='free1', field=self.BODY_PARTS[0])
            self.free1_duration_sec = self.free1_n_frames * Params.BIN_SIZE
        except (ValueError, KeyError):
            self.free1_n_frames = 0
            self.free1_duration_sec = 0.0
        
        try:
            self.intertrial_n_frames = get_n_time(self.df, trial_name='intertrial', field=self.BODY_PARTS[0])
            self.intertrial_duration_sec = self.intertrial_n_frames * Params.BIN_SIZE
        except (ValueError, KeyError):
            self.intertrial_n_frames = 0
            self.intertrial_duration_sec = 0.0
        
        # Count trials per direction
        self.n_trials_per_direction = {
            direction: len(self._trials_by_direction[direction])
            for direction in self.directions
        }
    
    @property
    def animal_id(self) -> str:
        """Animal ID from metadata."""
        return self.metadata.animal_id
    
    @property
    def condition(self) -> str:
        """Condition from metadata."""
        return self.metadata.condition
    
    def get_trials(self, direction: Optional[int] = None, 
                   trial_type: str = 'trial') -> pd.DataFrame:
        """
        Get trials filtered by direction and trial type.
        
        Parameters
        ----------
        direction : int or None
            Perturbation direction (0-11). If None, return all directions.
        trial_type : str
            One of: 'trial', 'free0', 'free1', 'intertrial'
        
        Returns
        -------
        pd.DataFrame
            Filtered dataframe with selected trials.
        """
        mask = self.df['trial_name'] == trial_type
        
        if direction is not None and trial_type == 'trial':
            direction_mask = self.df['values_Sol_direction'] == direction
            mask = mask & direction_mask
        
        return self.df[mask].reset_index(drop=True)
    
    def get_kinematics(self, body_part: str, direction: Optional[int] = None,
                       trial_type: str = 'trial') -> np.ndarray:
        """
        Get concatenated kinematics for a body part.
        
        Parameters
        ----------
        body_part : str
            One of: 'left_foot', 'right_foot', 'hip_center', 'shoulder_center', 'left_paw', 'right_paw'
        direction : int or None
            Perturbation direction (0-11). If None, return all directions.
        trial_type : str
            One of: 'trial', 'free0', 'free1', 'intertrial'
        
        Returns
        -------
        np.ndarray
            Shape: (total_frames, 3) - concatenated across all selected trials
        """
        df_filtered = self.get_trials(direction=direction, trial_type=trial_type)
        
        if len(df_filtered) == 0:
            raise ValueError(f"No trials found for direction={direction}, trial_type={trial_type}")
        
        if body_part not in self.BODY_PARTS:
            raise ValueError(f"Unknown body part: {body_part}. Must be one of {self.BODY_PARTS}")
        
        # Concatenate arrays from all trials
        concatenated = np.concatenate(df_filtered[body_part].values, axis=0)
        return concatenated
    
    def align_to_perturbation(self, arrays: np.ndarray, direction: int,
                              pre_ms: float = 200, post_ms: float = 1500,
                              pad_mode: str = 'nan') -> Tuple[np.ndarray, int]:
        """
        Align trials to perturbation onset (idx_sol_on).
        
        Parameters
        ----------
        arrays : np.ndarray
            Concatenated kinematic data, shape (total_frames, 3)
        direction : int
            Direction (needed to get correct trial indices)
        pre_ms : float
            Milliseconds before perturbation to include (default: 200)
        post_ms : float
            Milliseconds after perturbation to include (default: 1500)
        pad_mode : str
            How to handle variable-length trials: 'nan' to pad with NaN, 'truncate' to use shortest
        
        Returns
        -------
        aligned_arrays : np.ndarray
            Shape: (n_trials, n_frames_per_trial, 3)
        perturbation_frame_idx : int
            Frame index at which perturbation occurs (in the alignment window)
        """
        # Convert ms to frame indices
        pre_frames = int(pre_ms / (Params.BIN_SIZE * 1000))
        post_frames = int(post_ms / (Params.BIN_SIZE * 1000))
        window_frames = pre_frames + post_frames
        
        # Get trials for this direction
        df_trials = self.get_trials(direction=direction, trial_type='trial')
        
        if len(df_trials) == 0:
            raise ValueError(f"No trials found for direction {direction}")
        
        # Get perturbation onset indices
        idx_sol_on = df_trials['idx_sol_on'].values.astype(int)
        
        # Split concatenated arrays back into trials
        trial_arrays = []
        frame_idx = 0
        for idx_onset in idx_sol_on:
            trial_len = len(df_trials.iloc[trial_arrays.__len__()][self.BODY_PARTS[0]])
            trial_arrays.append(arrays[frame_idx:frame_idx + trial_len])
            frame_idx += trial_len
        
        # Align each trial to perturbation onset
        aligned = []
        for trial_data, onset in zip(trial_arrays, idx_sol_on):
            # Extract window around onset
            start_idx = max(0, onset - pre_frames)
            end_idx = min(len(trial_data), onset + post_frames)
            
            window = trial_data[start_idx:end_idx]
            
            # Pad if necessary
            if pad_mode == 'nan':
                if len(window) < window_frames:
                    pad_width = [(0, window_frames - len(window)), (0, 0)]
                    padded = np.pad(window, pad_width, mode='constant', constant_values=np.nan)
                    aligned.append(padded)
                else:
                    aligned.append(window[:window_frames])
            elif pad_mode == 'truncate':
                aligned.append(window)
        
        # Convert list to array, handling variable lengths if truncate mode
        if pad_mode == 'nan':
            aligned_array = np.array(aligned)  # All same length
        else:
            # Keep as list of arrays with jagged shape
            aligned_array = aligned
        
        # Perturbation frame index in the window (relative to window start)
        perturbation_frame_idx = pre_frames
        
        return aligned_array, perturbation_frame_idx
    
    def get_continuous_data(self, trial_type: str, body_part: str) -> np.ndarray:
        """
        Get continuous kinematic data (no alignment).
        
        For free0, free1, intertrial - returns raw concatenated data.
        
        Parameters
        ----------
        trial_type : str
            One of: 'free0', 'free1', 'intertrial'
        body_part : str
            One of the BODY_PARTS
        
        Returns
        -------
        np.ndarray
            Shape: (total_frames, 3)
        """
        if trial_type not in ['free0', 'free1', 'intertrial']:
            raise ValueError(f"Use get_kinematics() for '{trial_type}'. This method is for continuous data.")
        
        return self.get_kinematics(body_part, direction=None, trial_type=trial_type)
    
    def extract_window(self, array: np.ndarray, start_idx: int, 
                       duration_sec: float) -> np.ndarray:
        """
        Extract a time window from continuous data.
        
        Parameters
        ----------
        array : np.ndarray
            Continuous kinematic data, shape (total_frames, 3)
        start_idx : int
            Starting frame index
        duration_sec : float
            Duration of window in seconds
        
        Returns
        -------
        np.ndarray
            Shape: (n_frames_in_window, 3)
        """
        n_frames = int(duration_sec / Params.BIN_SIZE)
        end_idx = min(start_idx + n_frames, len(array))
        return array[start_idx:end_idx]
    
    def center_xz(self, arrays: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Center X and Z coordinates around their mean.
        Removes locomotor drift, reveals response kinematics.
        
        Parameters
        ----------
        arrays : np.ndarray
            Kinematic data, shape (n_trials, n_frames, 3) or (n_frames, 3)
        axis : int
            Axis over which to compute mean. 0=across trials, None=overall
        
        Returns
        -------
        np.ndarray
            Centered version of input, same shape
        """
        if arrays.ndim == 2:
            # Single trial or concatenated: (n_frames, 3)
            # Center X (col 0) and Z (col 2) around their means
            arrays_centered = arrays.copy()
            arrays_centered[:, 0] -= np.nanmean(arrays[:, 0])
            arrays_centered[:, 2] -= np.nanmean(arrays[:, 2])
            return arrays_centered
        elif arrays.ndim == 3:
            # Multiple trials: (n_trials, n_frames, 3)
            arrays_centered = arrays.copy()
            if axis == 0:
                # Compute mean per frame across trials, then center each trial
                mean_x = np.nanmean(arrays[:, :, 0], axis=0, keepdims=True)
                mean_z = np.nanmean(arrays[:, :, 2], axis=0, keepdims=True)
                arrays_centered[:, :, 0] -= mean_x
                arrays_centered[:, :, 2] -= mean_z
            else:
                # Center each trial around its own mean
                for i in range(arrays.shape[0]):
                    arrays_centered[i, :, 0] -= np.nanmean(arrays[i, :, 0])
                    arrays_centered[i, :, 2] -= np.nanmean(arrays[i, :, 2])
            return arrays_centered
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arrays.shape}")
    
    def compute_statistics(self, arrays: np.ndarray, 
                          axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and SEM across trials.
        
        Parameters
        ----------
        arrays : np.ndarray
            Shape: (n_trials, n_frames, 3) for trial data
        axis : int
            Axis over which to compute (default: 0 = across trials)
        
        Returns
        -------
        mean : np.ndarray
            Shape: (n_frames, 3)
        sem_vals : np.ndarray
            Shape: (n_frames, 3)
        """
        mean = np.nanmean(arrays, axis=axis)
        sem_vals = sem(arrays, axis=axis, nan_policy='omit')
        return mean, sem_vals
    
    def get_velocity(self, arrays: np.ndarray) -> np.ndarray:
        """
        Compute velocity from position data.
        
        Parameters
        ----------
        arrays : np.ndarray
            Position data, shape (..., n_frames, 3) or (n_frames, 3)
        
        Returns
        -------
        np.ndarray
            Velocity magnitude, shape (..., n_frames) or (n_frames,)
        """
        # Compute differences along time axis
        if arrays.ndim == 3:
            # (n_trials, n_frames, 3)
            diff = np.diff(arrays, axis=1)  # Shape: (n_trials, n_frames-1, 3)
            vel = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n_trials, n_frames-1)
            # Pad to original length
            vel = np.pad(vel, ((0, 0), (0, 1)), mode='edge')
        elif arrays.ndim == 2:
            # (n_frames, 3)
            diff = np.diff(arrays, axis=0)  # Shape: (n_frames-1, 3)
            vel = np.sqrt(np.sum(diff**2, axis=1))  # Shape: (n_frames-1,)
            vel = np.pad(vel, (0, 1), mode='edge')
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arrays.shape}")
        
        # Convert from cm/bin to cm/s
        vel = vel / Params.BIN_SIZE
        return vel
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BehaviorDataset({self.session_name})\n"
            f"  Animal: {self.animal_id}\n"
            f"  Condition: {self.condition}\n"
            f"  Directions: {self.directions}\n"
            f"  Trials/direction: {self.n_trials_per_direction}\n"
            f"  Free0: {self.free0_duration_sec:.1f}s\n"
            f"  Free1: {self.free1_duration_sec:.1f}s\n"
            f"  Intertrial: {self.intertrial_duration_sec:.1f}s"
        )
