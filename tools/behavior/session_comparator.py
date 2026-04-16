"""
SessionComparator: Compare responses across normal day 1, day 2, and muscimol sessions.
Phase 2, Step 1 of the behavior analysis platform.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tools.behavior.behavior_dataset import BehaviorDataset
from tools.params import Params


class SessionComparator:
    """Compare kinematics across normal day 1, day 2, and muscimol sessions for same animal."""
    
    def __init__(self, normal_day1_session_name: str, normal_day2_session_name: str, muscimol_session_name: str):
        """
        Load three sessions for comparison (normal day 1, normal day 2, muscimol).
        
        Parameters
        ----------
        normal_day1_session_name : str
            Session name for normal day 1 (e.g., "M103_2026_02_18_15_30")
        normal_day2_session_name : str
            Session name for normal day 2 (e.g., "M103_2026_02_19_15_30")
        muscimol_session_name : str
            Session name for muscimol condition
        """
        self.day1_dataset = BehaviorDataset(normal_day1_session_name)
        self.day2_dataset = BehaviorDataset(normal_day2_session_name)
        self.muscimol_dataset = BehaviorDataset(muscimol_session_name)
        
        # Verify same animal
        animals = {self.day1_dataset.animal_id, self.day2_dataset.animal_id, self.muscimol_dataset.animal_id}
        if len(animals) > 1:
            raise ValueError(f"Sessions must be from same animal. Got: {animals}")
        
        self.animal_id = self.day1_dataset.animal_id
        self.day1_name = normal_day1_session_name
        self.day2_name = normal_day2_session_name
        self.muscimol_name = muscimol_session_name
    
    def compute_response_metrics(self,
                                 direction: int,
                                 trial_type: str = 'trial',
                                 pre_ms: int = 200,
                                 post_ms: int = 1500,
                                 baseline_window: Tuple[int, int] = (-200, 0)) -> Dict:
        """
        Compute per-body-part response metrics for all three conditions.
        
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
        baseline_window : Tuple[int, int]
            Time window for baseline (ms relative to perturbation onset)
        
        Returns
        -------
        metrics : Dict
            {condition: {body_part: {metric_name: value}}}
        """
        metrics = {
            'Normal Day 1': {},
            'Normal Day 2': {},
            'Muscimol': {}
        }
        
        for condition, dataset in [('Normal Day 1', self.day1_dataset), 
                                   ('Normal Day 2', self.day2_dataset),
                                   ('Muscimol', self.muscimol_dataset)]:
            for body_part in dataset.BODY_PARTS:
                metric_dict = {}
                
                # Get kinematics
                kin_concat = dataset.get_kinematics(body_part, direction, trial_type)
                aligned, perturb_idx = dataset.align_to_perturbation(
                    kin_concat, direction, pre_ms, post_ms
                )
                
                # Compute per-trial metrics and average
                trial_metrics = []
                for trial_idx in range(aligned.shape[0]):
                    trial_kin = aligned[trial_idx, :, :]  # (n_frames, 3)
                    trial_dict = self._compute_single_trial_metrics(
                        trial_kin, perturb_idx, baseline_window, body_part
                    )
                    trial_metrics.append(trial_dict)
                
                # Average across trials
                for key in trial_metrics[0].keys():
                    values = [m[key] for m in trial_metrics if m[key] is not None]
                    if values:
                        metric_dict[key] = np.mean(values)
                        metric_dict[f'{key}_sem'] = np.std(values) / np.sqrt(len(values))
                    else:
                        metric_dict[key] = np.nan
                        metric_dict[f'{key}_sem'] = np.nan
                
                metrics[condition][body_part] = metric_dict
        
        return metrics
    
    def _compute_single_trial_metrics(self, 
                                      kin: np.ndarray,  # (n_frames, 3)
                                      perturb_idx: int,
                                      baseline_window: Tuple[int, int],
                                      body_part: str) -> Dict:
        """Compute metrics for a single trial."""
        metrics = {}
        
        # Baseline position
        baseline_start = int((baseline_window[0] / 1000) / Params.BIN_SIZE + perturb_idx)
        baseline_end = int((baseline_window[1] / 1000) / Params.BIN_SIZE + perturb_idx)
        baseline_start = max(0, baseline_start)
        baseline_end = min(kin.shape[0], baseline_end)
        
        if baseline_start < baseline_end:
            baseline_pos = kin[baseline_start:baseline_end, :].mean(axis=0)
        else:
            baseline_pos = kin[0, :]
        
        # Response phase [0, 500ms]
        response_end_frame = int((0.5 / Params.BIN_SIZE) + perturb_idx)
        response_end_frame = min(response_end_frame, kin.shape[0])
        response_kin = kin[perturb_idx:response_end_frame, :]
        
        # Peak deviation (3D distance from baseline)
        deviations = np.linalg.norm(response_kin - baseline_pos, axis=1)
        peak_dev_idx = np.argmax(deviations)
        peak_latency_ms = peak_dev_idx * Params.BIN_SIZE * 1000
        peak_magnitude = deviations[peak_dev_idx]
        
        metrics['peak_latency_ms'] = peak_latency_ms
        metrics['peak_magnitude_cm'] = peak_magnitude
        
        # Peak position vector
        peak_pos = response_kin[peak_dev_idx]
        peak_vec = peak_pos - baseline_pos
        metrics['peak_x_cm'] = peak_vec[0]
        metrics['peak_y_cm'] = peak_vec[1]
        metrics['peak_z_cm'] = peak_vec[2]
        
        # Recovery (return to baseline: within 1.5cm)
        recovery_threshold = 1.5
        full_kin = kin[perturb_idx:, :]  # From solenoid on
        deviations_full = np.linalg.norm(full_kin - baseline_pos, axis=1)
        
        # Find first time after peak that returns to threshold
        recovery_frames = np.where(deviations_full[peak_dev_idx:] < recovery_threshold)[0]
        if len(recovery_frames) > 0:
            recovery_latency_ms = (peak_dev_idx + recovery_frames[0]) * Params.BIN_SIZE * 1000
            metrics['recovery_latency_ms'] = recovery_latency_ms
        else:
            metrics['recovery_latency_ms'] = None
        
        # Correction latency: when does trajectory reverse direction?
        # Compute velocity during response phase
        if response_kin.shape[0] > 1:
            velocity = np.diff(response_kin, axis=0)  # (frames-1, 3)
            velocity_mag = np.linalg.norm(velocity, axis=1)
            
            # Initial direction (deviation vector at peak)
            initial_direction = peak_vec / np.linalg.norm(peak_vec) if np.linalg.norm(peak_vec) > 0 else np.zeros(3)
            
            # Dot product of velocity with initial direction
            # Negative = moving back toward baseline
            corrections = []
            for frame_idx in range(peak_dev_idx, len(velocity)):
                if velocity_mag[frame_idx] > 0:
                    direction = velocity[frame_idx] / velocity_mag[frame_idx]
                    dot = np.dot(direction, initial_direction)
                    if dot < -0.3:  # Moving significantly against initial direction
                        corrections.append(frame_idx)
            
            if corrections:
                first_correction = corrections[0]
                correction_latency_ms = first_correction * Params.BIN_SIZE * 1000
                metrics['correction_latency_ms'] = correction_latency_ms
            else:
                metrics['correction_latency_ms'] = None
        else:
            metrics['correction_latency_ms'] = None
        
        return metrics
    
    def compare_conditions(self,
                          direction: int,
                          trial_type: str = 'trial',
                          pre_ms: int = 200,
                          post_ms: int = 1500) -> pd.DataFrame:
        """
        Create a comparison table of metrics across all three conditions and body parts.
        
        Returns
        -------
        df : pd.DataFrame
            Metrics comparison with columns: Body Part, Metric, Normal Day 1, Normal Day 2, Muscimol
        """
        metrics = self.compute_response_metrics(direction, trial_type, pre_ms, post_ms)
        
        rows = []
        metric_names = ['peak_latency_ms', 'peak_magnitude_cm', 'recovery_latency_ms', 
                       'correction_latency_ms', 'peak_x_cm', 'peak_y_cm', 'peak_z_cm']
        
        conditions = ['Normal Day 1', 'Normal Day 2', 'Muscimol']
        body_parts = metrics['Normal Day 1'].keys()
        
        for body_part in body_parts:
            for metric_name in metric_names:
                row = {
                    'Body Part': body_part.replace('_', ' ').title(),
                    'Metric': metric_name
                }
                
                for condition in conditions:
                    val = metrics[condition][body_part].get(metric_name)
                    row[condition] = val
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_trial_level_metrics(self,
                               direction: int,
                               trial_type: str = 'trial',
                               pre_ms: int = 200,
                               post_ms: int = 1500,
                               baseline_window: Tuple[int, int] = (-200, 0),
                               manual_annotations: Optional[Dict] = None,
                               use_manual_annotations: bool = False) -> Dict:
        """
        Get trial-level metrics (not averaged) for scatter plotting.
        Can use either auto-detected or manually-annotated latencies.
        
        Parameters
        ----------
        direction : int
            Perturbation direction
        trial_type : str
            Trial type ('trial', etc.)
        pre_ms : int
            Milliseconds before perturbation
        post_ms : int
            Milliseconds after perturbation
        baseline_window : Tuple[int, int]
            Baseline window for computing metrics
        manual_annotations : Dict, optional
            Manual annotations dict (only used if use_manual_annotations=True)
        use_manual_annotations : bool
            If True, use manually annotated latencies (from manual_annotations dict)
            If False, use auto-detected latencies
        
        Returns
        -------
        dict : {condition: {body_part: {metric_name: [trial_values]}}}
        """
        if use_manual_annotations and manual_annotations:
            return self._get_manual_annotation_metrics(manual_annotations, direction)
        
        # Auto-detected metrics (original behavior)
        trial_metrics = {
            'Normal Day 1': {},
            'Normal Day 2': {},
            'Muscimol': {}
        }
        
        for condition, dataset in [('Normal Day 1', self.day1_dataset), 
                                   ('Normal Day 2', self.day2_dataset),
                                   ('Muscimol', self.muscimol_dataset)]:
            for body_part in dataset.BODY_PARTS:
                metric_dict = {}
                
                # Get kinematics
                kin_concat = dataset.get_kinematics(body_part, direction, trial_type)
                aligned, perturb_idx = dataset.align_to_perturbation(
                    kin_concat, direction, pre_ms, post_ms
                )
                
                # Compute per-trial metrics
                trial_values = {}
                for trial_idx in range(aligned.shape[0]):
                    trial_kin = aligned[trial_idx, :, :]  # (n_frames, 3)
                    trial_dict = self._compute_single_trial_metrics(
                        trial_kin, perturb_idx, baseline_window, body_part
                    )
                    
                    # Collect each metric value across trials
                    for key, value in trial_dict.items():
                        if key not in trial_values:
                            trial_values[key] = []
                        trial_values[key].append(value)
                
                trial_metrics[condition][body_part] = trial_values
        
        return trial_metrics
    
    def _get_manual_annotation_metrics(self, manual_annotations: Dict, direction: int) -> Dict:
        """
        Convert manual annotations to trial-level metrics format.
        Filters annotations by direction and organizes by condition and body part.
        
        Parameters
        ----------
        manual_annotations : Dict
            Annotation dict with keys like "triplet|direction|condition|trial|bodypart"
        direction : int
            Only return annotations for this direction
        
        Returns
        -------
        dict : {condition: {body_part: {metric_name: [trial_values]}}}
        """
        trial_metrics = {
            'Normal Day 1': {},
            'Normal Day 2': {},
            'Muscimol': {}
        }
        
        # Initialize empty lists for all body parts
        for condition in trial_metrics.keys():
            for body_part in self.day1_dataset.BODY_PARTS:
                trial_metrics[condition][body_part] = {
                    'peak_latency_ms': [],
                    'recovery_latency_ms': [],
                    'correction_latency_ms': []
                }
        
        # Parse annotations and extract matching ones
        for key, annotation_data in manual_annotations.items():
            # Parse the key: "triplet|direction|condition|trial|bodypart"
            parts = key.split('|')
            if len(parts) < 5:
                continue
            
            ann_direction = int(parts[1])
            ann_condition = parts[2]
            ann_bodypart = parts[4]
            
            # Filter by direction
            if ann_direction != direction:
                continue
            
            # Only include valid conditions
            if ann_condition not in trial_metrics:
                continue
            
            # Extract latency values
            if isinstance(annotation_data, dict):
                peak_lat = annotation_data.get('peak_latency_ms')
                recovery_lat = annotation_data.get('recovery_latency_ms')
                correction_lat = annotation_data.get('correction_latency_ms')
                
                # Add to metrics (only if values exist)
                if ann_bodypart in trial_metrics[ann_condition]:
                    if peak_lat is not None:
                        trial_metrics[ann_condition][ann_bodypart]['peak_latency_ms'].append(peak_lat)
                    if recovery_lat is not None:
                        trial_metrics[ann_condition][ann_bodypart]['recovery_latency_ms'].append(recovery_lat)
                    if correction_lat is not None:
                        trial_metrics[ann_condition][ann_bodypart]['correction_latency_ms'].append(correction_lat)
        
        # Remove empty body parts
        for condition in trial_metrics.keys():
            empty_bps = [bp for bp, metrics in trial_metrics[condition].items() 
                        if all(len(v) == 0 for v in metrics.values())]
            for bp in empty_bps:
                del trial_metrics[condition][bp]
        
        return trial_metrics

    def get_asymmetry_metrics(self,
                             direction: int,
                             trial_type: str = 'trial',
                             pre_ms: int = 200,
                             post_ms: int = 1500) -> Dict:
        """
        Analyze left vs right limb asymmetry across all three conditions.
        Muscimol is unilateral right, so expect left limbs (especially upper) to be more affected.
        
        Returns
        -------
        asymmetry : Dict
            {condition: {metric: value}}
        """
        metrics = self.compute_response_metrics(direction, trial_type, pre_ms, post_ms)
        asymmetry = {}
        
        for condition in ['Normal Day 1', 'Normal Day 2', 'Muscimol']:
            cond_metrics = metrics[condition]
            
            # Left vs right limbs
            left_limbs = [bp for bp in cond_metrics.keys() if 'left' in bp]
            right_limbs = [bp for bp in cond_metrics.keys() if 'right' in bp]
            
            asym = {}
            
            for metric_key in ['peak_magnitude_cm', 'peak_latency_ms', 'recovery_latency_ms']:
                left_vals = [cond_metrics[bp].get(metric_key) for bp in left_limbs]
                right_vals = [cond_metrics[bp].get(metric_key) for bp in right_limbs]
                
                left_mean = np.nanmean([v for v in left_vals if v is not None])
                right_mean = np.nanmean([v for v in right_vals if v is not None])
                
                if not np.isnan(left_mean) and not np.isnan(right_mean):
                    asym[f'{metric_key}_left'] = left_mean
                    asym[f'{metric_key}_right'] = right_mean
                    asym[f'{metric_key}_asymmetry'] = left_mean - right_mean
            
            asymmetry[condition] = asym
        
        return asymmetry
