# Phase 1 Notebooks - Usage Guide

## Quick Start

### 1. Test Notebook (Validation & Demos)
```bash
jupyter lab notebooks/phase1_behavior_dataset_test.ipynb
```

**What it does:**
- Tests `BehaviorDataset` with M103_2026_02_18_15_30 session
- Demonstrates all 8 core query methods
- Shows `TimeSeriesPlotter` in action (Section 9)
- **Status**: ✅ All cells working, plots generated

**Key outputs visible:**
- 18-subplot kinematics grid with mean ± SEM bands
- 6-subplot 2D trajectories (X-Z plane)
- Statistics table at perturbation onset
- Velocity computations

---

### 2. Interactive Explorer Notebook
```bash
jupyter lab notebooks/phase1_single_session_explorer.ipynb
```

**What it does:**
- Interactive session/direction/trial-type selector
- Dynamic plot windows that update with sliders
- Real-time statistics display

**Controls available:**
- **Session Dropdown**: Select any of 25 sessions
- **Direction Slider**: 0-11 (perturbation directions)
- **Trial Type**: trial / free0 / free1 / intertrial
- **Time Windows**: Adjust pre/post milliseconds (slider range: 50-500ms pre, 500-3000ms post)

**Three interactive views:**
1. Kinematics Time Series (18 subplots)
2. 2D Trajectories (X-Z plane)
3. Statistics Table (position/SEM at perturbation onset)

---

## System Architecture

```
tools/behavior/
├── session_metadata.py    → SessionMetadata, SessionRegistry
├── behavior_dataset.py    → BehaviorDataset (core data interface)
├── time_series_plotter.py → TimeSeriesPlotter (visualization)
└── __init__.py           → Package exports

notebooks/
├── phase1_behavior_dataset_test.ipynb        → Testing & validation
└── phase1_single_session_explorer.ipynb     → Interactive exploration
```

---

## BehaviorDataset Methods (Core Toolkit)

| Method | Purpose |
|--------|---------|
| `get_trials(direction, trial_type)` | Filter trials by direction & type |
| `get_kinematics(body_part, direction, trial_type)` | Extract & concatenate kinematic arrays |
| `align_to_perturbation(arrays, direction, pre_ms, post_ms)` | Align to perturbation onset (t=0) |
| `get_continuous_data(trial_type, body_part)` | Raw continuous data for free periods |
| `extract_window(array, start_idx, duration_sec)` | Time windowing utility |
| `center_xz(arrays, axis)` | Remove locomotor drift (X, Z coordinates) |
| `compute_statistics(arrays)` | Mean & SEM across trials |
| `get_velocity(arrays)` | Velocity magnitude (cm/s) |

---

## TimeSeriesPlotter Methods (Visualization)

| Method | Output |
|--------|--------|
| `plot_kinematics_grid()` | 18-subplot grid (6 body parts × 3 coords) |
| `plot_trajectories_2d()` | 6 spatial trajectory plots (X-Z plane) |
| `get_statistics_table()` | Pandas DataFrame with stats |
| `plot_statistics_snapshot()` | Formatted matplotlib table figure |

---

## Example: Loading & Plotting a Session

```python
from tools.behavior import BehaviorDataset, TimeSeriesPlotter

# Load session
dataset = BehaviorDataset("M103_2026_02_18_15_30")
print(dataset)
# Output: M103_2026_02_18_15_30 (animal: M103, condition: normal)

# Create plotter
plotter = TimeSeriesPlotter(dataset)

# Generate plots
fig, axes = plotter.plot_kinematics_grid(
    direction=4,           # Perturbation direction
    trial_type='trial',    # Perturbation trials
    pre_ms=200,           # 200ms before
    post_ms=1500          # 1500ms after
)

# Get statistics
df = plotter.get_statistics_table(direction=4, trial_type='trial')
print(df)
```

---

## Data Structure Reference

**Sessions**: 25 total
- **Animals**: M103, M106
- **Conditions**: control, normal, muscimol
- **Directions**: 0-11 (12 perturbation directions)
- **Trial Types**: trial (perturbation), free0, free1, intertrial

**Kinematics**: 6 body parts × 3 coordinates per frame
- Body parts: left_foot, right_foot, hip_center, shoulder_center, left_paw, right_paw
- Coordinates: X (forward/back), Y (vertical), Z (left/right) [cm]
- Sampling: 100 Hz (10ms bin size)

**Free Periods**: Continuous unperturbed motion
- Free0: Before trial block
- Free1: After trial block
- Intertrial: Between perturbation trials

---

## Next Steps: Phase 2

**Planned**: Multi-animal, multi-condition dashboards
- Comparison across animals/conditions
- Population statistics
- Response trajectories
- Condition-specific filters

All Phase 1 infrastructure is ready - core data handling and visualization pipeline complete! 🚀
