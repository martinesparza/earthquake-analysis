# Single Session Explorer - README

## Overview
The **Single Session Explorer** is an interactive Jupyter notebook for deep exploration of a single session's behavioral kinematics data. It provides multiple synchronized views of movement across 6 body parts during perturbation trials, free running, and intertrial periods.

**Current Focus**: Behavioral kinematics only (no neural data). Multi-animal, multi-condition comparisons are handled in separate dashboards.

## Features

### 1. Session Selection
- Load any session from the registry
- Auto-displays session metadata: animal ID, condition, date, trial counts
- Session summary shows: free0 duration, trial count, perturbation timing

### 2. Perturbation Trial Analysis
**Time Alignment**: Perturbation always occurs at **t=2.0 seconds** within the 6-second trial window.

**Default visualization**: -0.2s to +1.5s relative to perturbation onset
- **Pre-perturbation**: 200ms of baseline before stimulus
- **Stimulus response**: 1500ms capture of response dynamics

**User Control**: Adjustable time window via slider
- Pre-onset: adjustable from -0.5s to -0.1s
- Post-onset: adjustable from +0.5s to +3.0s
- Real-time plot updates as sliders change

**Display Modes**:
- **Average**: Mean ± SEM across all trials of selected direction
- **All Trials**: Individual trial trajectories (semi-transparent) with mean overlaid (bold)

**Direction Selection**:
- Slider: 0-11 for specific direction
- "All Directions" option for population view (averaged across all 12)

**Spatial Centering** (for trials only):
- Toggle to center the body's X-Z position within each trial before averaging
- Removes locomotor drift and reveals response kinematics more clearly
- Useful for comparing across trials with different starting positions

### 3. Free Running Analysis (Free0, Free1, Intertrial)

**Sliding Window Visualization**:
- Display 5-second window of continuous behavior
- Slider to move window through full free period (e.g., 4-20 min duration)
- Real-time update as slider moves

**Sliding Window Statistics**:
- Compute rolling averages of kinematic metrics every 100ms
- Display overlaid on raw data:
  - Velocity (computed from position derivatives)
  - Range of motion per coordinate
  - Activity level (sparsity of movement)
- Identify movement bouts, rest periods, gait patterns

**Window Size Adjustment**:
- Dropdown: 2s, 5s, 10s window
- Useful for examining movement at different timescales

### 4. Kinematic Visualization - 18 Time-Series Plots

**Body Parts** (6 total, organized as columns):
- Left Foot
- Right Foot
- Hip Center
- Shoulder Center
- Left Paw
- Right Paw

**Coordinates** (3 per body part, organized as rows):
- **X**: Forward/backward (locomotor axis)
- **Y**: Vertical (up/down)
- **Z**: Left/right (lateral)

**Plot Elements**:
- Line(s): Mean trajectory (blue or red for left/right, black for center)
- Shaded band: ±SEM around mean
- Individual trials (optional, low opacity in "All Trials" mode)
- **Perturbation markers** (Trials only):
  - Vertical dashed line at onset
  - Light gray shaded region during stimulus window
  - Time axis labeled with t=0 at perturbation onset

**Time Axis**:
- In seconds, synchronized across all 18 plots
- For Trials: relative to perturbation onset (t=0 at stimulus)
- For Free0/Free1: absolute time from session start

**Y-Axis**:
- Position in cm
- Auto-scaled per coordinate per trial type (prevents overcrowding)

### 5. 2D Spatial Projections - X-Z Plane

**6 subplots** (one per body part), showing forward/lateral movement:

**Layout**:
```
Left Foot              Right Foot             Hip Center
(avg ± SEM)            (avg ± SEM)            (avg ± SEM)

Z ▲                    Z ▲                    Z ▲
  │ ●                    │ ●                    │ ●
  │ │\                   │ │\                   │ │
  │ │ └─→                │ │ └─→                │ │ └─→
  └─────→ X             └─────→ X              └─────→ X
  
Left Paw               Right Paw              Shoulder Center
```

**Visual Encoding**:
- **Trajectory path**: Colored by time phase
  - Black/gray: pre-perturbation (trials only)
  - Blue/red: during/after perturbation response
  - Darker colors: later in response window
- **Markers**: 
  - ● at perturbation onset
  - Arrow at trajectory end
- **Shaded envelope** (optional): shows ±SEM in 2D space

**Features for Trials**:
- Automatic centering option (removes locomotor drift)
- Reveals response kinematics independent of locomotor trajectory

**Features for Free0/Free1**:
- Shows exploratory movement pattern / gait statistics
- Trajectory may be longer (>1 minute windowed data)
- Can identify preferred movement directions, territory use

**Toggle**: Show/hide all trajectories, or individual body parts

### 6. Statistics Panel

**Table Format**: Body part × statistic

**Metrics**:
| Body Part | X: Mean ± SD | Y: Mean ± SD | Z: Mean ± SD | Range (X,Y,Z) | Peak Speed |
|-----------|--------------|--------------|--------------|---------------|-----------|
| Left Foot | 2.1±0.3 | 5.0±0.2 | -1.2±0.4 | (1.5,1.0,1.2) | 8.3 cm/s |
| ... | ... | ... | ... | ... | ... |

**Includes**:
- Mean position ± standard deviation per coordinate
- Range of motion (min to max position)
- Peak velocity (maximum instantaneous speed during window)
- Computed only for selected trial type/direction

**Updates**: Dynamically as sliders/selectors change

---

## User Interface (Ipywidgets)

### Control Panel Layout

```
╔═══════════════════════════════════════════════════════════════════════════╗
║ SINGLE SESSION EXPLORER                                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║ SESSION: [M103_2026_02_18_15_30 ▾ Reload]  Animal: M103 | Condition: Normal ║
║                                                                           ║
║ ┌─────────────────────────────────────────────────────────────────────┐  ║
║ │ TRIAL SELECTION (for Perturbation Trials only)                      │  ║
║ │                                                                      │  ║
║ │ Direction: ┌─────────────────────────────────────────┐              │  ║
║ │            │ All ◆ ◉ 0   1   2   3   4   5 ... 11   │              │  ║
║ │            └┬────────────────────────────────────────┘              │  ║
║ │             └ Current: Direction 4 (n=50 trials)                    │  ║
║ │                                                                      │  ║
║ │ Display Mode: ◉ Average  ○ All Trials                               │  ║
║ │                                                                      │  ║
║ │ Time Window:                                                         │  ║
║ │   Pre-onset:  ─0.50s ─────────┬───── ─0.10s                         │  ║
║ │   Post-onset: +0.50s ─────────┬───── +3.00s                         │  ║
║ │   [Current: -0.20s to +1.50s]                                       │  ║
║ │                                                                      │  ║
║ │ Options:                                                             │  ║
║ │   ☑ Center X-Z position before averaging                            │  ║
║ │   ☑ Show ±SEM bands                                                 │  ║
║ │                                                                      │  ║
║ └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║ ┌─────────────────────────────────────────────────────────────────────┐  ║
║ │ FREE RUNNING ANALYSIS (Free0, Free1, Intertrial)                    │  ║
║ │                                                                      │  ║
║ │ Trial Type: ◉ Trials  ○ Free0  ○ Free1  ○ Intertrial               │  ║
║ │             [when Free0/Free1/Intertrial selected:]                 │  ║
║ │                                                                      │  ║
║ │ Window Size: [5 seconds ▾]  Total Duration: 480 sec (8.0 min)     │  ║
║ │                                                                      │  ║
║ │ Position in Free Period:                                             │  ║
║ │           ┌───────────────────────────────────────┐                 │  ║
║ │           │ ◀ [====|====================================] ▶          │  ║
║ │           └───────────────────────────────────────┘                 │  ║
║ │           0s                  [120-125s window]    480s             │  ║
║ │                                                                      │  ║
║ │ ☑ Show sliding window averages (velocity, range of motion)         │  ║
║ │ ☑ Show individual trials overlay (if Intertrial selected)          │  ║
║ │                                                                      │  ║
║ └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║ ┌─────────────────────────────────────────────────────────────────────┐  ║
║ │ VISUALIZATION OPTIONS                                               │  ║
║ │                                                                      │  ║
║ │ Body Parts:  Body Parts:  ☑ Left Foot  ☑ Right Foot  ☑ Hip Center  │  ║
║ │              ☑ Shoulder   ☑ Left Paw   ☑ Right Paw                 │  ║
║ │                                                                      │  ║
║ │ Panels to Display:                                                   │  ║
║ │   ☑ Kinematic Time-Series (18 plots)                                │  ║
║ │   ☑ 2D Trajectories (X-Z plane)                                     │  ║
║ │   ☑ Statistics Table                                                │  ║
║ │                                                                      │  ║
║ └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║ [Generate Figure]  [Export Data]  [Save Settings]                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Data Pipeline

### 1. Session Loading
```
User selects session
    ↓
Load via SessionRegistry (metadata only)
    ↓
Load full session data using dt.load_sessions()
    ↓
Preprocess (free0/free1 labeling, trial type separation)
    ↓
BehaviorDataset wraps preprocessed DataFrame
    ↓
Cache in memory for fast interaction
```

### 2. Trial-Based Analysis (Perturbation Response)
```
User selects direction + time window
    ↓
Query: get_trials(direction=4, trial_type='trial')
    ↓
Find perturbation onset: idx_sol_on per trial
    ↓
Align all trials to onset (-0.2s to +1.5s default)
    ↓
Optional: Center X-Z position
    ↓
Compute mean ± SEM across trials
    ↓
Convert to time axis (seconds)
    ↓
Plot with 18 subplots + 2D trajectories + stats
```

### 3. Free Running Analysis (Continuous Behavior)
```
User selects Free0/Free1/Intertrial + window size
    ↓
Concatenate all continuous data for that type
    ↓
User drags slider to select 5-second window (e.g., t=120-125s)
    ↓
Extract window from continuous data
    ↓
Compute sliding window statistics (rolling mean velocity, ROM)
    ↓
Plot raw data + overlaid rolling averages
    ↓
Update instantly as slider moves
```

---

## Technical Implementation

### Core Classes (to be built)

1. **BehaviorDataset**
   - Input: session name, preprocessed DataFrame
   - Methods:
     - `get_trials(direction, trial_type)`
     - `align_to_perturbation(arrays, pre_ms, post_ms)`
     - `compute_statistics(arrays)`
     - `extract_window(array, start_idx, duration)`

2. **TimeSeriesPlotter**
   - Methods:
     - `plot_kinematics_grid(direction, trial_type, mode='average')`
     - `plot_trajectories_2d(direction, centered=False)`
     - `plot_free_running_window(trial_type, window_start, window_duration)`
     - `get_statistics_table()`

3. **Notebook Interface** (Ipywidgets callbacks)
   - Synchronize all sliders/dropdowns
   - Update figures in real-time
   - Cache computed data for speed

### File Structure
```
notebooks/
├── phase1_session_metadata.ipynb         [Phase 1 Step 1 - DONE]
└── phase1_single_session_explorer.ipynb  [Phase 1 Step 2 - TO BUILD]

tools/behavior/
├── __init__.py
├── session_metadata.py                   [DONE]
├── behavior_dataset.py                   [TO BUILD]
├── time_series_plotter.py                [TO BUILD]
└── free_running_analyzer.py              [TO BUILD - optional]

metadata/
└── sessions.csv                          [DONE]
```

---

## First Test Case

**Session**: M103_2026_02_18_15_30
- Animal: M103
- Condition: Normal (day 1 post-control)
- Perturbations: 12 directions × ~50 trials
- Free periods: free0 & free1 (4-8 min each)

**Initial Test**:
1. Load session
2. Plot Direction 4, average mode, default time window (-0.2 to +1.5s)
3. Check: 18 plots render correctly, colors are sensible
4. Try: Direction slider, time window sliders
5. Toggle: center X-Z position → should see clearer response kinematics
6. Switch to Free0, drag window slider → should update in real-time

---

## Next Phases

**Phase 2**: Condition Comparison (same animal, control | normal | muscimol side-by-side)
- Builds on Single Session Explorer
- Adds statistical comparison layer

**Phase 3**: Multi-Animal Comparison (same condition, all animals)
- Population-level consistency checks

**Phase 4**: Summary Statistics & Reporting
- ANOVA tables, effect sizes, automated report generation

---

## Dependencies

Must already be available:
- pyaldata
- pandas, numpy
- matplotlib
- scipy (for statistics)
- ipywidgets
- tools.params.Params (BIN_SIZE)
- tools.dataTools (load_sessions, get_n_time, add_history_to_df, etc.)

---

## Performance Notes

- **Cache**: Preprocessed session stays in memory (typical: 500MB-2GB)
- **Responsiveness**: Time window sliders update in <500ms (no recomputation)
- **Memory**: All data for typical session fits in laptop RAM
- **Computation**: First load ~5-10s, then interactive

---

## Future Enhancements

1. **Video export**: Generate MP4 animations of selected trial direction
2. **Dimensionality reduction**: PCA on kinematics data for latent dynamics
3. **Gait analysis**: Automatic step detection, stride length, cadence
4. **Neural overlay** (Phase 2+): Display brain activity aligned with kinematics
5. **Cross-session comparison**: Overlay this session's avg against other sessions
6. **Batch analysis**: Run analysis on multiple sessions at once, generate CSV summary

---

## Questions / Troubleshooting

**Q**: Why does the X-Z centering only work for Trials?
**A**: Free running data has legitimate spatial drift (locomotion through space). Centering removes that information. For Trials, the animal stays in roughly the same place, so centering reveals response kinematics.

**Q**: Can I change the 5-second window size for free periods?
**A**: Yes, dropdown menu: 2s, 5s, 10s. Smaller windows catch fast movements, larger windows show overall trends.

**Q**: Why is the sliding window average overlay important?
**A**: Raw kinematics are noisy. Rolling averages reveal gait patterns, movement bouts, and when the animal transitions between behavioral states.

**Q**: Can I export the computed statistics?
**A**: Yes, statistics table can be exported as CSV. Individual trial data can be saved for downstream analysis.

---

**Contact**: For issues or feature requests, see BEHAVIOR_ANALYSIS_WORKFLOW.md or DASHBOARD_ARCHITECTURE.md for broader context.
