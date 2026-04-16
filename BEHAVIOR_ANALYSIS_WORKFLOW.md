# Behavior Analysis & Visualization Platform - Workflow Plan

## Executive Summary
Your data contains **3D kinematic trajectories** (body part positions), **neural recordings** (spike rates from 4 brain areas), and **behavioral trials** (perturbation responses in 12 directions + free running). You need a unified platform to visualize and analyze these across temporal, directional, and behavioral states.

---

## I. DATA ORGANIZATION & STRUCTURE

### Current DataFrame Organization
- **Type**: PyAlData (pandas subclass) with tabular trials
- **Kinematics**: Stored as numpy arrays per trial (shape: N_samples × 3 for 3D coords)
- **Neural**: Spike rates per trial (shape: N_samples × N_neurons per area)
- **Time resolution**: 10ms bins (BIN_SIZE = 0.01s)

### Data Slicing Patterns
```
trials              → response to perturbation (12 directions × 30+ repetitions)
intertrials         → transitions between trials
free0/free1         → unperturbed locomotion
```

### Key Metadata Available
- `values_Sol_direction`: Perturbation direction (0-11)
- `idx_sol_on`: Frame index of perturbation onset
- `trial_name`: Trial type identifier

---

## II. VISUALIZATION TIERS & USE CASES

### **Tier 1: Overview / Batch Analysis**
**Purpose**: Understand behavior across conditions quickly

#### 1.1 **Condition Heatmaps**
- Direction × metric (e.g., peak velocity, limb asymmetry) heatmaps
- Free running metrics (gait frequency, stride length, symmetry)
- Comparison: Control vs Perturbation vs Muscimol conditions
- **Tools**: Matplotlib heatmap or Seaborn
- **Output**: Static images for reports

#### 1.2 **Trial-Averaged Trajectories (2D Projections)**
- X-Z plane (left-right and forward-backward)
- Y-Z plane (vertical component)
- Multiple body parts overlaid (feet, hips, paws)
- **Tools**: Matplotlib plot with color-coded phases (pre-perturb, during, post-recovery)
- **Data**: Average ± SEM across trials per direction

---

### **Tier 2: Detailed Time-Series Inspection**
**Purpose**: Understand kinematic dynamics across time and conditions

#### 2.1 **Interactive 1D Time-Series Dashboard**
A single multi-panel display showing:
- **Panel A (top)**: Time axis (x), coordinate value (y), one plot per body part
  - X/Y/Z coordinates for each of 6 body parts = 18 subplots
  - Color indicates trial type or perturbation state
  - **Add visual markers**: Perturbation onset, recovery onset
  
- **Panel B (bottom)**: Neural activity
  - Each brain area as separate subplot
  - RasterPlot or colored time series (MOp, SSp, CP, VAL)
  - Same time axis aligned with kinematics

**Interactive Features**:
- Slider to select direction (0-11)
- Toggle: show average vs. individual trials
- Checkbox: show/hide specific body parts
- Dropdown: select trial type (trial/intertrial/free0/free1)

**Tools**: Matplotlib + Ipywidgets (in notebook) OR plotly (interactive HTML)

**Data handling**:
- Pre-compute averaged trials per direction
- Smooth data with rolling window for clarity

---

### **Tier 3: 3D Interactive Visualization**
**Purpose**: Understand full 3D body kinematics and movement space

#### 3.1 **Static 3D Scatter Plot** (Quick setup)
- Skeleton connectivity (body part positions linked)
- Animation frame by frame for single trial
- Color = time or bodypart
- **Tools**: Matplotlib 3D or Plotly 3D scatter
- **Output**: PNG images for inspection, HTML for interactive exploration

#### 3.2 **3D Animation / Video** (Medium complexity)
- Real-time animated skeleton for each trial
- Overlaid perturbation force vector (if available)
- Playback controls: speed, direction selection
- **Tools**: 
  - Matplotlib animation (creates MP4)
  - Plotly Dash (interactive web app)
  - Vispy or Napari (high-performance 3D)
- **Output**: MP4 videos or interactive web app

#### 3.3 **3D Avatar with Biomechanical Model** (Advanced)
- Skeleton linkages with segment lengths
- Joint angle visualization (if computed from positions)
- Overlay neural activity as color/intensity on joints
- **Tools**: Vispy, Three.js via plotly Dash, or custom WebGL
- **Requires**: Either direct joint angles or inverse kinematics computation

---

### **Tier 4: Multi-Trial Comparative Analysis**
**Purpose**: Compare responses across directions, conditions, or animals

#### 4.1 **Trajectory Overlay**
- Multiple trials on same 2D/3D plot with transparency
- Color = trial number, direction, or outcome (successful/failed)
- Desktop tool for drawing ROIs or detecting outliers
- **Tools**: Matplotlib (static) or Plotly (interactive)

#### 4.2 **Velocity/Acceleration Fields**
- Heatmap: direction × time → velocity magnitude or direction
- Trajectory colored by velocity
- Identify fast vs. slow response phases
- **Tools**: Matplotlib quiver, Plotly heatmap

#### 4.3 **Phase-Space Portraits**
- Joint angle vs. velocity, or position vs. velocity
- Limit cycles for periodic behaviors (walking)
- Divergence from normal for perturbation response
- **Tools**: Matplotlib scatter/streamplot

---

## III. RECOMMENDED IMPLEMENTATION SEQUENCE

### **Phase 1: Foundation (Week 1)**
Build utilities for data standardization and querying.

**Deliverables:**
1. **BehaviorDataset class** (Python)
   - Input: Single or multiple sessions
   - Methods:
     - `get_trials(direction, trial_type)` → returns trials as list
     - `get_aligned_kinematics(direction, trial_type, relative_to="perturb_onset")` → time-aligned arrays
     - `get_concatenated_neural(areas, trial_type)` → concatenated spike rates
   - Properties: `trial_types`, `directions`, `body_parts`, `neural_areas`

2. **Time alignment standardization**
   - Compute all epochs relative to perturbation onset (for perturbation trials)
   - For free running: segment into windows or use entire session
   - Handle trials of different lengths (pad/truncate strategy)

3. **Quality control filters**
   - Remove trials with missing data
   - Remove frames where animal is stationary (define threshold)
   - Log statistics: n_valid_trials, n_frames, etc.

**Code location**: `tools/behavior/dataset.py`

---

### **Phase 2: 1D Time-Series Dashboard (Week 2)**
Most versatile for rapid exploration.

**Deliverables:**
1. **TimerSeriesPlotter class** (Python + Jupyter)
   - Input: BehaviorDataset, direction, trial_type, display_mode
   - Outputs:
     - matplotlib figure with synchronized kinematics + neural panels
     - Ipywidgets sliders for real-time control
   - Features:
     - SEM bands on averaged trials
     - Perturbation window highlight
     - Customizable body part selection

2. **Interactive notebook widget**
   - One cell with sliders/dropdowns
   - Live plot updates
   - Save current plot as PNG

**Code location**: `tools/behavior/time_series_viz.py`

**Use case**: Day-to-day exploration of new sessions

---

### **Phase 3: 2D Trajectory Visualization (Week 2-3)**
Quick to implement, high insight.

**Deliverables:**
1. **TrajectoryPlotter class**
   - Methods:
     - `plot_direction_comparison(direction)` → direction-averaged traj with ±SEM
     - `plot_trial_overlay(direction, n_trials)` → multiple trials transparent
     - `plot_free_running(start_time, duration)` → walking periods

2. **Specialized plots**
   - X-Z plane (forward/lateral)
   - Y-Z plane (vertical/lateral)
   - X-Y plane (forward/vertical)
   - Foot-to-foot distance over time
   - Hip height over time

3. **Annotation tools**
   - Mark recovery time (manual or automatic)
   - Label different locomotor phases (stepping, stance, swing)
   - Identify outlier trials

**Code location**: `tools/behavior/trajectory_viz.py`

**Use case**: Manuscript figures, quick hypothesis testing

---

### **Phase 4: 3D Visualization (Week 3-4)**
Build from simpler to more complex.

**Step 4a: Static 3D Scatter (Day 1)**
- Plot skeleton as points + line segments
- Show single trial per plot
- Save PNG per trial/direction

**Step 4b: Frame-by-Frame Animation (Day 2-3)**
- Loop through frames, update scatter positions
- Save as MP4 using matplotlib.animation
- Overlay perturbation timestamp

**Step 4c: Interactive 3D (Optional, Week 4)**
- Plotly 3D scatter or Dash app
- Dropdown to select trial
- Slider for frame navigation
- Button to animate

**Code location**: `tools/behavior/skeleton_viz.py`

**Use case**: Presentations, supplementary materials, detailed case studies

---

### **Phase 5: Comparative Analysis Dashboard (Week 4-5)**
Synthesize Phases 1-3 into unified platform.

**Deliverables:**
1. **Multi-condition comparisons**
   - Heatmaps: Direction × metric (e.g., peak velocity)
   - Grouped bar plots: Control vs. Muscimol across directions
   - Statistical overlays (p-values, significance stars)

2. **Automated metrics extraction**
   - Peak velocity per trial
   - Time to recovery
   - Step symmetry (left vs. right limb)
   - Gait frequency
   - Perturbation magnitude (estimated from deviation)

3. **Web dashboard (Plotly Dash)**
   - Session selector
   - Condition tabs (perturbation vs. free running)
   - Real-time metric visualization
   - Export options (CSV, PNG, MP4)

**Code location**: `tools/behavior/dashboard.py` (Dash app)

---

## IV. TECHNICAL STACK RECOMMENDATIONS

### **For Static Plotting**
- Matplotlib (already in use, good for papers)
- Seaborn (enhanced heatmaps, violin plots)
- Plotly (interactive but exportable as PNG)

### **For Interactive Web Apps**
- Plotly Dash (recommended if you want a web interface)
- Streamlit (simpler, faster deployment)
- Jupyter + Ipywidgets (recommended for Jupyter workflow)

### **For 3D Visualization**
- Matplotlib 3D (limited, but works)
- Plotly 3D (interactive, web-friendly)
- Vispy (high-performance, real-time)
- Napari (scientific image viewer, supports 3D)

### **For Animation**
- Matplotlib.animation (MP4 output, integrates with matplotlib)
- Plotly (frame-based animation in HTML)
- OpenCV (if you need frame-by-frame control)

### **For Data Processing**
- NumPy (already in use)
- Pandas (already in use)
- PyAlData (already in use)
- Scikit-learn (dimensionality reduction if needed)

---

## V. DATA FLOW DIAGRAM

```
Raw Session Data
     ↓
[Load + Preprocess]
     ↓
BehaviorDataset (Phase 1)
     ├─→ get_trials(direction, trial_type)
     ├─→ get_aligned_kinematics()
     └─→ get_concatenated_neural()
     ↓
[Multiple Visualization Paths]
     ├─→ TimeSeriesDashboard (Tier 2, Phase 2) ← START HERE
     ├─→ TrajectoryPlotter (Tier 2.1, Phase 3)
     ├─→ SkeletonVisualizer (Tier 3, Phase 4)
     └─→ ComparativeDashboard (Tier 5, Phase 5)
     ↓
[Outputs]
     ├─→ PNG (matplotlib exports)
     ├─→ MP4 (animations)
     ├─→ HTML (Plotly interactive)
     └─→ CSV (metrics tables)
```

---

## VI. IMMEDIATE NEXT STEPS (This Week)

### **Priority 1: Build BehaviorDataset class**
This unblocks all downstream work. 
- Signature: `dataset = BehaviorDataset(session_list=['M103_2026_02_17_14_00', ...])`
- Methods: `get_trials()`, `get_aligned_kinematics()`, `get_metrics()`
- Test on perturb.ipynb data

### **Priority 2: Create TimeSeriesDashboard prototype**
- Input: BehaviorDataset, direction=4
- Output: Figure with 18 kinematics subplots + 4 neural panels
- Test: Can you easily switch direction with a slider?

### **Priority 3: Decide on scaling**
- Single-session analysis (laptop)?
- Multi-session batch processing (larger machine)?
- Live updating during collection? (requires streaming architecture)

---

## VII. LONG-TERM EXTENSIBILITY

Once Phase 1 is solid, you can easily add:
- **Machine learning**: Train models on free-running data, apply to perturbation trials
- **Statistics**: ANOVA across directions, post-hoc tests, effect sizes
- **Video integration**: Overlay 3D skeleton on filmed video
- **Real-time monitoring**: Stream during experiments
- **Multi-animal comparison**: Merge datasets across animals, align to template

---

## VIII. QUICK REFERENCE: DATA SHAPES

```python
# After extraction from perturb.ipynb
df = load_sessions(['M103_2026_02_17_14_00'])

# Kinematics (one body part)
df['left_foot'][0]          # shape: (n_frames, 3) for one trial
np.concatenate(df['left_foot'].values, axis=0)  # all frames: (total_frames, 3)

# Neural (one area)
df['MOp_rates'][0]          # shape: (n_frames, n_neurons)

# Metadata
df['trial_name']            # array of trial type labels
df['values_Sol_direction']  # array of direction indices (0-11)
df['idx_sol_on']            # array of frame indices for perturbation onset

# Query: Get direction 4 perturbation trials
dir_4_trials = pyal.select_trials(df, "values_Sol_direction == 4")
```

---

## SUMMARY TABLE

| Phase | Component | Focus | Tools | Timeline |
|-------|-----------|-------|-------|----------|
| 1 | BehaviorDataset | API for data access | Python/NumPy | 2-3 days |
| 2 | TimeSeries Dashboard | 1D exploration | Matplotlib/Ipywidgets | 3-5 days |
| 3 | Trajectory Plotter | 2D trajectories | Matplotlib/Plotly | 3-5 days |
| 4 | SkeletonVisualizer | 3D kinematics | Matplotlib 3D/Plotly | 5-7 days |
| 5 | Comparative Dashboard | Multi-condition | Plotly Dash/Streamlit | 5-7 days |

**Estimated total**: 3-4 weeks for full stack, but you can start using results after Phase 2 (~1 week).
