# Multi-Animal, Multi-Condition Dashboard Architecture

## Overview
You have multiple animals, each with multiple sessions under different conditions (control, normal, muscimol). This document plans dashboards that support:
- Single-session exploration
- Condition comparison (same animal)
- Cross-animal comparison (same condition)
- Population-level statistics (pooled across sessions/animals)

---

## I. METADATA & DATA HIERARCHY

### Session Naming Convention
```
M{animal_id}_{YYYY}_{MM}_{DD}_{HH}_{MM}

Examples:
M103_2026_02_17_14_00  → Animal M103, Feb 17, 2026, 2:00 PM
M106_2026_02_24_15_00  → Animal M106, Feb 24, 2026, 3:00 PM
```

### Session Metadata Schema (CSV or JSON)
```csv
session_name,animal_id,condition,date,notes,n_trials,n_perturbations
M103_2026_02_17_14_00,M103,control,2026-02-17,solenoids off,120,50
M103_2026_02_18_15_30,M103,normal,2026-02-18,standard perturbation,130,60
M103_2026_02_19_15_30,M103,normal,2026-02-19,standard perturbation,125,55
M103_2026_02_20_16_00,M103,muscimol,2026-02-20,muffled solenoid+MCx inactivated,115,45
M106_2026_02_24_15_00,M106,control,2026-02-24,solenoids off,118,48
M106_2026_02_25_15_00,M106,normal,2026-02-25,standard perturbation,128,58
...
```

### Data Hierarchy
```
All Data
├─ Animal M103
│  ├─ Condition: Control
│  │  └─ Session: M103_2026_02_17_14_00
│  │     ├─ Direction 0 → [Trial 1..N] → Frames → Kinematics/Neural
│  │     ├─ Direction 1 → [Trial 1..N]
│  │     └─ ... Direction 11
│  ├─ Condition: Normal
│  │  ├─ Session: M103_2026_02_18_15_30
│  │  └─ Session: M103_2026_02_19_15_30
│  └─ Condition: Muscimol
│     └─ Session: M103_2026_02_20_16_00
├─ Animal M106
│  ├─ Condition: Control → Condition: Normal → Condition: Muscimol
└─ ... other animals
```

### Required Enhancements to Data Pipeline

**1. SessionMetadata class**
```python
class SessionMetadata:
    session_name: str           # "M103_2026_02_17_14_00"
    animal_id: str              # "M103"
    condition: str              # "control", "normal", "muscimol"
    date: datetime
    solenoid_status: str        # "on", "off", "muffled"
    inactivation_target: str    # "MCx", "none"
    notes: str
    
    @property
    def label(self) -> str:     # For display: "M103 Control"
        return f"{self.animal_id} {self.condition.capitalize()}"
```

**2. Enhanced BehaviorDataset**
```python
class BehaviorDataset:
    sessions: List[Tuple[SessionMetadata, DataFrame]]
    
    def query(self, animal_ids=None, conditions=None, session_names=None):
        # Return filtered dataset
        return BehaviorDataset(filtered_sessions)
    
    def aggregate(self, by="condition"):
        # Return aggregated statistics grouped by condition/animal
        return AggregatedBehavior(grouped_data)
```

**3. AggregatedBehavior class**
```python
class AggregatedBehavior:
    """Holds statistics across pooled trials"""
    
    direction: int
    condition: str  # or ["control", "normal", "muscimol"]
    animal_ids: List[str]
    
    mean_trajectory: np.ndarray      # (n_frames, 3) averaged
    std_trajectory: np.ndarray       # ± SEM
    mean_neural: Dict[str, np.ndarray]  # per brain area
    metrics: Dict[str, float]        # peak_velocity, recovery_time, etc.
```

---

## II. DASHBOARD TYPES & LAYOUTS

### **Dashboard Type 1: Single Session Explorer** (Starting point)
**Purpose**: Detailed exploration of one session's kinematics & neural activity

**Inputs**:
- Session selector (dropdown)
- Direction selector (slider 0-11 or dropdown)
- Trial type toggle (trials vs. free0/free1/intertrials)
- Display mode (average ± SEM vs. all trials overlaid)

**Layout**:
```
[Session: M103_02_17 | Condition: Control] [Direction: 4 ↕] [Trial Type: ▼] [Mode: ○Avg ○Overlay]

╔════════════════════════════════════════════════════════════════╗
║ KINEMATICS: Direction 4 Perturbation Response                  ║
║ ┌─────────────┬─────────────┬──────────────┐                  ║
║ │ Left Foot   │ Right Foot  │ Hip Center   │                  ║
║ │  (X, Y, Z)  │  (X, Y, Z)  │   (X, Y, Z)  │                  ║
║ │             │             │              │                  ║
║ │ [3 subplots]│ [3 subplots]│ [3 subplots] │ Time →            ║
║ └─────────────┴─────────────┴──────────────┘                  ║
║ ┌──────────────┬──────────────┐                               ║
║ │ Left Paw     │ Right Paw    │                               ║
║ │ (X, Y, Z)    │ (X, Y, Z)    │                               ║
║ │              │              │                               ║
║ │ [3 subplots] │ [3 subplots] │                               ║
║ └──────────────┴──────────────┘                               ║
║ [|███|] Perturbation Onset __|                                ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║ NEURAL ACTIVITY: Bin Size = 10ms                               ║
║ ┌────────────────┬────────────────┐                           ║
║ │ MOp (n=243)    │ SSp (n=156)    │                           ║
║ │ [raster/time]  │ [raster/time]  │                           ║
║ └────────────────┴────────────────┘                           ║
║ ┌────────────────┬────────────────┐                           ║
║ │ CP (n=189)     │ VAL (n=103)    │                           ║
║ │ [raster/time]  │ [raster/time]  │                           ║
║ └────────────────┴────────────────┘                           ║
║ [|███|] Perturbation Onset __|                                ║
╚════════════════════════════════════════════════════════════════╝

│ Save Figure │ Export Metrics │
```

**Outputs**: PNG exports, CSV of metrics for that direction

---

### **Dashboard Type 2: Condition Comparison** (Same animal, different conditions)
**Purpose**: See how control vs. normal vs. muscimol responses differ

**Inputs**:
- Animal selector
- Direction selector (0-11)
- Metric to display (kinematics, neural, derived metrics)
- Trial type (perturbation trials only)

**Layout** (3 columns side-by-side):

```
[Animal: M103 ▼] [Direction: 4 ↕] [Show: Kinematics ▼]

╔═══════════════════════════╦══════════════════════════╦═══════════════════════════╗
║ CONTROL                   ║ NORMAL                   ║ MUSCIMOL                  ║
║ (Solenoid OFF)            ║ (Standard Perturbation)  ║ (MCx Inactivated)         ║
║                           ║                          ║                           ║
║ Left Foot X-Z Trajectory  ║ Left Foot X-Z Trajectory ║ Left Foot X-Z Trajectory  ║
║                           ║                          ║                           ║
║ [2D plot: perturb onset   ║ [2D plot]                ║ [2D plot]                 ║
║  marked, avg ± SEM]       ║                          ║                           ║
║ n=50 trials               ║ n=115 trials             ║ n=45 trials               ║
║                           ║                          ║                           ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════╣
║ Peak Velocity: 2.1 ± 0.3  ║ Peak Velocity: 1.8 ± 0.2 ║ Peak Velocity: 0.9 ± 0.4  ║
║ Recovery Time: 1.2 ± 0.1s ║ Recovery Time: 0.8 ± 0.1s║ Recovery Time: 1.8 ± 0.3s ║
║ Max Deviation: 8.2 cm     ║ Max Deviation: 6.1 cm    ║ Max Deviation: 4.5 cm     ║
╚═══════════════════════════╩══════════════════════════╩═══════════════════════════╝

│ Statistical Test │  Generate Report  │
```

**Visualization options (toggle)**:
- Kinematic trajectory (2D or 3D)
- Velocity heat map
- Peak velocity across directions (bar chart)
- Recovery time comparison (box plots)
- Neural activity heatmap

**Statistical layer**:
- ANOVA across conditions
- Significance markers on plots
- Effect size (Cohen's d)

---

### **Dashboard Type 3: Multi-Animal Comparison** (Same condition, different animals)
**Purpose**: See population-level consistency; identify individual differences

**Inputs**:
- Condition selector (control / normal / muscimol)
- Direction selector
- Display mode (individual animals vs. population mean)

**Layout**:

```
[Condition: Normal ▼] [Direction: 4 ↕] [Show: ○Individual ○Population Mean]

╔═══════════════════════════════════════════════════════════════════════════╗
║ Peak Foot Velocity by Direction - NORMAL CONDITION                       ║
║                                                                           ║
║  Velocity (cm/s)                                                          ║
║  ▲                                                                        ║
║  │     M103 (◆)    M106 (▲)    M078 (■)    Mean ± SEM (thick line)      ║
║  │      │          │           │                                         ║
║  3.0 ─ ◆─────────▲──────────■───────────────[——]                        ║
║  │     │ \      /│\        / │                                           ║
║  2.5 ─ │──◆────▲──│─▲──────■──│────────────[──]                         ║
║  │     │   \  / │  │ \    /   │                                          ║
║  2.0 ─ │────◆───│──────▲──────■─────────[──]                            ║
║  │     │        │        │                                               ║
║  1.5 ─ │        │        │    ────────────[──]                          ║
║  │                                                                        ║
║  └────┴──┴──┴──┴──┴──┴──┴──┴──┴──────────────→ Direction (0-11)        ║
║        0  2  4  6  8 10                                                   ║
║                                                                           ║
║  Legend: Shaded regions = ±SEM per animal                                ║
║  Population mean overlaid as thick black line with error bands           ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════╦══════════════════════════╦════════════════════╗
║ M103 (n=2 sessions)       ║ M106 (n=2 sessions)      ║ M078 (n=1 session) ║
║ Peak Vel: 2.1 ± 0.2 cm/s  ║ Peak Vel: 2.3 ± 0.3 cm/s║ Peak Vel: 2.0 ± -- ║
║ Recovery: 0.8 ± 0.1s      ║ Recovery: 0.9 ± 0.15s   ║ Recovery: 0.75 ± --║
│ p-values (ANOVA): (see table) ...                                         │
╚═══════════════════════════╩══════════════════════════╩════════════════════╝
```

**Display options**:
- Line plot with individual + population
- Box plots (animal × direction)
- Heatmap (animal × direction, color = metric value)
- Radar plot (multi-metric per animal)

---

### **Dashboard Type 4: Summary Statistics Dashboard** (Population level)
**Purpose**: Quick overview of all conditions & animals; identify main effects

**Inputs**:
- Metrics to display (checkboxes: peak velocity, recovery time, symmetry, etc.)
- Grouping (by animal, by condition, or cross-tabulated)
- Statistical test (ANOVA, t-test)

**Layout**:

```
[Metrics: ☑ Peak Vel ☑ Recovery ☑ Symmetry] [Grouped By: ○Animal ○Condition ○Both]

╔═════════════════════════════════════════════════════════════════════════════╗
║ PEAK FOOT VELOCITY (cm/s)                                                   ║
║ Condition × Animal                                                          ║
║                                                                             ║
║           │  M103   │  M106   │  M078   │  MEAN   │  p-value               ║
║───────────┼─────────┼─────────┼─────────┼─────────┼──────────────          ║
║ Control   │ 2.1±0.3 │ 2.0±0.4 │ 2.2±0.2 │ 2.1±0.2 │ ANOVA:                ║
║ Normal    │ 1.8±0.2 │ 2.3±0.3 │ 2.0±0.1 │ 2.0±0.2 │ F(2,33)=8.4*          ║
║ Muscimol  │ 0.9±0.4 │ 0.8±0.3 │ 1.0±0.2 │ 0.9±0.3 │ p<0.01                ║
║───────────┴─────────┴─────────┴─────────┴─────────┴──────────────          ║
║                                                                             ║
║ Effect: Muscimol significantly reduces peak velocity                       ║
║ Main effect group: F(2,33)=24.1, p<0.0001                                 ║
║ No animal main effect or interaction                                       ║
╚═════════════════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════════════════╗
║ RECOVERY TIME (s) - Same as above                                           ║
╚═════════════════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════════════════╗
║ GAIT SYMMETRY (L/R foot displacement ratio, 0=asymmetric, 1=perfect)       ║
╚═════════════════════════════════════════════════════════════════════════════╝

│ Export as CSV │ Generate Summary Report │
```

---

### **Dashboard Type 5: Trajectories & Heatmaps** (Spatial view)
**Purpose**: See where animal moves in 2D space across conditions

**Inputs**:
- Animal selector
- Condition selector
- Heatmap metric (frequency of visit, velocity at position, error)

**Layout**:

```
[Animal: M103 ▼] [Condition: Normal ▼] [Metric: ○Freq ○Velocity ○Error]

╔═════════════════════════════╦═════════════════════════════╗
║ LEFT FOOT TRAJECTORY        ║ RIGHT FOOT TRAJECTORY       ║
║ (All directions pooled)     ║ (All directions pooled)     ║
║                             ║                             ║
║  Z (vertical)               ║  Z (vertical)               ║
║  ▲                          ║  ▲                          ║
║ 20 ┼                        ║ 20 ┼                        ║
║    │     [Heat map]         ║    │     [Heat map]         ║
║ 10 ┼     [showing path      ║ 10 ┼     [showing path      ║
║    │      frequency]        ║    │      frequency]        ║
║  0 ┼                        ║  0 ┼                        ║
║    └────────────────→ X     ║    └────────────────→ X     ║
║             (horizontal)    ║             (horizontal)    ║
╚═════════════════════════════╩═════════════════════════════╝

Direction-specific:
[Direction: 4]
[Show: ○All Trials Overlay ●Avg Traj ○Heatmap]

  [Overlay of all trials, semi-transparent, with mean in bold]
```

---

## III. Dashboard Mode Selector (Top-Level Navigation)

```
╔════════════════════════════════════════════════════════════════╗
║ BEHAVIOR ANALYSIS PLATFORM                                    ║
║                                                                ║
║ [ Single Session ]  [ Condition Comp ]  [ Multi-Animal Comp ] ║
║ [ Summary Stats ]   [ Trajectories ]    [ Video Export ]      ║
╚════════════════════════════════════════════════════════════════╝
```

Each mode is a separate self-contained dashboard with its own:
- Data loading logic
- Query/filter controls
- Visualization pipeline
- Export options

---

## IV. META-FEATURES: Export & Reporting

### A. **Automatic Report Generation**
```
Generate Report: [Condition Comparison for M103]
├─ Select sessions: ☑ Control ☑ Normal ☑ Muscimol
├─ Select directions: ☑ All (0-11)
├─ Metrics: ☑ All
├─ Include statistics: ☑ ANOVA, Cohen's d
├─ Format: ○PDF ○HTML ◉Markdown
└─ [Generate]
```

Output: Markdown file with embedded figures, tables, p-values

### B. **Metric Export (CSV)**
```
session, animal, condition, direction, trial_num, 
peak_velocity, velocity_time, recovery_time, max_deviation,
limb_asymmetry, gait_frequency, ...
```

### C. **Video/Animation Export**
```
[Export as MP4]
- Resolution: 720p / 1080p
- Speed: 1x / 2x / 4x
- Overlay: ○Kinematics ○Neural ●Both
- Include timestep labels: ☑ Yes
```

---

## V. Unified Codebase Structure

```
tools/
├─ behavior/
│  ├─ __init__.py
│  ├─ dataset.py                    # BehaviorDataset, SessionMetadata
│  ├─ aggregation.py                # AggregatedBehavior, statistics
│  ├─ metrics.py                    # Compute: peak_velocity, recovery_time, etc.
│  │
│  ├─ dashboards/
│  │  ├─ __init__.py
│  │  ├─ single_session.py         # Dashboard Type 1
│  │  ├─ condition_comparison.py   # Dashboard Type 2
│  │  ├─ multi_animal.py           # Dashboard Type 3
│  │  ├─ summary_stats.py          # Dashboard Type 4
│  │  └─ trajectories.py           # Dashboard Type 5
│  │
│  └─ viz/
│     ├─ __init__.py
│     ├─ kinematics.py             # 2D/3D trajectory plots
│     ├─ neural.py                 # Raster, heatmap
│     ├─ overlays.py               # Multi-metric combined plots
│     ├─ comparisons.py            # Side-by-side condition plots
│     └─ animations.py             # Video export helpers
│
└─ metadata/
   ├─ sessions.csv                 # Session → Condition mapping
   └─ load_metadata.py             # Parse and validate
```

---

## VI. Data Flow for Each Dashboard

### **Single Session Explorer**
```
User selects session
    ↓
Load session + metadata
    ↓
User picks direction, trial type, display mode
    ↓
Query: get_trials(direction=4, trial_type="trials")
    ↓
Extract & align kinematics/neural relative to idx_sol_on
    ↓
Compute: mean, SEM per direction
    ↓
Render 18 kinem subplots + 4 neural subplots
    ↓
User can save PNG or export metrics CSV
```

### **Condition Comparison**
```
User selects animal + direction
    ↓
Load metadata: fetch all sessions for that animal
    ↓
For each condition [control, normal, muscimol]:
    └─ Query all sessions under condition
    └─ Concatenate trials across sessions
    └─ Compute aggregated statistics
    ↓
Render 3 side-by-side kinematic plots + metrics table
    ↓
Compute ANOVA/statistics
    ↓
Display results
```

### **Multi-Animal Comparison**
```
User selects condition + direction
    ↓
Load metadata: fetch all animals with that condition
    ↓
For each animal:
    └─ Concatenate all sessions for that animal + condition
    └─ Compute mean trajectory
    ↓
Render line plot with individual + population mean
    ↓
Statistical tests (ANOVA across animals)
    ↓
Display legend with n_sessions per animal
```

---

## VII. Handling Data Aggregation Challenges

### **Problem 1: Different Trial Counts Across Sessions**
**Solution**: Weighted averaging by number of trials
```python
# Instead of:
mean_traj = np.mean([session1_traj, session2_traj], axis=0)

# Do:
n_trials = [len(session1), len(session2)]
weights = np.array(n_trials) / sum(n_trials)
mean_traj = weighted_average([session1_traj, session2_traj], weights)
```

### **Problem 2: Different Trial Durations**
**Solution**: Adaptive windowing around perturbation onset
```python
# All trials aligned to idx_sol_on with consistent pre-/post- windows
pre_frames = 20   # e.g., 200ms before
post_frames = 150 # e.g., 1500ms after
aligned_traj = extract_window(trajectory, idx_sol_on - pre_frames, post_frames)
```

### **Problem 3: Different Neural Unit Counts Across Animals**
**Solution**: Dimensionality reduction for comparison
```python
# Pool all neural data within condition, fit PCA, apply to all sessions
pca = fit_pca(pooled_neural_data, n_components=10)
pc_scores = [pca.transform(session_neural) for session in sessions]
# Compare in PC space
```

OR: Use summary statistics per area (mean firing rate, variance)
```python
neural_summary = {
    "MOp_mean_fr": np.mean(mop_rates),
    "MOp_cv": np.std(mop_rates) / np.mean(mop_rates),
    ...
}
```

---

## VIII. Recommended Build Order

### **Phase 1: Core Infrastructure (Critical)**
- [ ] `SessionMetadata` class + sessions.csv
- [ ] Enhanced `BehaviorDataset` with query/filter methods
- [ ] `AggregatedBehavior` class with statistical computations
- [ ] Metric computation module (peak velocity, recovery time, etc.)

### **Phase 2a: Single-Session Dashboard (Quick win)**
- [ ] Notebook interface with sliders
- [ ] Kinematic + neural visualization
- [ ] Export functionality

### **Phase 2b: Condition Comparison (High priority)**
- [ ] 3-column side-by-side layout
- [ ] ANOVA computation and display
- [ ] Metric table generation

### **Phase 3: Multi-Animal Dashboard (Once 2a+2b work)**
- [ ] Population-level aggregation
- [ ] Line plots with individual + mean
- [ ] Group statistics

### **Phase 4: Summary Stats (Report generation)**
- [ ] Automated table generation
- [ ] ANOVA tables
- [ ] Effect size reporting

### **Phase 5: Advanced Viz (Polish)**
- [ ] Heatmap trajectories
- [ ] Video export
- [ ] Interactive web dashboard (optional, use Dash/Streamlit)

---

## IX. Example User Workflows

### **Workflow A: "How does Muscimol affect direction 4 response in M103?"**
1. Open **Condition Comparison** dashboard
2. Select Animal: M103
3. Select Direction: 4 (or 11 for most prominent)
4. See 3-column comparison: Control | Normal | Muscimol
5. Read metrics table for peak velocity, recovery time changes
6. Export figure for presentation

### **Workflow B: "Is the effect consistent across animals?"**
1. Open **Multi-Animal Comparison** dashboard
2. Select Condition: Muscimol
3. Select Direction: 4
4. See line plot of M103, M106, M078 with population mean
5. Check p-value for individual difference
6. Note which animals are outliers

### **Workflow C: "Generate a comprehensive report on control vs. muscimol"**
1. Open **Summary Stats** dashboard
2. Select Metrics: ☑ Peak Velocity, ☑ Recovery Time, ☑ Symmetry
3. Group By: Condition
4. Run ANOVA
5. Export as Markdown/PDF report with tables and statistics

### **Workflow D: "Which direction is most affected by perturbation?"**
1. Open **Condition Comparison** dashboard
2. Cycle through Direction: 0, 1, 2, ... 11
3. Compare peak velocity in Muscimol vs. Control
4. Identify direction with largest reduction
5. Look at 3D skeleton for that direction

---

## X. UI/UX Principles

### **Consistency**
- All dashboards use same color scheme (condition colors: control=green, normal=blue, muscimol=red)
- All dashboards have similar control layouts (selectors at top, visualization below)

### **Clarity**
- Always show: current session/animal/condition/direction in title
- Always show: n_trials, n_sessions used in computation
- Perturbation onset marked consistently (vertical line + shaded region)

### **Interactivity (Jupyter-first)**
- Ipywidgets sliders for quick exploration
- Plotly for hover-over information (trial count, exact values)
- Matplotlib for publication-ready figures

### **Scalability**
- Can handle 1 animal or 10 animals
- Can handle 1 session or 100 sessions
- Automatic aggregation and weighting

---

## Summary: Dashboard Comparison Table

| Dashboard | Focus | Input | Key Feature | Users |
|-----------|-------|-------|-------------|-------|
| **Single Session** | Details | 1 session | 18 kinem + 4 neural subplots | Day-to-day exploration |
| **Condition Comp** | Effect of manipulation | 1 animal, all conditions | 3-column side-by-side | Focused analysis |
| **Multi-Animal** | Generalization | All animals, 1 condition | Population mean ± SEM | Population stats |
| **Summary Stats** | High-level overview | All data | ANOVA tables, effect sizes | Reports & publications |
| **Trajectories** | Spatial behavior | Selectable | 2D heatmaps, frequency maps | Exploration new sessions |

---

## Next Steps

1. **Define sessions.csv**: Map each session to animal + condition
2. **Sketch the SessionMetadata & BehaviorDataset APIs**: Finalize what methods/properties you need
3. **Identify critical metrics**: What's most important? (peak velocity, recovery, symmetry?)
4. **Choose primary UI framework**: Jupyter+Ipywidgets (recommended) vs. Dash/Streamlit web app
5. **Start Phase 1 (infrastructure)**: Once infrastructure is solid, all dashboards follow naturally
