# Phase 1, Step 2: Single Session Explorer - Design Plan

## Overview
A single, interactive notebook interface for deep exploration of one session's behavior (kinematics only, no neural data yet).

**Goal**: Visualize all kinematic data for a session in a systematic, exploratory way. Should feel like "opening a session and being able to understand it quickly."

---

## I. DATA INPUTS & PROCESSING PIPELINE

### A. Input: Single Session Name
User selects one session (e.g., "M103_2026_02_17_14_00") via dropdown or text input.

### B. Data Loading Flow
```
User selects session
    ↓
SessionRegistry.get(session_name) → SessionMetadata
    ↓
Load full session data using dt.load_sessions([session_name])
    ↓
Preprocess (handles free0/free1 renaming, trial type standardization)
    ↓
BehaviorDataset wraps the preprocessed DataFrame
    ↓
Display: Session summary + available trial types & directions
```

### C. Session Summary (Auto-Computed)
After loading, show a quick info box:
```
╔════════════════════════════════════════════╗
║ SESSION SUMMARY                            ║
╠════════════════════════════════════════════╣
║ Animal: M103                               ║
║ Condition: Control (solenoids OFF)         ║
║ Date: 2026-02-17                           ║
║ Duration: ~50 min total                    ║
║                                            ║
║ Free0: 480 frames (4.8 sec)  [480ms bins] ║
║ Trials: 50 total × 12 directions           ║
║ Intertrials: 50 × variable duration       ║
║ Free1: 600 frames (6.0 sec)                ║
║                                            ║
║ Perturbation parameters:                   ║
║  - Direction: 12 (0-11)                    ║
║  - Onset index: varies ~200 frames         ║
╚════════════════════════════════════════════╝
```

### D. Key Computations (Done Once)
For the session, pre-compute and cache:
1. **Trial grouping**: Which trials belong to which direction, trial type
2. **Alignment**: Index where perturbation onset occurs per trial
3. **Time axis**: Convert frame indices to time in seconds using Params.BIN_SIZE
4. **Concatenated trials**: For averaging (e.g., all direction 4 trials concatenated)

---

## II. USER INTERFACE & CONTROLS

### A. Control Panel (Top of notebook)
Five key selectors:

```
╔═══════════════════════════════════════════════════════════╗
║ SINGLE SESSION EXPLORER - M103_2026_02_17_14_00           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║ 1. Session:   [M103_2026_02_17_14_00 ▾]  [Reload]       ║
║                                                           ║
║ 2. Direction: [4 ←→ slide 0-11] or [All ▾]               ║
║                                                           ║
║ 3. Trial Type: [○ Trials  ○ Free0  ○ Free1  ○ Intertrial]║
║                                                           ║
║ 4. Display:   [○ Average  ○ All Trials]                  ║
║                                                           ║
║ 5. Features:  [☑ Body Parts  ☑ Trajectories  ☑ Stats]    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

### B. Control Logic (State Management)

**Session selector**:
- User picks session → reload everything
- Show session metadata in summary box

**Direction selector** (only for Trials & Intertrial):
- Slider 0-11 for perturbation direction
- "All" option to show all directions averaged
- Free0/Free1 bypass this (no direction concept)

**Trial Type selector**:
- **Trials**: Perturbation responses (aligned to idx_sol_on)
- **Free0**: Free running before perturbation block (segment 1)
- **Free1**: Free running after perturbation block (segment 2)
- **Intertrial**: Transitions between trials (no clear alignment point, just show as-is)

**Display Mode** (two options, mutually exclusive):
- **Average** (default): Mean ± SEM across all trials of that direction/type
- **All Trials**: Overlay individual trials (with transparency) + mean in bold

**Feature toggles** (show/hide panels):
- Body Parts: Toggles visibility of 6 body parts
- Trajectories: Show 2D plots or just time-series?
- Stats: Show computed statistics (peak velocity, range, etc.)

---

## III. VISUALIZATION OUTPUT: THE MAIN DISPLAY

### A. Overall Layout

**Three sections, stacked vertically:**

```
┌─────────────────────────────────────────────────────────────┐
│ KINEMATICS PANEL                                            │
│ (18 subplots: 6 body parts × 3 coordinates)                │
│ [Time-series view]                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2D TRAJECTORIES PANEL (Optional)                            │
│ X-Z plane for each body part (6 subplots)                  │
│ [Spatial view]                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STATISTICS TABLE (Optional)                                 │
│ Metrics per body part and trial type                        │
└─────────────────────────────────────────────────────────────┘
```

### B. Kinematics Panel Detail

**6 body parts** (columns): left_foot, right_foot, hip_center, shoulder_center, left_paw, right_paw

**3 coordinates** (rows per body part): X, Y, Z

**18 small time-series plots** arranged in a grid:

```
                X Coordinate         Y Coordinate         Z Coordinate
                (Forward/Back)       (Vertical)           (Left/Right)
                
Left Foot    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
             │              │    │              │    │              │
             │ [plot]       │    │ [plot]       │    │ [plot]       │
             └──────────────┘    └──────────────┘    └──────────────┘
             
Right Foot   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
             │              │    │              │    │              │
             └──────────────┘    └──────────────┘    └──────────────┘

Hip Center   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
             │              │    │              │    │              │
             └──────────────┘    └──────────────┘    └──────────────┘

[... 3 more body parts ...]

X-axis: Time (seconds), all aligned
Y-axis: Position (cm) - auto-scaled per coordinate
```

### C. Key Visual Elements in Each Time-Series Plot

**For Trials (with perturbation onset)**:
1. Line(s): Mean trajectory in color (blue for left, red for right, black for center)
2. Shaded band: ±SEM around mean (light blue/red)
3. Individual trials (optional, low opacity if "All Trials" mode)
4. Vertical line at perturbation onset (dotted, labeled "Perturb")
5. Shaded region: Perturbation period (e.g., -200ms to +1500ms, light gray background)
6. Horizontal axis label: Time (s), with t=0 at perturbation onset

**For Free0/Free1**:
- No perturbation line (continuous behavior)
- Can optionally split into 5-10 second windows for easier visual parsing

**For Intertrial**:
- Show full duration from end of one trial to start of next
- No special alignment point

### D. 2D Trajectories Panel Detail

**6 subplots** (one per body part), X-Z plane (forward/lateral):

```
Left Foot Trajectory        Right Foot Trajectory    Hip Center
(avg ± SEM)                 (avg ± SEM)              (avg ± SEM)

Z ▲                         Z ▲                      Z ▲
  │    [before perturb]       │    [before perturb]    │
  │    ───────────           │    ───────────        │
  │       ╱╲ [after]        │       ╱╲ [after]      │
  │      ╱  ╲                │      ╱  ╲             │
  └──────────→ X            └──────────→ X          └──────────→ X

Color coding:
- Black: before perturbation
- Blue/Red: during/after perturbation
- Dot: perturbation onset marker
```

**Only shown for Trials mode** (perturbation-aligned data).

### E. Statistics Table

**Rows**: Body parts (6) + computed metrics
**Columns**: X, Y, Z coordinates + summary

Example layout:
```
Body Part      │ X: Mean ± SD   │ Y: Mean ± SD   │ Z: Mean ± SD   │ Range (X,Y,Z)
───────────────┼────────────────┼────────────────┼────────────────┼──────────────
Left Foot      │ 2.1 ± 0.3 cm   │ 5.0 ± 0.2 cm   │ -1.2 ± 0.4 cm  │ (1.5, 1.0, 1.2)
Right Foot     │ -2.0 ± 0.3 cm  │ 4.8 ± 0.2 cm   │ -1.3 ± 0.4 cm  │ (1.4, 1.1, 1.3)
Hip Center     │ 0.05 ± 0.1 cm  │ 10.0 ± 0.1 cm  │ 0.1 ± 0.2 cm   │ (0.2, 0.5, 0.4)
[...]
───────────────┴────────────────┴────────────────┴────────────────┴──────────────
Perturbation:  Direction 4 | Onset: ~200ms | n=50 trials
```

---

## IV. DATA TRANSFORMATION LOGIC

### A. For Trials (Perturbation Response)

**Input**: DataFrame with selected direction, trial_type='trial'

**Steps**:
1. Filter: `pyal.select_trials(df, f"values_Sol_direction == {direction}")`
2. Get onset indices: Extract `idx_sol_on` from filtered trials
3. Concatenate kinematics: `np.concatenate(df['left_foot'].values, axis=0)` → shape (total_frames, 3)
4. Reshape by trial: `reshape(n_trials, frames_per_trial, 3)`
5. Align to onset: 
   - Pre: from (idx_sol_on - 200) to idx_sol_on
   - Post: from idx_sol_on to (idx_sol_on + 1500)
   - Pad/truncate to fixed length
6. Average: `mean_traj = traj.mean(axis=0)`, `sem_traj = sem(traj, axis=0)`
7. Convert to time: `time_axis = np.arange(mean_traj.shape[0]) * Params.BIN_SIZE - 0.2` (0.2s pre-window offset)

### B. For Free0/Free1 (Continuous Behavior)

**Input**: DataFrame with trial_type='free0' or trial_type='free1'

**Steps**:
1. Filter: `pyal.select_trials(df, "trial_name == 'free0'")`
2. Concatenate all: `concat_data = np.concatenate(df['left_foot'].values, axis=0)` → single array
3. Option A - Show raw: Plot the entire continuous sequence
4. Option B - Segment: Split into 5-second windows, average within windows
5. Time axis: `time_axis = np.arange(concat_data.shape[0]) * Params.BIN_SIZE`

### C. For Intertrial (Between Trials)

**Input**: DataFrame with trial_type='intertrial'

**Steps**:
1. Filter: `pyal.select_trials(df, "trial_name == 'intertrial'")`
2. Same as Free0/Free1 - just show raw sequence
3. Can optionally overlay trial boundaries or direction changes

---

## V. IMPLEMENTATION STEPS (No Code Yet)

### Phase 1.2a: Core Classes Needed
1. **BehaviorDataset class** 
   - Initialize with session name + preprocessed DataFrame
   - Methods:
     - `get_kinematics(direction, trial_type)` → raw arrays
     - `align_to_perturbation(arrays, idx_sol_on, pre_ms, post_ms)` → aligned arrays
     - `compute_statistics(arrays)` → mean, SEM, stats

2. **TimeSeriesPlotter class**
   - Initialize with BehaviorDataset
   - Methods:
     - `plot_kinematics_grid(direction, trial_type, mode='average')` → figure with 18 subplots
     - `plot_trajectories_2d(direction)` → 6 spatial plots
     - `get_statistics_table(direction, trial_type)` → pandas DataFrame

3. **Notebook interface** (Ipywidgets)
   - Session dropdown
   - Direction slider
   - Trial type radio buttons
   - Display mode radio buttons
   - Feature toggles
   - Update callbacks

### Phase 1.2b: Notebook Structure
Cells in order:
1. Imports + setup
2. Session selector widget
3. Session summary display (updated on selection)
4. Direction/trial type/display mode controls
5. Main visualization (kinematics panel)
6. Secondary visualization (trajectories + stats)
7. (Optional) Export section

---

## VI. DESIGN DECISIONS TO CONFIRM

### A. Time Alignment Window
For perturbation trials, how much pre/post window?
- Current plan: -200ms before to +1500ms after
- Could also be: -500ms to +2000ms (longer context)
- Or user-configurable?

### B. Handling Variable Trial Lengths
Some trials might end before 1500ms post-perturbation. Options:
- Truncate to shortest trial
- Pad with NaN (then compute SEM ignoring NaN)
- Keep separate, don't average

### C. 2D Projection
Which planes to show?
- Current plan: X-Z (forward/lateral) for body parts
- Could add: Y-Z (vertical/lateral), X-Y (forward/vertical)
- Or all three options available via toggle?

### D. Free0/Free1 Visualization
Show raw, or segment into windows?
- Raw is simpler but might be cluttered if session is long (e.g., 20 min)
- Windowed (5-10 sec windows, averaged) is cleaner for comparison
- Could offer both options

### E. Statistics to Display
Which statistics are most useful?
- Mean ± SD per coordinate
- Range of motion (min to max)
- Peak velocity
- Computed dynamically per trial type/direction/window

---

## VI. USER-CONFIRMED DESIGN DECISIONS

### A. Time Alignment Window ✓ CONFIRMED
For perturbation trials:
- **Perturbation always at t=2.0 seconds** within the 6-second trial window
- **Default visualization**: -0.2s to +1.5s (200ms pre, 1500ms post)
- **User-adjustable sliders**:
  - Pre-onset: -0.5s to -0.1s
  - Post-onset: +0.5s to +3.0s
  - Real-time plot update as sliders change

### B. Free0/Free1/Intertrial Visualization ✓ CONFIRMED
**Sliding window approach**:
- Display 5-second window of continuous behavior
- Slider bar to move window through full free period
- Window size adjustable (2s, 5s, 10s dropdown)
- **Sliding window statistics** overlay:
  - Computed over rolling 100ms bins
  - Shows: velocity magnitude, range of motion per coordinate
  - Identifies movement bouts, rest periods, gait patterns
- Updates in real-time as slider moves

### C. 2D Projection ✓ CONFIRMED
- **X-Z plane only** (forward/lateral movement)
- Show all 6 body parts as separate subplots
- **Toggle visibility** for each body part
- **Spatial centering option** (Trials only):
  - Centers X-Z center-of-mass position before averaging
  - Removes locomotor drift, reveals response kinematics
  - NOT used for Free0/Free1 (removes legitimate locomotor information)

### D. Body Parts ✓ CONFIRMED
Display all 6, with optional visibility toggles:
- left_foot, right_foot, hip_center, shoulder_center, left_paw, right_paw

### E. Starting Session ✓ CONFIRMED
- **M103_2026_02_18_15_30** (M103, Normal, Day 1)
- Good test case with perturbations and free periods

---

## VII. DATA CACHING & PERFORMANCE

To keep the notebook responsive:
- **Cache preprocessed session data** in memory after first load
- **Pre-compute aligned trials** for each direction when session is loaded
- **Lazy-load 2D trajectories & stats** (only compute when toggled on)

---

## VIII. IMPLEMENTATION READINESS

All design decisions confirmed. Ready to code:
- **BehaviorDataset class** with direction/trial_type queries and alignment
- **TimeSeriesPlotter class** with multi-panel visualization
- **Notebook interface** with Ipywidgets for interactive exploration
- **Free running analyzer** with sliding window support
- **Spatial centering** option for trials
- **Statistics computation** (mean, SD, range, peak velocity)

---

## Summary of Single Session Explorer

| Aspect | Details |
|--------|---------|
| **Input** | Single session name |
| **Primary Output** | 18 time-series plots (6 body parts × 3 coordinates) |
| **Secondary Outputs** | 6 × 2D trajectories (X-Z plane) + statistics table |
| **Trial Types** | Trials (perturbation) vs. Free0 vs. Free1 vs. Intertrial |
| **Perturbation Timing** | Always at t=2.0s (aligned to -0.2s to +1.5s by default) |
| **Time Window** | User-adjustable via sliders: -0.5s to -0.1s pre, +0.5s to +3.0s post |
| **Directions** | 0-11 slider or "All" for population view (Trials only) |
| **Display Modes** | Average ± SEM or All Trials overlay |
| **Spatial Centering** | Optional (Trials only): centers X-Z position to remove locomotor drift |
| **Free Running Window** | 5-second sliding window with adjustable size (2s/5s/10s) |
| **Free Running Stats** | Rolling averages of velocity and range of motion |
| **Body Parts** | 6 (all shown by default, individually toggleable) |
| **2D Projections** | X-Z plane (forward/lateral) with spatial trajectories |
| **Statistics** | Mean ± SD, range of motion, peak velocity per body part |
| **Interactive** | Ipywidgets sliders/dropdowns, real-time updates in Jupyter |
| **Performance** | <500ms update time for slider changes (cached data) |
| **Next Step** | Condition Comparison (same animal, control | normal | muscimol) |

---

Does this cover everything you wanted in the Single Session Explorer?
