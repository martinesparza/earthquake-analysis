# CLAUDE.md — earthquake-analysis

## Jupyter notebooks
Dont run the notebooks until told explicitely

## Printing results in notebook cells
Always add print statements to notebook cells that produce figures or compute statistics. Claude cannot view images or plots directly — the only way to interpret results is through text output. Every cell with a plot or numerical finding should print the key numbers (e.g. medians, counts, p-values, percentiles) so Claude can read and interpret them from cell outputs.


## 1. The Task

Mice run freely on a **spherical treadmill** while multi-area neural activity is recorded via Neuropixels probes. During locomotion, the treadmill delivers unexpected **mechanical perturbations** (solenoid jolts) that briefly destabilise the animal's gait. Each perturbation has a **direction** (0–11, encoding 12 angles × 2 intensity levels) and an **onset time** (`idx_sol_on`).

The central scientific question is: **how do different brain areas encode, predict, and communicate about perturbation-driven movement disruption?**

Recorded areas: `MOp` (primary motor), `MOs` (secondary motor), `SSp` (somatosensory), `CP` (striatum), `VAL` (thalamus).

Direction metadata:
- `values_Sol_direction`: integer 0–11
- `Params.sol_dir_to_angle`: maps direction → angle in degrees
- `Params.sol_dir_to_contra_ipse`: maps direction → 0 (ipsilateral) or 1 (contralateral)
- `Params.sol_dir_to_level`: maps direction → intensity level (0 or 1)

---

## 2. Data Structure — pyalData Trial Table

Data uses the **pyalData** format (BeNeuroLab fork: `https://github.com/BeNeuroLab/pyaldata`, branch `matlab73`). Installed as a local editable package at `/home/me24/repos/pyaldata` — never pin a version for it.

### Core structure
The data is a `pandas.DataFrame` where **each row is one trial**. Load with:
```python
df = pyal.load_pyaldata(path)
```

### Key column conventions
| Column | Description |
|---|---|
| `trial_name` | `'trial'` (locomotion + perturbation) or `'intertrial'` (rest between trials) |
| `trial_id` | Unique integer per trial |
| `bin_size` | Duration of each time bin in seconds (raw = 0.01 s = 10 ms) |
| `<area>_spikes` | `T × N` array of raw spike counts |
| `<area>_rates` | `T × N` smoothed firing rates (added by `pyal.add_firing_rates()`) |
| `<area>_rates_pca` | `T × K` PCA-reduced rates (added by `dt.add_pca_df()`) |
| `bhv` | `T × 36` array of 3D keypoint positions (12 keypoints × xyz) |
| `idx_sol_on` | Time-bin index of perturbation onset within the trial |
| `values_Sol_direction` | Integer 0–11 encoding perturbation direction |
| `sol_level_id` | Intensity level derived from direction (0 or 1) |
| `sol_contra_ipsi` | Laterality derived from direction (0 = ipsi, 1 = contra) |
| `disturb_score` | `(12,)` array: oscillation response per keypoint (added by `kin.compute_perturbation_metric()`) |
| `disturb_mean` | Scalar mean of `disturb_score` across keypoints |


### Useful pyaldata functions
```python
pyal.select_trials(df, df.trial_name == 'trial')   # filter rows
pyal.restrict_to_interval(df, epoch_fun)            # slice time axis
pyal.generate_epoch_fun(start_point_name='idx_sol_on', rel_start=0, rel_end=50)
pyal.remove_low_firing_neurons(df, signal, 1)       # drop neurons < 1 Hz
pyal.combine_time_bins(df, n)                       # merge n bins → one
pyal.sqrt_transform_signal(df, signal)              # variance stabilisation
pyal.add_firing_rates(df, 'smooth', std=0.05)       # Gaussian-smoothed rates
```

### Standard analysis bin
Even though some code says to use 30 ms bins, we will use 10ms for most analysis unless specificied otherwise

---

## 3. Loading & Trial-Dropping Pipeline

This is the canonical pipeline implemented in `tools/dsp/preprocessing.py:load_and_process_session()` and reflected across the `notebooks/standardising/` notebooks.

### Step-by-step

```python
import pyaldata as pyal
import tools.dsp as dsp
import tools.dataTools as dt
import tools.kinematics as kin
from tools.params import Params

# 1. Load raw session
df = pyal.load_pyaldata(f'/data/raw/{session[:4]}/{session}')

# 2. Preprocess — keeps 10 ms bins (combine_time_bins=False)
#    - removes neurons firing < 1 Hz
#    - sqrt-transforms spikes
#    - adds smoothed firing rates
#    - adds sol_level_id and sol_contra_ipsi columns
df = dsp.preprocess(df, only_trials=False, combine_time_bins=False)

# 3. Fit PCA — skip row 0 (free locomotion period before any trials)
df = dt.add_pca_df(df.iloc[1:])

# 4. Compute perturbation metric
#    - builds bhv field from Params.oscillating_key_points (12 limb keypoints)
#    - computes Morlet power at dominant gait frequency per keypoint
#    - adds 'disturb_score' (T×12), 'disturb_mean', 'disturb_sum'
df = kin.compute_perturbation_metric(df, bhv_fields=Params.oscillating_key_points)

# 5. Select perturbation trials only
df_tr = pyal.select_trials(df, df.trial_name == 'trial')

# 6. Drop immobile trials
#    - pools log-speed across all trials
#    - fits Otsu threshold (parameter-free, no bimodality assumption)
#    - drops trials where speed < thresh for >= min_immobile_bins consecutive
#      samples anywhere inside pre_perturb_window (default: bins 100–200 = 1–2 s)
df_tr = kin.drop_immobile_trials(df_tr, min_immobile_bins=5)

# 7. Drop trials with NaN disturb_score
mask = df_tr['disturb_score'].apply(
    lambda x: isinstance(x, np.ndarray) and not np.any(np.isnan(x))
)
df_tr = df_tr.loc[mask]

# 8. Sort by disturbance (ascending)
disturbances = np.sum(np.stack(df_tr.disturb_score.values), axis=1)
dstrb_idx = np.argsort(disturbances)
```

All of this is wrapped in `dsp.load_and_process_session(session, data_dir)`.

### Trial selection for analysis

After loading, further restrict using `disturb_mean` + SEM to keep only well-perturbed trials:
```python
threshold = -scipy.stats.sem(df_tr['disturb_mean'])   # negative = strong perturbation
df_tr_restricted = df_tr[df_tr['disturb_mean'] < threshold]
```

### Batch loading for trial-by-trial analyses

For multi-session analyses, use `dsp.load_sessions_for_trial_analyses()`. It wraps `load_and_process_session` + `drop_unperturbed_trials` + `pyal.restrict_to_interval` for every session in a list, returning a dict keyed by session name.

```python
sessions = [
    'M061_2025_03_05_14_00',
    'M061_2025_03_06_14_00',
    'M063_2025_03_13_14_00',
    'M078_2025_08_06_15_00',
    'M103_2026_02_18_15_30',
    'M106_2026_02_25_15_00',
]

# Returns {session: {'td': perturb_td} | None}
# None means the session failed to load
all_session_processed = dsp.load_sessions_for_trial_analyses(sessions)
```

Key parameters of `load_sessions_for_trial_analyses`:
- `data_dir` — path to raw data (default `/data/bnd-data/raw/`)
- `rel_start` / `rel_end` — time window in 10 ms bins relative to `idx_sol_on` (defaults: -200 / +300 = −2 s to +3 s)
- `thresh_val` — hard lower bound on `disturb_mean` passed to `drop_unperturbed_trials` (default -2.0)

The resulting `perturb_td` stored in each `{'td': perturb_td}` is already:
- restricted to well-perturbed trials
- sliced to the perturbation window (at 10 ms bins — call `pyal.combine_time_bins(td, 3)` before neural analyses)

Iterating across sessions:
```python
for sess, val in all_session_processed.items():
    if val is None:
        continue
    td = val['td']
    # ... per-session analysis ...
```

### Key parameters (`tools/params.py`)
| Name | Value | Meaning |
|---|---|---|
| `Params.BIN_SIZE` | `0.03` s | Standard neural analysis bin |
| `Params.WINDOW_perturb` | `(0, 1.5)` s | Post-perturbation analysis window |
| `Params.WINDOW_perturb_long` | `(-1, 3)` s | Extended window (includes baseline) |
| `Params.oscillating_key_points` | 12 limb keypoints | Used for perturbation metric |
| `Params.areas` | `['MOp', 'SSp', 'CP', 'VAL']` | Standard recorded areas |

### Session naming & data paths
- Session format: `M0XX_YYYY_MM_DD_HH_MM`
- Raw data: `/data/raw/<animal_id>/<session>`
- Results (pickles, CSVs): `results/<analysis_type>/`
- Analysis notebooks: `notebooks/pipelines/`
- Per-animal exploration: `notebooks/M0XX/`

### Notebook boilerplate
Every new notebook under `notebooks/` must start with:
```python
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('../../')
```

## Running scripts
Always run Python scripts with:
```bash
poetry run python <script_path>
```
Never use `python` or `conda run` directly.
