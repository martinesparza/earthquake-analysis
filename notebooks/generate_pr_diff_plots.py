#!/usr/bin/env python3
"""
Plot PR difference (intertrial - free0) grouped by session type
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

os.chdir('/home/il620/earthquake-analysis/notebooks')

# Session categorization by animal and order
sessions_by_type = {
    'control': [
        'M062_2025_03_19_14_00',  # M062 day 1
        'M061_2025_03_04_10_00',  # M061 day 1
        'M063_2025_03_12_14_00',  # M063 day 1
        'M078_2025_08_05_15_30',  # M078 day 1
        'M086_2025_12_09_16_00',  # M086 day 1
        'M103_2026_02_17_14_00',  # M103 day 1
        'M106_2026_02_24_15_00',  # M106 day 1
    ],
    'first': [
        'M062_2025_03_20_14_00',  # M062 day 2
        'M061_2025_03_05_14_00',  # M061 day 2
        'M063_2025_03_13_14_00',  # M063 day 2
        'M078_2025_08_06_15_00',  # M078 day 2
        'M086_2025_12_10_15_00',  # M086 day 2
        'M103_2026_02_18_15_30',  # M103 day 2
        'M106_2026_02_25_15_00',  # M106 day 2
    ],
    'second': [
        'M062_2025_03_21_14_00',  # M062 day 3
        'M061_2025_03_06_14_00',  # M061 day 3
        'M063_2025_03_14_15_30',  # M063 day 3
        'M078_2025_08_07_13_30',  # M078 day 3
        'M086_2025_12_11_15_00',  # M086 day 3
        'M103_2026_02_19_15_30',  # M103 day 3
        'M106_2026_02_26_16_00',  # M106 day 3
    ],
    'third': [
        'M078_2025_08_08_10_30',  # M078 day 4 (muscimol)
        'M103_2026_02_20_16_00',  # M103 day 4 (muscimol)
        'M106_2026_02_27_16_00',  # M106 day 4 (muscimol)
    ]
}

# Plot for each brain region
fields_to_plot = ['CP_rates', 'MOp_rates', 'VAL_rates', 'SSp_rates']
window_size_s = 180
colors_animal = {'M062': '#1f77b4', 'M061': '#ff7f0e', 'M063': '#2ca02c', 'M078': '#d62728',
                 'M086': '#9467bd', 'M103': '#8c564b', 'M106': '#e377c2'}

for field in fields_to_plot:
    diff_data = []
    
    # Compute differences for each session
    for session_type in ['control', 'first', 'second', 'third']:
        for sess_name in sessions_by_type[session_type]:
            try:
                with open(f'pr_subsampled_{sess_name}_{field}.pkl', 'rb') as f:
                    pr_dict = pickle.load(f)
                
                if window_size_s not in pr_dict:
                    continue
                
                # Get PR results at max neuron count
                n_neurons_dict = pr_dict[window_size_s]
                max_neurons = max(n_neurons_dict.keys())
                
                free0_pr = pr_dict[window_size_s][max_neurons].get('free0', None)
                intertrial_pr = pr_dict[window_size_s][max_neurons].get('intertrial', None)
                
                if free0_pr is not None and intertrial_pr is not None:
                    free0_mean = free0_pr.mean()
                    intertrial_mean = intertrial_pr.mean()
                    diff = intertrial_mean - free0_mean
                    animal = sess_name.split('_')[0]
                    diff_data.append({
                        'session_type': session_type,
                        'session': sess_name,
                        'animal': animal,
                        'pr_diff': diff,
                    })
                    print(f"  {sess_name}: {diff:.4f}")
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  {sess_name}: ERROR - {e}")
                continue
    
    if not diff_data:
        print(f"No data for {field}\n")
        continue
    
    diff_df = pd.DataFrame(diff_data)
    print(f"\n{field}:")
    print(diff_df.groupby('session_type')['pr_diff'].agg(['mean', 'std', 'count']))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    session_types = ['control', 'first', 'second', 'third']
    x_pos = np.arange(len(session_types))
    bar_width = 0.6
    
    # Calculate means and stds per session type
    means = []
    stds = []
    for st in session_types:
        st_df = diff_df[diff_df['session_type'] == st]
        if len(st_df) > 0:
            means.append(st_df['pr_diff'].mean())
            stds.append(st_df['pr_diff'].std())
        else:
            means.append(0)
            stds.append(0)
    
    # Plot bars
    bars = ax.bar(x_pos, means, bar_width, yerr=stds, capsize=5, 
                  color='lightgray', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Overlay individual session points
    for i, st in enumerate(session_types):
        st_data = diff_df[diff_df['session_type'] == st]
        for _, row in st_data.iterrows():
            ax.scatter(i, row['pr_diff'], color=colors_animal.get(row['animal'], 'gray'),
                      s=120, alpha=0.8, edgecolor='black', linewidth=1.5, zorder=3)
    
    # Create legend
    legend_elements = [Patch(facecolor=colors_animal[animal], edgecolor='black', label=animal)
                       for animal in sorted(colors_animal.keys())]
    ax.legend(handles=legend_elements, title='Animal', loc='best', fontsize=11)
    ax.set_xlabel('Session Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('PR Difference (Intertrial - Free0)', fontsize=13, fontweight='bold')
    ax.set_title(f'Dimensionality Change During Intertrial Periods\n{field} (Window {window_size_s}s)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Control', 'First', 'Second', 'Third'])
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='No difference')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pr_diff_{field}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: pr_diff_{field}.png\n")

print("Done!")
