#!/usr/bin/env python3
"""
Plot PR difference (intertrial - free0) grouped by session type
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the PR data
pr_loaded = pickle.load(open('notebooks/pr_.pkl', 'rb'))

# Load the sessions metadata
import sys
sys.path.insert(0, '/home/il620/earthquake-analysis')
import notebooks.session_summary as ss_module

# Get sessions from the notebook's kernel
# For now, hardcode them based on typical structure
sessions = {
    'control': [],
    'first': [],
    'second': [],
    'third': []
}

# Build sessions dict from pr_loaded
for session_name in pr_loaded[180]['CP_rates'].keys():
    # Determine session type based on naming pattern
    # Typical pattern: ANIMAL_DATE_TIME
    parts = session_name.split('_')
    animal = parts[0]
    
    # You might need to adjust this based on your actual naming convention
    # For now, let's just try to categorize them
    if 'control' in session_name.lower():
        sessions['control'].append(session_name)
    elif 'first' in session_name.lower():
        sessions['first'].append(session_name)
    # Add more conditions as needed
    else:
        # If unsure, put in a category (might need manual adjustment)
        sessions['control'].append(session_name)

print(f"Found sessions:")
for st, sess_list in sessions.items():
    print(f"  {st}: {len(sess_list)} sessions")

# Now generate the plots
field = "CP_rates"  
window_size_s = 180

for field in ['CP_rates', 'MOp_rates', 'SSp_rates', 'VAL_rates']:
    if field not in pr_loaded[window_size_s]:
        continue
        
    # Build dataframe with differences per session
    diff_data = []
    for session_type in ['control', 'first', 'second', 'third']:
        if session_type not in sessions:
            continue
        for sess_name in sessions[session_type]:
            if sess_name not in pr_loaded[window_size_s][field]:
                continue
            
            # Get mean PR for each condition
            free0_pr = pr_loaded[window_size_s][field][sess_name].get('free0', None)
            intertrial_pr = pr_loaded[window_size_s][field][sess_name].get('intertrial', None)
            
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
                    'free0': free0_mean,
                    'intertrial': intertrial_mean
                })

    if not diff_data:
        print(f"No data for {field}, skipping")
        continue
        
    diff_df = pd.DataFrame(diff_data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_animal = {'M062': '#1f77b4', 'M061': '#ff7f0e', 'M063': '#2ca02c', 'M078': '#d62728', 
                     'M086': '#9467bd', 'M103': '#8c564b', 'M106': '#e377c2'}
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
    bars = ax.bar(x_pos, means, bar_width, yerr=stds, capsize=5, color='lightgray', alpha=0.7, edgecolor='black', linewidth=1.5)

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
    ax.set_xticklabels(session_types)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='No difference')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'notebooks/figures/pr_diff_{field}.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot for {field}")
    plt.close()

    print(f"\nPR Difference Summary ({field}):")
    print("-" * 60)
    for st in session_types:
        st_data = diff_df[diff_df['session_type'] == st]
        if len(st_data) > 0:
            print(f"{st.upper():12} (n={len(st_data)}): {st_data['pr_diff'].mean():7.4f} ± {st_data['pr_diff'].std():7.4f}")

print("\nPlots saved to notebooks/figures/")
