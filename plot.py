

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_CONFIG = {
    "FPA": {
        'sinr1': 'ue_1_sinr_fpa.txt', 'sinr2': 'ue_2_sinr_fpa.txt',
        'power1': 'ue_1_power_fpa.txt', 'power2': 'ue_2_power_fpa.txt'
    },
    "Tabular Q-Learning": {
        'sinr1': 'ue_1_sinr_tabular.txt', 'sinr2': 'ue_2_sinr_tabular.txt',
        'power1': 'ue_1_power_tabular.txt', 'power2': 'ue_2_power_tabular.txt'
    },
    "DQN": {
        'sinr1': 'ue_1_sinr_dqn.txt', 'sinr2': 'ue_2_sinr_dqn.txt',
        'power1': 'ue_1_power_dqn.txt', 'power2': 'ue_2_power_dqn.txt'
    },
    "Optimal": {
        'sinr1': 'ue_1_sinr_optimal.txt', 'sinr2': 'ue_2_sinr_optimal.txt',
        'power1': 'ue_1_power_optimal.txt', 'power2': 'ue_2_power_optimal.txt'
    }
}

METHOD_COLORS = {
    "FPA": "red",
    "Tabular Q-Learning": "green",
    "DQN": "orange",
    "Optimal": "blue"
}

METHODS_TO_PLOT = ["FPA", "Tabular Q-Learning", "DQN", "Optimal"]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

def read_and_flatten(file_path1, file_path2):
  
    if not (os.path.exists(file_path1) and os.path.exists(file_path2)):
        return None
    
    try:
        df1 = pd.read_csv(file_path1, header=None, on_bad_lines='skip').T
        df2 = pd.read_csv(file_path2, header=None, on_bad_lines='skip').T
        combined_df = pd.concat([df1, df2], ignore_index=True).astype(float)
        flat_data = combined_df.values.flatten()
        return flat_data[~np.isnan(flat_data)]
    except Exception as e:
        print(f"  Can not read file: {e}")
        return None

def load_data_for_method(method_name):
    print(f"--- Loading data: {method_name} ---")
    
    files = METHOD_CONFIG.get(method_name)
    if not files:
        print(f"  No Configuration {method_name}")
        return None

    data = {'sinr': [], 'power': []}

    sinr_data = read_and_flatten(files['sinr1'], files['sinr2'])
    if sinr_data is not None:
        data['sinr'] = sinr_data
        print(f"  -> SINR: {len(sinr_data)} mẫu")
    else:
        print(f"  No file SINR for {method_name}")

    power_data = read_and_flatten(files['power1'], files['power2'])
    if power_data is not None:
        data['power'] = power_data
        print(f"  -> Power: {len(power_data)} mẫu")

    return data

def save_plot(fig, filename):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    path = f'figures/{filename}.png'
    fig.savefig(path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Save in path: {path}")

# --- Plotting ---
def plot_ccdf():
    fig, ax = plt.subplots(figsize=(10, 7))
    has_data = False

    for method in METHODS_TO_PLOT:
        dataset = load_data_for_method(method)
        
        if dataset is None or len(dataset['sinr']) == 0:
            continue
            
        sinr_data = dataset['sinr']
        
        # --- Read data ---
        cutoff = 16.4
        sinr_data = sinr_data[sinr_data <= cutoff]
        sinr_data = sinr_data[sinr_data >= -1]

        if len(sinr_data) > 0:
            has_data = True
            
            # Calculating CCDF
            sorted_data = np.sort(sinr_data)
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ccdf = 1 - y_vals
            
            # Configuration
            color = METHOD_COLORS.get(method, 'black') 

            # Coloring
            ax.plot(sorted_data, ccdf, label=method, linewidth=2, color=color)

    if not has_data:
        print("No file to draw.")
        return


    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('SINR (dB)')
    ax.set_ylabel('CCDF (Probability SINR > x)')
    ax.set_title('CCDF of SINR')
    ax.legend()

    save_plot(fig, 'ccdf_sinr_comparison')

# --- MAIN ---
if __name__ == "__main__":
    print("=== START ===")
    plot_ccdf()
    print("=== COMPLETE ===")