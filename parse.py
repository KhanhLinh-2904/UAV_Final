
import glob
import re
import json
import numpy as np
import pandas as pd


measurement_files = glob.glob('plots/measurements*.json')
reward_files = glob.glob('figures/optimal*.txt')
print("Found {} measurement files and {} reward files.".format(len(measurement_files), len(reward_files)))
f1 = open('ue_1_sinr.txt', 'a')
f2 = open('ue_2_sinr.txt', 'a')
f3 = open('ue_1_power.txt', 'a')
f4 = open('ue_2_power.txt', 'a')

# ---  REWARD (OPTIMAL FILES) ---
rewards = []
episodes = []


num_pattern = r"[-+]?\d*\.\d+|\d+" 

for filename in reward_files:
    with open(filename, 'r') as f:
        text = f.readline()
        
    # Input : "Episode 4512/5000 generated highest reward 1314.80."
    # findall return list: ['4512', '5000', '1314.80']
    found_numbers = re.findall(num_pattern, text)
    print("In file {}, found numbers: {}".format(filename, found_numbers))
    if len(found_numbers) >= 3:
        episodes.append(found_numbers[0]) # 
        rewards.append(found_numbers[2])  # 

# Save file convergence
if len(episodes) > 0:
    episodes = np.array(episodes).astype(int)
    rewards = np.array(rewards).astype(float)

    pd.DataFrame(data={
        'episode': episodes,
        'reward': rewards
    }).to_csv("convergence.txt", index=False, header=False)
    print(f"Processed {len(episodes)} reward files.")

# --- MEASUREMENTS (JSON FILES) ---
count_processed = 0
count_dropped = 0

for filename in measurement_files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        #  JSON structure
        # Structure: root -> "metrics" -> "sinr_ue1" (list)
        metrics = data.get('metrics', {})
        sinr1 = metrics.get('sinr_ue1', [])
        sinr2 = metrics.get('sinr_ue2', [])
        # Mapping: power_bs1 -- tx1, power_bs2 -- tx2
        tx1 = metrics.get('power_bs1', []) 
        tx2 = metrics.get('power_bs2', [])

        # check NAN 
        all_values = sinr1 + sinr2 + tx1 + tx2
        if np.isnan(np.sum(all_values)):
            count_dropped += 1
            continue

        
        f1.write(','.join(map(str, sinr1)) + ',')
        f2.write(','.join(map(str, sinr2)) + ',')
        f3.write(','.join(map(str, tx1)) + ',')
        f4.write(','.join(map(str, tx2)) + ',')
        
        count_processed += 1

    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {filename}")
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")

f1.close()
f2.close()
f3.close()
f4.close()

print(f"Measurements: Processed {count_processed}, Dropped {count_dropped} (due to NaN).")