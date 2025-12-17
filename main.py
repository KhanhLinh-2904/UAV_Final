import numpy as np
import os
from colorama import Fore, Style
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import random
import time
from environment import RadioEnvironment as radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner
from QLearningAgent import QLearningAgent as QLearner_table
MAX_EPISODES = 5000
UAV_FRAMES = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, seed=0):
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "episode": episode_index,
        "seed": seed,
        "metrics": {
            "sinr_ue1": sinr_progress,
            "sinr_ue2": sinr_ue2_progress,
            "power_bs1": serving_tx_power_progress,
            "power_bs2": interfering_tx_power_progress
        }
    }
    json_path = os.path.join(save_dir, f'measurements_ep{episode_index}_seed{seed}.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved raw data to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON data. Error: {e}")
   
    try:
        plt.figure(figsize=(10, 10)) 

        # -- Subplot 1: SINR --
        plt.subplot(2, 1, 1)
        plt.plot(sinr_progress, label='Serving SINR (UE1)', color='blue', marker='o', markersize=3, alpha=0.7)
        plt.plot(sinr_ue2_progress, label='Interfering SINR (UE2)', color='orange', linestyle='--', linewidth=2)
        plt.title(f'Episode {episode_index} - SINR Performance')
        plt.ylabel('SINR (dB)')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)

        # -- Subplot 2: Power --
        plt.subplot(2, 1, 2)
        plt.plot(serving_tx_power_progress, label='BS1 Power', color='green', linewidth=2)
        plt.plot(interfering_tx_power_progress, label='BS2 Power', color='red', linestyle='-.', linewidth=2)
        plt.title('Transmit Power Levels')
        plt.xlabel('Time Step (ms)')
        plt.ylabel('Power (dBm)') 
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()

        # -- Lưu ảnh --
        img_path = os.path.join(save_dir, f'plot_ep{episode_index}_seed{seed}.png')
        plt.savefig(img_path, dpi=150) # dpi=150 cho ảnh nét hơn
        # plt.show()
        plt.close() # Quan trọng: Đóng figure để giải phóng RAM
        
        print(f"Saved plot to: {img_path}")

    except Exception as e:
        print(f"ERROR plotting: {e}")
        # In ra stack trace nếu cần thiết để debug
        import traceback
        traceback.print_exc()


def plot_actions(actions, max_timesteps_per_episode, episode_index, seed=0):
    actions = [x.item() if hasattr(x, 'item') else x for x in actions]
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    
    plt.step(range(len(actions)), actions, where='mid', color='purple', linewidth=2)
    
    plt.title(f'Episode {episode_index} - Agent Actions')
    plt.xlabel('Time Step')
    plt.ylabel('Action Index')
    plt.grid(True, alpha=0.5)
    
    img_path = os.path.join(save_dir, f'actions_ep{episode_index}_seed{seed}.png')
    plt.savefig(img_path)
    plt.close()
    
    print(f"Saved action plot to {img_path}")

def plot_performance_function_deep(values, episode_count, is_loss=False):
    if not values or len(values) < 2:
        return
    
    values = [x.item() if hasattr(x, 'item') else x for x in values]

    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)

    if is_loss:
        title_Label = "Training Loss"
        y_label = "Loss Value"
        color_raw = 'indianred'      
        color_smooth = 'maroon'      
        file_prefix = 'loss_history'
    else:
        title_Label = "Agent Performance"
        y_label = "Value (Reward / Q-Value)"
        color_raw = 'royalblue'      
        color_smooth = 'navy'        
        file_prefix = 'performance_history'

    episodes_x = np.arange(1, len(values) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(episodes_x, values, 
             color=color_raw, 
             alpha=0.6,          
             linewidth=1.5,      
             label='Raw Data')

    window_size = max(int(len(values) * 0.05), 5)
    
    if len(values) >= window_size:
        weights = np.ones(window_size) / window_size
        smoothed_values = np.convolve(values, weights, mode='valid')
        smoothed_x = episodes_x[window_size - 1:]
        
        plt.plot(smoothed_x, smoothed_values, 
                 color=color_smooth, 
                 linewidth=3.5,  
                 label=f'Trend (Moving Avg, w={window_size})')

    plt.title(f'{title_Label} up to Episode {episode_count}', fontsize=14, fontweight='bold')
    plt.xlabel('Episode Index', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7) 
    
    plt.legend(fontsize=10, loc='best', framealpha=0.9) 
    plt.tight_layout()

    filename = os.path.join(save_dir, f'{file_prefix}_ep{episode_count}.png')
    
    plt.savefig(filename, dpi=200) 
    plt.close()
    
  
def run_agent_fpa(env, plotting=True):
    
    max_episodes = MAX_EPISODES
    max_timesteps = UAV_FRAMES
    successful_episodes_list = []
    
    best_episode_idx = -1
    best_reward = -np.inf
    print(f"{'Ep.':<5000} | {'TS':<3} | {'SINR (srv)':<15} | {'SINR (int)':<15} | {'Srv Pwr':<10} | {'Int Pwr':<10} | {'Reward':<10}")
    print('-' * 100)
    
    for episode_idx in range(1, max_episodes + 1):
        observation = env.reset()
        (_, _, _, _, pt_serving, pt_interferer) = observation
        print(f"{episode_idx}/{max_episodes} | 0   | {'-':<15} | {'-':<15} | {pt_serving:.2f} W    | {pt_interferer:.2f} W    | 0.00      |")
        
        action = -1 # constant Power
        current_episode_reward = 0
        is_aborted = False
        
        history_actions = []
        history_sinr = []
        history_sinr_ue2 = []
        history_srv_pwr = []
        history_int_pwr = []
        
        for timestep in range(1, max_timesteps +1):
            next_observation, reward, done, abort = env.step(action)
            
            current_episode_reward += reward
            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            (_, _, _, _, pt_serving, pt_interferer) = next_observation
            
            history_actions.append(action)
            history_sinr.append(received_sinr)
            history_sinr_ue2.append(received_ue2_sinr)
            history_srv_pwr.append(env.serving_transmit_power_dBm)
            history_int_pwr.append(env.interfering_transmit_power_dBm)
            
            print(f"{episode_idx}/{max_episodes} | {timestep:<3} | {received_sinr:.2f} dB       | {received_ue2_sinr:.2f} dB       | {pt_serving:.2f} W    | {pt_interferer:.2f} W    | {current_episode_reward:.2f}      | ", end='')
            
            if abort:
                print("ABORTED")
                is_aborted = True
                break
            else:
                print()
        
        is_successful = (current_episode_reward > 0) and (not is_aborted)
        if is_successful:
            print(Fore.GREEN + 'SUCCESS.' + Style.RESET_ALL)
            successful_episodes_list.append(episode_idx)
            if plotting:
                plot_measurements(history_sinr, history_sinr_ue2, history_srv_pwr, history_int_pwr, max_timesteps, episode_idx, episode_idx)
                 # plot_actions(history_actions, max_timesteps, episode_idx, episode_idx)
            
            if current_episode_reward > best_reward:
                best_reward = current_episode_reward
                best_episode_idx = episode_idx
                
            print('Successful episodes so far:', successful_episodes_list)
            optimal_msg = f'Episode {best_episode_idx}/{max_episodes} generated the highest reward {best_reward:.2f}.'
            print(Fore.CYAN + optimal_msg + Style.RESET_ALL)
            
            os.makedirs('figures', exist_ok=True)
            with open('figures/optimal_fpa.txt', 'w') as file:
                file.write(optimal_msg)

        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.' + Style.RESET_ALL)

    if not successful_episodes_list:
        print(f"Goal cannot be reached after {max_episodes} episodes. Try to increase maximum episodes.")
        
    
def run_agent_tabular(env, agent, plotting=True):
 
    max_episodes = MAX_EPISODES
    max_timesteps = UAV_FRAMES
    
    successful_episodes = [] 
    history_avg_q_values = [] 
    
    best_episode_idx = -1
    best_reward = -np.inf

    header = f"{'Ep.':<8} | {'TS':<3} | {'SINR (srv)':<12} | {'SINR (int)':<12} | {'Srv Pwr':<9} | {'Int Pwr':<9} | {'Reward':<8} | {'Epsilon':<6} | {'Action':<6}"
    print(header)
    print('-' * len(header))

    for episode_idx in range(1, max_episodes + 1):
        
        observation = env.reset()
        (_, _, _, _, pt_serving, pt_interferer) = observation

        action = agent.begin_episode(observation)

        print(f"{episode_idx:<8} | 0   | {'-':<12} | {'-':<12} | {pt_serving:.2f} W   | {pt_interferer:.2f} W   | 0.00     | {agent.exploration_rate:.2f}   | {action:<6}")

        total_reward = 0
        is_aborted = False
   
        history_actions = [action]
        history_sinr = []
        history_sinr_ue2 = []
        history_srv_pwr = []
        history_int_pwr = []
        history_step_q_values = [] 

        for timestep in range(1, max_timesteps + 1):
           
            next_observation, reward, done, abort = env.step(action)
            action = agent.act(next_observation, reward)
           
            current_q_value = agent.get_performance()
            history_step_q_values.append(current_q_value)

            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            (_, _, _, _, pt_serving, pt_interferer) = next_observation
            
            total_reward += reward
            
            print(f"{episode_idx:<8} | {timestep:<3} | {received_sinr:.2f} dB    | {received_ue2_sinr:.2f} dB    | {pt_serving:.2f} W   | {pt_interferer:.2f} W   | {total_reward:.2f}     | {agent.exploration_rate:.2f}   | {action:<6}")

            history_actions.append(action)
            history_sinr.append(received_sinr)
            history_sinr_ue2.append(received_ue2_sinr)
            history_srv_pwr.append(env.serving_transmit_power_dBm)
            history_int_pwr.append(env.interfering_transmit_power_dBm)

            if abort:
                print(Fore.YELLOW + 'ABORTED.' + Style.RESET_ALL)
                is_aborted = True
                break

        avg_q_value = np.mean(history_step_q_values) if history_step_q_values else 0
        history_avg_q_values.append(avg_q_value)

        is_successful = (total_reward > 0) and (not is_aborted)

        if is_successful:
            print(Fore.GREEN + f'SUCCESS. Total reward = {total_reward:.2f}.' + Style.RESET_ALL)
            successful_episodes.append(episode_idx)

            if total_reward > best_reward:
                best_reward = total_reward
                best_episode_idx = episode_idx
                
                optimal_msg = f'Episode {best_episode_idx}/{max_episodes} generated the highest reward {best_reward:.2f}.'
                os.makedirs('figures', exist_ok=True)
                with open('figures/optimal_tabular.txt', 'w') as file:
                    file.write(optimal_msg)
                agent.save('figures/tabular_q_table_best.pkl')
                print(Fore.CYAN + optimal_msg + Style.RESET_ALL)

            if plotting:
                plot_measurements(history_sinr, history_sinr_ue2, history_srv_pwr, history_int_pwr, max_timesteps, episode_idx, episode_idx)
                # plot_actions(history_actions, max_timesteps, episode_idx, episode_idx)
                # plot_performance_function_deep(history_avg_q_values, episode_idx, is_loss=False)

        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.' + Style.RESET_ALL)

    if not successful_episodes:
        print(f"Goal cannot be reached after {max_episodes} episodes.")
    else:
        print(f"First successful episode occurred at: {successful_episodes[0]}")


def run_agent_deep(env, agent, plotting=True):
  
    max_episodes = MAX_EPISODES
    max_timesteps = UAV_FRAMES
    batch_size = 32 
    
    successful_episodes = []
    history_losses = []
    history_q_values = []
    
    best_reward = -np.inf
    best_episode_idx = -1

    header = f"{'Ep.':<8} | {'TS':<3} | {'SINR (srv)':<10} | {'SINR (int)':<10} | {'Srv Pwr':<9} | {'Int Pwr':<9} | {'Reward':<8} | {'Loss':<6} | {'Epsilon':<6}"
    print(header)
    print('-' * len(header))

    for episode_idx in range(1, max_episodes + 1):
        
        observation_numpy = env.reset()
        observation_tensor = torch.from_numpy(observation_numpy).float().to(device)

        
        (_, _, _, _, pt_serving, pt_interferer) = observation_numpy
        
        action = agent.begin_episode(observation_tensor)
        
        print(f"{episode_idx:<8} | 0   | {'-':<10} | {'-':<10} | {pt_serving:.2f} W   | {pt_interferer:.2f} W   | 0.00     | {'-':<6} | {agent.exploration_rate:.2f}")

        total_reward = 0
        is_aborted = False
        
        episode_losses = []
        episode_qs = []
        
        history_actions = []
        history_sinr = []
        history_sinr_ue2 = []
        history_srv_pwr = []
        history_int_pwr = []

        for timestep in range(1, max_timesteps + 1):
            next_observation_numpy, reward, done, abort = env.step(action)
            next_observation_tensor = torch.from_numpy(next_observation_numpy).float().to(device)
            print("next_observation_tensor: ", next_observation_tensor)
         
            agent.remember(observation_tensor, action, reward, next_observation_tensor, done)
            
            sample_size = min(len(agent.memory), batch_size)
            loss, q = agent.replay(sample_size)
            
            if loss is not None: episode_losses.append(loss)
            if q is not None: episode_qs.append(q)

            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            (_, _, _, _, pt_serving, pt_interferer) = next_observation_numpy
            
            total_reward += reward
            
            loss_str = f"{loss:.4f}" if loss is not None else "-"
            print(f"{episode_idx:<8} | {timestep:<3} | {received_sinr:.2f} dB  | {received_ue2_sinr:.2f} dB  | {pt_serving:.2f} W   | {pt_interferer:.2f} W   | {total_reward:.2f}     | {loss_str:<6} | {agent.exploration_rate:.2f}")

            history_actions.append(action)

            sinr_val = received_sinr
            if hasattr(sinr_val, 'item'): 
                sinr_val = sinr_val.item()
            history_sinr.append(sinr_val)

            sinr_val2 = received_ue2_sinr
            if hasattr(sinr_val2, 'item'): 
                sinr_val2 = sinr_val2.item()
            history_sinr_ue2.append(sinr_val2)

            pw_srv_val = env.serving_transmit_power_dBm
            if hasattr(pw_srv_val, 'item'):
                pw_srv_val = pw_srv_val.item()
            history_srv_pwr.append(pw_srv_val)

            pw_int_val = env.interfering_transmit_power_dBm
            if hasattr(pw_int_val, 'item'):
                pw_int_val = pw_int_val.item()
            history_int_pwr.append(pw_int_val)

            if abort:
                print(Fore.YELLOW + 'ABORTED.' + Style.RESET_ALL)
                is_aborted = True
                break
            
            observation_tensor = next_observation_tensor
            observation_numpy = next_observation_numpy 
            action = agent.act(observation_tensor)

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_qs) if episode_qs else 0
        
        history_losses.append(avg_loss)
        history_q_values.append(avg_q)
        
        is_successful = (total_reward > 0) and (not is_aborted)
        if is_successful:
            print(Fore.GREEN + f'SUCCESS. Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f}' + Style.RESET_ALL)
            successful_episodes.append(episode_idx)
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode_idx = episode_idx
                
                agent.save('deep_rl_best.model')
                
                optimal_msg = f'Episode {best_episode_idx}/{max_episodes} generated highest reward {best_reward:.2f}.'
                os.makedirs('figures', exist_ok=True)
                with open('figures/optimal_deeplearning.txt', 'w') as file:
                    file.write(optimal_msg)
                print(Fore.CYAN + ">>> New Best Model Saved! <<<" + Style.RESET_ALL)

            if plotting:
                plot_measurements(history_sinr, history_sinr_ue2, history_srv_pwr, history_int_pwr, max_timesteps, episode_idx, episode_idx)
                # plot_actions(history_actions, max_timesteps, episode_idx, episode_idx)
                # plot_performance_function_deep(history_losses, episode_idx, is_loss=True)
                # plot_performance_function_deep(history_q_values, episode_idx, is_loss=False)

        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.' + Style.RESET_ALL)

    if not successful_episodes:
        print(f"Goal cannot be reached after {max_episodes} episodes.")
    else:
        print(f"Training Complete. First success at episode: {successful_episodes[0]}")
        
def run_agent_optimal(env, plotting=True):
    
    max_episodes = MAX_EPISODES
    max_timesteps = UAV_FRAMES
    
    #  (dB)
    power_changes = [-3, -1, 1, 3] 
    best_reward = -np.inf 
    best_episode_idx = -1
    best_history = None 

    print(f"{'Ep.':<5} | {'TS':<3} | {'SINR (srv)':<10} | {'SINR (int)':<10} | {'Srv Pwr':<9} | {'Int Pwr':<9}")
    print('-' * 70)

    for episode_idx in range(1, max_episodes + 1):
        observation = env.reset()
        (_, _, _, _, pt_serving, pt_interferer) = observation
        
        action = -1 
        sinr_progress = []
        sinr_ue2_progress = []
        srv_pwr_progress = []
        int_pwr_progress = []
        
        is_aborted = False
        total_reward = 0
        for timestep in range(1, max_timesteps + 1):
            next_observation, reward, done, abort = env.step(action)
            (x_ue_1, y_ue_1, x_ue_2, y_ue_2, _, _) = next_observation 
            total_reward += reward
            best_total_sinr = -np.inf
            best_sinr_ue1 = -np.inf
            best_sinr_ue2 = -np.inf
            best_pt_serving = pt_serving
            best_pt_interferer = pt_interferer

            for pc_1 in power_changes:
                cand_pt_serving = pt_serving * (10 ** (pc_1 / 10.0))
                
                for pc_2 in power_changes:
                    cand_pt_int = pt_interferer * (10 ** (pc_2 / 10.0))
                  
                    _, _, sinr_1 = env._compute_rf(x_ue_1, y_ue_1, cand_pt_serving, cand_pt_int, is_ue_2=False)
                    _, _, sinr_2 = env._compute_rf(x_ue_2, y_ue_2, cand_pt_serving, cand_pt_int, is_ue_2=True)
                    
                    current_total = sinr_1 + sinr_2
                    
                    if current_total > best_total_sinr:
                        best_total_sinr = current_total
                        best_sinr_ue1 = sinr_1
                        best_sinr_ue2 = sinr_2
                        best_pt_serving = cand_pt_serving
                        best_pt_interferer = cand_pt_int

            pt_serving = best_pt_serving
            pt_interferer = best_pt_interferer
            received_sinr = best_sinr_ue1
            received_ue2_sinr = best_sinr_ue2
            
            print(f"{episode_idx:<5} | {timestep:<3} | {received_sinr:.2f} dB  | {received_ue2_sinr:.2f} dB  | {pt_serving:.2f} W   | {pt_interferer:.2f} W")

            cond_power_max = (pt_serving > env.max_tx_power) or (pt_interferer > env.max_tx_power_interference) or\
                            (pt_serving < 0) or (pt_interferer < 0)
            cond_sinr_min = (received_sinr < env.sinr_min) or (received_ue2_sinr < env.sinr_min)
            # cond_sinr_max = (received_sinr > env.sinr_max) or (received_ue2_sinr > env.sinr_max)
            
            if cond_power_max or cond_sinr_min :
                print(Fore.YELLOW + "ABORTED: Constraints violated." + Style.RESET_ALL)
                is_aborted = True
                break

            sinr_progress.append(received_sinr)
            sinr_ue2_progress.append(received_ue2_sinr)
            srv_pwr_progress.append(10 * np.log10(pt_serving * 1e3))
            int_pwr_progress.append(10 * np.log10(pt_interferer * 1e3))

        final_check = (
            (not is_aborted) and 
            (timestep == max_timesteps)
        )

        if final_check:
            print(Fore.GREEN + 'SUCCESS.' + Style.RESET_ALL)
            

            if total_reward > best_reward:
                best_reward = total_reward
                best_episode_idx = episode_idx
                best_history = (sinr_progress, sinr_ue2_progress, srv_pwr_progress, int_pwr_progress)
            if plotting:
                plot_measurements(
                    sinr_progress, 
                    sinr_ue2_progress, 
                    srv_pwr_progress, 
                    int_pwr_progress, 
                    max_timesteps, 
                    episode_idx,    
                    episode_idx   
                )
        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.' + Style.RESET_ALL)       
    
    # if plotting and best_history is not None:
    #     print(f"Plotting best episode: {best_episode_idx} with reward: {best_reward:.2f}")
    #     h_sinr, h_sinr2, h_srv, h_int = best_history
    #     plot_measurements(h_sinr, h_sinr2, h_srv, h_int, max_timesteps, best_episode_idx, best_episode_idx)
    # elif plotting:
    #     print("No successful episodes to plot.")
        
SEEDS_TO_RUN = [98]
DEFAULT_STATE_SIZE = 6
DEFAULT_ACTION_SIZE = 16

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_env_dimensions(env):
    try:
        return env.state_size, env.num_actions
    except AttributeError:
        print(Fore.YELLOW + "Warning: env.state_size or env.num_actions not found. Using default values." + Style.RESET_ALL)
        return DEFAULT_STATE_SIZE, DEFAULT_ACTION_SIZE

def save_execution_time(duration_ms, antenna_count):
    os.makedirs('figures', exist_ok=True)
    filename = f'figures/timing_M={antenna_count}.txt'
    with open(filename, 'w') as f:
        msg = f'Execution time: {duration_ms:.4f} ms.\n'
        f.write(msg)
        print(msg)

def main():
    start_time = time.time()
    
    last_env_antenna_count = 0 

    for seed in SEEDS_TO_RUN:
        print(f"\n--- Running Experiment with Seed {seed} ---")
        set_reproducibility(seed)
        env = radio_environment(seed=seed)
        
        last_env_antenna_count = env.num_antennas 
        
        s_size, a_size = get_env_dimensions(env)

        agent = QLearner(
            state_size=s_size, 
            action_size=a_size, 
            seed=seed, 
        )
        agent_table = QLearner_table(seed=seed)
        # Select the Algorithm to Run
        
        # run_agent_fpa(env)      # Fixed Power Allocation
        run_agent_tabular(env, agent_table)  # Tabular Q-Learning (Classical RL)
        # run_agent_optimal(env)  # Exhaustive Search (Optimal Benchmark)
        # run_agent_deep(env, agent)       # Deep Q-Network (Deep RL)

    end_time = time.time()
    duration = 1000. * (end_time - start_time)
    
    save_execution_time(duration, last_env_antenna_count)

# Standard entry point for Python scripts
if __name__ == "__main__":
    main()