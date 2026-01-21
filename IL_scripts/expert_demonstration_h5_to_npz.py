import os
import h5py
import numpy as np

def convert_h5_to_npz(h5_folder, output_file):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all .h5 files in the directory
    h5_file_paths = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith('.h5')]
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    all_dones = []

    # Loop through each h5 file and extract states, actions, rewards, next_states, and dones
    for h5_file in h5_file_paths:
        print(f"Processing file: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            # Loop through each trajectory group in the file
            for trajectory_key in f.keys():
                print(f"Processing trajectory: {trajectory_key}")
                trajectory_group = f[trajectory_key]

                # Check if 'states', 'actions', 'rewards', 'next_states', and 'dones' exist and have data
                if all(key in trajectory_group for key in ['states', 'actions', 'rewards', 'next_states', 'dones']):
                    obs = trajectory_group['states'][:]
                    act = trajectory_group['actions'][:]
                    rew = trajectory_group['rewards'][:]
                    next_obs = trajectory_group['next_states'][:]
                    done = trajectory_group['dones'][:]

                    # Print shape of extracted arrays to ensure they contain data
                    print(f"States shape: {obs.shape}")
                    print(f"Actions shape: {act.shape}")
                    print(f"Rewards shape: {rew.shape}")
                    print(f"Next States shape: {next_obs.shape}")
                    print(f"Dones shape: {done.shape}")

                    # Append to the respective lists if data exists
                    if all(arr.size > 0 for arr in [obs, act, rew, next_obs, done]):
                        all_observations.append(obs)
                        all_actions.append(act)
                        all_rewards.append(rew)
                        all_next_observations.append(next_obs)
                        all_dones.append(done)
                else:
                    print(f"Warning: One or more keys missing in {trajectory_key}")

    # Ensure data was appended to all the lists
    if not all([all_observations, all_actions, all_rewards, all_next_observations, all_dones]):
        print("No valid data found in any of the files.")
        return

    # Concatenate all data into single arrays
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    next_observations = np.concatenate(all_next_observations, axis=0)
    dones = np.concatenate(all_dones, axis=0)

    # Print the final shapes of the concatenated arrays to verify
    print(f"Final observations shape: {observations.shape}")
    print(f"Final actions shape: {actions.shape}")
    print(f"Final rewards shape: {rewards.shape}")
    print(f"Final next_observations shape: {next_observations.shape}")
    print(f"Final dones shape: {dones.shape}")

    # Save the flattened arrays as a .npz file
    np.savez(output_file, 
             observations=observations, 
             actions=actions, 
             rewards=rewards, 
             next_observations=next_observations, 
             dones=dones)
    print(f"Saved expert data to {output_file}")

# Folder containing your .h5 expert data files
h5_folder = '/home/hyun/AS_IL_GAIL_RL/data/expert_demonstration/session_10_12/'

# Output npz file
output_file = '/home/hyun/AS_IL_GAIL_RL/data/expert_demonstration/session_10_12/expert_demonstration.npz'
convert_h5_to_npz(h5_folder, output_file)
