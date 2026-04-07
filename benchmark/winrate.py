import pandas as pd
import numpy as np

# Load
df = pd.read_csv("~/work/gail_ppo_ridesharing-main/data/benchmark_episode_comparison.csv")

# -------------------------------------------------
# RL win indicators (1 = RL better, 0 = not better)
# Lower is better for these time metrics
# -------------------------------------------------

# RL vs Offline DARP
df['rl_vs_offline_wait_t_win'] = np.where(
    df[['rl_avg_wait_time', 'offline_avg_wait_time']].notna().all(axis=1),
    (df['rl_avg_wait_time'] < df['offline_avg_wait_time']).astype(int),
    np.nan
)

df['rl_vs_offline_invehicle_t_win'] = np.where(
    df[['rl_avg_in_vehicle_time', 'offline_avg_in_vehicle_time']].notna().all(axis=1),
    (df['rl_avg_in_vehicle_time'] < df['offline_avg_in_vehicle_time']).astype(int),
    np.nan
)

df['rl_vs_offline_service_t_win'] = np.where(
    df[['rl_avg_service_time', 'offline_avg_service_time']].notna().all(axis=1),
    (df['rl_avg_service_time'] < df['offline_avg_service_time']).astype(int),
    np.nan
)

# RL vs Online Nearest
df['rl_vs_nearest_wait_t_win'] = np.where(
    df[['rl_avg_wait_time', 'online_nearest_avg_wait_time']].notna().all(axis=1),
    (df['rl_avg_wait_time'] < df['online_nearest_avg_wait_time']).astype(int),
    np.nan
)

df['rl_vs_nearest_invehicle_t_win'] = np.where(
    df[['rl_avg_in_vehicle_time', 'online_nearest_avg_in_vehicle_time']].notna().all(axis=1),
    (df['rl_avg_in_vehicle_time'] < df['online_nearest_avg_in_vehicle_time']).astype(int),
    np.nan
)

df['rl_vs_nearest_service_t_win'] = np.where(
    df[['rl_avg_service_time', 'online_nearest_avg_service_time']].notna().all(axis=1),
    (df['rl_avg_service_time'] < df['online_nearest_avg_service_time']).astype(int),
    np.nan
)

# RL vs Online Wait-Time-Aware
df['rl_vs_wait_t_aware_wait_t_win'] = np.where(
    df[['rl_avg_wait_time', 'online_wait_aware_avg_wait_time']].notna().all(axis=1),
    (df['rl_avg_wait_time'] < df['online_wait_aware_avg_wait_time']).astype(int),
    np.nan
)

df['rl_vs_wait_t_aware_invehicle_t_win'] = np.where(
    df[['rl_avg_in_vehicle_time', 'online_wait_aware_avg_in_vehicle_time']].notna().all(axis=1),
    (df['rl_avg_in_vehicle_time'] < df['online_wait_aware_avg_in_vehicle_time']).astype(int),
    np.nan
)

df['rl_vs_wait_t_aware_service_t_win'] = np.where(
    df[['rl_avg_service_time', 'online_wait_aware_avg_service_time']].notna().all(axis=1),
    (df['rl_avg_service_time'] < df['online_wait_aware_avg_service_time']).astype(int),
    np.nan
)

# -------------------------------------------------
# Summary win rates
# Mean of 0/1 columns = RL win rate
# -------------------------------------------------
winrate_summary = pd.DataFrame({
    'comparison_metric': [
        'rl_vs_offline_wait_t_win',
        'rl_vs_offline_invehicle_t_win',
        'rl_vs_offline_service_t_win',
        'rl_vs_nearest_wait_t_win',
        'rl_vs_nearest_invehicle_t_win',
        'rl_vs_nearest_service_t_win',
        'rl_vs_wait_t_aware_wait_t_win',
        'rl_vs_wait_t_aware_invehicle_t_win',
        'rl_vs_wait_t_aware_service_t_win',
    ],
    'rl_win_rate': [
        df['rl_vs_offline_wait_t_win'].mean(),
        df['rl_vs_offline_invehicle_t_win'].mean(),
        df['rl_vs_offline_service_t_win'].mean(),
        df['rl_vs_nearest_wait_t_win'].mean(),
        df['rl_vs_nearest_invehicle_t_win'].mean(),
        df['rl_vs_nearest_service_t_win'].mean(),
        df['rl_vs_wait_t_aware_wait_t_win'].mean(),
        df['rl_vs_wait_t_aware_invehicle_t_win'].mean(),
        df['rl_vs_wait_t_aware_service_t_win'].mean(),
    ],
    'n_valid_episodes': [
        df['rl_vs_offline_wait_t_win'].notna().sum(),
        df['rl_vs_offline_invehicle_t_win'].notna().sum(),
        df['rl_vs_offline_service_t_win'].notna().sum(),
        df['rl_vs_nearest_wait_t_win'].notna().sum(),
        df['rl_vs_nearest_invehicle_t_win'].notna().sum(),
        df['rl_vs_nearest_service_t_win'].notna().sum(),
        df['rl_vs_wait_t_aware_wait_t_win'].notna().sum(),
        df['rl_vs_wait_t_aware_invehicle_t_win'].notna().sum(),
        df['rl_vs_wait_t_aware_service_t_win'].notna().sum(),
    ]
})

# -------------------------------------------------
# Save outputs
# -------------------------------------------------
episode_output_path = "~/work/gail_ppo_ridesharing-main/data/benchmark_episode_comparison_with_winrates.csv"
summary_output_path = "~/work/gail_ppo_ridesharing-main/data/benchmark_winrate_summary.csv"

df.to_csv(episode_output_path, index=False)
winrate_summary.to_csv(summary_output_path, index=False)

print("Saved episode-level file to:", episode_output_path)
print("Saved win-rate summary to:", summary_output_path)
print()
print(winrate_summary)