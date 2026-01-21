import os
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import matplotlib
import matplotlib.pyplot as plt
import sys

# Force matplotlib to use 'TkAgg' backend for visualization
matplotlib.use('TkAgg')

# Import ShuttleEnv and add the module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

# Register the environment
register(
    id='ShuttleEnv-v0',
    entry_point='env.env_GAIL:ShuttleEnv',  # Path to your environment class
)

def make_env(render_mode="human"):  # Set render mode here
    env = ShuttleEnv(env_config={}, render_mode=render_mode)
    return env

def evaluate_gail_model(saved_model_path, episodes=10):
    # Load the trained PPO model
    model = PPO.load(saved_model_path)
    
    # Create environment with the render mode enabled for visualization (e.g., 'matplotlib' or 'human')
    env = make_env(render_mode="matplotlib")
    
    # Evaluate the agent for the given number of episodes
    for episode in range(episodes):
        obs, _ = env.reset()  # Reset environment without seed
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            env.render()  # Visualize the environment
            
            # Get the action from the trained model
            action, _ = model.predict(obs, deterministic=False)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if the episode has finished
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode+1}: Total reward: {total_reward}, Steps taken: {steps}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    # Path to the saved GAIL model
    saved_model_path = '/tf/rl_project/RL_models/11_17/RL2.zip'
    
    # Visualize the trained agent for 10 episodes
    evaluate_gail_model(saved_model_path, episodes=10)
