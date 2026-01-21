import pygame
import os
import sys
import csv
import numpy as np
import logging
import matplotlib
import h5py
import time
matplotlib.use('TkAgg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_GAIL import ShuttleEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manual_control.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ShuttleEnvManualControl:
    def __init__(self, env_config):
        self.env = ShuttleEnv(env_config)
        self.env.reset()
        self.recording = False
        self.recorded_data = []  # List to store (state, action, reward, next_state, done)

        # Ensure the base directory exists
        self.base_dir = '../data/expert_demonstration/11_17'
        os.makedirs(self.base_dir, exist_ok=True)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Shuttle Manual Control Interface')
        clock = pygame.time.Clock()
        running = True

        font = pygame.font.SysFont(None, 24)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Toggle recording
                        self.recording = not self.recording
                        status = "started" if self.recording else "stopped"
                        print(f"Recording {status}.")
                        logger.info(f"Recording {status}.")

                        if not self.recording:
                            # Save recorded data
                            self.save_recording()

                    # Map keys to shuttle actions
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0  # Move forward
                    elif event.key == pygame.K_LEFT:
                        action = 1  # Turn left
                    elif event.key == pygame.K_RIGHT:
                        action = 2  # Turn right

                    if action is not None:
                        observation, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated

                        if self.recording:
                            state = self.env.get_observation(agent_id=0)
                            next_state = observation
                            self.recorded_data.append((state, action, reward, next_state, done))
                            #logger.debug(f"Recorded state-action pair: {state}, {action}, {reward}, {next_state}, {done}")

                        if done:
                            print(f"Episode done. Saving recorded data.")
                            logger.info(f"Episode done. Saving recorded data.")
                            self.save_recording()
                            self.env.reset()

            # Render the environment including the blocked paths
            self.env.render()

            # Display recording status on Pygame window
            screen.fill((255, 255, 255))  # Fill with white
            status_text = "Recording: ON" if self.recording else "Recording: OFF"
            text_surface = font.render(status_text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()

            clock.tick(30)  # Limit to 30 FPS

        pygame.quit()


    def save_recording(self):
        """
        Save the recorded state-action-reward-next_state-done pairs to an HDF5 file.
        Each trajectory is stored as a separate group within the HDF5 file.
        """
        if not self.recorded_data:
            print("No data to save.")
            logger.warning("Attempted to save recording, but no data was recorded.")
            return

        # Append a timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(self.base_dir, f"expert_data_{timestamp}.h5")

        # Open HDF5 file for writing the datasets
        with h5py.File(file_path, 'w') as hf:
            trajectory_group = hf.create_group('trajectory_1')  # Starting with trajectory_1

            # Ensure `self.recorded_data` has the right structure
            assert all(len(record) == 5 for record in self.recorded_data), "Each record should have (state, action, reward, next_state, done)."

            # Separate out each component
            states = np.array([record[0] for record in self.recorded_data], dtype=np.float32)
            actions = np.array([record[1] for record in self.recorded_data], dtype=np.int32)
            rewards = np.array([record[2] for record in self.recorded_data], dtype=np.float32)
            next_states = np.array([record[3] for record in self.recorded_data], dtype=np.float32)
            dones = np.array([record[4] for record in self.recorded_data], dtype=bool)

            # Create datasets within the trajectory group
            trajectory_group.create_dataset('states', data=states)
            trajectory_group.create_dataset('actions', data=actions)
            trajectory_group.create_dataset('rewards', data=rewards)
            trajectory_group.create_dataset('next_states', data=next_states)
            trajectory_group.create_dataset('dones', data=dones)

        print(f"Recorded data saved to {file_path}.")
        logger.info(f"Recorded data saved to {file_path}.")
        self.recorded_data = []  # Clear after saving

def render(self):
    """Renders the current state of the environment, including shuttles, passengers, and blocked paths."""
    if self.render_mode == 'matplotlib':
        self.ax1.clear()

        # Draw the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.ax1.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='white'))

        # Draw shuttles
        for shuttle_id, shuttle in self.shuttles.items():
            self.ax1.add_patch(plt.Circle((shuttle['position'][0], shuttle['position'][1]), 0.3, color='blue'))
            self.ax1.text(shuttle['position'][0], shuttle['position'][1], f"S{shuttle_id}", color='white', ha='center', va='center')

        # Draw passengers
        for passenger in self.passengers:
            color = 'green' if passenger['picked_up'] else 'red'
            self.ax1.add_patch(plt.Circle((passenger['origin'][0], passenger['origin'][1]), 0.2, color=color))
            self.ax1.text(passenger['origin'][0], passenger['origin'][1], f"P{passenger['id']}", color='black', ha='center', va='center')

        # Highlight blocked paths
        for blocked_pos in self.blocked_positions:
            self.ax1.add_patch(patches.Rectangle(blocked_pos, 1, 1, edgecolor='black', facecolor='grey'))

        # Set grid limits and labels
        self.ax1.set_xlim(0, self.grid_size)
        self.ax1.set_ylim(0, self.grid_size)
        self.ax1.set_xticks(range(self.grid_size))
        self.ax1.set_yticks(range(self.grid_size))
        self.ax1.grid(True)

        # Update the plot
        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    env_config = {
        'grid_size': 10,
        'num_shuttles': 1,
        'render_mode': "matplotlib",
        'total_time': 500,
        'max_passengers': 6,
    }

    manual_control = ShuttleEnvManualControl(env_config)
    manual_control.run()
