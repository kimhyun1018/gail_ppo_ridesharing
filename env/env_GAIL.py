import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from gymnasium import spaces
from gymnasium.spaces import Box, MultiDiscrete
import matplotlib.patches as patches
import logging
from torch.distributions import Categorical
from collections import deque
import random
import heapq
from gymnasium.utils import seeding
import os
from datetime import datetime

# Configure the logger to display debug messages
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs are output to the console
    ]
)
logger = logging.getLogger(__name__)

class ShuttleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'matplotlib']}

    def __init__(self, env_config=None, render_mode=None):
        # Default config handling
        if env_config is None:
            env_config = {}

        # Allow flexible control over the render mode
        self.render_mode = render_mode or env_config.get('render_mode', False)
        self.np_random = None  # Will be initialized in reset()

        self.grid_size = env_config.get('grid_size', 10)
        self.num_shuttles = env_config.get('num_shuttles', 1)  # Configurable number of shuttles

        # Initialize the shuttles dictionary here
        self.shuttles = {
            shuttle_id: {
                'position': (self.grid_size // 2, self.grid_size // 2),
                'orientation': 'north',
                'passengers': [],
                'has_passenger': False  # Track whether the shuttle has picked up a passenger
            } for shuttle_id in range(self.num_shuttles)
        }

        self.total_time = env_config.get('total_time', 500)
        self.max_passengers = env_config.get('max_passengers', 6)
        self.max_active_passengers = env_config.get('max_active_passengers', 10)
        self.steps = 0
        self.episode = 0
        self.grid_width = self.grid_size
        self.grid_height = self.grid_size
        self.all_passengers = []
        self.use_exploration_penalty = True  # Set to False to disable the penalty
        self.blocked_positions = set()

        self.recent_positions = {shuttle_id: [] for shuttle_id in range(self.num_shuttles)}
        self.consecutive_successes = 0
        self.level = env_config.get('initial_level', 0)
        self.episode_rewards = {shuttle_id: [] for shuttle_id in range(self.num_shuttles)}
        self.cumulative_rewards = {shuttle_id: 0 for shuttle_id in range(self.num_shuttles)}

        self.orientation_mapping = {'north': 0, 'east': 1, 'south': 2, 'west': 3}
        self.passengers = []
        self.time = 0
        self.passenger_log = []
        self.generated_passengers = 0

        # New: Track dynamically blocked paths for each shuttle
        self.blocked_paths = {shuttle_id: [] for shuttle_id in range(self.num_shuttles)}

        # Updated Observation Space Dimensions
        # 1 for orientation to nearest passenger
        # 1 for orientation to drop-off (if a passenger is onboard)
        # 1 for distance to nearest passenger
        # 1 for distance to nearest drop-off (if a passenger is onboard)
        # 2 for current shuttle position (x, y)
        # 1 for current shuttle orientation
        # 2 for next passenger position (x, y)
        # 2 for next drop-off position (x, y)
        observation_dim = 1 + 1 + 1 + 1 + 2 + 1 + 2 + 2  # Total: 11 features

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32  # Normalized to [0, 1]
        )
        logger.info(f"Updated observation space shape: {self.observation_space.shape}")

        # Only initialize matplotlib when render mode is set to 'matplotlib'
        if self.render_mode == 'matplotlib':
            self._init_matplotlib()

        self.action_space = spaces.Discrete(3)


    def _init_matplotlib(self):
        """Initialize matplotlib plotting when in render mode 'matplotlib'."""
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 16))
        self.fig.suptitle('Shuttle Environment', fontsize=16)
        plt.show(block=False)

    def reset(self, *, seed=None, options=None):
        # Print debug info
        print(f"Reset called with seed={seed} and options={options}")
        
        self.np_random, seed = seeding.np_random(seed)
        
        # Call super().reset() if necessary (for compatibility)
        # Ensure it doesn't break anything in Gymnasium's API.
        
        # Reset internal episode parameters
        self.episode += 1
        self.time = 0
        self.cumulative_rewards = {shuttle_id: 0 for shuttle_id in range(self.num_shuttles)}  # Reset for all shuttles
        
        # Reset shuttle positions, orientations, and clear logs
        initial_shuttle_positions = [(self.grid_size // 2, self.grid_size // 2) for _ in range(self.num_shuttles)]
        initial_orientations = ['north' if shuttle_id % 2 == 0 else 'south' for shuttle_id in range(self.num_shuttles)]

        for shuttle_id in range(self.num_shuttles):
            # Ensure that self.shuttles is already initialized correctly in __init__
            self.shuttles[shuttle_id]['position'] = initial_shuttle_positions[shuttle_id]
            self.shuttles[shuttle_id]['orientation'] = initial_orientations[shuttle_id]
            self.shuttles[shuttle_id]['passengers'] = []

        self.all_passengers = []
        self.passengers = []
        self.generated_passengers = 0
        self.passenger_log = []
        
        # Generate initial passengers
        self.passengers = self.generate_passengers()
        
        # If rendering mode is 'matplotlib', plot the grid
        if self.render_mode == 'matplotlib':
            self.plot_grid()
        
        # Dynamically create observations for each shuttle
        observation = self.get_observation(agent_id=0)


        return observation, {}

    def step(self, action):
        self.time += 1
        shuttle_id = 0
        shuttle = self.shuttles[shuttle_id]
        current_position = shuttle['position']
        current_orientation = shuttle['orientation']
        previous_position = current_position

        #print(f"Action taken: {action}")

        # Log the action and current status before movement
        #logger.debug(f"Shuttle {shuttle_id} - Current Pos: {current_position}, Orientation: {current_orientation}, Action: {action}")

        if action == 0:  # Move forward
            new_position = self.move_forward(current_position, current_orientation, shuttle_id)
            shuttle['position'] = new_position
        elif action == 1:  # Turn left and move forward
            new_orientation = self.turn_left(current_orientation, shuttle_id)
            shuttle['orientation'] = new_orientation
            new_position = self.move_forward(current_position, new_orientation, shuttle_id)
            shuttle['position'] = new_position
        elif action == 2:  # Turn right and move forward
            new_orientation = self.turn_right(current_orientation, shuttle_id)
            shuttle['orientation'] = new_orientation
            new_position = self.move_forward(current_position, new_orientation, shuttle_id)
            shuttle['position'] = new_position

        # Log the new position after movement
        #logger.debug(f"Shuttle {shuttle_id} - New Pos: {shuttle['position']}, New Orientation: {shuttle['orientation']}")

        # Update blocked paths and handle other steps
        self.update_blocked_paths(shuttle_id, shuttle['position'])

        pickups = self.handle_pickup(shuttle_id, shuttle['position'])
        dropoffs = self.handle_dropoff(shuttle_id, shuttle['position'])

        # Update environment state after action
        self.track_recent_positions()
        self.update_passenger_times()

        # Generate new passengers if needed
        if len(self.all_passengers) < 6:
            steps_to_add_passenger = 10
            if self.time >= steps_to_add_passenger and (self.time - steps_to_add_passenger) % 10 == 0:
                new_passenger = self.generate_passenger()
                if new_passenger:
                    self.passengers.append(new_passenger)

        # Calculate reward based on movements, pickups, and dropoffs
        reward = self.calculate_total_reward({shuttle_id: previous_position}, {shuttle_id: pickups}, {shuttle_id: dropoffs}, shuttle_id)

        # Initialize step_reward to the calculated reward
        step_reward = reward

        # Check if the episode is done (terminated)
        terminated = self.check_if_done()

        # Add final rewards if episode is terminating
        if terminated:
            # Add early completion bonus if applicable
            if len(self.passengers) == 0 and all(len(s['passengers']) == 0 for s in self.shuttles.values()):
                early_completion_bonus = self.calculate_early_completion_bonus()
                step_reward += early_completion_bonus  # Fixed: step_reward is defined

            # Log final episode reward
            logger.info(f"Episode ended with total reward: {self.cumulative_rewards[shuttle_id]}")

            # Automatically log statistics to file
            self.log_passenger_statistics(save_logs=True)


        # Get observation
        observation = self.get_observation(agent_id=shuttle_id)

        # Assume no truncation by default
        truncated = False

        # Get the normalized observation for the current state
        observation = self.get_observation(agent_id=shuttle_id)
        #print(f"Step {self.time} - Observation: {observation}")


        # Info dictionary (can be empty or include additional info)
        info = {}

        # Return the observation, reward, termination flag, truncation flag, and info
        return observation, reward, terminated, truncated, info

    def update_blocked_paths(self, shuttle_id, current_position):
        """
        Dynamically calculate and block paths that are not part of any valid shortest paths
        to the nearest passenger or nearest drop-off (as defined in the observation space).
        """
        shuttle = self.shuttles[shuttle_id]

        # List to store valid paths
        valid_paths = []

        # Include paths to the nearest passenger (if any passengers exist)
        if self.passengers:
            nearest_passenger = min(
                self.passengers,
                key=lambda p: self.calculate_passenger_cost(current_position, p)
            )
            passenger_paths = self.calculate_all_shortest_paths(current_position, nearest_passenger['origin'])
            valid_paths.extend(passenger_paths)

        # Include paths to the nearest drop-off (if any onboard passengers exist)
        if shuttle['passengers']:
            nearest_dropoff = min(
                shuttle['passengers'],
                key=lambda p: self.calculate_dropoff_cost(current_position, p)
            )
            dropoff_paths = self.calculate_all_shortest_paths(current_position, nearest_dropoff['destination'])
            valid_paths.extend(dropoff_paths)

        # Block all positions not part of any valid shortest paths
        self.block_unnecessary_paths(current_position, valid_paths)



    def block_unnecessary_paths(self, current_position, valid_paths):
        """
        Allow movement along any valid path and block all other positions.
        """
        self.blocked_positions = set()  # Reset blocked positions
        grid_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # Gather all valid positions from the shortest paths
        valid_positions = set(pos for path in valid_paths for pos in path)

        # Block any position not in valid positions
        for pos in grid_positions:
            if pos not in valid_positions:
                self.blocked_positions.add(pos)


    def valid_paths_for_shuttle(self, shuttle_id):
        """
        Return a set of valid positions for the shuttle to move to, based on the nearest
        passenger and nearest drop-off (as defined in the observation space).
        """
        shuttle = self.shuttles[shuttle_id]
        current_position = shuttle['position']
        valid_positions = set()

        # Add valid paths to the nearest passenger (if any passengers exist)
        if self.passengers:
            nearest_passenger = min(
                self.passengers,
                key=lambda p: self.calculate_passenger_cost(current_position, p)
            )
            passenger_paths = self.calculate_all_shortest_paths(current_position, nearest_passenger['origin'])
            for path in passenger_paths:
                valid_positions.update(path)

        # Add valid paths to the nearest drop-off (if any onboard passengers exist)
        if shuttle['passengers']:
            nearest_dropoff = min(
                shuttle['passengers'],
                key=lambda p: self.calculate_dropoff_cost(current_position, p)
            )
            dropoff_paths = self.calculate_all_shortest_paths(current_position, nearest_dropoff['destination'])
            for path in dropoff_paths:
                valid_positions.update(path)

        return valid_positions


    def calculate_all_shortest_paths(self, start_position, end_position):
        """
        Calculate all shortest paths between two points (start_position and end_position) on a grid.
        This function uses a breadth-first search (BFS) algorithm to find all possible shortest paths,
        treating horizontal and vertical movements equally.
        """
        x1, y1 = start_position
        x2, y2 = end_position

        # If the start and end positions are the same, return an empty path
        if start_position == end_position:
            return [[]]

        # Define the possible movements: up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (dx, dy): up, down, right, left
        
        # Use BFS to find all shortest paths
        queue = deque()
        queue.append((x1, y1, []))  # Queue contains tuples of (current_x, current_y, current_path)
        
        visited = {}
        shortest_paths = []
        shortest_path_length = None

        while queue:
            current_x, current_y, current_path = queue.popleft()
            current_position = (current_x, current_y)
            path_length = len(current_path)

            # If we have found the shortest path length and current path is longer, skip
            if shortest_path_length is not None and path_length > shortest_path_length:
                continue

            # If we reached the end position
            if current_position == end_position:
                if shortest_path_length is None:
                    shortest_path_length = path_length
                if path_length == shortest_path_length:
                    shortest_paths.append(current_path + [current_position])
                continue  # Continue to find other paths of the same length

            # Mark the current position as visited with the path length
            if current_position in visited:
                if visited[current_position] < path_length:
                    continue  # We've already found a shorter path to this position
                elif visited[current_position] == path_length:
                    pass  # Allow paths of the same length
            visited[current_position] = path_length

            # Explore all possible directions
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                new_position = (new_x, new_y)

                # Check if the new position is within bounds
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    if new_position not in visited or visited[new_position] >= path_length + 1:
                        queue.append((new_x, new_y, current_path + [current_position]))

        return shortest_paths



    def check_level_up(self, success_rate, current_level):
        """
        Only advance to the next level if the agent has passed the success rate
        threshold 3 times in a row.
        """
        success_threshold = 0.8

        if success_rate >= success_threshold:
            self.consecutive_successes += 1
        else:
            # Reset the counter if the agent fails to meet the success rate
            self.consecutive_successes = 0

        if self.consecutive_successes >= 1:  # Adjust this threshold as needed
            self.consecutive_successes = 0  # Reset for the next level
            return current_level + 1
        else:
            return current_level

    def advance_level(self):
        """
        Advance to the next level by increasing difficulty or resetting with new parameters.
        """
        self.level += 1
        logger.info(f"Advancing to Level {self.level}.")

        # Adjust environment parameters based on the new level
        self.adjust_difficulty(self.level)

        # Reset environment for the new level
        self.reset()

        # Notify any listeners about the level advancement
        if hasattr(self, 'level_change_callback') and callable(self.level_change_callback):
            self.level_change_callback(self.level)

    def get_zone(self, position):
        x, y = position
        return y * self.grid_size + x + 1

    def calculate_average_times(self):
        if not self.passengers:
            return 0.0, 0.0, 0.0  # Return zeros if no passengers

        total_wait_time = 0
        total_travel_time = 0
        total_elapsed_time = 0
        count = len(self.passengers)

        for passenger in self.passengers:
            total_wait_time += passenger['elapsed_wait_time']
            total_travel_time += passenger['elapsed_travel_time']
            total_elapsed_time += passenger['elapsed_time']

        avg_wait_time = total_wait_time / count
        avg_travel_time = total_travel_time / count
        avg_elapsed_time = total_elapsed_time / count

        return avg_wait_time, avg_travel_time, avg_elapsed_time

    def calculate_average_distances(self):
        avg_distances = np.zeros(self.num_shuttles)

        for shuttle_id in range(self.num_shuttles):
            total_distance = 0
            for passenger in self.passengers:
                total_distance += self.calculate_travel_time(self.shuttles[shuttle_id]['position'], passenger['origin'])

            avg_distances[shuttle_id] = total_distance / len(self.passengers) if self.passengers else 0

        return avg_distances
        
    def move_forward(self, position, orientation, shuttle_id):
        x, y = position

        # Determine the next position based on the current orientation
        if orientation == 'north':
            new_position = (x, y + 1)
        elif orientation == 'east':
            new_position = (x + 1, y)
        elif orientation == 'south':
            new_position = (x, y - 1)
        elif orientation == 'west':
            new_position = (x - 1, y)

        # Ensure the move is valid (not out of bounds or blocked)
        if (0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size) and \
        (new_position not in self.blocked_positions):
        #    logger.debug(f"Shuttle {shuttle_id} moved to {new_position}")
            return new_position  # Move is valid, continue to new position

        # If the move is invalid or blocked, stay in the current position
        #logger.debug(f"Shuttle {shuttle_id} cannot move to {new_position}. Staying at {position}.")
        return position

    def turn_left(self, orientation, shuttle_id):
        orientations = ['north', 'west', 'south', 'east']
        idx = orientations.index(orientation)
        new_orientation = orientations[(idx + 1) % 4]
        return new_orientation

    def turn_right(self, orientation, shuttle_id):
        orientations = ['north', 'east', 'south', 'west']
        idx = orientations.index(orientation)
        new_orientation = orientations[(idx + 1) % 4]
        return new_orientation

    def get_next_shortest_path_position(self, shuttle_id):
        """
        Get the next position the shuttle should move towards based on the shortest path.
        This is either towards the next passenger's pickup location or the drop-off location.
        """
        shuttle = self.shuttles[shuttle_id]

        if shuttle['passengers']:
            # If there are passengers onboard, move towards the nearest drop-off location
            passenger = shuttle['passengers'][0]
            destination = passenger['destination']
            shortest_paths = self.calculate_all_shortest_paths(shuttle['position'], destination)
        else:
            # Move towards the nearest pickup location if no passengers are onboard
            nearest_passenger = min(self.passengers, key=lambda p: self.calculate_travel_time(shuttle['position'], p['origin']))
            shortest_paths = self.calculate_all_shortest_paths(shuttle['position'], nearest_passenger['origin'])

        # Check if shortest_paths is not empty before accessing the first element
        if shortest_paths and shortest_paths[0]:
            return shortest_paths[0][0]  # Return the next position in the shortest path
        else:
            # Handle the edge case where no valid path is found
            print(f"Warning: No valid shortest path found for shuttle {shuttle_id}")
            return shuttle['position']  # Stay in the same position as fallback


    def is_orientation_towards_target(self, current_position, target_position, orientation):
        """
        Check if the current orientation moves the shuttle closer to the target position.
        """
        x, y = current_position
        target_x, target_y = target_position

        if orientation == 'north' and y < target_y:
            return True
        elif orientation == 'south' and y > target_y:
            return True
        elif orientation == 'east' and x < target_x:
            return True
        elif orientation == 'west' and x > target_x:
            return True
        return False


    def track_recent_positions(self):
        """
        Track the recent positions of shuttles to detect repeated visits.
        We'll store a list of recent positions for each shuttle, and limit the length.
        """
        max_recent_steps = 10  # Limit the history to the last 10 steps
        
        for shuttle_id, shuttle in self.shuttles.items():
            # Update recent positions for each shuttle
            if len(self.recent_positions[shuttle_id]) >= max_recent_steps:
                self.recent_positions[shuttle_id].pop(0)  # Remove the oldest position

            # Add the current position to the list of recent positions
            self.recent_positions[shuttle_id].append(shuttle['position'])
        
        return self.recent_positions
    
    def check_if_done(self):
        """
        Check if the episode is done.
        
        The episode is done if:
        - The maximum time limit (500) is reached, or
        - All passengers have been served (picked up and dropped off)
        """
        # Check if the total time limit has been reached
        time_limit_reached = self.time >= self.total_time

        # Check if all passengers have been served
        all_passengers_served = len(self.passengers) == 0 and all(len(shuttle['passengers']) == 0 for shuttle in self.shuttles.values())

        return time_limit_reached or all_passengers_served

    
    def update_passenger_times(self):
        """
        Update the elapsed times for all passengers.
        This method increases the elapsed time for each passenger based on how long they have been waiting or traveling.
        """
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                passenger['elapsed_wait_time'] = self.time - passenger['request_time']
                passenger['elapsed_time'] = passenger['elapsed_wait_time']
            elif passenger['status'] == 'onboard':
                passenger['elapsed_travel_time'] = self.time - passenger['pickup_time']
                passenger['elapsed_time'] = passenger['elapsed_wait_time'] + passenger['elapsed_travel_time']

        # Update times for passengers onboard shuttles
        for shuttle in self.shuttles.values():
            for passenger in shuttle['passengers']:
                passenger['elapsed_travel_time'] = self.time - passenger['pickup_time']
                passenger['elapsed_time'] = passenger['elapsed_wait_time'] + passenger['elapsed_travel_time']


    
    def handle_pickup(self, shuttle_id, current_position):
        shuttle = self.shuttles[shuttle_id]
        passengers_picked = []

        for passenger in self.passengers.copy():  # Use copy to avoid modification during iteration
            if not passenger['picked_up'] and passenger['origin'] == current_position:
                if len(shuttle['passengers']) < self.max_passengers:
                    shuttle['passengers'].append(passenger)
                    passenger['picked_up'] = True
                    passenger['status'] = 'onboard'
                    passenger['pickup_time'] = self.time
                    passenger['shuttle'] = shuttle_id
                    passengers_picked.append(passenger)
                    #logger.info(f"Shuttle {shuttle_id} picked up Passenger {passenger['name']} at {current_position}")
                else:
                    logger.warning(f"Shuttle {shuttle_id} is full. Cannot pick up Passenger {passenger['name']}.")

        # Remove picked up passengers from the general waiting list
        for passenger in passengers_picked:
            self.passengers.remove(passenger)

        return passengers_picked  # Return the list of passengers picked up this step

    
    def handle_dropoff(self, shuttle_id, current_position):
        shuttle = self.shuttles[shuttle_id]
        passengers_dropped = []

        for passenger in shuttle['passengers'].copy():  # Use copy to avoid modification during iteration
            if passenger['destination'] == current_position:
                shuttle['passengers'].remove(passenger)
                passenger['dropoff_time'] = self.time
                passenger['status'] = 'served'

                # Log passenger statistics
                self.passenger_log.append({
                    'name': passenger['name'],
                    'shuttle_id': shuttle_id,
                    'wait_time': passenger['pickup_time'] - passenger['request_time'],
                    'travel_time': passenger['dropoff_time'] - passenger['pickup_time'],
                    'total_time': passenger['dropoff_time'] - passenger['request_time'],
                    'pickup_time': passenger['pickup_time'],
                    'dropoff_time': passenger['dropoff_time'],
                    'mwt': self.calculate_mwt(passenger['origin'], passenger['destination']),
                    'request_time': passenger['request_time'],
                    'matd': passenger['matd'],
                    'origin': passenger['origin'],
                    'destination': passenger['destination'],
                    'status': passenger['status']
                })

                passengers_dropped.append(passenger)
                logger.info(f"Shuttle {shuttle_id} dropped off Passenger {passenger['name']} at {current_position}")

        # Optionally, remove served passengers from general passenger list
        self.passengers = [p for p in self.passengers if p not in passengers_dropped]

        return passengers_dropped  # Return the list of passengers dropped off this step


    def calculate_travel_time(self, start_pos, end_pos):
        """
        Calculate the Manhattan distance between two points on the grid,
        which can be used as a proxy for travel time.
        """
        return abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])
    
    
    def get_observation(self, agent_id):
        """
        Generate the observation for the given agent, including information about
        the nearest passenger and the nearest drop-off (if applicable), using custom cost functions.
        """
        shuttle = self.shuttles[agent_id]
        current_position = shuttle['position']
        current_orientation = shuttle['orientation']
        grid_max_distance = (2 * self.grid_size) - 2  # Maximum Manhattan distance on the grid

        # Orientation and distance to the nearest passenger (using passenger cost)
        if self.passengers:
            nearest_passenger = min(
                self.passengers,
                key=lambda p: self.calculate_passenger_cost(current_position, p)
            )
            orientation_toward_passenger = int(
                self.is_orientation_towards_target(current_position, nearest_passenger['origin'], current_orientation)
            )
            distance_to_passenger = self.calculate_travel_time(current_position, nearest_passenger['origin']) / grid_max_distance  # Normalized
            next_passenger_position = np.array(nearest_passenger['origin']) / self.grid_size  # Normalized position
        else:
            orientation_toward_passenger = 0
            distance_to_passenger = 0
            next_passenger_position = np.array(current_position) / self.grid_size

        # Orientation and distance to the nearest drop-off (using drop-off cost)
        if shuttle['passengers']:
            nearest_dropoff = min(
                shuttle['passengers'],
                key=lambda p: self.calculate_dropoff_cost(current_position, p)
            )
            orientation_toward_dropoff = int(
                self.is_orientation_towards_target(current_position, nearest_dropoff['destination'], current_orientation)
            )
            distance_to_dropoff = self.calculate_travel_time(current_position, nearest_dropoff['destination']) / grid_max_distance  # Normalized
            next_dropoff_position = np.array(nearest_dropoff['destination']) / self.grid_size  # Normalized position
        else:
            orientation_toward_dropoff = 0
            distance_to_dropoff = 0
            next_dropoff_position = np.array(current_position) / self.grid_size

        # Shuttle's current position (normalized by grid size)
        position = np.array(current_position) / self.grid_size

        # Shuttle's current orientation (normalized by total orientations)
        orientation = np.array([self.orientation_mapping[current_orientation] / (len(self.orientation_mapping) - 1)])

        # Construct the observation array
        observation = np.concatenate([
            [orientation_toward_passenger],  # Orientation towards the nearest passenger
            [orientation_toward_dropoff],    # Orientation towards the nearest drop-off
            [distance_to_passenger],         # Normalized distance to the nearest passenger
            [distance_to_dropoff],           # Normalized distance to the nearest drop-off
            position,                        # Normalized shuttle position
            orientation,                     # Normalized shuttle orientation
            next_passenger_position,         # Normalized position of the nearest passenger
            next_dropoff_position            # Normalized position of the nearest drop-off
        ])

        return observation

    def calculate_passenger_cost(self, current_position, passenger):
        """
        Calculate the cost for a passenger based on:
        - Distance from the shuttle to the passenger's origin.
        - Elapsed wait time (higher weight for longer waits).
        Lower cost indicates higher priority for the passenger.
        """
        alpha = 1.0  # Weight for elapsed wait time
        distance = self.calculate_travel_time(current_position, passenger['origin'])
        wait_time = passenger['elapsed_wait_time']
        # Combine distance and wait time into the cost function
        return distance - alpha * wait_time

    def calculate_dropoff_cost(self, current_position, passenger):
        """
        Calculate the cost for a drop-off based on:
        - Distance from the shuttle to the passenger's destination.
        - Elapsed travel time (higher weight for longer travel times).
        Lower cost indicates higher priority for the drop-off.
        """
        beta = 1.0  # Weight for elapsed travel time
        distance = self.calculate_travel_time(current_position, passenger['destination'])
        travel_time = passenger['elapsed_travel_time']
        # Combine distance and travel time into the cost function
        return distance + beta * travel_time



        
    def calculate_agent_specific_times(self, agent_id):
        """
        Calculate the average wait time, travel time, and elapsed time for a specific agent (shuttle).
        Only calculates the times based on the agent's own transported passengers.
        """
        if len(self.shuttles[agent_id]['passengers']) == 0:
            return 0.0, 0.0, 0.0  # Return zeros if no passengers have been transported by this agent

        total_wait_time = 0
        total_travel_time = 0
        total_elapsed_time = 0
        count = len(self.shuttles[agent_id]['passengers'])  # Only consider passengers onboard this agent

        for passenger in self.shuttles[agent_id]['passengers']:
            total_wait_time += passenger['elapsed_wait_time']
            total_travel_time += passenger['elapsed_travel_time']
            total_elapsed_time += passenger['elapsed_time']

        avg_wait_time = total_wait_time / count
        avg_travel_time = total_travel_time / count
        avg_elapsed_time = total_elapsed_time / count

        return avg_wait_time, avg_travel_time, avg_elapsed_time

    def calculate_average_distance_to_passengers(self, agent_id):
        """
        Calculate the average distance between the agent (shuttle) and the passengers.
        This information should only be available to the agent observing it.
        """
        total_distance = 0
        for passenger in self.passengers:
            total_distance += self.calculate_manhattan_distance(self.shuttles[agent_id]['position'], passenger['origin'])

        if len(self.passengers) == 0:
            return 0  # Avoid division by zero

        return total_distance / len(self.passengers)

    def calculate_manhattan_distance(self, pos1, pos2):
        """
        Calculate the Manhattan distance between two points.
        pos1 and pos2 should be tuples representing (x, y) coordinates.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])



    def calculate_total_reward(self, previous_positions, pickups, dropoffs, shuttle_id):
        # Define constants
        VICINITY_RADIUS = 5  # Define the vicinity radius for additional rewards
        VICINITY_TIMEOUT = 4  # Time threshold after which no vicinity rewards are given
        TIMEOUT_RESET_LAG = 3  # Steps shuttle needs to be away from vicinity before reset
        BASE_REWARDS = {
            'pickup': .20,    # Base reward for successful pickup
            'dropoff': .15    # Base reward for successful dropoff
        }
        VICINITY_REWARDS = {
            'pickup': .5,      # Reward factor for pickup vicinity
            'dropoff': .5      # Reward factor for dropoff vicinity
        }

        reward = 0  # Initialize the reward for this shuttle

        def process_vicinity(passenger, distance, reward_type):
            """Helper function to process vicinity rewards."""
            nonlocal reward
            vicinity_reward = VICINITY_REWARDS[reward_type] / (distance + 1)
            reward += vicinity_reward
            logger.debug(
                f"Vicinity {reward_type.capitalize()} Reward: {vicinity_reward:.4f} "
                f"for Shuttle {shuttle_id} near Passenger {passenger['name']}'s {reward_type}."
            )

        def calculate_inverse_speed_bonus(elapsed_time, max_time, base_reward):
            """Calculate the inverse speed bonus based on elapsed time and the maximum allowed time."""
            if elapsed_time <= 0:
                return 0
            speed_bonus = base_reward * (2 - (elapsed_time / max_time))
            return max(speed_bonus, 0)  # Ensure the bonus doesn't go negative

        # Handle Dropoff Rewards
        for passenger in dropoffs.get(shuttle_id, []):
            task_position = passenger['destination']
            distance_to_task = self.calculate_travel_time(self.shuttles[shuttle_id]['position'], task_position)

            # Initialize timeout and lag counters if not present
            passenger.setdefault('vicinity_timeout_counter', 0)
            passenger.setdefault('vicinity_lag_counter', 0)

            if distance_to_task == 0:
                # Successful Dropoff
                base_reward = BASE_REWARDS['dropoff']
                # Calculate MATD for the passenger
                matd = self.calculate_matd(passenger['origin'], passenger['destination'])
                # Calculate speed bonus inversely related to the travel time (shorter travel time, higher reward)
                speed_bonus = calculate_inverse_speed_bonus(self.time - passenger['pickup_time'], matd, base_reward)
                reward += base_reward + speed_bonus

                #logger.info(f"Shuttle {shuttle_id} dropped off passenger {passenger['name']} with reward {base_reward + speed_bonus}")
                passenger['vicinity_timeout_counter'] = 0  # Reset counters after successful dropoff

            elif distance_to_task <= VICINITY_RADIUS:
                if passenger['vicinity_timeout_counter'] < VICINITY_TIMEOUT:
                    process_vicinity(passenger, distance_to_task, 'dropoff')
                    passenger['vicinity_timeout_counter'] += 1
                else:
                    logger.debug(f"Shuttle {shuttle_id} in vicinity but timeout exceeded for dropoff.")

        # Handle Pickup Rewards
        for passenger in pickups.get(shuttle_id, []):
            task_position = passenger['origin']
            distance_to_task = self.calculate_travel_time(self.shuttles[shuttle_id]['position'], task_position)

            # Initialize timeout and lag counters if not present
            passenger.setdefault('vicinity_timeout_counter', 0)
            passenger.setdefault('vicinity_lag_counter', 0)

            if distance_to_task == 0:
                # Successful Pickup
                base_reward = BASE_REWARDS['pickup']
                # Calculate MWT for the passenger
                mwt = self.calculate_mwt(passenger['origin'], passenger['destination'])
                # Calculate speed bonus inversely related to the wait time (shorter wait, higher reward)
                speed_bonus = calculate_inverse_speed_bonus(self.time - passenger['request_time'], mwt, base_reward)
                reward += base_reward + speed_bonus

                logger.info(f"Shuttle {shuttle_id} picked up passenger {passenger['name']} with reward {base_reward + speed_bonus}")
                passenger['vicinity_timeout_counter'] = 0  # Reset counters after successful pickup

            elif distance_to_task <= VICINITY_RADIUS:
                if passenger['vicinity_timeout_counter'] < VICINITY_TIMEOUT:
                    process_vicinity(passenger, distance_to_task, 'pickup')
                    passenger['vicinity_timeout_counter'] += 1
                else:
                    logger.debug(f"Shuttle {shuttle_id} in vicinity but timeout exceeded for pickup.")

        # Apply exploration penalty
        exploration_penalty = self.calculate_exploration_penalty(shuttle_id)
        reward += exploration_penalty

        # Check if the episode is done (early completion bonus)
        if self.check_if_done():
            early_completion_bonus = self.calculate_early_completion_bonus()
            reward += early_completion_bonus
            logger.info(f"Early completion bonus awarded: {early_completion_bonus}")

        return reward  # Return the final reward for this shuttle

    def calculate_early_completion_bonus(self):
        """Calculate a bonus for completing the episode early by transporting all passengers."""
        # Define the total time limit for the episode
        max_episode_time = self.total_time  # Total time, typically 500

        # Calculate the remaining time (how much time is left in the episode)
        remaining_time = max_episode_time - self.time

        # Reward the agent based on the remaining time (more remaining time = bigger reward)
        early_completion_bonus = remaining_time * 0.01  # Scaling factor for the reward
        return early_completion_bonus


    def calculate_speed_bonus(self, time_diff, base_reward):
        """
        Calculate the speed bonus based on the time taken for the action (pickup/dropoff).
        """
        if time_diff > 0:
            return 30 / time_diff
        return 0  # No speed bonus if time_diff is zero or negative


    def calculate_exploration_penalty(self, shuttle_id):
        """
        Calculate a penalty for the shuttle staying within a 3-step vicinity of its previous positions
        and for staying in the same quadrant for more than 5 steps.
        """
        if not self.use_exploration_penalty:
            return 0  # If penalty is disabled, return 0

        recent_positions = self.recent_positions[shuttle_id]
        penalty = 0  # Initialize the penalty

        # Penalty for staying too close to previous positions
        #if len(recent_positions) >= 3:
        #    current_position = recent_positions[-1]
        #    for past_position in recent_positions[-3:]:
        #        distance = abs(current_position[0] - past_position[0]) + abs(current_position[1] - past_position[1])
        #        if distance <= 3:
        #            penalty -= 0.015  # Apply penalty for proximity

        # Penalty for staying in the same quadrant for too long
        if len(recent_positions) >= 15:
            current_position = recent_positions[-1]
            current_quadrant = self.get_quadrant(current_position)
            steps_in_same_quadrant = sum(1 for pos in recent_positions[-5:] if self.get_quadrant(pos) == current_quadrant)
            if steps_in_same_quadrant >= 5:
                penalty -= 0.02 * (steps_in_same_quadrant - 4)  # Increase penalty the longer they stay in the same quadrant

        return penalty



    def calculate_unserved_passenger_penalty(self):
        total_unserved = len(self.passengers) + sum(len(shuttle['passengers']) for shuttle in self.shuttles.values())
        penalty_per_passenger = -40

        # Distribute penalty equally among shuttles
        total_penalty = total_unserved * penalty_per_passenger
        per_shuttle_penalty = total_penalty / self.num_shuttles if self.num_shuttles > 0 else 0

        return {shuttle_id: per_shuttle_penalty for shuttle_id in self.shuttles.keys()}


        
    def get_quadrant(self, position):
        """
        Helper function to determine which quadrant the given position is in.
        The grid is divided into four equal quadrants.
        """
        x, y = position
        mid_x = self.grid_size // 2
        mid_y = self.grid_size // 2

        if x < mid_x and y < mid_y:
            return "bottom_left"
        elif x < mid_x and y >= mid_y:
            return "top_left"
        elif x >= mid_x and y < mid_y:
            return "bottom_right"
        else:
            return "top_right"

    def check_level_up(self, success_rate, current_level):
            """
            Only advance to the next level if the agent has passed the success rate
            threshold 3 times in a row.
            """
            success_threshold = 0.8

            if success_rate >= success_threshold:
                self.consecutive_successes += 1
            else:
                # Reset the counter if the agent fails to meet the success rate
                self.consecutive_successes = 0

            if self.consecutive_successes >= 1:
                self.consecutive_successes = 0  # Reset for the next level
                return current_level + 1
            else:
                return current_level
            
    def adjust_difficulty(self, success_rate):
        """
        Adjust the environment's difficulty by increasing the maximum number of passengers 
        or other challenge factors based on the agent's success rate.
        """
        if success_rate > 0.7:  # Example threshold for increasing difficulty
            if self.max_passengers < 10:  # Cap the number of passengers to avoid overwhelming the system
                self.max_passengers += 1  # Gradually increase the max number of passengers
    
            print(f"Difficulty increased: {self.max_passengers} max passengers")

    def calculate_success_rate(self):
        total_passengers = len(self.passenger_log)
        if total_passengers == 0:
            return 0

        successful_pickups = 0
        successful_dropoffs = 0
        shared_rides = 0
        pickup_performance = 0  # To measure how much earlier than MWT the pickup was
        dropoff_performance = 0  # To measure how much earlier than MATD the dropoff was

        for passenger in self.passenger_log:
            # Ensure 'pickup_time' exists and is not None
            if 'pickup_time' in passenger and passenger['pickup_time'] is not None:
                # Check if the pickup was within the MWT
                if passenger['pickup_time'] <= passenger['mwt']:
                    successful_pickups += 1

                    # The earlier the pickup compared to the MWT, the better
                    pickup_diff = passenger['mwt'] - passenger['pickup_time']
                    pickup_performance += pickup_diff / passenger['mwt']  # Normalize by MWT

                # Ensure 'dropoff_time' exists and is not None
                if 'dropoff_time' in passenger and passenger['dropoff_time'] is not None:
                    travel_time = passenger['dropoff_time'] - passenger['pickup_time']
                    if travel_time <= passenger['matd']:
                        successful_dropoffs += 1

                        # The earlier the dropoff compared to the MATD, the better
                        dropoff_diff = passenger['matd'] - travel_time
                        dropoff_performance += dropoff_diff / passenger['matd']  # Normalize by MATD

            # Track ride-sharing performance (if ride-sharing occurred)
            if passenger.get('shared_ride', False):
                shared_rides += 1

        # Calculate individual metrics (normalized values)
        pickup_rate = successful_pickups / total_passengers
        dropoff_rate = successful_dropoffs / total_passengers
        ride_sharing_bonus = shared_rides / total_passengers

        # Average pickup and dropoff performances across all passengers
        avg_pickup_performance = pickup_performance / total_passengers if total_passengers > 0 else 0
        avg_dropoff_performance = dropoff_performance / total_passengers if total_passengers > 0 else 0

        # Adjust success rate calculation to include performance metrics
        success_rate = (
            (pickup_rate * 0.1) +            # Weighting raw pickup success
            (avg_pickup_performance * 0.3) + # Weighting early pickups
            (dropoff_rate * 0.1) +           # Weighting raw dropoff success
            (avg_dropoff_performance * 0.3) +# Weighting early dropoffs
            (ride_sharing_bonus * 0.2)       # Reward for ride-sharing
        )

        return success_rate

    def check_success(self, success_rate, current_time, max_time):
        """
        Check if the agent has successfully completed the episode by transporting
        all passengers within the time limit and satisfying the MATD and MWT constraints.
        """
        all_passengers_served = True
        within_time_limit = current_time <= max_time
        
        # Iterate over all passengers and check if their pickup/dropoff times are within MATD and MWT
        for passenger in self.all_passengers:
            if passenger['status'] == 'served':
                wait_time = passenger['pickup_time'] - passenger['request_time']
                travel_time = passenger['dropoff_time'] - passenger['pickup_time']
                
                # Check if wait time exceeds MWT
                if wait_time > self.calculate_mwt(passenger['origin'], passenger['destination']):
                    logger.info(f"Passenger {passenger['name']} exceeded MWT: Waited {wait_time}, MWT was {self.calculate_mwt(passenger['origin'], passenger['destination'])}.")
                    all_passengers_served = False
                
                # Check if travel time exceeds MATD
                if travel_time > self.calculate_matd(passenger['origin'], passenger['destination']):
                    logger.info(f"Passenger {passenger['name']} exceeded MATD: Traveled {travel_time}, MATD was {self.calculate_matd(passenger['origin'], passenger['destination'])}.")
                    all_passengers_served = False
            else:
                # If any passenger has not been served, mark as failure
                logger.info(f"Passenger {passenger['name']} was not served in time.")
                all_passengers_served = False

        # Check success based on passenger serving status and time constraints
        if all_passengers_served and within_time_limit and success_rate >= 0.8:
            logger.info(f"Success! All passengers were transported successfully within {current_time} steps and met MATD/MWT constraints.")
            return True
        else:
            logger.info(f"Episode failed. Not all passengers were served within MATD/MWT or time limit exceeded.")
            return False


    def generate_passenger(self):
        """
        Generate a new passenger with their origin, destination, MATD (Maximum Allowable Time Deviation),
        and initial status. Limits the total number of generated passengers to 10.
        """
        if len(self.all_passengers) >= 10:
            logger.info("Maximum number of passengers reached. No more passengers will be generated.")
            return None  # No more passengers should be generated if we reach the limit of 10

        # Replace 'randint' with 'integers'
        origin = tuple(self.np_random.integers(0, self.grid_size, size=2))  # Random origin within the grid
        destination = tuple(self.np_random.integers(0, self.grid_size, size=2))  # Random destination within the grid


        # Ensure the destination is different from the origin
        while destination == origin:
            destination = tuple(self.np_random.integers(0, self.grid_size, size=2))

        matd = self.calculate_matd(origin, destination)  # Calculate allowable time deviation based on distance

        passenger_name = f"P{self.generated_passengers + 1}"

        passenger = {
            'name': passenger_name,
            'origin': origin,
            'destination': destination,
            'request_time': self.time,
            'pickup_time': None,
            'dropoff_time': None,
            'elapsed_wait_time': 0,
            'elapsed_travel_time': 0,
            'elapsed_time': 0,
            'picked_up': False,
            'shuttle': None,
            'status': 'waiting',
            'matd': matd
        }

        self.generated_passengers += 1  # Increment total passengers generated

        # Add the passenger to the list of waiting passengers
        self.all_passengers.append(passenger)
        self.passengers.append(passenger)

        #logger.info(f"Generated Passenger: {passenger_name}, Origin: {origin}, Destination: {destination}")
        return passenger



    def generate_passengers(self):
        """
        Generate initial random passengers.
        """
        passengers = []
        num_initial_passengers = 3  # Number of passengers to generate initially

        for _ in range(num_initial_passengers):
            passenger = self.generate_passenger()
            if passenger:
                passengers.append(passenger)

        return passengers



        
    def generate_fixed_passenger(self, origin, destination):
        """
        Generates a passenger with a fixed origin and destination.
        """
        passenger_name = f"P{self.generated_passengers + 1}"

        passenger = {
            'name': passenger_name,
            'origin': origin,
            'destination': destination,
            'request_time': self.time,
            'pickup_time': None,
            'dropoff_time': None,
            'elapsed_wait_time': 0,
            'elapsed_travel_time': 0,
            'elapsed_time': 0,
            'picked_up': False,
            'shuttle': None,
            'status': 'waiting',
            'matd': self.calculate_matd(origin, destination)    
        }

        self.generated_passengers += 1  # Increment the total passengers generated

        # Add the passenger to the list of waiting passengers
        self.all_passengers.append(passenger)
        self.passengers.append(passenger)

        return passenger

    def generate_random_position_passenger(self):
        """Generate a passenger with a random position for harder levels."""
        return self.generate_passenger()  # Additional logic for randomization can be added here.
        
    def calculate_matd(self, origin, destination):
        """
        Calculate Maximum Allowable Time Deviation (MATD) based on the Manhattan distance between
        origin and destination. MATD is 150% of the expected travel time.
        """
        travel_time = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
        matd = travel_time * 2.5
        return matd

    def calculate_mwt(self, origin, destination):
        """
        Calculate Maximum Wait Time (MWT) based on the Manhattan distance between
        origin and destination. MWT is defined as half of the expected travel time.
        """
        travel_time = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
        mwt = travel_time * 1.75  
        return mwt

    def update_passenger_status(self):
        """
        Update the status of passengers based on their interactions with shuttles.
        """
        for shuttle_id, shuttle in self.shuttles.items():
            for passenger in shuttle['passengers']:
                passenger['status'] = 'onboard'  # Update the status of onboard passengers

        # Passengers still in the waiting state (not onboard any shuttle)
        for passenger in self.passengers:
            if passenger['status'] == 'waiting':
                passenger['elapsed_time'] = self.time - passenger['request_time']
            elif passenger['status'] == 'served':
                # Served passengers are already dropped off and won't be updated further
                continue

    def render(self, mode="human"):
        if self.render_mode == "matplotlib":
            self.plot_grid()
        elif mode == "rgb_array":
            raise NotImplementedError("The 'rgb_array' mode is not implemented yet.")
        else:
            raise NotImplementedError(f"Unsupported render mode: {mode}")

    def log_passenger_statistics(self, save_logs=True, total_simulation_steps=None):
        """
        Log passengers that have been successfully served at the end of the episode.
        This log includes the shuttle that served them, their waiting time, traveling time,
        total time, the distance of their trip, pickup and drop-off locations, the step they were generated,
        and the total simulation steps at termination.
        """
        global logger  # Use the globally configured logger

        if save_logs:
            # Define your desired save directory
            save_dir = "../data/passenger_logs"  # Replace with your directory path
            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

            # Generate a filename with today's date and time
            log_filename = os.path.join(save_dir, f"passenger_stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

            # Set up the logger with a FileHandler to write to the log file
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            # Create a file handler that writes to the log file
            file_handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Clear existing handlers and add the new file handler
            logger.handlers.clear()
            logger.addHandler(file_handler)

        # Log the episode summary header
        logger.info("\nEpisode Summary - Successfully Served Passengers:\n")
        logger.info(f"{'Passenger':<10} {'Trip Distance':<15} {'Waiting Time':<15} {'Traveling Time':<15} {'Total Time':<15} {'Shuttle':<10}")
        logger.info(f"{'Pickup':<25} {'Dropoff':<25} {'Generate Step':<20}")
        logger.info("=" * 140)

        # Iterate over the passenger log and log details of served passengers
        for passenger in self.passenger_log:
            if 'pickup_time' in passenger and 'dropoff_time' in passenger:
                shuttle = passenger.get('shuttle_id', 'N/A')
                waiting_time = passenger.get('wait_time', 'N/A')
                traveling_time = passenger.get('travel_time', 'N/A')
                total_time = passenger.get('total_time', 'N/A')
                trip_distance = self.calculate_manhattan_distance(passenger['origin'], passenger['destination'])
                pickup_location = passenger.get('origin', 'Unknown')
                dropoff_location = passenger.get('destination', 'Unknown')
                generate_step = passenger.get('request_time', 'Unknown')  # Step when the passenger was generated
                pickup_step = passenger.get('pickup_time', 'N/A')  # Step when the passenger was picked up
                dropoff_step = passenger.get('dropoff_time', 'N/A')  # Step when the passenger was dropped off

                logger.info(
                    f"{passenger['name']:<10} {trip_distance:<15} {waiting_time:<15} {traveling_time:<15} "
                    f"{total_time:<15} {shuttle:<10} {str(pickup_location):<25} {str(dropoff_location):<25} "
                    f"{generate_step:<20} Pickup Step: {pickup_step:<10} Dropoff Step: {dropoff_step:<10}"
                )

            else:
                logger.warning(f"Passenger {passenger['name']} missing 'pickup_time' or 'dropoff_time'. Possibly not served.")

        logger.info("=" * 140)

        # Log total simulation steps at termination
        if total_simulation_steps is not None:
            logger.info(f"\nTotal Simulation Steps Taken During Simulation: {total_simulation_steps}")

        logger.info("\nEnd of Episode Summary\n")

        # Remove file handler if logging to a file to avoid duplicate logs
        if save_logs:
            logger.removeHandler(file_handler)


    def plot_grid(self):
        """
        Visualize the environment using matplotlib.
        This will draw the shuttles, passengers, and display additional state information.
        """
        if not self.render_mode == "matplotlib":
            return  # Only plot if the render mode is set to matplotlib
        
        #print("plot_grid called")  # Debug statement

        self.ax1.clear()
        self.ax1.set_xlim(-1, self.grid_size + 1)
        self.ax1.set_ylim(-1, self.grid_size + 1)
        self.ax1.set_xticks(np.arange(0, self.grid_size + 1, 1))
        self.ax1.set_yticks(np.arange(0, self.grid_size + 1, 1))
        self.ax1.grid(True)

        # Define triangle size for shuttle representation
        triangle_base = 0.2
        triangle_height = 0.3

        # Plot shuttles
        for shuttle_id, shuttle in self.shuttles.items():
            x, y = shuttle['position']
            orientation = shuttle['orientation']

            # Calculate vertices for the shuttle's orientation
            if orientation == 'north':
                vertices = [(x, y), (x - triangle_base / 2, y - triangle_height), (x + triangle_base / 2, y - triangle_height)]
            elif orientation == 'east':
                vertices = [(x, y), (x - triangle_height, y - triangle_base / 2), (x - triangle_height, y + triangle_base / 2)]
            elif orientation == 'south':
                vertices = [(x, y), (x - triangle_base / 2, y + triangle_height), (x + triangle_base / 2, y + triangle_height)]
            elif orientation == 'west':
                vertices = [(x, y), (x + triangle_height, y - triangle_base / 2), (x + triangle_height, y + triangle_base / 2)]

            # Draw the shuttle as a triangle
            triangle = patches.Polygon(vertices, closed=True, facecolor='blue', edgecolor='blue')
            self.ax1.add_patch(triangle)

            # Annotate the shuttle with info
            self.ax1.annotate(
                f'AS{shuttle_id}\nP: {len(shuttle["passengers"])}',
                xy=(x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10,
                color='blue'
            )

        # Plot passengers' destinations (red cross) and origins (red circle)
        for passenger in self.all_passengers:
            # Only plot the destination if the passenger has not been served
            if passenger['status'] != 'served':
                dx, dy = passenger['destination']
                self.ax1.plot(dx, dy, 'rx')
                self.ax1.annotate(
                    f'{passenger["name"]}\nD:({dx},{dy})',
                    xy=(dx, dy),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10,
                    color='red'
                )

            # If passenger is waiting or picked up, show origin
            if passenger['status'] == 'waiting' or passenger['status'] == 'picked_up':
                ox, oy = passenger['origin']
                self.ax1.plot(ox, oy, 'ro')
                color = 'red' if passenger['status'] == 'waiting' else 'purple'
                self.ax1.annotate(
                    f'{passenger["name"]}\nO:({ox},{oy})\nMATD:{passenger["matd"]:.1f}\nT:{passenger["elapsed_time"]:.1f}',
                    xy=(ox, oy),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10,
                    color=color
                )

        # Display simulation time
        self.ax1.text(
            self.grid_size, -1,
            f'Time: {self.time}',
            fontsize=12,
            ha='right',
            color='black'
        )

        self.ax1.text(self.grid_size, -1, f'Time: {self.time}', fontsize=12, ha='right', color='black')


        # Clear ax2 and plot detailed shuttle and passenger information
        self.ax2.clear()
        self.ax2.axis('off')

        # Prepare AS information table (unchanged)
        as_info = ['AS Information\n']
        for shuttle_id, shuttle in self.shuttles.items():
            onboard_passengers = '\n'.join([f'{p["name"]}: Dest: {p["destination"]}, MATD: {p["matd"]:.1f}' for p in shuttle['passengers']])
            as_info.append(f'AS{shuttle_id}: Loc: {shuttle["position"]}, Capacity: {len(shuttle["passengers"])}/{self.max_passengers}\nOnboard Passengers:\n{onboard_passengers if onboard_passengers else "None"}\n')

        # Prepare current passenger request table (updated)
        passenger_requests_info = ['Passenger Requests\n']
        for passenger in self.passengers:
            info = f'{passenger["name"]}: Pickup: {passenger["origin"]}, Dropoff: {passenger["destination"]}, ' \
                   f'MATD: {passenger["matd"]:.1f}, Elapsed Time: {passenger["elapsed_time"]:.1f}'
            if passenger['status'] == 'picked_up':
                passenger_requests_info.append(f'\033[91m{info}\033[0m')  # Red color for picked-up passengers
            else:
                passenger_requests_info.append(info)

        # Prepare transferred passengers table (unchanged)
        transferred_passengers_info = ['Transferred Passengers\n']
        for log in self.passenger_log:
            if log["name"] in [p["name"] for p in self.all_passengers if p["status"] == "served"]:
                transferred_passengers_info.append(
                    f'\033[92m{log["name"]}: Shuttle {log["shuttle_id"]}, '
                    f'Wait: {log["wait_time"]:.1f}, Travel: {log["travel_time"]:.1f}, '
                    f'Total: {log["total_time"]:.1f}, MATD: {log["matd"]:.1f}\033[0m'
                )
            else:
                transferred_passengers_info.append(
                    f'{log["name"]}: Shuttle {log["shuttle_id"]}, '
                    f'Wait: {log["wait_time"]:.1f}, Travel: {log["travel_time"]:.1f}, '
                    f'Total: {log["total_time"]:.1f}, MATD: {log["matd"]:.1f}'
                )

        # Display information
        self.ax2.text(0.05, 0.95, '\n'.join(as_info), va='top', ha='left', fontsize=10)
        self.ax2.text(0.32, 0.95, '\n'.join(passenger_requests_info), va='top', ha='left', fontsize=10)
        self.ax2.text(0.65, 0.95, '\n'.join(transferred_passengers_info), va='top', ha='left', fontsize=10)

        # Plot blocked positions (e.g., as grey squares)
        for bx, by in self.blocked_positions:
            blocked_square = patches.Rectangle((bx - 0.5, by - 0.5), 1, 1, facecolor='grey', edgecolor='grey', alpha=0.5)
            self.ax1.add_patch(blocked_square)


        self.fig.canvas.draw()
        plt.pause(0.01)