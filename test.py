import random
import time
import math
import pickle
from collections import deque
import numpy as np

# ========================================
# DEEP Q-LEARNING RACING GAME
# Neural Network AI learns to race!
# Watch it train in real-time!
# ========================================

# Simple Neural Network implementation (no external dependencies!)
class NeuralNetwork:
    """A simple feedforward neural network for DQN"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        # He initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.w3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    def forward(self, x):
        """Forward pass through network"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        return self.z3

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def backward(self, x, y_true):
        """Backward pass and update weights"""
        m = x.shape[0]

        # Output layer gradients
        dz3 = self.a3 - y_true
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Hidden layer 2 gradients
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    def predict(self, x):
        """Make prediction"""
        self.a3 = self.forward(x)
        return self.a3

    def train_step(self, x, y):
        """Single training step"""
        self.a3 = self.forward(x)
        self.backward(x, y)

    def copy_from(self, other_network):
        """Copy weights from another network"""
        self.w1 = other_network.w1.copy()
        self.b1 = other_network.b1.copy()
        self.w2 = other_network.w2.copy()
        self.b2 = other_network.b2.copy()
        self.w3 = other_network.w3.copy()
        self.b3 = other_network.b3.copy()


class Car:
    """The racing car"""

    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # in degrees
        self.speed = 0
        self.max_speed = 3.0
        self.acceleration = 0.2
        self.friction = 0.95
        self.turn_speed = 5

        # Sensor distances
        self.sensor_distances = []

    def update(self, action):
        """
        Update car state based on action
        action: 0=nothing, 1=accelerate, 2=left, 3=right, 4=brake
        """
        # Apply action
        if action == 1:  # Accelerate
            self.speed = min(self.max_speed, self.speed + self.acceleration)
        elif action == 2:  # Turn left
            self.angle -= self.turn_speed
        elif action == 3:  # Turn right
            self.angle += self.turn_speed
        elif action == 4:  # Brake
            self.speed *= 0.8

        # Apply friction
        self.speed *= self.friction

        # Update position
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

        # Keep angle in [0, 360)
        self.angle = self.angle % 360


class RaceTrack:
    """A procedural race track"""

    def __init__(self, width=60, height=40):
        self.width = width
        self.height = height
        self.track_width = 8

        # Create oval track
        self.create_oval_track()

    def create_oval_track(self):
        """Create an oval racing track"""
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.radius_x = self.width // 2 - 5
        self.radius_y = self.height // 2 - 3

        # Start position (right side of oval)
        self.start_x = self.center_x + self.radius_x - 10
        self.start_y = self.center_y
        self.start_angle = 180  # Facing left

    def is_on_track(self, x, y):
        """Check if position is on the track"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        # Calculate distance from center of oval
        dx = (x - self.center_x) / self.radius_x
        dy = (y - self.center_y) / self.radius_y
        dist_from_center = math.sqrt(dx*dx + dy*dy)

        # On track if within the track width
        outer_bound = 1.0
        inner_bound = 1.0 - (self.track_width / min(self.radius_x, self.radius_y))

        return inner_bound <= dist_from_center <= outer_bound

    def get_progress(self, x, y):
        """Get progress around track (0 to 1)"""
        # Angle from center
        dx = x - self.center_x
        dy = y - self.center_y
        angle = math.atan2(dy, dx)

        # Normalize to [0, 1]
        progress = (angle + math.pi) / (2 * math.pi)
        return progress


class RacingGame:
    """The racing game environment"""

    def __init__(self):
        self.track = RaceTrack()
        self.car = None
        self.reset()

        # Sensor configuration (5 distance sensors)
        self.sensor_angles = [-60, -30, 0, 30, 60]  # Relative to car angle
        self.max_sensor_distance = 15

    def reset(self):
        """Reset the game"""
        self.car = Car(self.track.start_x, self.track.start_y, self.track.start_angle)
        self.steps = 0
        self.max_progress = 0
        self.last_progress = 0
        self.laps_completed = 0
        self.crashed = False

        return self.get_state()

    def get_state(self):
        """Get current state (sensor readings + speed + angle)"""
        # Update sensors
        self.car.sensor_distances = self.get_sensor_readings()

        # Normalize values
        normalized_sensors = [d / self.max_sensor_distance for d in self.car.sensor_distances]
        normalized_speed = self.car.speed / self.car.max_speed

        # Angle as sin/cos for continuity
        rad = math.radians(self.car.angle)
        angle_sin = math.sin(rad)
        angle_cos = math.cos(rad)

        state = normalized_sensors + [normalized_speed, angle_sin, angle_cos]
        return np.array(state, dtype=np.float32)

    def get_sensor_readings(self):
        """Get distance readings from sensors"""
        distances = []

        for sensor_angle in self.sensor_angles:
            # Sensor direction
            angle = self.car.angle + sensor_angle
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)

            # Cast ray
            distance = 0
            for step in range(self.max_sensor_distance):
                test_x = self.car.x + dx * step
                test_y = self.car.y + dy * step

                if not self.track.is_on_track(test_x, test_y):
                    distance = step
                    break
            else:
                distance = self.max_sensor_distance

            distances.append(distance)

        return distances

    def step(self, action):
        """Take action and return (state, reward, done)"""
        self.steps += 1

        # Update car
        old_x, old_y = self.car.x, self.car.y
        self.car.update(action)

        # Check if still on track
        if not self.track.is_on_track(self.car.x, self.car.y):
            self.crashed = True
            return self.get_state(), -100, True

        # Calculate reward
        reward = 0

        # Reward for speed (encourage going fast)
        reward += self.car.speed * 0.5

        # Reward for progress
        progress = self.track.get_progress(self.car.x, self.car.y)

        # Handle lap completion
        if progress < 0.2 and self.last_progress > 0.8:
            self.laps_completed += 1
            reward += 500  # Big reward for completing lap!
            self.max_progress = 0

        # Reward for making progress
        if progress > self.max_progress:
            reward += (progress - self.max_progress) * 100
            self.max_progress = progress

        self.last_progress = progress

        # Small step penalty to encourage efficiency
        reward -= 0.1

        # Episode limits
        done = False
        if self.steps > 1000:  # Time limit
            done = True
            if self.laps_completed == 0:
                reward -= 50  # Penalty for not completing

        if self.laps_completed >= 2:  # Completed 2 laps
            done = True
            reward += 200  # Bonus for completing 2 laps

        return self.get_state(), reward, done


class DQNAgent:
    """Deep Q-Network agent"""

    def __init__(self, state_size=8, action_size=5, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Networks
        self.model = NeuralNetwork(state_size, 64, action_size, learning_rate)
        self.target_model = NeuralNetwork(state_size, 64, action_size, learning_rate)
        self.target_model.copy_from(self.model)

        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_laps = []

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Predict Q-values
        state_batch = state.reshape(1, -1)
        q_values = self.model.predict(state_batch)[0]
        return np.argmax(q_values)

    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        # Predict current Q-values
        current_q = self.model.predict(states)

        # Predict next Q-values using target network
        next_q = self.target_model.predict(next_states)

        # Update Q-values with Bellman equation
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Train network
        self.model.train_step(states, target_q)

    def update_target_network(self):
        """Copy weights from model to target model"""
        self.target_model.copy_from(self.model)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward, steps, laps):
        """Record episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_laps.append(laps)


class Visualizer:
    """Visualize the racing game"""

    def __init__(self):
        self.colors = {
            'track': '\033[90m',      # Dark gray
            'grass': '\033[32m',      # Green
            'car': '\033[93m',        # Yellow
            'sensor': '\033[96m',     # Cyan
            'wall': '\033[91m',       # Red
            'reset': '\033[0m',
            'info': '\033[96m',
            'score': '\033[95m',
            'white': '\033[97m',
            'dim': '\033[2m',
        }

    def clear_screen(self):
        print("\033[2J\033[H", end='')

    def hide_cursor(self):
        print("\033[?25l", end='')

    def show_cursor(self):
        print("\033[?25h", end='')

    def draw_game(self, game, episode, agent, total_reward):
        """Draw the current game state"""
        self.clear_screen()

        # Create display grid
        grid = [[' ' for _ in range(game.track.width)] for _ in range(game.track.height)]

        # Draw track
        for y in range(game.track.height):
            for x in range(game.track.width):
                if game.track.is_on_track(x, y):
                    grid[y][x] = '░'

        # Draw sensors
        for i, (sensor_angle, distance) in enumerate(zip(game.sensor_angles, game.car.sensor_distances)):
            angle = game.car.angle + sensor_angle
            rad = math.radians(angle)
            for step in range(int(distance)):
                sx = int(game.car.x + math.cos(rad) * step)
                sy = int(game.car.y + math.sin(rad) * step)
                if 0 <= sx < game.track.width and 0 <= sy < game.track.height:
                    grid[sy][sx] = '·'

        # Draw car
        car_x, car_y = int(game.car.x), int(game.car.y)
        if 0 <= car_x < game.track.width and 0 <= car_y < game.track.height:
            # Car direction indicator
            angle_chars = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']
            angle_index = int((game.car.angle + 22.5) / 45) % 8
            grid[car_y][car_x] = angle_chars[angle_index]

        # Print grid
        print(self.colors['info'] + '╔' + '═' * game.track.width + '╗' + self.colors['reset'])
        for y, row in enumerate(grid):
            print(self.colors['info'] + '║' + self.colors['reset'], end='')
            for x, cell in enumerate(row):
                if cell == '░':
                    print(self.colors['track'] + cell + self.colors['reset'], end='')
                elif cell == '·':
                    print(self.colors['sensor'] + cell + self.colors['reset'], end='')
                elif cell in ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']:
                    print(self.colors['car'] + cell + self.colors['reset'], end='')
                else:
                    print(self.colors['grass'] + cell + self.colors['reset'], end='')
            print(self.colors['info'] + '║' + self.colors['reset'])
        print(self.colors['info'] + '╚' + '═' * game.track.width + '╝' + self.colors['reset'])

        # Stats
        print()
        print(f"{self.colors['info']}Episode: {self.colors['white']}{episode}{self.colors['reset']}  ", end='')
        print(f"{self.colors['score']}Laps: {self.colors['white']}{game.laps_completed}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Steps: {self.colors['white']}{game.steps}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Reward: {self.colors['white']}{total_reward:.1f}{self.colors['reset']}")

        print(f"{self.colors['info']}Speed: {self.colors['white']}{game.car.speed:.2f}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Progress: {self.colors['white']}{game.max_progress*100:.1f}%{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Epsilon: {self.colors['white']}{agent.epsilon:.3f}{self.colors['reset']}")

        # Recent performance
        if len(agent.episode_laps) > 0:
            recent = min(50, len(agent.episode_laps))
            avg_reward = sum(agent.episode_rewards[-recent:]) / recent
            avg_laps = sum(agent.episode_laps[-recent:]) / recent
            max_laps = max(agent.episode_laps[-recent:])

            print()
            print(f"{self.colors['info']}Last {recent} episodes:{self.colors['reset']}")
            print(f"  Avg Reward: {self.colors['white']}{avg_reward:.1f}{self.colors['reset']}  ", end='')
            print(f"Avg Laps: {self.colors['white']}{avg_laps:.2f}{self.colors['reset']}  ", end='')
            print(f"Best: {self.colors['white']}{max_laps} laps{self.colors['reset']}")

        # Progress chart
        if len(agent.episode_laps) >= 10:
            self._draw_progress_chart(agent.episode_laps[-40:])

    def _draw_progress_chart(self, laps):
        """Draw progress chart"""
        if not laps:
            return

        print()
        print(f"{self.colors['info']}Laps Progress (last {len(laps)} episodes):{self.colors['reset']}")

        max_laps = max(max(laps), 1)
        height = 6
        width = min(40, len(laps))

        # Sample if needed
        if len(laps) > width:
            step = len(laps) / width
            sampled = [laps[int(i * step)] for i in range(width)]
        else:
            sampled = laps

        # Draw chart
        for h in range(height, 0, -1):
            threshold = (h / height) * max_laps
            line = ''
            for lap_count in sampled:
                if lap_count >= threshold:
                    line += self.colors['score'] + '█' + self.colors['reset']
                else:
                    line += self.colors['dim'] + '░' + self.colors['reset']
            print(f"  {line}")

        print(f"  {self.colors['dim']}{'─' * width}{self.colors['reset']}")

    def show_intro(self):
        """Show intro screen"""
        self.clear_screen()
        print()
        print(self.colors['score'] + "  ╔═══════════════════════════════════════════════════╗")
        print("  ║                                                   ║")
        print("  ║       DEEP Q-LEARNING RACING GAME                 ║")
        print("  ║                                                   ║")
        print("  ║          Neural Network AI Learning               ║")
        print("  ║                                                   ║")
        print("  ╚═══════════════════════════════════════════════════╝" + self.colors['reset'])
        print()
        print(f"{self.colors['info']}Watch a neural network learn to race!{self.colors['reset']}")
        print()
        print(f"{self.colors['white']}Legend:{self.colors['reset']}")
        print(f"  {self.colors['car']}→↓←↑{self.colors['reset']} Car (direction)")
        print(f"  {self.colors['sensor']}·{self.colors['reset']} Distance sensors")
        print(f"  {self.colors['track']}░{self.colors['reset']} Race track")
        print()
        print(f"{self.colors['info']}The AI uses Deep Q-Learning:{self.colors['reset']}")
        print("  • Neural network processes sensor data")
        print("  • Experience replay for stable learning")
        print("  • Learns to stay on track and complete laps")
        print()
        print(f"{self.colors['dim']}Starting training in 3 seconds...{self.colors['reset']}")
        time.sleep(3)


def train_agent(episodes=300, render_every=1, delay=0.02):
    """Train the DQN agent"""
    game = RacingGame()
    agent = DQNAgent(state_size=8, action_size=5)
    viz = Visualizer()

    try:
        viz.hide_cursor()
        viz.show_intro()

        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0
            done = False

            while not done:
                # Choose action
                action = agent.get_action(state, training=True)

                # Take step
                next_state, reward, done = game.step(action)
                total_reward += reward

                # Remember experience
                agent.remember(state, action, reward, next_state, done)

                # Train on replay memory
                agent.replay()

                state = next_state

                # Render
                if episode % render_every == 0:
                    viz.draw_game(game, episode, agent, total_reward)
                    time.sleep(delay)

            # Record episode
            agent.record_episode(total_reward, game.steps, game.laps_completed)
            agent.decay_epsilon()

            # Update target network periodically
            if episode % 10 == 0:
                agent.update_target_network()

            # Print progress
            if episode % 5 == 0 and episode % render_every != 0:
                viz.draw_game(game, episode, agent, total_reward)

        print()
        print(f"{viz.colors['score']}Training complete!{viz.colors['reset']}")
        print()

        if len(agent.episode_laps) > 0:
            recent = min(50, len(agent.episode_laps))
            avg_laps = sum(agent.episode_laps[-recent:]) / recent
            max_laps = max(agent.episode_laps)

            print(f"  Best performance: {viz.colors['white']}{max_laps} laps{viz.colors['reset']}")
            print(f"  Recent average: {viz.colors['white']}{avg_laps:.2f} laps{viz.colors['reset']}")

        print()

    finally:
        viz.show_cursor()
        print(viz.colors['reset'])


def interactive_menu():
    """Interactive menu"""
    viz = Visualizer()

    while True:
        viz.clear_screen()
        print()
        print(viz.colors['score'] + "  ╔═══════════════════════════════════════════════════╗")
        print("  ║                                                   ║")
        print("  ║       DEEP Q-LEARNING RACING GAME                 ║")
        print("  ║                                                   ║")
        print("  ╚═══════════════════════════════════════════════════╝" + viz.colors['reset'])
        print()
        print(f"{viz.colors['info']}Choose an option:{viz.colors['reset']}")
        print()
        print(f"  {viz.colors['white']}1.{viz.colors['reset']} Quick demo (50 episodes)")
        print(f"  {viz.colors['white']}2.{viz.colors['reset']} Train agent (300 episodes, watch all)")
        print(f"  {viz.colors['white']}3.{viz.colors['reset']} Train agent (300 episodes, fast)")
        print(f"  {viz.colors['white']}4.{viz.colors['reset']} Long training (1000 episodes)")
        print(f"  {viz.colors['white']}5.{viz.colors['reset']} Exit")
        print()

        choice = input(f"{viz.colors['info']}Enter choice (1-5): {viz.colors['reset']}")

        if choice == '1':
            train_agent(episodes=50, render_every=1, delay=0.03)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '2':
            train_agent(episodes=300, render_every=1, delay=0.02)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '3':
            train_agent(episodes=300, render_every=5, delay=0.01)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '4':
            train_agent(episodes=1000, render_every=10, delay=0.005)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '5':
            print(f"\n{viz.colors['info']}Thanks for watching!{viz.colors['reset']}\n")
            break
        else:
            print(f"{viz.colors['wall']}Invalid choice!{viz.colors['reset']}")
            time.sleep(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            train_agent(episodes=50, render_every=1, delay=0.03)
        elif sys.argv[1] == '--train':
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            train_agent(episodes=episodes, render_every=5, delay=0.01)
    else:
        interactive_menu()
