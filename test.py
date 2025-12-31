import random
import time
import math
import os
import pickle
from collections import deque
from enum import Enum

# ========================================
# REINFORCEMENT LEARNING SNAKE GAME
# Q-Learning AI that learns to play Snake
# Watch it train in real-time!
# ========================================

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGame:
    """The Snake game environment"""

    def __init__(self, width=20, height=15):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        # Snake starts in the middle
        mid_x = self.width // 2
        mid_y = self.height // 2
        self.snake = deque([(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)])
        self.direction = Direction.RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        return self.get_state()

    def _place_food(self):
        """Place food at random location not on snake"""
        while True:
            food = (random.randint(0, self.width - 1),
                   random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        """
        Take an action (0=straight, 1=right turn, 2=left turn)
        Returns: (new_state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0, True

        # Convert action to direction
        self._update_direction(action)

        # Move snake
        head_x, head_y = self.snake[0]

        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)

        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            return self.get_state(), -100, True

        # Check collision with self
        if new_head in self.snake:
            self.game_over = True
            return self.get_state(), -100, True

        self.snake.appendleft(new_head)
        self.steps += 1
        self.steps_since_food += 1

        # Check if ate food
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._place_food()
            reward = 100
        else:
            self.snake.pop()
            reward = 0

        # Penalty for taking too long
        if self.steps_since_food > 100:
            self.game_over = True
            return self.get_state(), -100, True

        # Small reward for getting closer to food
        old_dist = self._distance_to_food(1)
        new_dist = self._distance_to_food(0)
        if new_dist < old_dist:
            reward += 1

        return self.get_state(), reward, self.game_over

    def _distance_to_food(self, index=0):
        """Manhattan distance from snake head to food"""
        head = self.snake[index]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def _update_direction(self, action):
        """Update direction based on action (0=straight, 1=right, 2=left)"""
        if action == 0:  # Straight
            return
        elif action == 1:  # Right turn
            self.direction = Direction((self.direction.value + 1) % 4)
        elif action == 2:  # Left turn
            self.direction = Direction((self.direction.value - 1) % 4)

    def get_state(self):
        """
        Get current state as a tuple for Q-table
        State includes:
        - Danger straight, right, left
        - Current direction (4 values)
        - Food location relative to head (8 directions)
        """
        head = self.snake[0]

        # Check danger in 3 directions
        danger_straight = self._is_danger(0)
        danger_right = self._is_danger(1)
        danger_left = self._is_danger(2)

        # Current direction (one-hot encoded)
        dir_up = self.direction == Direction.UP
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT

        # Food location (relative to head)
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        state = (
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_down, food_left, food_right
        )

        return state

    def _is_danger(self, action):
        """Check if there's danger in the direction of action"""
        # Create a copy of direction
        test_dir = self.direction

        if action == 1:  # Right turn
            test_dir = Direction((test_dir.value + 1) % 4)
        elif action == 2:  # Left turn
            test_dir = Direction((test_dir.value - 1) % 4)

        head_x, head_y = self.snake[0]

        if test_dir == Direction.UP:
            test_pos = (head_x, head_y - 1)
        elif test_dir == Direction.DOWN:
            test_pos = (head_x, head_y + 1)
        elif test_dir == Direction.LEFT:
            test_pos = (head_x - 1, head_y)
        else:  # RIGHT
            test_pos = (head_x + 1, head_y)

        # Check wall collision
        if (test_pos[0] < 0 or test_pos[0] >= self.width or
            test_pos[1] < 0 or test_pos[1] >= self.height):
            return True

        # Check self collision
        if test_pos in self.snake:
            return True

        return False


class QLearningAgent:
    """Q-Learning agent for playing Snake"""

    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def get_action(self, state, training=True):
        """
        Get action using epsilon-greedy policy
        Returns: action (0=straight, 1=right, 2=left)
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 2)
        else:
            # Exploitation: best action
            q_values = [self.get_q_value(state, a) for a in range(3)]
            max_q = max(q_values)

            # If multiple actions have same Q-value, choose randomly
            best_actions = [a for a in range(3) if q_values[a] == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        current_q = self.get_q_value(state, action)

        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(3)]
            target_q = reward + self.discount_factor * max(next_q_values)

        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward, score, steps):
        """Record episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_scores.append(score)
        self.episode_steps.append(steps)

    def save(self, filename='snake_agent.pkl'):
        """Save agent to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_scores': self.episode_scores,
                'episode_steps': self.episode_steps
            }, f)

    def load(self, filename='snake_agent.pkl'):
        """Load agent from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
                self.episode_rewards = data['episode_rewards']
                self.episode_scores = data['episode_scores']
                self.episode_steps = data['episode_steps']
            return True
        except FileNotFoundError:
            return False


class Visualizer:
    """Visualize the game and training progress"""

    def __init__(self):
        self.colors = {
            'snake': '\033[92m',      # Green
            'head': '\033[93m',       # Yellow
            'food': '\033[91m',       # Red
            'wall': '\033[94m',       # Blue
            'reset': '\033[0m',
            'info': '\033[96m',       # Cyan
            'score': '\033[95m',      # Magenta
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

        # Draw top border
        print(self.colors['wall'] + '╔' + '═' * (game.width * 2) + '╗' + self.colors['reset'])

        # Draw game area
        for y in range(game.height):
            print(self.colors['wall'] + '║' + self.colors['reset'], end='')
            for x in range(game.width):
                pos = (x, y)
                if pos == game.snake[0]:
                    print(self.colors['head'] + '██' + self.colors['reset'], end='')
                elif pos in game.snake:
                    print(self.colors['snake'] + '██' + self.colors['reset'], end='')
                elif pos == game.food:
                    print(self.colors['food'] + '●●' + self.colors['reset'], end='')
                else:
                    print('  ', end='')
            print(self.colors['wall'] + '║' + self.colors['reset'])

        # Draw bottom border
        print(self.colors['wall'] + '╚' + '═' * (game.width * 2) + '╝' + self.colors['reset'])

        # Draw stats
        print()
        print(f"{self.colors['info']}Episode: {self.colors['white']}{episode}{self.colors['reset']}  ", end='')
        print(f"{self.colors['score']}Score: {self.colors['white']}{game.score}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Steps: {self.colors['white']}{game.steps}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Reward: {self.colors['white']}{total_reward:.1f}{self.colors['reset']}")

        print(f"{self.colors['info']}Epsilon: {self.colors['white']}{agent.epsilon:.3f}{self.colors['reset']}  ", end='')
        print(f"{self.colors['info']}Q-Table Size: {self.colors['white']}{len(agent.q_table)}{self.colors['reset']}")

        # Show recent performance
        if len(agent.episode_scores) > 0:
            recent = min(100, len(agent.episode_scores))
            avg_score = sum(agent.episode_scores[-recent:]) / recent
            max_score = max(agent.episode_scores[-recent:]) if agent.episode_scores else 0

            print()
            print(f"{self.colors['info']}Last {recent} episodes:{self.colors['reset']}")
            print(f"  Avg Score: {self.colors['white']}{avg_score:.2f}{self.colors['reset']}  ", end='')
            print(f"Max Score: {self.colors['white']}{max_score}{self.colors['reset']}")

        # Progress bar
        if len(agent.episode_scores) >= 10:
            self._draw_progress_chart(agent.episode_scores[-50:])

    def _draw_progress_chart(self, scores):
        """Draw a simple ASCII chart of recent scores"""
        if not scores:
            return

        print()
        print(f"{self.colors['info']}Score Progress (last {len(scores)} episodes):{self.colors['reset']}")

        max_score = max(scores) if scores else 1
        height = 8
        width = min(50, len(scores))

        # Sample scores if we have too many
        if len(scores) > width:
            step = len(scores) / width
            sampled = [scores[int(i * step)] for i in range(width)]
        else:
            sampled = scores

        # Draw chart
        for h in range(height, 0, -1):
            threshold = (h / height) * max_score
            line = ''
            for score in sampled:
                if score >= threshold:
                    line += self.colors['score'] + '█' + self.colors['reset']
                else:
                    line += self.colors['dim'] + '░' + self.colors['reset']
            print(f"  {line}")

        print(f"  {self.colors['dim']}{'─' * width}{self.colors['reset']}")
        print(f"  0{' ' * (width - 4)}{max_score}")

    def show_intro(self):
        """Show introduction screen"""
        self.clear_screen()
        print()
        print(self.colors['score'] + "  ╔═══════════════════════════════════════════════════╗")
        print("  ║                                                   ║")
        print("  ║        REINFORCEMENT LEARNING SNAKE GAME          ║")
        print("  ║                                                   ║")
        print("  ║              Q-Learning AI Training               ║")
        print("  ║                                                   ║")
        print("  ╚═══════════════════════════════════════════════════╝" + self.colors['reset'])
        print()
        print(f"{self.colors['info']}Watch the AI learn to play Snake in real-time!{self.colors['reset']}")
        print()
        print(f"{self.colors['white']}Legend:{self.colors['reset']}")
        print(f"  {self.colors['head']}██{self.colors['reset']} Snake Head")
        print(f"  {self.colors['snake']}██{self.colors['reset']} Snake Body")
        print(f"  {self.colors['food']}●●{self.colors['reset']} Food")
        print()
        print(f"{self.colors['info']}The AI uses Q-Learning to learn:{self.colors['reset']}")
        print("  • Avoid walls and itself")
        print("  • Find and eat food")
        print("  • Maximize score")
        print()
        print(f"{self.colors['dim']}Starting training in 3 seconds...{self.colors['reset']}")
        time.sleep(3)

    def show_training_complete(self, agent, episodes):
        """Show training completion screen"""
        self.clear_screen()
        print()
        print(self.colors['score'] + "  ╔═══════════════════════════════════════════════════╗")
        print("  ║                                                   ║")
        print("  ║              TRAINING COMPLETE!                   ║")
        print("  ║                                                   ║")
        print("  ╚═══════════════════════════════════════════════════╝" + self.colors['reset'])
        print()

        if len(agent.episode_scores) > 0:
            total_episodes = len(agent.episode_scores)
            avg_score = sum(agent.episode_scores) / total_episodes
            max_score = max(agent.episode_scores)

            recent = min(100, total_episodes)
            recent_avg = sum(agent.episode_scores[-recent:]) / recent

            print(f"{self.colors['info']}Training Statistics:{self.colors['reset']}")
            print(f"  Total Episodes: {self.colors['white']}{total_episodes}{self.colors['reset']}")
            print(f"  Overall Avg Score: {self.colors['white']}{avg_score:.2f}{self.colors['reset']}")
            print(f"  Max Score: {self.colors['white']}{max_score}{self.colors['reset']}")
            print(f"  Last {recent} Avg: {self.colors['white']}{recent_avg:.2f}{self.colors['reset']}")
            print(f"  Q-Table Size: {self.colors['white']}{len(agent.q_table)}{self.colors['reset']}")
            print(f"  Final Epsilon: {self.colors['white']}{agent.epsilon:.3f}{self.colors['reset']}")
            print()

        print(f"{self.colors['dim']}Agent saved to 'snake_agent.pkl'{self.colors['reset']}")
        print()


def train_agent(episodes=500, render_every=1, delay=0.01):
    """
    Train the Q-learning agent

    Args:
        episodes: Number of training episodes
        render_every: Render every N episodes (1 = render all)
        delay: Delay between frames (seconds)
    """
    game = SnakeGame()
    agent = QLearningAgent()
    viz = Visualizer()

    try:
        viz.hide_cursor()
        viz.show_intro()

        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0
            done = False

            while not done:
                # Get action and take step
                action = agent.get_action(state, training=True)
                next_state, reward, done = game.step(action)
                total_reward += reward

                # Update Q-table
                agent.update(state, action, reward, next_state, done)
                state = next_state

                # Render if needed
                if episode % render_every == 0:
                    viz.draw_game(game, episode, agent, total_reward)
                    time.sleep(delay)

            # Record episode
            agent.record_episode(total_reward, game.score, game.steps)
            agent.decay_epsilon()

            # Print progress every 10 episodes
            if episode % 10 == 0 and episode % render_every != 0:
                viz.draw_game(game, episode, agent, total_reward)

        # Show completion
        viz.show_training_complete(agent, episodes)

        # Save agent
        agent.save()

    finally:
        viz.show_cursor()
        print(viz.colors['reset'])


def watch_agent(episodes=5, delay=0.05):
    """
    Watch a trained agent play

    Args:
        episodes: Number of episodes to watch
        delay: Delay between frames (seconds)
    """
    game = SnakeGame()
    agent = QLearningAgent()
    viz = Visualizer()

    # Load trained agent
    if not agent.load():
        print("No trained agent found! Train one first.")
        return

    try:
        viz.hide_cursor()

        print(f"\n{viz.colors['info']}Watching trained agent play...{viz.colors['reset']}\n")
        time.sleep(2)

        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0
            done = False

            while not done:
                # Get action (no exploration)
                action = agent.get_action(state, training=False)
                next_state, reward, done = game.step(action)
                total_reward += reward
                state = next_state

                # Render
                viz.draw_game(game, episode, agent, total_reward)
                time.sleep(delay)

            print(f"\n{viz.colors['score']}Episode {episode} finished! Score: {game.score}{viz.colors['reset']}")
            time.sleep(2)

    finally:
        viz.show_cursor()
        print(viz.colors['reset'])


def interactive_menu():
    """Show interactive menu"""
    viz = Visualizer()

    while True:
        viz.clear_screen()
        print()
        print(viz.colors['score'] + "  ╔═══════════════════════════════════════════════════╗")
        print("  ║                                                   ║")
        print("  ║        REINFORCEMENT LEARNING SNAKE GAME          ║")
        print("  ║                                                   ║")
        print("  ╚═══════════════════════════════════════════════════╝" + viz.colors['reset'])
        print()
        print(f"{viz.colors['info']}Choose an option:{viz.colors['reset']}")
        print()
        print(f"  {viz.colors['white']}1.{viz.colors['reset']} Train new agent (500 episodes, fast)")
        print(f"  {viz.colors['white']}2.{viz.colors['reset']} Train new agent (500 episodes, watch all)")
        print(f"  {viz.colors['white']}3.{viz.colors['reset']} Train new agent (2000 episodes, fast)")
        print(f"  {viz.colors['white']}4.{viz.colors['reset']} Watch trained agent play")
        print(f"  {viz.colors['white']}5.{viz.colors['reset']} Quick demo (50 episodes)")
        print(f"  {viz.colors['white']}6.{viz.colors['reset']} Exit")
        print()

        choice = input(f"{viz.colors['info']}Enter choice (1-6): {viz.colors['reset']}")

        if choice == '1':
            train_agent(episodes=500, render_every=10, delay=0.01)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '2':
            train_agent(episodes=500, render_every=1, delay=0.01)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '3':
            train_agent(episodes=2000, render_every=20, delay=0.005)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '4':
            watch_agent(episodes=5, delay=0.05)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '5':
            train_agent(episodes=50, render_every=1, delay=0.02)
            input(f"\n{viz.colors['info']}Press Enter to continue...{viz.colors['reset']}")
        elif choice == '6':
            print(f"\n{viz.colors['info']}Thanks for watching the AI learn!{viz.colors['reset']}\n")
            break
        else:
            print(f"{viz.colors['food']}Invalid choice!{viz.colors['reset']}")
            time.sleep(1)


if __name__ == "__main__":
    # Check if we should run in auto mode or interactive
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--train':
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
            train_agent(episodes=episodes, render_every=10, delay=0.01)
        elif sys.argv[1] == '--watch':
            watch_agent(episodes=5, delay=0.05)
        elif sys.argv[1] == '--demo':
            train_agent(episodes=50, render_every=1, delay=0.02)
    else:
        # Interactive menu
        interactive_menu()
