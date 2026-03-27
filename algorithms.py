"""
Reinforcement Learning Algorithms
Comparative Analysis: Value Iteration, Q-Learning, SARSA
"""

import numpy as np
import time
from collections import defaultdict


class GridWorld:
    """Standard GridWorld Environment"""
    
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # UP, DOWN, LEFT, RIGHT
    ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    def __init__(self, size=8, obstacles=None, start=None, goal=None, seed=42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)
        
        self.start = start or (0, 0)
        self.goal = goal or (size - 1, size - 1)
        
        if obstacles is None:
            self.obstacles = self._generate_obstacles()
        else:
            self.obstacles = set(obstacles)
        
        self.state = self.start
        self.step_count = 0
        self.max_steps = size * size * 4
    
    def _generate_obstacles(self):
        obstacles = set()
        rng = np.random.RandomState(self.seed)
        num_obs = int(self.size * self.size * 0.15)
        for _ in range(num_obs):
            r, c = rng.randint(0, self.size, 2)
            pos = (r, c)
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        return obstacles
    
    def reset(self):
        self.state = self.start
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        dr, dc = self.ACTIONS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        
        # Boundary check
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.state = (nr, nc)
        
        # Reward structure
        if self.state == self.goal:
            reward = 100.0
            done = True
        elif self.step_count >= self.max_steps:
            reward = -10.0
            done = True
        else:
            reward = -1.0
            done = False
        
        return self.state, reward, done
    
    def get_all_states(self):
        states = []
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.obstacles:
                    states.append((r, c))
        return states
    
    def is_valid(self, state):
        r, c = state
        return (0 <= r < self.size and 0 <= c < self.size and 
                state not in self.obstacles)
    
    def get_transitions(self, state, action):
        """For model-based planning"""
        dr, dc = self.ACTIONS[action]
        nr, nc = state[0] + dr, state[1] + dc
        
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            next_state = (nr, nc)
        else:
            next_state = state
        
        if next_state == self.goal:
            reward = 100.0
            done = True
        else:
            reward = -1.0
            done = False
        
        return [(1.0, next_state, reward, done)]


class ValueIteration:
    """
    Model-Based Planning: Value Iteration
    Uses full knowledge of the environment model (MDP)
    """
    
    def __init__(self, env, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
        self.training_metrics = {
            'iterations': 0,
            'delta_history': [],
            'time_taken': 0,
            'convergence_iteration': 0,
            'episodes_to_solve': 0,
            'rewards_per_episode': [],
            'steps_per_episode': []
        }
    
    def train(self, callback=None):
        start_time = time.time()
        states = self.env.get_all_states()
        
        # Initialize V
        for s in states:
            self.V[s] = 0.0
        self.V[self.env.goal] = 0.0
        
        iteration = 0
        while True:
            delta = 0
            for s in states:
                if s == self.env.goal:
                    continue
                v = self.V[s]
                action_values = []
                for a in range(4):
                    transitions = self.env.get_transitions(s, a)
                    val = sum(prob * (r + self.gamma * (0 if done else self.V.get(ns, 0)))
                              for prob, ns, r, done in transitions)
                    action_values.append(val)
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            self.training_metrics['delta_history'].append(delta)
            iteration += 1
            
            if callback:
                callback(iteration, delta)
            
            if delta < self.theta:
                self.training_metrics['convergence_iteration'] = iteration
                break
            
            if iteration > 5000:
                break
        
        # Extract policy
        for s in states:
            if s == self.env.goal:
                self.policy[s] = 0
                continue
            action_values = []
            for a in range(4):
                transitions = self.env.get_transitions(s, a)
                val = sum(prob * (r + self.gamma * (0 if done else self.V.get(ns, 0)))
                          for prob, ns, r, done in transitions)
                action_values.append(val)
            self.policy[s] = int(np.argmax(action_values))
        
        self.training_metrics['iterations'] = iteration
        self.training_metrics['time_taken'] = time.time() - start_time
        
        # Simulate episodes with learned policy
        self._simulate_episodes(20)
        
        return self.training_metrics
    
    def _simulate_episodes(self, n=20):
        for _ in range(n):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done and steps < self.env.max_steps:
                action = self.policy.get(state, 0)
                state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
            self.training_metrics['rewards_per_episode'].append(total_reward)
            self.training_metrics['steps_per_episode'].append(steps)
        self.training_metrics['episodes_to_solve'] = n
    
    def get_action(self, state):
        return self.policy.get(state, 0)
    
    def get_value_grid(self):
        grid = np.zeros((self.env.size, self.env.size))
        for (r, c), v in self.V.items():
            grid[r][c] = v
        return grid


class QLearning:
    """
    Model-Free Learning: Q-Learning (Off-Policy TD)
    Learns directly from experience without environment model
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.Q = defaultdict(lambda: np.zeros(4))
        self.training_metrics = {
            'iterations': 0,
            'rewards_per_episode': [],
            'steps_per_episode': [],
            'epsilon_history': [],
            'q_value_history': [],
            'time_taken': 0,
            'convergence_iteration': 0,
            'delta_history': []
        }
    
    def train(self, callback=None):
        start_time = time.time()
        prev_max_q = 0
        
        for ep in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Epsilon-greedy
                if np.random.random() < self.epsilon:
                    action = np.random.randint(4)
                else:
                    action = int(np.argmax(self.Q[state]))
                
                next_state, reward, done = self.env.step(action)
                
                # Q-Learning update (off-policy: uses max Q of next state)
                best_next = np.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next * (1 - done)
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.training_metrics['rewards_per_episode'].append(total_reward)
            self.training_metrics['steps_per_episode'].append(steps)
            self.training_metrics['epsilon_history'].append(self.epsilon)
            
            max_q = max(np.max(v) for v in self.Q.values()) if self.Q else 0
            delta = abs(max_q - prev_max_q)
            self.training_metrics['delta_history'].append(delta)
            self.training_metrics['q_value_history'].append(max_q)
            prev_max_q = max_q
            
            if callback:
                callback(ep + 1, delta)
        
        self.training_metrics['iterations'] = self.episodes
        self.training_metrics['time_taken'] = time.time() - start_time
        self.training_metrics['convergence_iteration'] = self._find_convergence()
        
        return self.training_metrics
    
    def _find_convergence(self):
        rewards = self.training_metrics['rewards_per_episode']
        window = 50
        if len(rewards) < window:
            return len(rewards)
        for i in range(window, len(rewards)):
            avg = np.mean(rewards[i-window:i])
            if avg > -50:
                return i
        return len(rewards)
    
    def get_action(self, state):
        return int(np.argmax(self.Q[state]))
    
    def get_value_grid(self):
        grid = np.zeros((self.env.size, self.env.size))
        for (r, c) in self.env.get_all_states():
            grid[r][c] = np.max(self.Q[(r, c)])
        return grid


class SARSA:
    """
    Model-Free Learning: SARSA (On-Policy TD)
    Learns from the actual policy being followed (including exploratory actions)
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.Q = defaultdict(lambda: np.zeros(4))
        self.training_metrics = {
            'iterations': 0,
            'rewards_per_episode': [],
            'steps_per_episode': [],
            'epsilon_history': [],
            'q_value_history': [],
            'time_taken': 0,
            'convergence_iteration': 0,
            'delta_history': []
        }
    
    def _choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.Q[state]))
    
    def train(self, callback=None):
        start_time = time.time()
        prev_max_q = 0
        
        for ep in range(self.episodes):
            state = self.env.reset()
            action = self._choose_action(state)
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self._choose_action(next_state)
                
                # SARSA update (on-policy: uses actual next action)
                td_target = reward + self.gamma * self.Q[next_state][next_action] * (1 - done)
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.training_metrics['rewards_per_episode'].append(total_reward)
            self.training_metrics['steps_per_episode'].append(steps)
            self.training_metrics['epsilon_history'].append(self.epsilon)
            
            max_q = max(np.max(v) for v in self.Q.values()) if self.Q else 0
            delta = abs(max_q - prev_max_q)
            self.training_metrics['delta_history'].append(delta)
            self.training_metrics['q_value_history'].append(max_q)
            prev_max_q = max_q
            
            if callback:
                callback(ep + 1, delta)
        
        self.training_metrics['iterations'] = self.episodes
        self.training_metrics['time_taken'] = time.time() - start_time
        self.training_metrics['convergence_iteration'] = self._find_convergence()
        
        return self.training_metrics
    
    def _find_convergence(self):
        rewards = self.training_metrics['rewards_per_episode']
        window = 50
        if len(rewards) < window:
            return len(rewards)
        for i in range(window, len(rewards)):
            avg = np.mean(rewards[i-window:i])
            if avg > -50:
                return i
        return len(rewards)
    
    def get_action(self, state):
        return int(np.argmax(self.Q[state]))
    
    def get_value_grid(self):
        grid = np.zeros((self.env.size, self.env.size))
        for (r, c) in self.env.get_all_states():
            grid[r][c] = np.max(self.Q[(r, c)])
        return grid


def get_comparison_stats(metrics_dict):
    """Generate comparative statistics for all three algorithms"""
    stats = {}
    for name, m in metrics_dict.items():
        rewards = m['rewards_per_episode']
        steps = m['steps_per_episode']
        stats[name] = {
            'avg_reward': np.mean(rewards[-50:]) if rewards else 0,
            'best_reward': max(rewards) if rewards else 0,
            'avg_steps': np.mean(steps[-50:]) if steps else 0,
            'best_steps': min(steps) if steps else 0,
            'time_taken': m['time_taken'],
            'convergence': m['convergence_iteration'],
            'total_episodes': m['iterations']
        }
    return stats
