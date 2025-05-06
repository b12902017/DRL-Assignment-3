import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque
import sys

# --- ENVIRONMENT SETUP ---
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
action_size = env.action_space.n
obs_shape = (4, 96, 96)  # 4 stacked frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

def preprocess(obs, visualize=False):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = obs[30:, :]
    obs = cv2.resize(obs, (96, 96))

    if visualize:
        cv2.imshow("Grayscale Frame", obs)
        cv2.waitKey(1)
    
    obs = np.expand_dims(obs, axis=-1)  # (96,96,1)
    return np.ascontiguousarray(obs, dtype=np.float32)

def skip_step(env, action, frame_stack, skip=4, initial_life=None, visualize=False):
    total_reward = 0.0
    done = False
    info = None
    obs = None

    for _ in range(skip):
        obs, reward, step_done, info = env.step(action)
        total_reward += reward

        state = frame_stack.step(np.ascontiguousarray(obs), visualize=visualize)

        dead = info['life'] < initial_life if initial_life is not None else False
        if step_done or dead:
            done = True
            break

    return state, total_reward, done, info

class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, obs):
        obs = preprocess(obs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=-1)

    def step(self, obs, visualize=False):
        obs = preprocess(obs, visualize=visualize)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=-1)

# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32) / 255.0,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32) / 255.0,
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# --- SIMPLE CNN DQN ---
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),   # (96x96, 4) -> (23x23)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # (10x10)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # (8x8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return self.net(x)

# --- AGENT ---
class DQNAgent:
    def __init__(self, action_size, epsilon_decay=0.99):
        self.action_size = action_size
        self.q_net = DQN(action_size).to(device)
        self.target_net = DQN(action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer()
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64
        self.tau = 0.005
        self.jump_actions = [2, 4, 5]
        # self.jump_hold = 0 

    def get_action(self, state, deterministic=False):
        '''
        if self.jump_hold > 0:
            self.jump_hold -= 1
            return self.last_jump_action
        '''

        if not deterministic and random.random() < self.epsilon:
            
            if random.random() < 0.3 and self.epsilon > 0.5:
                self.last_jump_action = random.choice(self.jump_actions)
                # self.jump_hold = random.randint(5, 10)  # random hold 5~10 frames
                return self.last_jump_action
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            q = self.q_net(state)
        return q.argmax().item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_net(states).gather(1, actions)
        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()
        # self.decay_epsilon()
    
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- TRAINING LOOP ---
agent = DQNAgent(action_size, epsilon_decay=0.997)
frame_stack = FrameStack()
episodes = 1000
max_steps = 4000
min_buffer_size = 5000
stagnation_limit = 500

for ep in range(episodes):
    state = env.reset()
    state = frame_stack.reset(np.ascontiguousarray(state))
    total_reward = 0
    done = False
    step_count = 0
    max_x_pos = 0
    stagnation_steps = 0
    prev_x = 0
    last_progress_x = 0

    pipe_positions = [40, 80, 120, 160]
    pipe_passed = [False] * len(pipe_positions)
    gap_positions = [190, 260, 310, 370, 430, 460]
    gap_passed = [False] * len(gap_positions)

    info = env.step(0)[-1]
    initial_life = info['life']

    while not done:
        action = agent.get_action(state)
        # env.render()
        next_state, reward, done, info = skip_step(env, action, frame_stack, initial_life=initial_life)
        # next_state = frame_stack.step(np.ascontiguousarray(next_state))
        total_reward += reward
        dead = info['life'] < initial_life
        
        # shaping
        original_reward = reward / 100.0
        shaped = 0.0

        # forward
        forward_reward = (info['x_pos'] - prev_x) * 0.05
        shaped += forward_reward
        prev_x = info['x_pos']

        # tubes and gaps
        for i, px in enumerate(pipe_positions):
            if not pipe_passed[i] and info['x_pos'] > px + 5:
                shaped += 1.0
                pipe_passed[i] = True
        for i, gx in enumerate(gap_positions):
            if not gap_passed[i] and info['x_pos'] > gx + 10:
                shaped += 1.2
                gap_passed[i] = True

        # checkpoints
        if info['x_pos'] >= last_progress_x + 100:
            shaped += 0.3
            last_progress_x = int(info['x_pos'] // 100) * 100

        # flags
        if done:
            if info.get('flag_get', False):
                shaped += 5.0
            else:
                shaped -= 1.0

        if stagnation_steps >= stagnation_limit:
            shaped -= 2.0

        # die
        if dead:
            shaped -= 5.0
            done = True

        reward = original_reward + shaped
        # reward = np.clip(reward, -1, 1)

        if info['x_pos'] > max_x_pos:
            max_x_pos = info['x_pos']
            stagnation_steps = 0
        else:
            stagnation_steps += 1
            if stagnation_steps >= stagnation_limit:
                done = True

        agent.replay.add(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        
    agent.decay_epsilon()
    truncation_reason = "stagnation" if stagnation_steps >= stagnation_limit else "max_steps" if step_count >= max_steps else "environment"
    print(f"Episode {ep+1}, Reward: {total_reward:.2f}, X-Pos: {info['x_pos']}, Max X-Pos: {max_x_pos}, Steps: {step_count}, Epsilon: {agent.epsilon:.3f}, Done: {truncation_reason}")
    sys.stdout.flush()
