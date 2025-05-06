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
import imageio
import os
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict import TensorDict

# --- ENVIRONMENT SETUP ---
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
action_size = env.action_space.n
obs_shape = (4, 84, 84)  # 4 stacked frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess(obs, visualize=False):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = obs[36:, :]
    obs = cv2.resize(obs, (84, 84))
    
    if visualize:
        cv2.imshow("Grayscale Frame", obs)
        cv2.waitKey(1)
    
    obs = np.expand_dims(obs, axis=-1)
    return np.ascontiguousarray(obs, dtype=np.float32)

def skip_step(env, action, frame_stack, skip=4, visualize=False):
    total_reward = 0.0
    done = False
    info = None
    obs = None

    for _ in range(skip):
        obs, reward, step_done, info = env.step(action)
        total_reward += reward

        state = frame_stack.step(np.ascontiguousarray(obs), visualize=visualize)

        if step_done:
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
class RB:
    def __init__(self, capacity=100000, device='cuda'):
        self.device = device
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=capacity, device=device),
            collate_fn=None  # TensorDict handles batch collation itself
        )

    def add(self, state, action, reward, next_state, done):
        # Wrap transition in a TensorDict
        transition = TensorDict({
            "state": torch.as_tensor(state, dtype=torch.uint8, device=self.device),
            "action": torch.as_tensor(action, dtype=torch.int64, device=self.device),
            "reward": torch.as_tensor(reward, dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(next_state, dtype=torch.uint8, device=self.device),
            "done": torch.as_tensor(done, dtype=torch.float32, device=self.device),
        }, batch_size=[])
        self.buffer.add(transition)

    def sample(self, batch_size=32):
        batch = self.buffer.sample(batch_size)
        return (
            batch["state"].float() / 255.0,
            batch["action"],
            batch["reward"],
            batch["next_state"].float() / 255.0,
            batch["done"]
        )

    def __len__(self):
        return len(self.buffer)

# --- SIMPLE CNN DQN ---
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
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

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.00025)
        self.loss_fn = nn.MSELoss()
        self.replay = RB()
        self.gamma = 0.99

        self.epsilon_base = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.batch_size = 32
        self.tau = 0.005

    def get_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon_base:
            return random.randint(0, 6)

        state = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            q = self.q_net(state)
        return q.argmax().item()

    def decay_epsilon_base(self):
        if self.epsilon_base > self.epsilon_min:
            self.epsilon_base *= self.epsilon_decay


    def train(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

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
        self.decay_epsilon_base()
    
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def evaluate(agent, env, frame_stack, save_gif=False, gif_name="eval.gif"):
    state = env.reset()
    state = frame_stack.reset(np.ascontiguousarray(state))
    done = False
    total_reward = 0
    max_x_pos = 0
    step_count = 0


    frames = []

    while not done:
        action = agent.get_action(state,  deterministic=True)
        next_state, reward, done, info = skip_step(env, action, frame_stack)
        state = next_state
        step_count += 1
        total_reward += reward
        max_x_pos = max(max_x_pos, info['x_pos'])

        # 儲存畫面（灰階）
        if save_gif:
            frame = state[:, :, -1]
            frames.append(frame.astype(np.uint8))

    print(f"[Eval] Reward: {total_reward:.2f}, Max X: {max_x_pos}, Steps: {step_count}")

    if save_gif:
        import imageio, os
        os.makedirs("eval_gifs", exist_ok=True)
        path = f"eval_gifs/{gif_name}"
        with imageio.get_writer(path, mode='I', duration=0.1) as writer:
            for f in frames:
                writer.append_data(f)
        print(f"Saved GIF to {path}")

# --- TRAINING LOOP ---
agent = DQNAgent(action_size, epsilon_decay=0.99999975)
frame_stack = FrameStack()
episodes = 5000
max_steps = 4000
stagnation_limit = 500
evaluate(agent, env, frame_stack)

for ep in range(episodes):
    print(ep+1)
    sys.stdout.flush()
    state = env.reset()
    state = frame_stack.reset(np.ascontiguousarray(state))
    frames = []
    total_reward = 0
    step_count = 0
    max_x_pos = 0

    while True:
        action = agent.get_action(state)
        # env.render()
        next_state, reward, done, info = skip_step(
            env, action, frame_stack,
        )
        step_count += 1
        total_reward += reward

        if info["x_pos"] > max_x_pos:
            max_x_pos = info["x_pos"]
        
        if step_count % 50 == 0:
            print(f"Step: {step_count}, Action: {action}, Original Reward: {reward:.2f}, X-Pos: {info['x_pos']}, Max X-Pos: {max_x_pos}, Epsilon: {agent.epsilon_base:.3f}")
            sys.stdout.flush()

        agent.replay.add(state, action, reward, next_state, done)
        agent.train()

        frames.append(state[:, :, -1])
        state = next_state

        if done or info["flag_get"]:
            break
    
    print(f"Episode {ep+1}, Reward: {total_reward:.2f}, X-Pos: {info['x_pos']}, Max X-Pos: {max_x_pos}, Steps: {step_count}, Epsilon: {agent.epsilon_base:.3f}")
    sys.stdout.flush()
    
    if (ep+1) % 50 == 0 and ep+1 >= 50:
        evaluate(agent, env, frame_stack, save_gif=True, gif_name=f"episode_{ep+1:03d}.gif")

    if (ep+1) % 100 == 0:
        torch.save(agent.q_net.state_dict(), f"mario_dqn_{ep+1}.pth")
        print(f"Model saved at episode {ep+1}")
        sys.stdout.flush()
