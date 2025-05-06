import gym
import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque

def preprocess(obs, visualize=False):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = obs[36:, :]
    obs = cv2.resize(obs, (84, 84))
    
    if visualize:
        cv2.imshow("Grayscale Frame", obs)
        cv2.waitKey(1)
    
    obs = np.expand_dims(obs, axis=-1)
    return np.ascontiguousarray(obs, dtype=np.float32)

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
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC → NCHW
        return self.net(x)

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frame_stack = FrameStack(num_frames=4)
        self.step_count = 0
        self.q_net = DQN(self.action_space.n)
        self.q_net.load_state_dict(torch.load('mario_dq_200.pth', map_location='cpu'))
        self.q_net.eval()
        self.start = True
        self.prev_action = None

    def act(self, observation):
        if self.start:
            state = self.frame_stack.reset(observation)
            self.start = False
        else:
            state = self.frame_stack.step(observation)
        
        if self.step_count % 4 == 0:
            state = torch.FloatTensor(state).unsqueeze(0) / 255.0
            with torch.no_grad():
                q = self.q_net(state)
            action = q.argmax().item()
            self.prev_action = action
        else:
            action = self.prev_action

        self.step_count += 1
        return action