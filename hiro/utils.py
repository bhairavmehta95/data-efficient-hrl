import numpy as np

import torch
from torchvision import transforms

from os import listdir
from os.path import join, isdir

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

totensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def var(tensor):
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6, batch_size=100):
        self.storage = []
        self.maxsize = maxsize
        self.next_idx = 0
        self.batch_size = batch_size

    # Expects tuples of (x, x', g, u, r, d, x_seq, a_seq)
    def add(self, data):
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        x, y, g, u, r, d, x_seq, a_seq = [], [], [], [], [], [], [], []          

        for i in ind: 
            X, Y, G, U, R, D, obs_seq, acts = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

            # For off-policy goal correction
            x_seq.append(np.array(obs_seq, copy=False))
            a_seq.append(np.array(acts, copy=False))
        
        return np.array(x), np.array(y), np.array(g), \
            np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
            x_seq, a_seq


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape[0])).clip(-max_action, max_action)

        return action

class OUNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def perturb_action(self, action, max_action=np.inf):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return (self.X + action).clip(-max_action, max_action)

