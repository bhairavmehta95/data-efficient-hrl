import numpy as np

import torch
from torchvision import transforms
import pickle as pkl

from os import listdir
from os.path import join, isdir

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

totensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6, batch_size=100):
        self.storage = [[] for _ in range(8)]
        self.maxsize = maxsize
        self.next_idx = 0
        self.batch_size = batch_size

    # Expects tuples of (x, x', g, u, r, d, x_seq, a_seq)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, g, u, r, d, x_seq, a_seq = [], [], [], [], [], [], [], []          

        for i in ind: 
            X, Y, G, U, R, D, obs_seq, acts = (array[i] for array in self.storage)
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

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            y=self.storage[1], g=self.storage[2], u=self.storage[3],
                            r=self.storage[4], d=self.storage[5], xseq=self.storage[6],
                            aseq=self.storage[7])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data["idx"][0])
            self.storage = [data["x"], data["y"], data["g"], data["u"], data["r"],
                            data["d"], data["xseq"], data["aseq"]]
            self.storage = [list(l) for l in self.storage]

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

