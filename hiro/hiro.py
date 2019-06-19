import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from math import ceil
from os.path import join, exists
from os import makedirs

from hiro.models import ControllerActor, ControllerCritic, ManagerActor, ManagerCritic

totensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Manager(object):
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr, critic_lr, candidate_goals):
        self.actor = ManagerActor(state_dim, goal_dim, action_dim)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.candidate_goals = candidate_goals

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def value_estimate(self, state, goal, subgoal):
        state = state
        goal = goal
        subgoal = subgoal

        return self.critic(state, goal, subgoal)

    def actor_loss(self, state, goal):
        state = state
        goal = goal

        return -self.critic.Q1(state, goal, self.actor(state, goal)).mean()

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq):
        # TODO: Doesn't include subgoal transitions!!
        # return subgoals

        # new_subgoals = controller_policy.multi_subgoal_transition(x_seq, subgoals)
        
        first_x = [x[0] for x in x_seq] # First x
        last_x = [x[-1] for x in x_seq] # Last x

        diff_goal = (np.array(last_x) - np.array(first_x))[:, np.newaxis, :] # Shape: (batchsz, 1, subgoaldim)
        original_goal = np.array(subgoals)[:, np.newaxis, :] # Shape: (batchsz, 1, subgoaldim)
        random_goals = np.random.normal(loc=diff_goal, size=(batch_size, self.candidate_goals, original_goal.shape[-1]))

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)


        x_seq = np.array(x_seq)[:, :-1, :]
        a_seq = np.array(a_seq)
        seq_len = len(x_seq[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = candidates.shape[1]

        true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
        observations = x_seq.reshape((new_batch_sz,) + obs_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(x_seq, candidates[:, c])
            candidate = candidate.reshape(new_batch_sz, *obs_dim)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, iterations, 
              batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.
        for it in range(iterations):
            # Sample replay buffer
            x, y, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            sg = self.off_policy_corrections(controller_policy, batch_size, sgorig, xobs_seq, a_seq)

            state = get_tensor(x)
            next_state = get_tensor(y)
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            reward = get_tensor(r)
            done = get_tensor(1 - d)

            # Q target = reward + discount * Q(next_state, pi(next_state))
            target_Q1, target_Q2 = self.critic_target(next_state, goal, self.actor_target(next_state, goal))

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, goal)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_ManagerActor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_ManagerCritic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_ManagerActor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_ManagerCritic.pth' % (directory, filename)))


class Controller(object):
    def __init__(self, state_dim, goal_dim,
        action_dim, max_action, actor_lr, critic_lr, ctrl_rew_type
    ):
        self.actor = ControllerActor(state_dim, goal_dim, action_dim)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
            lr=critic_lr)

        self.subgoal_transition = self.hiro_subgoal_transition

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state, sg, to_numpy=True):
        state = get_tensor(state)
        sg = get_tensor(sg)

        if to_numpy:
            return self.actor(state, sg).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, sg).squeeze()

    def value_estimate(self, state, sg, action):
        state = get_tensor(state)
        sg = get_tensor(sg)
        action = get_tensor(action)

        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        state = get_tensor(state)
        sg = get_tensor(sg)

        return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()

    def hiro_subgoal_transition(self, state, subgoal, next_state):
        return state + subgoal - next_state

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0])[:, None] - states
        return subgoals

    def train(self, replay_buffer, iterations,
        batch_size=100, discount=0.99, tau=0.005):

        avg_act_loss, avg_crit_loss = 0., 0.

        for it in range(iterations):
            # Sample replay buffer
            x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            state = x
            action = u
            next_state = y
            done = get_tensor(1 - d)
            reward = get_tensor(r)

            next_g = get_tensor(self.subgoal_transition(state, sg, next_state))

            # Q target = reward + discount * Q(next_state, pi(next_state))
            target_Q1, target_Q2 = self.critic_target(get_tensor(next_state), next_g,
                                          self.actor_target(get_tensor(next_state), next_g))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) + self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, sg)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations 

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_ControllerActor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_ControllerCritic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_ControllerActor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_ControllerCritic.pth' % (directory, filename)))


