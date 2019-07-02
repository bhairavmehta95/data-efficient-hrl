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

from hiro.models import ControllerActor, \
    ControllerCritic, \
    ManagerActor,\
    ManagerCritic

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
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5):
        self.scale = scale
        self.actor = ManagerActor(state_dim, goal_dim, action_dim,
                                  scale=scale)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim,
                                         scale=scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=0.0001)

        self.action_norm_reg = 0

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

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
        return self.critic(state, goal, subgoal)

    def actor_loss(self, state, goal):
        actions = self.actor(state, goal)
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        return eval + norm

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq,):
        # TODO: Doesn't include subgoal transitions!!
        # return subgoals

        # new_subgoals = controller_policy.multi_subgoal_transition(x_seq, subgoals)
        first_x = [x[0] for x in x_seq] # First x
        last_x = [x[-1] for x in x_seq] # Last x

        # Shape: (batchsz, 1, subgoaldim)
        diff_goal = (np.array(last_x) -
                     np.array(first_x))[:, np.newaxis, :self.action_dim]

        # Shape: (batchsz, 1, subgoaldim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

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
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(x_seq, candidates[:, c])
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)
        print(diff_goal[0, 0])
        print(original_goal[0, 0])
        print(candidates[0, max_indices[0]])

        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, iterations, 
              batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.
        for it in range(iterations):
            # Sample replay buffer
            x, y, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            if self.correction:
                sg = self.off_policy_corrections(controller_policy, batch_size,
                                                 sgorig, xobs_seq, a_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            reward = get_tensor(r)
            done = get_tensor(1 - d)

            # Q target = reward + discount * Q(next_state, pi(next_state))
            noise = torch.FloatTensor(sgorig).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)
            print(next_action[0])

            target_Q1, target_Q2 = self.critic_target(next_state, goal,
                                          next_action)
            # target_Q1, target_Q2 = self.critic_target(next_state, goal, self.actor_target(next_state, goal))

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

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
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir):
        torch.save(self.actor.state_dict(), '%s/ManagerActor.pth' % (dir))
        torch.save(self.critic.state_dict(), '%s/ManagerCritic.pth' % (dir))
        torch.save(self.actor_target.state_dict(), '%s/ManagerActorTarget.pth' % (dir))
        torch.save(self.critic_target.state_dict(), '%s/ManagerCriticTarget.pth' % (dir))
        torch.save(self.actor_optimizer.state_dict(), '%s/ManagerActorOptim.pth' % (dir))
        torch.save(self.critic_optimizer.state_dict(), '%s/ManagerCriticOptim.pth' % (dir))

    def load(self, dir):
        self.actor.load_state_dict(torch.load('%s/ManagerActor.pth' % (dir)))
        self.critic.load_state_dict(torch.load('%s/ManagerCritic.pth' % (dir)))
        self.actor_target.load_state_dict(torch.load('%s/ManagerActorTarget.pth' % (dir)))
        self.critic_target.load_state_dict(torch.load('%s/ManagerCriticTarget.pth' % (dir)))
        self.actor_optimizer.load_state_dict(torch.load('%s/ManagerActorOptim.pth' % (dir)))
        self.critic_optimizer.load_state_dict(torch.load('%s/ManagerCriticOptim.pth' % (dir)))


class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr,
                 critic_lr, ctrl_rew_type, repr_dim=15, no_xy=True,
                 policy_noise=0.2, noise_clip=0.5,
    ):
        self.actor = ControllerActor(state_dim, goal_dim, action_dim,
                                     scale=max_action)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim,
                                            scale=max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
            lr=critic_lr, weight_decay=0.0001)

        self.no_xy = no_xy

        self.subgoal_transition = self.hiro_subgoal_transition

        if torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic = self.critic.cuda()
            self.critic_target = self.critic_target.cuda()

        self.criterion = nn.SmoothL1Loss()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def select_action(self, state, sg, to_numpy=True):
        state = get_tensor(state)
        sg = get_tensor(sg)
        state = self.clean_obs(state)

        if to_numpy:
            return self.actor(state, sg).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, sg).squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()

    def hiro_subgoal_transition(self, state, subgoal, next_state):
        if len(state.shape) == 1:  # check if batched
            return state[:self.goal_dim] + subgoal - next_state[:self.goal_dim]
        else:
            return state[:, :self.goal_dim] + subgoal -\
                   next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals

    def train(self, replay_buffer, iterations,
        batch_size=100, discount=0.99, tau=0.005):

        avg_act_loss, avg_crit_loss = 0., 0.

        for it in range(iterations):
            # Sample replay buffer
            x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            next_g = get_tensor(self.subgoal_transition(x, sg, y))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y))

            # Q target = reward + discount * Q(next_state, pi(next_state))
            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_g,
                                          next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

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

    def save(self, dir):
        torch.save(self.actor.state_dict(), '%s/ControllerActor.pth' % (dir))
        torch.save(self.critic.state_dict(), '%s/ControllerCritic.pth' % (dir))
        torch.save(self.actor_target.state_dict(), '%s/ControllerActorTarget.pth' % (dir))
        torch.save(self.critic_target.state_dict(), '%s/ControllerCriticTarget.pth' % (dir))
        torch.save(self.actor_optimizer.state_dict(), '%s/ControllerActorOptim.pth' % (dir))
        torch.save(self.critic_optimizer.state_dict(), '%s/ControllerCriticOptim.pth' % (dir))

    def load(self, dir):
        self.actor.load_state_dict(torch.load('%s/ControllerActor.pth' % (dir)))
        self.critic.load_state_dict(torch.load('%s/ControllerCritic.pth' % (dir)))
        self.actor_target.load_state_dict(torch.load('%s/ControllerActorTarget.pth' % (dir)))
        self.critic_target.load_state_dict(torch.load('%s/ControllerCriticTarget.pth' % (dir)))
        self.actor_optimizer.load_state_dict(torch.load('%s/ControllerActorOptim.pth' % (dir)))
        self.critic_optimizer.load_state_dict(torch.load('%s/ControllerCriticOptim.pth' % (dir)))
