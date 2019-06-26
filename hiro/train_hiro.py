import numpy as np
import torch
import os
from math import ceil
import pickle as pkl
import gym

from tensorboardX import SummaryWriter

from scipy.special import huber

import torch.nn.functional as F

import hiro.utils as utils
import hiro.hiro as hiro

from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, writer, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5,):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()

            goal = obs['desired_goal']
            state = obs['observation']

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal)
                new_obs, reward, done, _ = env.step(action)
                # See if the environment goal was achieved
                if reward >= env.distance_threshold:
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True


                goal = new_obs['desired_goal']
                new_state = new_obs['observation']

                # Update subgoal
                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)

                state = new_state

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes: {} \nAvg Ctrl Reward: {}".format(eval_episodes, avg_reward,
                                                                             avg_controller_rew))
        print("Goals achieved: {}%".format(100*goals_achieved/eval_episodes))
        print('Average Steps to finish: {}'.format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


def hiro_controller_reward(z, subgoal, next_z, scale):
    reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
    return reward


def run_hiro(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.log_file)):
        can_load = False
        os.makedirs(os.path.join(args.log_dir, args.log_file))
    else:
        can_load = True
        print("Existing directory found; may be able to load weights.")
    output_dir = os.path.join(args.log_dir, args.log_file)
    print("Logging in {}".format(output_dir))

    if "-v" in args.env_name:
        env = gym.make(args.env_name)
        env.env.reward_type = args.reward_type
        env.distance_threshold = env.env.distance_threshold
        max_action = np.array([1.54302745e+00, 1.21865324e+00, 9.98163424e-01, 1.97805133e-04,
                               7.15193042e-05, 2.56647627e-02, 2.30302501e-02, 2.13756120e-02,
                               1.19019512e-02, 6.31742249e-03])
        min_action = np.array(
            [7.95019864e-01, - 5.56192570e-02, 3.32176206e-01, 0.00000000e+00, 0.00000000e+00, - 2.58566763e-02,
             - 2.46581777e-02, - 1.77669761e-02, - 1.13476014e-02, - 5.08970149e-04])
        scale = max_action - min_action
    else:
        env = EnvWithGoal(create_maze_env(args.env_name), args.env_name)

        scale = np.array([10, 10, 0.5, 1, 1, 1] + [60]*3 + [40]*3
                         + [60]*3 + [40]*3
                         + [60]*3 + [40]*3
                         + [60]*3 + [40]*3)

    obs = env.reset()

    goal = obs['desired_goal']
    state = obs['observation']

    # # Write Hyperparameters to file
    # print("---------------------------------------")
    # print("Current Arguments:")
    # with open(os.path.join(args.log_dir, args.log_file, "hps.txt"), 'w') as f:
    #     for arg in vars(args):
    #         print("{}: {}".format(arg, getattr(args, arg)))
    #         f.write("{}: {}\n".format(arg, getattr(args, arg)))
    # print("---------------------------------------\n")

    writer = SummaryWriter(logdir=os.path.join(args.log_dir, args.log_file))
    torch.cuda.set_device(0)

    env_name = type(env).__name__
    file_name = 'hiro_{}'.format(env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]

    if args.ctrl_direct_rewards:
        controller_g_dim = state_dim + goal_dim
    else:
        controller_g_dim = state_dim

    max_action = float(env.action_space.high[0])

    # Initialize policy, replay buffers
    controller_policy = hiro.Controller(
        state_dim=state_dim,
        goal_dim=controller_g_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        ctrl_rew_type=args.ctrl_rew_type
    )

    manager_policy = hiro.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=state_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=scale
    )
    calculate_controller_reward = hiro_controller_reward

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size)

    if can_load and args.load:
        try:
            manager_policy.load(output_dir)
            controller_policy.load(output_dir)
            manager_buffer.load(os.path.join(output_dir, "mbuf.npz"))
            controller_buffer.load(os.path.join(output_dir, "cbuf.npz"))
            with open(os.path.join(output_dir, "iter.pkl"), "rb") as f:
                iter = pkl.load(f) + 1
            print("Loaded successfully")
            just_loaded = True
        except Exception as e:
            iter = 0
            just_loaded = False
            print(e, "Not loading")
    else:
        iter = 0
        just_loaded = False

    # Logging Parameters
    total_timesteps = iter
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    while total_timesteps < args.max_timesteps:
        if args.save_every > 0 and (total_timesteps + 1) % args.save_every == 0:
            print("Saving")
            controller_policy.save(output_dir)
            manager_policy.save(output_dir)
            manager_buffer.save(os.path.join(output_dir, "mbuf.npz"))
            controller_buffer.save(os.path.join(output_dir, "cbuf.npz"))
            with open(os.path.join(output_dir, "iter.pkl"), "wb") as f:
                pkl.dump(total_timesteps, f)

        if done:
            if total_timesteps != 0 and not just_loaded:
                # print('Training Controller...')
                ctrl_act_loss, ctrl_crit_loss = controller_policy.train(controller_buffer, episode_timesteps,
                                                                        args.ctrl_batch_size, args.ctrl_discount,
                                                                        args.ctrl_tau)

                writer.add_scalar('data/controller_actor_loss', ctrl_act_loss, total_timesteps)
                writer.add_scalar('data/controller_critic_loss', ctrl_crit_loss, total_timesteps)

                writer.add_scalar('data/controller_ep_rew', episode_reward, total_timesteps)
                writer.add_scalar('data/manager_ep_rew', manager_transition[4], total_timesteps)

                # Train Manager
                if timesteps_since_manager >= args.train_manager_freq:
                    # print('Training Manager...')

                    timesteps_since_manager = 0
                    man_act_loss, man_crit_loss = manager_policy.train(controller_policy, manager_buffer,
                                                                       ceil(episode_timesteps/args.train_manager_freq),
                                                                       args.man_batch_size, args.discount, args.man_tau)

                    writer.add_scalar('data/manager_actor_loss', man_act_loss, total_timesteps)
                    writer.add_scalar('data/manager_critic_loss', man_crit_loss, total_timesteps)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish =\
                        evaluate_policy(env, writer, manager_policy, controller_policy, calculate_controller_reward,
                                        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations))

                    writer.add_scalar('eval/avg_ep_rew', avg_ep_rew, total_timesteps)
                    writer.add_scalar('eval/avg_controller_rew', avg_controller_rew, total_timesteps)
                    writer.add_scalar('eval/avg_steps_to_finish', avg_steps, total_timesteps)
                    writer.add_scalar('eval/perc_env_goal_achieved', avg_env_finish, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])

                    if args.save_models:
                        controller_policy.save(file_name+'_controller', directory="./pytorch_models")
                        manager_policy.save(file_name+'_manager', directory="./pytorch_models")

                    np.save("./results/%s" % file_name, evaluations)

                # Process final state/obs, store manager transition, if it was not just created
                if len(manager_transition[-2]) != 1:                    
                    manager_transition[1] = state
                    manager_transition[5] = float(True)

                    # Every manager transition should have same length of sequences
                    if len(manager_transition[-2]) <= args.manager_propose_freq:
                        while len(manager_transition[-2]) <= args.manager_propose_freq:
                            manager_transition[-1].append(np.inf)
                            manager_transition[-2].append(state)

                    manager_buffer.add(manager_transition)

            # Reset environment
            obs = env.reset()
            goal = obs['desired_goal']
            state = obs['observation']
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            # Create new manager transition
            subgoal = manager_policy.sample_goal(state, goal)

            timesteps_since_subgoal = 0
 
            # Create a high level transition
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        # TODO: Scale action to environment
        if args.ctrl_direct_rewards:
            action = controller_policy.select_action(state, np.concatenate([subgoal, goal], -1))
        else:
            action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, max_action)

        # Perform action, get (nextst, r, d)
        next_tup, manager_reward, env_done, _ = env.step(action)

        # Update cumulative reward (env. reward) for manager
        manager_transition[4] += manager_reward * args.man_rew_scale

        # Process
        next_goal = next_tup['desired_goal']
        next_state = next_tup['observation']

        # Append low level sequence for off policy correction
        manager_transition[-1].append(action)
        manager_transition[-2].append(next_state)

        # Calculate reward, transition subgoal
        # print(np.sum(np.abs(state - next_state)), subgoal)

        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

        if args.ctrl_direct_rewards:
            controller_reward = controller_reward + manager_reward
            controller_goal = np.concatenate([subgoal, goal], -1)
        else:
            controller_goal = subgoal
        # Is the episode over?
        if env_done:
            done = True

        episode_reward += controller_reward

        # Store low level transition
        controller_buffer.add(
            (state, next_state, controller_goal, action, controller_reward, float(done), [], [])
        )

        # Update state parameters
        state = next_state
        goal = next_goal

        # Update counters
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            # Finish, add transition
            manager_transition[1] = state
            manager_transition[5] = float(True)

            manager_buffer.add(manager_transition)

            subgoal = manager_policy.sample_goal(state, goal)
            subgoal = man_noise.perturb_action(subgoal, max_action=200)

            # Reset number of timesteps since we sampled a subgoal
            timesteps_since_subgoal = 0

            # Create a high level transition
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

    # Final evaluation
    evaluations.append([evaluate_policy(env, writer, manager_policy, controller_policy,
                                        calculate_controller_reward, args.ctrl_rew_scale,
                                        args.manager_propose_freq, len(evaluations))])

    if args.save_models:
        controller_policy.save(file_name+'_controller', directory="./pytorch_models")
        manager_policy.save(file_name+'_manager', directory="./pytorch_models")

    np.save("./results/%s" % (file_name), evaluations)
