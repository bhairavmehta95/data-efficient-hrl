import numpy as np
import torch
import os
from math import ceil
import gym

from tensorboardX import SummaryWriter

import torch.nn.functional as F

import hiro.utils as utils
import hiro.hiro as hiro

from envs import ParticleEnv

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, writer, manager_policy, controller_policy, calculate_controller_reward, 
    ctrl_rew_scale, manager_propose_frequency=10, eval_idx=0, eval_episodes=5
    ):
    print("Starting evaluation number {}...".format(eval_idx))

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        avg_step_count = 0
        global_steps = 0
        for eval_ep in range(eval_episodes):
            obs, s, goalobs, g = env.reset()
            z = controller_policy.get_encoding(obs)
            done = False
            step_count = 0
            env_goals_achieved = 0

            recon = controller_policy.get_reconstruction(obs)
            goalrecon = controller_policy.get_reconstruction(goalobs)

            if eval_ep == 0:
                writer.add_image('eval_images/original', obs, eval_idx * eval_episodes + eval_ep)
                writer.add_image('eval_images/goal', goalobs, eval_idx * eval_episodes + eval_ep)
                writer.add_image('eval_images/reconstructed', recon, eval_idx * eval_episodes + eval_ep)
                writer.add_image('eval_images/goal_reconstructed', goalrecon, eval_idx * eval_episodes + eval_ep)

            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(obs, goalobs)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(obs, subgoal)
                obstup, reward, done, _ = env.full_step(action)

                # See if the environment goal was achieved
                if done: env_goals_achieved += 1

                if step_count + 1 == env.max_steps:
                    done = True

                obs = obstup[0]
                # Update subgoal
                next_z = controller_policy.get_encoding(obs)
                subgoal = controller_policy.subgoal_transition(z, subgoal, next_z)
                controller_rew = calculate_controller_reward(z, subgoal, next_z, ctrl_rew_scale)

                z = next_z

                avg_reward += reward
                avg_controller_rew += controller_rew

            avg_step_count += step_count

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count /= eval_episodes
        avg_env_finish = env_goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes: {} \nAvg Ctrl Reward: {}".format(eval_episodes, avg_reward, avg_controller_rew))
        print('Average Steps to finish: {}'.format(step_count))
        print("---------------------------------------")

        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish

def hiro_controller_reward(z, subgoal, next_z, scale):
    return -1 * np.linalg.norm(z + subgoal - next_z) * scale

def run_hiro(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.log_file)):
        os.makedirs(os.path.join(args.log_dir, args.log_file))

    args.vae_dir = os.path.join(args.vae_dir, args.log_file)
    if not os.path.exists(args.vae_dir):
        os.makedirs(args.vae_dir)
    
    env = gym.make(args.env_name)
    obs = env.reset()

    goal = obs['desired_goal']
    state = obs['observation']

    # Write Hyperparameters to file
    print("---------------------------------------")
    print("Current Arguments:")
    with open(os.path.join(args.log_dir, args.log_file, "hps.txt"), 'w') as f:
        for arg in vars(args):
            print("{}: {}".format(arg, getattr(args, arg)))
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    print("---------------------------------------\n")

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.log_file))
    torch.cuda.set_device(0)

    env_name = type(env).__name__
    file_name = 'hiro_{}'.format(env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = s.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = int(env.action_space.high[0])

    # Initialize policy, replay buffers
    controller_policy = hiro.Controller(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        vae_lr=args.vae_lr,
        vae_beta1=args.vae_beta1,
        vae_beta2=args.vae_beta2,
        ctrl_rew_type=args.ctrl_rew_type
    )

    manager_policy = hiro.Manager(
        state_dim=state_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals
    )

    calculate_controller_reward = hiro_controller_reward

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(state_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size)

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_vae = 0
    timesteps_since_manager = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []


    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0:
                print('Training Controller...')
                ctrl_act_loss, ctrl_crit_loss, avg_mu, avg_logvar = controller_policy.train(controller_buffer, episode_timesteps,
                    args.ctrl_batch_size, args.discount, args.ctrl_tau)

                writer.add_scalar('data/controller_actor_loss', ctrl_act_loss, total_timesteps)
                writer.add_scalar('data/controller_critic_loss', ctrl_crit_loss, total_timesteps)

                writer.add_scalar('data/controller_ep_rew', episode_reward, total_timesteps)
                writer.add_scalar('data/manager_ep_rew', manager_transition[-5], total_timesteps)

                # Train Manager
                if timesteps_since_manager >= args.train_manager_freq:
                    print('Training Manager...')
                    
                    # Set to evaluation mode for batchnorm
                    manager_policy.set_train()
                
                    timesteps_since_manager = 0
                    man_act_loss, man_crit_loss, sg_orig, sg = manager_policy.train(controller_policy, 
                        manager_buffer, ceil(episode_timesteps / args.train_manager_freq) , args.man_batch_size, args.discount, args.man_tau)

                    writer.add_scalar('data/manager_actor_loss', man_act_loss, total_timesteps)
                    writer.add_scalar('data/manager_critic_loss', man_crit_loss, total_timesteps)
                    writer.add_scalar('debug/num_minibatches_manager', ceil(episode_timesteps / args.train_manager_freq), total_timesteps)
                    
                    if total_timesteps % args.train_manager_freq * 10 == 0:
                        writer.add_image('images/manager_original_sg', sg_orig, total_timesteps)
                        writer.add_image('images/manager_correct_sg', sg, total_timesteps)
                        writer.add_image('images/goal', goalobs, total_timesteps)
                        writer.add_image('images/current', obs, total_timesteps)

                    # Reset to training mode for batchnorm
                    manager_policy.set_eval()

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish = evaluate_policy(env, writer, manager_policy, controller_policy, calculate_controller_reward, args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations))

                    writer.add_scalar('eval/avg_ep_rew', avg_ep_rew, total_timesteps)
                    writer.add_scalar('eval/avg_controller_rew', avg_controller_rew, total_timesteps)
                    writer.add_scalar('eval/avg_steps_to_finish', avg_steps, total_timesteps)
                    writer.add_scalar('eval/perc_env_goal_achieved', avg_env_finish, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])

                    if args.save_models:
                        controller_policy.save(file_name+'_controller', directory="./pytorch_models")
                        manager_policy.save(file_name+'_manager', directory="./pytorch_models")

                    np.save("./results/%s" % (file_name), evaluations)

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
            episode_num += 1

            # Create new manager transition
            subgoal = manager_policy.sample_goal(state, goal)

            timesteps_since_subgoal = 0
 
            # Create a high level transition
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        # TODO: Scale action to environment
        action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, max_action)

        # Perform action, get (nextst, r, d)
        next_tup, manager_reward, env_done, _ = env.step(action)

        # Update cumulative reward (env. reward) for manager
        manager_transition[4] += manager_reward * args.man_rew_scale

        # Process
        next_goal = obs['desired_goal']
        next_state = obs['observation']

        # Append low level sequence for off policy correction
        manager_transition[-1].append(action)
        manager_transition[-2].append(next_state)

        # Calculate reward, transition subgoal
        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

        # Is the episode over?
        if env_done or episode_timesteps + 1 == env.max_steps:
            done = True

        episode_reward += controller_reward

        # Store low level transition
        controller_buffer.add(
            (
                state, next_state, subgoal, \
                action, controller_reward, float(done), \
                [], []
            )
        )

        # Update state parameters
        obs = next_obs
        state = next_state

        # Update counters
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_vae += 1
        timesteps_since_subgoal += 1

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            # Finish, add transition
            manager_transition[1] = state
            manager_transition[5] = float(True)

            manager_buffer.add(manager_transition)

            subgoal = manager_policy.sample_goal(state, goal)
            subgoal = man_noise.perturb_action(subgoal, max_action=np.inf)

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
