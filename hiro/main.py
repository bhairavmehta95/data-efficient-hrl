import argparse
import numpy as np
from hiro.train_hiro import run_hiro


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)                      # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--eval_freq", default=5e3, type=float)             # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=float)         # Max time steps to run environment for
parser.add_argument("--save_models", action="store_true")               # Whether or not models are saved
parser.add_argument("--env_name", default="Pendulum-v1", type=str)         # Environment name
parser.add_argument("--log_dir", default="logs/", type=str)             # Where logs are saved
parser.add_argument("--log_file", default="hiro", type=str)             # Where logs are saved
parser.add_argument("--random_params", action="store_true")             # Run HP search

# hiro Specific Params
parser.add_argument("--manager_propose_freq", default=10, type=int)     # #of env. steps at which we propose subgoals
parser.add_argument("--train_manager_freq", default=10, type=int)       # #of env. steps at which we train manager
parser.add_argument("--tune_vae_freq", default=10e3, type=int)           # #of env. steps at which we tune VAE
parser.add_argument("--discount", default=0.99, type=float)             # Discount factor
parser.add_argument("--z_dim", default=8, type=int)                    # Subgoal dimensionality

# Manager Parameters
parser.add_argument("--man_tau", default=0.001, type=float)              # Manager Target network update rate
parser.add_argument("--man_batch_size", default=128, type=int)          # Batch size for both actor and critic
parser.add_argument("--man_buffer_size", default=2e6, type=int)         # Replay Buffer size
parser.add_argument("--man_rew_scale", default=0.1, type=float)         # Reward Scaling
parser.add_argument("--man_act_lr", default=1e-4, type=float)           # Actor Learning Rate
parser.add_argument("--man_crit_lr", default=1e-3, type=float)          # Critic Learning Rate
parser.add_argument("--man_last_layer", default="fc", type=str)          # Critic Learning Rate
parser.add_argument("--candidate_goals", default=3, type=int)          # Critic Learning Rate

# Controller Parameters
parser.add_argument("--ctrl_tau", default=0.001, type=float)             # Controller Target network update rate
parser.add_argument("--ctrl_batch_size", default=128, type=int)         # Batch size for both actor and critic
parser.add_argument("--ctrl_buffer_size", default=2e6, type=int)        # Replay Buffer size
parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)        # Reward Scaling
parser.add_argument("--ctrl_rew_type", default="rig", type=str)         # What type of rew to use
parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)          # Actor Learning Rate
parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)         # Critic Learning Rate

# VAE Parameters
parser.add_argument("--pretrain_bvae", action="store_true")             # Pretrain the BVAE
parser.add_argument("--vae_data_dir", 
    default="pytorch-models/vae", type=str)                             # Where data is stored
parser.add_argument("--vae_dir", 
    default="pytorch-models/vae", type=str)                             # Directory to store VAE (Relative)
parser.add_argument("--vae_lr", default=1e-3, type=float)               # VAE Learning Rate
parser.add_argument("--vae_batch_size", default=128, type=int)          # VAE Batch Size
parser.add_argument("--vae_num_epochs", default=4, type=int)            # VAE # Epochs
parser.add_argument("--beta", default=1, type=float)                    # Beta in BVAE               
parser.add_argument("--vae_beta1", default=0.9, type=float)             # Beta1 in Adam Optimizer
parser.add_argument("--vae_beta2", default=0.999, type=float)           # Beta2 in Adam Optimizer
parser.add_argument("--load_pretrained", action="store_true")           # Is there a pretrained BVAE
parser.add_argument("--recon_loss_type", default="gaussian", type=str)  # bernoulli or gaussian recon. loss
parser.add_argument("--vae_pretrain_epochs", default=30, type=int)       # Num epochs to pretrain BVAE 

# Noise Parameters
parser.add_argument("--noise_type", default="ou", type=str)
parser.add_argument("--ctrl_noise_sigma", default=0.1, type=float)               # Std of Gaussian exploration noise
parser.add_argument("--man_noise_sigma", default=0.25, type=float)               # Std of Gaussian exploration noise

# Run the algorithm
args = parser.parse_args()

if args.random_params:
    args.z_dim = int(np.random.choice([32, 16, 8], p=[0.1, 0.4, 0.5])) 
    args.manager_propose_freq = np.random.choice([10, 5, 20], p=[0.6, 0.1, 0.3])
    args.train_manager_freq = args.manager_propose_freq
    # args.tune_vae_freq = np.random.choice([5e3, 7500, 10e3])
    args.noise_type = str(np.random.choice(['ou', 'normal']))
    
    args.ctrl_batch_size = np.random.choice([64, 128])
    args.ctrl_tau = np.random.choice([0.005, 0.001])
    args.ctrl_noise_sigma = np.random.choice([0.1, 0.2])
    args.ctrl_act_lr = np.random.choice([1e-3, 1e-4])
    
    args.man_batch_size = np.random.choice([64, 128])
    args.man_tau = args.ctrl_tau
    args.man_noise_sigma = np.random.choice([0.2, 0.25])
    args.man_act_lr = np.random.choice([1e-3, 1e-4])
    # args.man_last_layer = str(np.random.choice(['vae', 'fc']))

    args.log_file = "{}-{}-{}-{}-{}-{}-{}-{}".format(args.env_name, args.man_last_layer, args.z_dim, args.manager_propose_freq, args.noise_type, args.man_act_lr, args.ctrl_act_lr, np.random.randint(100))

# Seed AFTER the hyperparameters are set
np.random.seed(args.seed)
run_hiro(args)
