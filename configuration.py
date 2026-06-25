import argparse

parser = argparse.ArgumentParser(description='neural operator for longer time')
parser.add_argument('--noisy_scale', default=0, type=float)
parser.add_argument('--embed_dim', default=32, type=int)
parser.add_argument('--modes', default=50, type=int)
parser.add_argument('--width', default=32, type=int)

parser.add_argument('--crop_T', default=200, type=int)
parser.add_argument('--x_len', default=100, type=int)
parser.add_argument('--len_to_operator', default=2, type=int)
parser.add_argument('--calculate_metric', default=0, type=int)

parser.add_argument('--epochs', default=701, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)

##############################################################################
# Experiment selection: pass exactly one of --l96, --kse, or --l63
##############################################################################
parser.add_argument('--l96', action='store_true',
                    help='Run the Lorenz-96 experiment')
parser.add_argument('--kse', action='store_true',
                    help='Run the Kuramoto-Sivashinsky experiment')
parser.add_argument('--l63', action='store_true',
                    help='Run the Lorenz-63 experiment')

##############################################################################
# KS-specific settings
# state_dim for KS should match the spatial grid N used in data generation
# (default 256). For L96 it remains 60.
##############################################################################
parser.add_argument('--ks_N', default=256, type=int,
                    help='KS spatial grid size N (must match generated data)')
parser.add_argument('--ks_data_train', default=None, type=str,
                    help='Path to KS training data folder (overrides default)')
parser.add_argument('--ks_data_val', default=None, type=str,
                    help='Path to KS validation data folder (overrides default)')
parser.add_argument('--ks_data_test', default=None, type=str,
                    help='Path to KS test data folder (overrides default)')
parser.add_argument('--ks_eval_lengths', nargs='+', default=[1000], type=int,
                    help='KS rollout lengths to evaluate and store separately, '
                         'e.g. --ks_eval_lengths 100 1000')
parser.add_argument('--ks_noisy_eval_split', default='test', type=str,
                    choices=['test', 'validation'],
                    help='Split used for noisy KS evaluation. '
                         'Default is test so clean and noisy metrics come from the '
                         'same split for robustness studies.')
parser.add_argument('--ks_clean_eval_init_noise', default=0.0, type=float,
                    help='Noise level used for the initial condition stream during '
                         'clean KS evaluation. Default 0.0 keeps clean evaluation '
                         'truly clean.')

##############################################################################
# L63-specific settings
##############################################################################
parser.add_argument('--l63_data_train', default=None, type=str,
                    help='Path to L63 training data folder (overrides default)')
parser.add_argument('--l63_data_val', default=None, type=str,
                    help='Path to L63 validation data folder (overrides default)')
parser.add_argument('--l63_data_test', default=None, type=str,
                    help='Path to L63 test data folder (overrides default)')
parser.add_argument('--l63_eval_lengths', nargs='+', default=[100, 1000], type=int,
                    help='L63 rollout lengths to evaluate, '
                         'e.g. --l63_eval_lengths 100 1000')
parser.add_argument('--l63_noisy_eval_split', default='test', type=str,
                    choices=['test', 'validation'],
                    help='Split used for noisy L63 evaluation.')
parser.add_argument('--l63_clean_eval_init_noise', default=0.0, type=float,
                    help='Noise level used for initial condition stream during '
                         'clean L63 evaluation.')
parser.add_argument('--l63_plot_samples', default=3, type=int,
                    help='Number of L63 rollout samples saved as plots per eval run.')
parser.add_argument('--l63_eval_size_test', default=20, type=int,
                    help='Maximum number of L63 test trajectories used for evaluation.')
parser.add_argument('--l63_eval_size_val', default=20, type=int,
                    help='Maximum number of L63 validation trajectories used for noisy evaluation.')
parser.add_argument('--l63_ot_dims', nargs='+', default=[0, 1, 2], type=int,
                    help='State coordinates used by fixed OT summary for L63, '
                         'e.g. --l63_ot_dims 2 (z only) or --l63_ot_dims 0 1 2 (xyz).')
parser.add_argument('--l63_ot_feature_mode', default='state', type=str,
                    choices=['state', 'lagged_state'],
                    help='Feature construction for fixed OT summary in L63. '
                         '"state" uses u_t coordinates only; '
                         '"lagged_state" concatenates [u_t, u_{t-1}, ...] features.')
parser.add_argument('--l63_ot_lags', nargs='+', default=[0], type=int,
                    help='Temporal lags used when --l63_ot_feature_mode lagged_state. '
                         'Example: --l63_ot_lags 0 1 2 4')
parser.add_argument('--l63_ot_trim', default=2, type=int,
                    help='Temporal boundary trim for L63 OT summary extraction.')
parser.add_argument('--l63_attractor_loss_weight', default=0.0, type=float,
                    help='Weight of attractor-aware regularization for L63 training.')
parser.add_argument('--l63_attractor_max_lag', default=64, type=int,
                    help='Maximum lag for L63 autocorrelation regularizer and diagnostics.')
parser.add_argument('--l63_attractor_psd_weight', default=1.0, type=float,
                    help='Relative weight of PSD-matching term in L63 attractor regularizer.')
parser.add_argument('--l63_attractor_autocorr_weight', default=1.0, type=float,
                    help='Relative weight of autocorrelation-matching term in L63 attractor regularizer.')
parser.add_argument('--l63_attractor_regime_weight', default=0.5, type=float,
                    help='Relative weight of lobe occupancy-matching term in L63 attractor regularizer.')
parser.add_argument('--l63_attractor_switch_weight', default=0.5, type=float,
                    help='Relative weight of lobe switching-rate term in L63 attractor regularizer.')
parser.add_argument('--l63_attractor_regime_sharpness', default=8.0, type=float,
                    help='Sharpness used to approximate x-sign for differentiable regime penalties.')

##############################################################################
parser.add_argument('--subsample', action='store_true')
parser.add_argument('--ranked', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--with_geomloss', default=0, type=int)
parser.add_argument('--with_geomloss_kd', default=0, type=int,
                    help='Per-dimension KD selection for L96 OT loss. '
                         'Ignored for KS.')
parser.add_argument('--lambda_geomloss', default=0, type=float)
parser.add_argument('--blur', default=0.01, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--train_operator', action='store_true')
parser.add_argument('--eval_LE', action='store_true')
parser.add_argument('--training_size', default=2000, type=int)
parser.add_argument('--batches_per_epoch', default=1, type=int,
                    help='Number of gradient steps per epoch (KS single-traj mode). '
                         'Controls how many random crops are drawn per epoch regardless '
                         'of the number of segment files.')
parser.add_argument('--seed', default=34, type=int)

parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--output_folder', default=None, type=str,
                    help='Root directory for output files and checkpoints')
parser.add_argument('--multiprocessing_distributed', action='store_true')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str)
parser.add_argument('--local_rank', '--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_run_name', default=None, type=str)
parser.add_argument('--loss_mode', default='ot', type=str,
                    choices=['ot', 'learnable_ot', 'learnable_sinkhorn'])
parser.add_argument('--wgan_lr', default=1e-4, type=float)
parser.add_argument('--wgan_clip', default=0.01, type=float)
parser.add_argument('--wgan_critic_steps', default=1, type=int)
parser.add_argument('--wgan_warmup_epochs', default=0, type=int)
parser.add_argument('--summary_dim', default=3, type=int)
parser.add_argument('--summary_clip', default=None, type=float)
parser.add_argument('--summary_mode', default='pointwise', type=str,
                    choices=['physics', 'ks_physics', 'pointwise', 'local', 'statewise', 'linear'],
                    help='Mode for SummaryNet')
parser.add_argument('--summary_window_size', default=2, type=int,
                    help='Half-window size for local SummaryNet mode (input width = 2*w+1)')
parser.add_argument('--ot_start_epoch', default=50, type=int,
                    help='Epoch at which OT/adversarial loss activates.')
parser.add_argument('--ot_warmup_epochs', default=0, type=int,
                    help='Linear warmup epochs for OT weight after ot_start_epoch. '
                         '0 disables warmup.')
parser.add_argument('--ot_std_floor', default=0.01, type=float,
                    help='Minimum std used when normalizing OT statistics. '
                         'Prevents division by very small variance.')
parser.add_argument('--ot_stat_clip', default=0.0, type=float,
                    help='Optional clip value for normalized OT statistics. '
                         'Set <=0 to disable.')
parser.add_argument('--ot_weighted_cap', default=0.0, type=float,
                    help='Optional cap for weighted OT contribution in total loss. '
                         'Set <=0 to disable.')
parser.add_argument('--grad_clip_norm', default=0.0, type=float,
                    help='Optional global gradient-norm clipping for operator. '
                         'Set <=0 to disable.')
parser.add_argument('--state_dim', default=60, type=int,
                    help='Spatial dimension of the state. '
                         'Set to 256 (or --ks_N value) when running KS. '
                         'Set to 3 when running L63. '
                         'Used by statewise/linear SummaryNet modes.')

args = parser.parse_args()

# ── Auto-set state_dim for KS if user did not override it ────────────────────
# This saves the user from having to pass --state_dim 256 manually every time.
if args.kse and args.state_dim == 60:
    args.state_dim = args.ks_N
if args.l63 and args.state_dim == 60:
    args.state_dim = 3
