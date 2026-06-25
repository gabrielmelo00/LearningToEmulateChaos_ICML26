# TORCH_DISTRIBUTED_DEBUG=INFO
# TORCH_DISTRIBUTED_DEBUG=DETAIL
import builtins, json, warnings, pdb, os, time, sys, traceback, torch, torch.optim, torch.utils.data, torch.utils.data.distributed, random
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from log import create_folder_path
path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from timeit import default_timer
from utils import init_distributed_mode, HiddenPrints
from tqdm import tqdm
from configuration import args
from scripts.dataloader_init import init_dataloader
from scripts.train_utils import (
    LpLoss_,
    adjust_learning_rate_cos,
    visualiztion,
    plot_loss,
    save_operator,
    save_summary_checkpoint,
)
from models.fno_1d_new import FNO1d
from models.mlp_l63 import L63MLP
from scripts.summary import SummaryNet, Critic


def setup_wandb(args, output_path):
    if not args.wandb or not args.is_master:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f'wandb import failed: {exc}')
        return None
    run_name = args.wandb_run_name or args.prefix
    if args.kse:
        project_name = "OT4DYNSYS-KS"
    elif args.l63:
        project_name = "OT4DYNSYS-L63"
    else:
        project_name = "OT4DYNSYS-L96"
    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args),
        dir=output_path,
    )
    return wandb


def set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def _effective_ot_lambda(base_lambda, epoch, start_epoch, warmup_epochs):
    base = float(base_lambda)
    if base <= 0:
        return 0.0
    if epoch < int(start_epoch):
        return 0.0
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return base
    progress = (epoch - int(start_epoch) + 1) / float(warmup_epochs)
    return base * max(0.0, min(1.0, progress))


def main(args):
    print(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.gpu is not None and has_cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    elif args.gpu is not None and not has_cuda:
        print('CUDA is not available. Falling back to CPU execution.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = (args.world_size >= 1) and has_cuda
    ngpus_per_node = torch.cuda.device_count() if has_cuda else 0

    print('start')
    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        # Single-process runs should behave as rank 0 and keep stdout enabled.
        args.rank = 0
        if not has_cuda:
            args.world_size = 0
    if has_cuda:
        args.gpu = args.gpu % torch.cuda.device_count()
    else:
        args.gpu = 'cpu'
    print('world_size', args.world_size)
    print('rank', args.rank)
    print('device', args.gpu)
    if args.distributed and args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    args.distributed = has_cuda and (args.world_size >= 1 or args.multiprocessing_distributed)
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if ngpus_per_node > 0:
        args.is_master = args.rank % ngpus_per_node == 0 and args.gpu == 0
    else:
        args.is_master = args.rank == 0

    # ── Validate experiment flag ──────────────────────────────────────────────
    n_systems = int(args.l96) + int(args.kse) + int(args.l63)
    assert n_systems == 1, "Pass exactly one system flag: --l96, --kse, or --l63"

    # ── Import system-specific modules ────────────────────────────────────────
    if args.l96:
        from eval_scripts.eval_l96 import eval_l96
        from scripts.cal_stats_l96 import cal_stats_l96 as cal_stats
        from train_utils import long_length_predict_with_yinit
    elif args.kse:
        from eval_scripts.eval_ks import eval_ks
        from scripts.cal_stats_ks import cal_stats_ks as cal_stats
        from train_utils import long_length_predict_ks as long_length_predict_with_yinit
    else:
        from eval_scripts.eval_l63 import eval_l63
        from scripts.cal_stats_l63 import cal_stats_l63 as cal_stats
        from train_utils import long_length_predict_ks as long_length_predict_with_yinit

    train_dataset_operator, train_loader_operator, train_sampler, train_sampler_operator = init_dataloader(args)
    operators_path, summary_path, output_path = create_folder_path(args)
    wandb_logger = setup_wandb(args, output_path)

    ###########################################################################
    ################### Training the neural operator ##########################
    if args.l63:
        operator = L63MLP(hidden_dim=args.width)
    else:
        operator = FNO1d(args.modes, args.width)
    operator.to(args.gpu)
    learning_rate, epochs = args.learning_rate, args.epochs
    optimizer = torch.optim.AdamW(operator.parameters(), lr=args.learning_rate, weight_decay=0.0)
    train_epoch_ = 0
    if args.distributed:
        operator = torch.nn.parallel.DistributedDataParallel(
            operator, device_ids=[args.gpu],
            find_unused_parameters=False, broadcast_buffers=False
        )
    if args.gpu != 'cpu':
        operator = nn.SyncBatchNorm.convert_sync_batchnorm(operator)

    from scripts.OT_utils import OT_measure
    OT_measure = OT_measure(
        args.with_geomloss,
        args.blur,
        std_floor=getattr(args, 'ot_std_floor', 0.01),
        stat_clip=getattr(args, 'ot_stat_clip', 0.0),
    )

    summary_net = None
    critic = None
    summary_optimizer = None
    critic_optimizer = None
    print("USING LOSS MODE:",args.loss_mode)
    if args.loss_mode in ('learnable_ot', 'learnable_sinkhorn'):
        summary_net = SummaryNet(
            summary_dim=args.summary_dim,
            mode=args.summary_mode,
            state_dim=args.state_dim,
            window_size=getattr(args, 'summary_window_size', 2),
        ).to(args.gpu)
        summary_optimizer = torch.optim.Adam(summary_net.parameters(), lr=args.wgan_lr)
        if args.loss_mode == 'learnable_ot':
            critic = Critic(summary_dim=args.summary_dim).to(args.gpu)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.wgan_lr)
        if args.distributed:
            summary_net = torch.nn.parallel.DistributedDataParallel(
                summary_net, device_ids=[args.gpu],
                find_unused_parameters=False, broadcast_buffers=False
            )
            if critic is not None:
                critic = torch.nn.parallel.DistributedDataParallel(
                    critic, device_ids=[args.gpu],
                    find_unused_parameters=False, broadcast_buffers=False
                )

    ###########################################################################
    ################### Operator training loop ################################
    if args.train_operator:
        loss_list, ep_loss = [], []
        global_step = 0
        for ep in tqdm(range(epochs)):
            if args.distributed:
                train_sampler_operator.set_epoch(ep)
            train_x_len = int(args.x_len)
            lambda_ot_eff = _effective_ot_lambda(
                args.lambda_geomloss,
                ep,
                args.ot_start_epoch,
                getattr(args, 'ot_warmup_epochs', 0),
            )

            for batch_idx, batch in enumerate(train_loader_operator):
                operator.train()
                l2       = torch.tensor([0]).to(args.gpu).float()
                loss_OT  = torch.tensor([0]).to(args.gpu).float()
                loss_CL  = torch.tensor([0]).to(args.gpu).float()

                lr_ = adjust_learning_rate_cos(args.learning_rate, optimizer, ep, epochs, args)

                # ── Unpack batch ──────────────────────────────────────────────
                # L96: batch = (param, y)  where param is scalar F
                # KS:  batch = (ic, y)     where ic is the initial condition
                # L63: batch = (params, y) where params=(sigma, rho, beta)
                # In both cases the second element is the trajectory.
                # The operator does not receive param/ic directly for KS/L63
                # (it is a pure state-to-state map), but we keep the same
                # unpacking convention so the loop is uniform.
                param, y = batch
                param = param.to(args.gpu)
                y     = y.to(args.gpu).squeeze()

                assert train_x_len <= y.shape[1]
                assert y.shape[0] == args.batch_size

                # ── Forward rollout ───────────────────────────────────────────
                # long_length_predict_with_yinit signature:
                #   L96: (operator, y, param, x_len, len_to_operator)
                #   KS:  (operator, y,        x_len, len_to_operator)
                #   L63: (operator, y,        x_len, len_to_operator)
                #        params are ignored — operator takes only state
                if args.kse or args.l63:
                    y_predict = long_length_predict_with_yinit(
                        operator, y, train_x_len, args.len_to_operator
                    )
                else:
                    y_predict = long_length_predict_with_yinit(
                        operator, y, param, train_x_len, args.len_to_operator
                    )

                y_target = y[:, :train_x_len]
                l2 += LpLoss_(2).rel(y_predict, y_target)

                # ── OT loss ───────────────────────────────────────────────────
                if args.loss_mode == 'ot' and args.with_geomloss > 0 and ep >= args.ot_start_epoch:
                    if args.l63:
                        anchor_stats, out_stats = cal_stats(
                            y_target.squeeze(),
                            y_predict.squeeze(),
                            dims=getattr(args, 'l63_ot_dims', [0, 1, 2]),
                            feature_mode=getattr(args, 'l63_ot_feature_mode', 'state'),
                            lags=getattr(args, 'l63_ot_lags', [0]),
                            trim=getattr(args, 'l63_ot_trim', 2),
                        )
                    else:
                        anchor_stats, out_stats = cal_stats(
                            y_target.squeeze(), y_predict.squeeze()
                        )
                    # L96 supports per-dimension KD selection; KS does not
                    if args.l96 and args.with_geomloss_kd != 0:
                        anchor_stats = anchor_stats[:, :, np.array([args.with_geomloss_kd - 1])]
                        out_stats    = out_stats[:, :, np.array([args.with_geomloss_kd - 1])]
                    assert anchor_stats.shape[0] == args.batch_size
                    loss_OT = OT_measure.loss(anchor_stats, out_stats)

                # ── Learnable OT loss ─────────────────────────────────────────
                if args.loss_mode == 'learnable_ot' and ep >= args.ot_start_epoch:
                    set_requires_grad(summary_net, True)
                    set_requires_grad(critic, True)
                    for _ in range(args.wgan_critic_steps):
                        summary_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        anchor_stats = summary_net(y_target.squeeze())
                        out_stats    = summary_net(y_predict.detach().squeeze())
                        norm_std  = anchor_stats.std(dim=1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
                        norm_mean = anchor_stats.mean(dim=1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
                        out_stats    = (out_stats    - norm_mean) / (norm_std + 1e-6)
                        anchor_stats = (anchor_stats - norm_mean) / (norm_std + 1e-6)
                        wgan_loss = -(critic(anchor_stats).mean() - critic(out_stats).mean())
                        wgan_loss.backward()
                        summary_optimizer.step()
                        critic_optimizer.step()
                        for p in critic.parameters():
                            p.data.clamp_(-args.wgan_clip, args.wgan_clip)
                    set_requires_grad(summary_net, False)
                    set_requires_grad(critic, False)
                    anchor_stats = summary_net(y_target.squeeze())
                    out_stats    = summary_net(y_predict.squeeze())
                    norm_std  = anchor_stats.std(dim=1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
                    norm_mean = anchor_stats.mean(dim=1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
                    out_stats    = (out_stats    - norm_mean) / (norm_std + 1e-6)
                    anchor_stats = (anchor_stats - norm_mean) / (norm_std + 1e-6)
                    loss_OT = critic(anchor_stats).mean() - critic(out_stats).mean()

                # ── Learnable Sinkhorn loss ───────────────────────────────────
                if args.loss_mode == 'learnable_sinkhorn' and ep >= args.ot_start_epoch:
                    if ep <= 200:
                        set_requires_grad(summary_net, True)
                        for _ in range(args.wgan_critic_steps):
                            summary_optimizer.zero_grad()
                            anchor_stats = summary_net(y_target.squeeze())
                            out_stats = summary_net(y_predict.detach().squeeze())
                            loss_summary = -OT_measure.loss(anchor_stats, out_stats)
                            loss_summary.backward()
                            summary_optimizer.step()
                        set_requires_grad(summary_net, False)
                    anchor_stats = summary_net(y_target.squeeze())
                    out_stats    = summary_net(y_predict.squeeze())
                    loss_OT = OT_measure.loss(anchor_stats, out_stats)

                if not torch.isfinite(loss_OT):
                    if args.is_master:
                        print(f'[WARN] non-finite loss_OT at epoch={ep}, batch={batch_idx}; setting OT loss to 0.')
                    loss_OT = torch.zeros_like(loss_OT)

                weighted_ot = lambda_ot_eff * loss_OT
                ot_weighted_cap = float(getattr(args, 'ot_weighted_cap', 0.0))
                if ot_weighted_cap > 0:
                    weighted_ot = torch.clamp(weighted_ot, min=-ot_weighted_cap, max=ot_weighted_cap)

                # ── Optimizer step ────────────────────────────────────────────
                warmup_only = (
                    args.loss_mode in ('learnable_ot', 'learnable_sinkhorn')
                    and ep < args.wgan_warmup_epochs
                )
                if warmup_only:
                    total_loss = l2 + weighted_ot
                else:
                    total_loss = (l2
                                  + args.lambda_contra * loss_CL
                                  + weighted_ot)
                if not torch.isfinite(total_loss):
                    if args.is_master:
                        print(f'[WARN] non-finite total_loss at epoch={ep}, batch={batch_idx}; skipping step.')
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if wandb_logger is not None:
                    log_payload = {
                        'loss/reconstruction': l2.item(),
                        'loss/ot':             loss_OT.item(),
                        'loss/ot_weighted':    weighted_ot.item(),
                        'loss/total':          total_loss.item(),
                        'train/epoch':         ep,
                        'train/batch_idx':     batch_idx,
                        'train/lr':            lr_,
                        'train/ot_lambda_effective': lambda_ot_eff,
                        'train/train_x_len':   train_x_len,
                        'train/ot_weighted_cap': ot_weighted_cap,
                    }
                    wandb_logger.log(log_payload, step=global_step)
                global_step += 1
                ep_loss.append([
                    l2.item(),
                    (args.lambda_contra * loss_CL).cpu().data.numpy().item(),
                    weighted_ot.item(),
                ])
                loss_list.append(np.array(ep_loss).mean(axis=0).tolist())

                if warmup_only:
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                grad_clip_norm = float(getattr(args, 'grad_clip_norm', 0.0))
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(operator.parameters(), grad_clip_norm)
                optimizer.step()

            if ep % 50 == 0 and ep > 0:
                visualiztion(train_dataset_operator, operator, args,
                             img_pth=f'{output_path}/training_vis', ep=ep)
                plot_loss(loss_list, img_pth=f'{output_path}/training_loss_operator')

        if ep == epochs - 1:
            if args.is_master:
                save_operator(operator, optimizer,
                              saved_pth=f'{operators_path}/{args.prefix}/{ep:03d}', ep=ep)
                if summary_net is not None:
                    save_summary_checkpoint(
                        summary_net=summary_net,
                        summary_optimizer=summary_optimizer,
                        critic=critic,
                        critic_optimizer=critic_optimizer,
                        saved_pth=f'{summary_path}/{args.prefix}/{ep:03d}',
                        ep=ep,
                        args=args,
                    )

    ###########################################################################
    #################### Load model and evaluate ##############################
    ep = args.epochs - 1
    from eval_scripts.eval_utils import load_operator
    operator = load_operator(operator, saved_pth=f'{operators_path}/{args.prefix}/{ep:03d}')
    visualiztion(train_dataset_operator, operator, args,
                 img_pth=f'{output_path}/training_vis', ep=ep)

    if args.l96:
        eval_len_list = [1500]
        for eval_len in eval_len_list:
            eval_l96(operator, args, args.noisy_scale,
                     x_len=eval_len, calculate_l2=True, output_path=output_path)
            eval_l96(operator, args, 0,
                     x_len=eval_len, calculate_l2=True, output_path=output_path)

    if args.kse:
        eval_len_list = [int(x) for x in getattr(args, 'ks_eval_lengths', [1000])]
        for eval_len in eval_len_list:
            eval_ks(operator, args, args.noisy_scale,
                    x_len=eval_len, output_path=output_path)
            eval_ks(operator, args, 0,
                    x_len=eval_len, output_path=output_path)

    if args.l63:
        eval_len_list = [int(x) for x in getattr(args, 'l63_eval_lengths', [100, 1000])]
        for eval_len in eval_len_list:
            eval_l63(operator, args, args.noisy_scale,
                     x_len=eval_len, output_path=output_path)
            eval_l63(operator, args, 0,
                     x_len=eval_len, output_path=output_path)

    if args.eval_LE:
        if args.l96:
            from eval_scripts.eval_LE import cal_LE
            LE_results = cal_LE(operator, args)
        else:
            print('eval_LE is currently implemented only for L96; skipping.')

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == '__main__':
    try:
        main(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
