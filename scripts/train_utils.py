import torch, os
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

class LpLoss_:
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss_, self).__init__()
        # Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

def load_model(saved_pth, operator, optimizer):
    saved_pth_list = glob.glob(f'{saved_pth}_*')
    saved_pth_list.sort()
    if len(saved_pth_list) > 0:
        epoch_id = np.array([int(p.split('_')[-1]) for p in saved_pth_list])
        train_epoch_ = epoch_id.max() + 1
        if epoch_id.max() > 0:
            epoch_load = epoch_id.max()
            train_epoch_ = epoch_load
            load_pth = '{saved_pth}_{}'.format(visual_folder, prefix, args.special_prefix, epoch_load)
            print('load checkpoint: {}'.format(train_epoch_))
            checkpoint = torch.load(load_pth, map_location='cuda:{}'.format(args.gpu), weights_only=True)
            checkpoint_state = {key.replace('module.','') : val for key, val in checkpoint['state_dict'].items()}
            checkpoint_optimizer = {key.replace('module.','') : val for key, val in checkpoint['optimizer_state_dict'].items()}
            try:
                operator.module.load_state_dict(checkpoint_state)
                optimizer.load_state_dict(checkpoint_optimizer)
            except:
                operator.load_state_dict(checkpoint_state)
                optimizer.load_state_dict(checkpoint_optimizer)
        torch.cuda.empty_cache()
    return operator, optimizer

def save_operator(operator, optimizer, saved_pth, ep):
    torch.save({'epoch': ep, 'state_dict': operator.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, saved_pth)

def save_summary_checkpoint(summary_net, summary_optimizer, critic, critic_optimizer, saved_pth, ep, args):
    checkpoint = {
        'epoch': ep,
        'summary_state_dict': summary_net.state_dict() if summary_net is not None else None,
        'summary_optimizer_state_dict': summary_optimizer.state_dict() if summary_optimizer is not None else None,
        'loss_mode': args.loss_mode,
        'summary_mode': args.summary_mode,
        'summary_dim': args.summary_dim,
        'state_dim': args.state_dim,
    }
    if critic is not None:
        checkpoint['critic_state_dict'] = critic.state_dict()
    if critic_optimizer is not None:
        checkpoint['critic_optimizer_state_dict'] = critic_optimizer.state_dict()
    torch.save(checkpoint, saved_pth)

def long_length_predict_with_yinit(operator, y, param, x_len, len_to_operator):
    #####the following is only for speeding up#####
    batch_size, x_res = y.shape[0], y.shape[-1]
    initial_index = np.arange(0, x_len, len_to_operator)
    if initial_index[-1] == x_len - 1:
        initial_index = initial_index[:-1]
        match_length_flag = True
    else:
        match_length_flag = False
    y_initials = y[:, initial_index].reshape(batch_size*len(initial_index), x_res)
    prediction_list = [y_initials.reshape(batch_size, len(initial_index), x_res)]
    for ix in range(len_to_operator-1):
        params_ = param[:, None, 0].repeat(1, len(initial_index)).reshape(batch_size*len(initial_index), -1)
        predict_y = operator(y_initials.squeeze()[:, None, :], params_)
        prediction_list.append(predict_y.reshape(batch_size, len(initial_index), x_res))
        y_initials = predict_y
    prediction_out = torch.stack(prediction_list, axis = 2).squeeze().reshape(batch_size, -1, x_res)
    if match_length_flag:
        prediction_out = torch.cat([prediction_out.reshape(batch_size, -1, x_res), y[:, x_len-1, :][:, None, :]], dim = 1)
    out = prediction_out[:, :x_len, :]
    return out

def visualiztion(dataset, operator, args, img_pth, ep):
    operator.eval()
    os.makedirs(f'{img_pth}', exist_ok=True)
    for ix_data in range(3):
        with torch.no_grad():
            ix = np.random.randint(10)
            data = dataset[ix_data]
            param, x = data[0], data[1]
            x_0_start = 0
            x_end = 100
            x_len_ = x_end - x_0_start - 1
            data_list = []
            x_0 = x.squeeze()[x_0_start, :][None, None, :].to(args.gpu).float()
            data_list.append(x_0.squeeze())
            for ix in range(x_len_):
                if args.kse or getattr(args, 'l63', False):
                    x_1 = operator(x_0)
                else:
                    x_1 = operator(x_0, torch.from_numpy(param[None, :]))
                data_list.append(x_1.squeeze())
                x_0 = x_1
            out = torch.stack(data_list, dim = 0)
        out = out[:x_len_+1, :].cpu().data.numpy()
        y = x[x_0_start:x_end, :].squeeze()
        plot_3col([y, 'true'], [out, 'predict'], operator='-', im = f'{img_pth}/{ep:03d}_{ix_data}')

def adjust_learning_rate_cos(lr_ori, optimizer, epoch, total_epochs, args, warm_up_epochs = 10):
    """Decay the learning rate based on schedule"""
    lr = lr_ori
    if epoch < warm_up_epochs:
        lr = lr_ori * (epoch + 1) / warm_up_epochs
    else:
        lr *= 0.5 * (1 + math.cos(math.pi * (epoch - warm_up_epochs) / total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def plot_loss(loss_list, img_pth):
    if len(loss_list) == 0:
        return
    loss_arr = np.asarray(loss_list, dtype=float)
    if loss_arr.ndim == 1:
        loss_arr = loss_arr.reshape(-1, 1)
    if loss_arr.shape[1] < 3:
        padded = np.zeros((loss_arr.shape[0], 3), dtype=float)
        padded[:, :loss_arr.shape[1]] = loss_arr
        loss_arr = padded
    loss_arr = loss_arr[-100:]
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(loss_arr)), loss_arr[:, 0])
    plt.title('loss_mse')
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(loss_arr)), loss_arr[:, 1])
    plt.title('loss_OT')
    plt.tight_layout()
    plt.savefig(f'{img_pth}.pdf')
    plt.close()

def plot_3col(a_t, b_t, operator='-', im = '', cmap_ = 'bwr'):
    vals = np.concatenate([a_t[0], b_t[0], a_t[0] - b_t[0]]).reshape(-1)
    vmin_, vmax_ = vals.min(), vals.max()
    vmin_, vmax_ = -max(abs(vmin_), abs(vmax_)), max(abs(vmin_), abs(vmax_))
    fig, ax = plt.subplots(3, 1, sharex='all', sharey='all')
    ax[0].imshow(a_t[0].T, vmin=vmin_, vmax=vmax_, cmap=cmap_)
    ax[0].set_title(a_t[1])
    ax[0].set_ylabel('x')
    im1 = ax[1].imshow(b_t[0].T, vmin=vmin_, vmax=vmax_, cmap=cmap_)
    ax[1].set_title(b_t[1])
    ax[1].set_ylabel('x')
    im2 = ax[2].imshow((a_t[0] - b_t[0]).T, vmin=vmin_, vmax=vmax_, cmap=cmap_)
    ax[2].set_title('Diff.')
    ax[2].set_ylabel('x')
    ax[2].set_xlabel('t')
    cbar = fig.colorbar(im2, ax=ax.ravel().tolist())
    plt.savefig('{}.png'.format(im))
    plt.close()


def long_length_predict_ks(operator, y, x_len, len_to_operator):
    """
    L96-style anchored rollout for KS.

    For len_to_operator=2 (default), we anchor every 2 steps using ground
    truth and only predict the intermediate step. This mirrors the L96
    pipeline behavior and avoids long free-rollout drift during training.
    """
    batch_size, x_res = y.shape[0], y.shape[-1]
    initial_index = np.arange(0, x_len, len_to_operator)
    if initial_index[-1] == x_len - 1:
        initial_index = initial_index[:-1]
        match_length_flag = True
    else:
        match_length_flag = False

    y_initials = y[:, initial_index].reshape(batch_size * len(initial_index), x_res)
    prediction_list = [y_initials.reshape(batch_size, len(initial_index), x_res)]

    for _ in range(len_to_operator - 1):
        predict_y = operator(y_initials[:, None, :])
        prediction_list.append(predict_y.reshape(batch_size, len(initial_index), x_res))
        y_initials = predict_y

    prediction_out = torch.stack(prediction_list, axis=2).squeeze().reshape(batch_size, -1, x_res)
    if match_length_flag:
        prediction_out = torch.cat(
            [prediction_out.reshape(batch_size, -1, x_res), y[:, x_len - 1, :][:, None, :]],
            dim=1,
        )

    out = prediction_out[:, :x_len, :]
    assert out.shape == y[:, :x_len, :].shape
    return out
