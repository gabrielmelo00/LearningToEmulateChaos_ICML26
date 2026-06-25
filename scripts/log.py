import os
import json

def create_folder_path(args):
    if args.l96:
        args.prefix += f'l96'

    if args.prefix:
        pass
    else:
        prefix_for_op = f'xl_{args.x_len}'
        if getattr(args, 'loss_mode', 'ot') == 'learnable_ot':
            prefix_for_op += '_learnOT'
        elif getattr(args, 'loss_mode', 'ot') == 'learnable_sinkhorn':
            prefix_for_op += '_learnSinkhorn'
        elif args.with_geomloss > 0:
            prefix_for_op += f'_lmOT_{args.lambda_geomloss}'
        args.prefix = prefix_for_op

    print('\n', args.prefix)
    operators_path = 'saved_checkpoints/operators'
    summary_path = 'saved_checkpoints/summary'
    root = getattr(args, 'output_folder', None)
    if not root:
        if getattr(args, 'kse', False):
            root = 'ks_output_foulder'
        elif getattr(args, 'l63', False):
            root = 'l63_output_folder'
        else:
            root = 'output_folder'
    output_path = f'{root}/{args.prefix}'
    if args.is_master:
        os.makedirs(f'{operators_path}/{args.prefix}', exist_ok=True)
        os.makedirs(f'{summary_path}/{args.prefix}', exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        with open('{}/configuration.txt'.format(output_path), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return operators_path, summary_path, output_path
