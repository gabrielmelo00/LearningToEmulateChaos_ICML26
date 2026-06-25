import torch

def load_operator(operator, saved_pth):
    print('loading saved operator', saved_pth)
    try:
        device = next(operator.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    checkpoint = torch.load(saved_pth, map_location=device, weights_only=True)
    checkpoint = {key.replace('module.','') : val for key, val in checkpoint['state_dict'].items()}
    try:
        operator.module.load_state_dict(checkpoint)
    except:
        operator.load_state_dict(checkpoint)
    return operator
