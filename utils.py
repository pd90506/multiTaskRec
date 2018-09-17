"""
    Some handy functions for pytroch model training ...
"""
import torch
import os

# Checkpoints
def save_checkpoint(model, model_dir):
    #model_dir = os.path.join('/home/panda/github/neural-collaborative-filtering/src', model_dir)    
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    #model_dir = os.path.join('/home/panda/github/neural-collaborative-filtering/src', model_dir)    
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['adam_lr'], weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer