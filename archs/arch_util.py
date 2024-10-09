import torch
import numpy as np
from torch import nn as nn
from torch.nn import init as init

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
def load_weights(model, old_weights):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)
    return model

def load_optim(optim, optim_weights):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    # new_weights = optim.state_dict()
    
    # params_to_train = [param for param in model.parameters() if param.requires_grad]
    
    # # new_weights.update({k: v for k, v in optim_weights.items() if k in new_weights})
    # filtered_state_dict = {k: v for k, v in optim_weights.items() if k in params_to_train}
    
    # optim_weights['state'].update({k: v for k, v in filtered_state_dict['state'].items() if k in new_weights}) 
    # noes = {k: v for k, v in optim_weights.items() if k in new_weights}
    # print(noes['param_groups'])
    optim.load_state_dict(optim_weights)
    return optim


def save_checkpoint(model, optim, scheduler, metrics, paths, adapter = False, rank = None):


    if rank!=0: 
        return metrics, metrics['best_psnr']
    weights = model.state_dict()
    if adapter:
        weights = {k: v for k, v in weights.items() if 'adapter' in k}
    else:
        weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': np.mean(metrics['train_loss']),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(model_to_save, paths['new'])

    #save best model if new valid_psnr is higher than the best one
    if np.mean(metrics['valid_psnr']) >= metrics['best_psnr']:
        
        torch.save(model_to_save, paths['best'])
        
        metrics['best_psnr'] = np.mean(metrics['valid_psnr']) # update best psnr
    # elif not rank:
    #     weights = model.state_dict()
    #     if adapter:
    #         weights = {k: v for k, v in weights.items() if 'adapter' in k}
    #     else:
    #         weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    #     # Save the model after every epoch
    #     model_to_save = {
    #         'epoch': metrics['epoch'],
    #         'model_state_dict': weights,
    #         'optimizer_state_dict': optim.state_dict(),
    #         'loss': np.mean(metrics['train_loss']),
    #         'scheduler_state_dict': scheduler.state_dict()
    #     }
    #     torch.save(model_to_save, paths['new'])

    #     #save best model if new valid_psnr is higher than the best one
    #     if np.mean(metrics['valid_psnr']) >= metrics['best_psnr']:
            
    #         torch.save(model_to_save, paths['best'])
            
    #         metrics['best_psnr'] = np.mean(metrics['valid_psnr']) # update best psnr 
    # else:
    #     raise ValueError(f'Not valid value for rank: {rank}')      
    
    return metrics, metrics['best_psnr']

if __name__ == '__main__':
    
    pass