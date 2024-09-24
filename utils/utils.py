import torch


def load_weights(model, old_weights):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)
    return model

# def load_weights(model, adapter_weights):
#     '''
#     Loads the weights of a pretrained model, picking only the weights that are
#     in the new model.
#     '''
#     new_weights = model.state_dict()
#     new_weights.update({k: v for k, v in adapter_weights.items() if k in new_weights})
    
#     model = model.load_state_dict(new_weights)
#     return model

def load_optim(optim, optim_weights, model):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    # new_weights = optim.state_dict()
    
    params_to_train = [param for param in model.parameters() if param.requires_grad]
    
    # new_weights.update({k: v for k, v in optim_weights.items() if k in new_weights})
    filtered_state_dict = {k: v for k, v in optim_weights.items() if k in params_to_train}
    
    # optim_weights['state'].update({k: v for k, v in filtered_state_dict['state'].items() if k in new_weights}) 
    # noes = {k: v for k, v in optim_weights.items() if k in new_weights}
    # print(noes['param_groups'])
    optim.load_state_dict(optim_weights)
    return optim

def freeze_parameters(model,
                      substring:str,
                      reverse:bool = False):
    if reverse:
        for name, param in model.named_parameters():
            if substring not in name:
                param.requires_grad = False  
        return model
    else:
        for name, param in model.named_parameters():
            if substring in name:
                param.requires_grad = False  
        return model        

    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}") 

if __name__ == '__main__':
    
    pass