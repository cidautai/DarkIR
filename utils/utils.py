import torch


def load_weights(model, old_weights):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model = model.load_state_dict(new_weights)
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
        
def freeze_parameters(model,
                      substring:str):

    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad_(False)    

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}") 
