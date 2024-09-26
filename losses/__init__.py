from .loss import MSELoss, L1Loss, CharbonnierLoss, SSIM, VGGLoss, EdgeLoss, FrequencyLoss, EnhanceLoss

def create_loss(opt):
    
    '''
    Returns the needed losses for evaluating our model
    '''
    losses = dict()
    
    # first the pixel losses
    if opt['pixel_criterion'] == 'l1':
        pixel_loss = L1Loss()
    elif opt['pixel_criterion'] == 'l2':
        pixel_loss = MSELoss()
    elif opt['pixel_criterion'] == 'Charbonnier':
        pixel_loss = CharbonnierLoss()
    else:
        raise NotImplementedError(f'{opt['pixel_criterion']} is not implemented')

    losses['pixel_loss'] = pixel_loss
    
    # now the perceptual loss
    perceptual = opt['perceptual']
    perceptual_loss = VGGLoss(loss_weight = opt['perceptual_weight'],
                            criterion = opt['perceptual_criterion'],
                            reduction = opt['perceptual_reduction'])
    if perceptual:
        losses['perceptual_loss'] = perceptual_loss
  
    return losses
    