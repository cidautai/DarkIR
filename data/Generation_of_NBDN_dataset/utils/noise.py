import numpy as np
import scipy.io as scio
from glob import glob
from matplotlib import pyplot as plt


import torch
import torch.distributions as dist


def add_heteroscedastic_gnoise(image, sigma_1_range=(5e-3, 5e-2), sigma_2_range=(1e-3, 1e-2)):
    """
    Adds heteroscedastic Gaussian noise to an image.
    
    Parameters:
    - image: PyTorch tensor of the image.
    - sigma_1_range: Tuple indicating the range of sigma_1 values.
    - sigma_2_range: Tuple indicating the range of sigma_2 values.
    
    Returns:
    - Noisy image: Image tensor with added heteroscedastic Gaussian noise.
    """
    # Randomly choose sigma_1 and sigma_2 within the specified ranges
    sigma_1 = torch.empty(image.size()).uniform_(*sigma_1_range) #set the sigmas for the size of the image in a uniform distribution (r1, r2)
    sigma_2 = torch.empty(image.size()).uniform_(*sigma_2_range)
    
    # Calculate the variance for each pixel
    variance = (sigma_1 ** 2) * image + (sigma_2 ** 2)
    
    # Generate the Gaussian noise
    noise = torch.normal(mean=0.0, std=variance.sqrt())
    
    # Add the noise to the original image
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def add_gnoise(inp_img, var_low=5e-5, var_high=2e-4, variance=None):
    variance = var_low + np.random.rand() * (var_high - var_low)
    # Generate Gaussian noise with 0 mean and the specified variance
    noise = np.sqrt(variance) * torch.randn(inp_img.shape)
    return torch.clamp(inp_img + noise, 0., 1.)

def generate_poisson_(y, k=1):
    y = torch.poisson(y / k) * k
    return y

def generate_read_noise(shape, noise_type, scale, loc=0):
    noise_type = noise_type.lower() #string method for returning a string with all the characters in lowercase
    if noise_type == 'norm':
        read = torch.FloatTensor(shape).normal_(loc, scale)
    else:
        raise NotImplementedError('Read noise type error.')
    return read

#### Noise sampling based on UPI, CycleISP and Model-based Image Signal Processors

def random_noise_levels_dslr():
    """Generates random noise levels from a log-log linear distribution.
    This shot and read noise distribution covers a wide range of photographic scenarios using DSLR cameras.
    """
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.001]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10,log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.1]))
    read_noise = distribution.sample()
    line = lambda x: 1.34 * x + 0.22
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10,log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005, use_cuda=True):
    """Adds random shot (proportional to image) and read (independent) noise."""
    
    assert torch.all(image) >= 0
    assert torch.all(shot_noise) >= 0
    assert torch.all(read_noise) >= 0
    #print (111, read_noise, shot_noise)
    variance = image * shot_noise + read_noise
    scale = torch.sqrt(variance)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), scale)
    noise = distribution.sample()
    noisy_raw = image + noise
    noisy_raw = torch.clamp(noisy_raw,0,1)
    return noisy_raw

def add_natural_noise(inp_img, use_cuda=True):
    '''
    This function takes the other functions to add natural noise to the image.
    '''
    
    shot_noise, read_noise = random_noise_levels_dslr()
    return add_noise(inp_img, shot_noise, read_noise, use_cuda)


if __name__ == "__main__":
    pass