import numpy as np

from blur import apply_psf, add_blur
from noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from imutils import downsample_raw, convert_to_tensor


def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img)

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)
    
    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise > 0.3:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)
    
    return img