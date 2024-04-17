import numpy as np
import astropy
import matplotlib
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import zoom

import astropy.io.fits as _fits
from astropy.nddata import block_reduce, block_replicate
from astropy.cosmology import Planck15
import torch
import torchvision.transforms.functional as TF

pixel_conversion = {}
pixel_conversion['HSC'] = 0.17 * astropy.units.arcsec # arcsec / pixel
pixel_conversion['JWST'] = 0.03 * astropy.units.arcsec # arcsec / pixel

hist_kwargs = {'bins' : 'auto', 'histtype' : 'step', 'density' : True}

# Create an instance of the Planck 2015 cosmology
cosmo = Planck15

def summarize_cosmology():
    # Access various cosmological parameters
    H0 = cosmo.H0  # Hubble constant in km/s/Mpc
    Omega_m = cosmo.Om0  # Matter density parameter
    Omega_lambda = cosmo.Ode0  # Dark energy density parameter

    print("Hubble constant (H0):", H0)
    print("Matter density parameter (Omega_m):", Omega_m)
    print("Dark energy density parameter (Omega_lambda):", Omega_lambda)

def histogram(data, **hist_kwargs):

    data_hist = data.flatten()

    plt.hist(data_hist, **hist_kwargs)

def plot_image(image_data, vmin = None, vmax = None):

    if(vmin == None): vmin = np.min(image_data)
    if(vmax == None): vmax = np.max(image_data)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.imshow(image_data, cmap='jet', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.close()


def arcsec_to_radian(x):
    return np.pi / 648000 * x

def radian_to_arcsec(x):
    return x / (np.pi / 648000)

def get_down_scale_factor(z = 0.2, unit_pixel_in_kpc = 0.1, experiment = 'HSC'):

    """
    Returns the scale factor at which the image has to be decreased for a given redshift.
    If the scale factor is >1, we decrease the image size.

    """

    arcsec_per_kpc_at_z = Planck15.arcsec_per_kpc_proper(z)

    unit_pixel = unit_pixel_in_kpc * astropy.units.kpc

    down_scale_factor = 1 / unit_pixel * pixel_conversion[experiment] / arcsec_per_kpc_at_z

    if(down_scale_factor.unit == ''):
        return down_scale_factor.value
    else:
        raise "Error, factor carries a unit. "


def magnitude_to_flux_in_janskies(magnitude_values):

    """
    See https://en.wikipedia.org/wiki/AB_magnitude

    """


    return 10 ** ((magnitude_values - 8.9) / (-2.5))


def get_image_in_janski(image_data, z):

    unit_pixel_in_kpc = 0.1

    unit_pixel = unit_pixel_in_kpc * astropy.units.kpc

    pixel_width_in_arcsec = Planck15.arcsec_per_kpc_proper(z) * unit_pixel

    image_in_janskis = pixel_width_in_arcsec.value ** 2 * magnitude_to_flux_in_janskies(image_data)

    return image_in_janskis

def get_downscaled_image_at_z_in_janski(image_data, z, experiment = 'HSC', use_zoom_func = False):

    image_data_in_janski = get_image_in_janski(image_data, z = z)

    reduce_factor = get_down_scale_factor(z, experiment = experiment)

    if(use_zoom_func):
        # Open the image file
        # use zoom function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
        image_smaller = zoom(image_data_in_janski, 1 / reduce_factor)
    else:
        image_smaller = block_reduce(image_data_in_janski, reduce_factor)

    return image_smaller

def normalize_min_max_func(x):

    x_min = np.min(x)
    x_max = np.max(x)

    return (x - x_min) / (x_max - x_min)

'''
def ScaleImage(img, z = 0.1, experiment = 'HSC', new_size = 150, normalize_min_max = True):

    """
    A function that converts the image as an array to the desired
    """

    rescaled_image = get_downscaled_image_at_z_in_janski(
        image_data = img,
        z = z,
        experiment = experiment,
        use_zoom_func = True
    )

    image_desired_size = tf.image.resize_with_crop_or_pad(rescaled_image[:,:,None], new_size, new_size)[:,:,0].numpy()

    if(normalize_min_max):
        return normalize_min_max_func(image_desired_size)
    else:
        return image_desired_size
'''

def ScaleImage(img, z=0.1, experiment='HSC', new_size=150, normalize_min_max=True):
    """
    A function that converts the image as an array to the desired
    """

    rescaled_image = get_downscaled_image_at_z_in_janski(
        image_data=img,
        z=z,
        experiment=experiment,
        use_zoom_func=True
    )

    if not isinstance(rescaled_image, torch.Tensor):
        rescaled_image = torch.from_numpy(rescaled_image) if isinstance(rescaled_image, np.ndarray) else torch.tensor(rescaled_image)

    # Resize and crop or pad the image to the new size
    image_desired_size = TF.resize(rescaled_image.unsqueeze(0), [new_size, new_size])  # resize slightly larger to ensure cropping is possible
    image_desired_size = TF.center_crop(image_desired_size, [new_size, new_size]).squeeze(0).numpy()

    if normalize_min_max:
        return normalize_min_max_func(image_desired_size)
    else:
        return image_desired_size