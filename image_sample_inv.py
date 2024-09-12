"""
Solve the inverse problem multiple times and save the results as a large numpy array.
We solve the inverse problem applied to an idealized TNG image to which we artificially apply
the PSF, the noise and fix a certain magnitude.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import matplotlib.pyplot as plt
from PIL import Image
import astropy.units
from astropy.cosmology import Planck15
from scipy.ndimage import zoom
from astropy.nddata import block_reduce
from astropy.io import fits
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, SinhStretch, LogStretch, LinearStretch, AsinhStretch)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Pixel conversion factors
pixel_conversion = {
    'HSC': 0.17 * astropy.units.arcsec,  # arcsec / pixel
    'JWST': 0.03 * astropy.units.arcsec  # arcsec / pixel
}

# Create an instance of the Planck 2015 cosmology
cosmo = Planck15

# Average min and max values from the training set. They are used to reverse the normalization step made during training
max_r, max_g, max_b = 4.1556867948239864e-07, 1.3491573542387314e-06, 2.6782236060862626e-06
min_r, min_g, min_b = 3.6307805477010026e-17, 3.6307805477010026e-17, 3.6307805477010026e-17

def pilimg_to_tensor(pil_img):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  t = t.to(device)
  return(t)

def crop_center(img, cropx, cropy, offsetx=0, offsety=0):
    """
    Crop the center of an image with an optional offset.

    Parameters:
    img (numpy.ndarray or torch.Tensor): The input image.
    cropx (int): The width of the cropped image.
    cropy (int): The height of the cropped image.
    offsetx (int, optional): The horizontal offset for the center. Default is 0.
    offsety (int, optional): The vertical offset for the center. Default is 0.

    Returns:
    numpy.ndarray or torch.Tensor: The cropped image.
    """
    y, x = img.shape[-2:]  # Handle both 2D and 3D images
    centerx = x // 2 + offsetx
    centery = y // 2 + offsety
    startx = max(centerx - cropx // 2, 0)
    starty = max(centery - cropy // 2, 0)
    endx = min(centerx + cropx // 2, x)
    endy = min(centery + cropy // 2, y)
    return img[..., starty:endy, startx:endx]

###########################################################################################################

# Get the PSF from the TNG Realisitc image and use it after to convolve the Idealized one

cur_dir = os.getcwd()
file = fits.open(cur_dir + 'TNG/shalo_089-1_v0_HSC_GRIZY.fits')

PSF_g = file[3].data
PSF_r = file[7].data
PSF_i = file[11].data

x_smallest = min(PSF_g.shape[0], PSF_r.shape[0], PSF_i.shape[0])
y_smallest = min(PSF_g.shape[1], PSF_r.shape[1], PSF_i.shape[1])

PSF_g = crop_center(PSF_g, y_smallest, x_smallest)
PSF_r = crop_center(PSF_r, y_smallest, x_smallest)
PSF_i = crop_center(PSF_i, y_smallest, x_smallest)

psf_kernel = torch.from_numpy(np.dstack((PSF_g, PSF_r, PSF_i)).transpose((2,0,1))).float()

# Define cosmology quantities for the f_{z,p} function

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
    plt.hist(data_hist.numpy(), **hist_kwargs)

def plot_image(image_data, vmin=None, vmax=None, title=''):
    if vmin is None: vmin = torch.min(image_data)
    if vmax is None: vmax = torch.max(image_data)
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.imshow(image_data.permute(1, 2, 0).cpu().numpy(), vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()

def arcsec_to_radian(x):
    return np.pi / 648000 * x

def radian_to_arcsec(x):
    return x / (np.pi / 648000)

def get_down_scale_factor(z=0.1, unit_pixel_in_kpc=0.4, experiment='HSC'):
    """
    Returns the scale factor at which the image has to be decreased for a given redshift.
    If the scale factor is >1, we decrease the image size.
    """
    arcsec_per_kpc_at_z = Planck15.arcsec_per_kpc_proper(z)
    unit_pixel = unit_pixel_in_kpc * astropy.units.kpc
    down_scale_factor = 1 / unit_pixel * pixel_conversion[experiment] / arcsec_per_kpc_at_z

    if down_scale_factor.unit == '':
        return down_scale_factor.value
    else:
        raise ValueError("Error, factor carries a unit.")

def get_image_in_janski(image_data, z):
    unit_pixel_in_kpc = 0.4
    unit_pixel = unit_pixel_in_kpc * astropy.units.kpc
    pixel_width_in_arcsec = Planck15.arcsec_per_kpc_proper(z) * unit_pixel
    image_in_janskis = pixel_width_in_arcsec.value ** 2 * image_data
    return image_in_janskis

def get_downscaled_image_at_z_in_janski(image_data, z, experiment='HSC', use_zoom_func=False):
    image_data_in_janski = get_image_in_janski(image_data, z=z)
    reduce_factor = get_down_scale_factor(z, experiment=experiment)

    if use_zoom_func:
        scale_factor = 1 / reduce_factor
        new_size = (int(image_data_in_janski.shape[-2] * scale_factor), int(image_data_in_janski.shape[-1] * scale_factor))
        image_smaller = F.interpolate(image_data_in_janski, size=new_size, mode='bilinear', align_corners=False)
    else:
        # Not yet implemented in pytorch version
        image_smaller = block_reduce(image_data_in_janski, (1, 1, reduce_factor, reduce_factor))

    return image_smaller

def normalize_min_max_func(x):
    """
    Normalize each channel of the tensor to the range [0, 1].
    
    Parameters:
    x (torch.Tensor): The input tensor.
    
    Returns:
    torch.Tensor: The normalized tensor.
    """
    min_val = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    return (x - min_val) / (max_val - min_val + 1e-18)

def _ScaleImage(img, z=0.1, experiment='HSC', new_size=256, normalize_min_max=True):
    """
    A function that converts the image as a tensor to the desired format.
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
    #rescaled_image = rescaled_image.unsqueeze(0).unsqueeze(0) if rescaled_image.dim() == 2 else rescaled_image.unsqueeze(0)
    #rescaled_image = F.interpolate(rescaled_image, size=(new_size, new_size), mode='bilinear', align_corners=False)

    if normalize_min_max:
        return normalize_min_max_func(rescaled_image)
    else:
        return rescaled_image

z = file[0].header['REDSHIFT']

def pre_process(img, z = z, experiment='HSC', new_size=256, normalize_min_max=True):
    return _ScaleImage((img+1)/2, experiment=experiment, new_size=new_size, z=z, normalize_min_max=normalize_min_max)*2 - 1

def pre_process2(img, z = z, experiment='HSC', new_size=256, normalize_min_max=False):
    return _ScaleImage(img, experiment=experiment, new_size=new_size, z=z, normalize_min_max=normalize_min_max)

# Define the other parts of the operator A, the PSF and the functions to apply/reverse normalization and stretching

def apply_psf_kernel(x, psf_kernel=psf_kernel):
    """
    Applies the provided PSF kernel to each channel of a tensor x.
    
    Parameters:
    - x (torch.Tensor): The input tensor of shape [N, C, H, W].
    - psf_kernel (torch.Tensor): The PSF kernel of shape [C, kernel_size, kernel_size].
    
    Returns:
    - torch.Tensor: The tensor after applying the PSF kernel.
    """

    device = x.device  # Get the device from the input tensor
    channels = x.size(1)  # Assuming x has shape [N, C, H, W]
    original_size = x.shape[2:]
    
    # Ensure the PSF kernel has the shape [C, 1, kernel_size, kernel_size]
    if psf_kernel.dim() == 3:
        psf_kernel = psf_kernel.unsqueeze(1).to(device)
    
    # Apply convolution using the multi-channel PSF kernel
    kernel_size = psf_kernel.size(2)  # Assuming square kernels
    padding = kernel_size // 2
    x_psf = F.conv2d(x, psf_kernel, padding=padding, groups=channels)
    x_psf = TF.center_crop(x_psf, original_size)
    
    return x_psf.float()

# This minmax function considers channel-wise normalization
def normalize_min_max_func2(x):
    """
    Normalize each channel of the tensor to the range [0, 1].
    
    Parameters:
    x (torch.Tensor): The input tensor.
    
    Returns:
    torch.Tensor: The normalized tensor.
    """
    min_val = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    return (x - min_val) / (max_val - min_val + 1e-18)

def inverse_normalize_min_max_func2(x, min_r=min_r, min_g=min_g, min_b=min_b, max_r=max_r, max_g=max_g, max_b=max_b):
    """
    Reverse the normalization applied by normalize_min_max_func2.
    
    Parameters:
    x (torch.Tensor): The input tensor to be denormalized.
    min_r, min_g, min_b (float): Minimum values for the red, green, and blue channels respectively.
    max_r, max_g, max_b (float): Maximum values for the red, green, and blue channels respectively.
    
    Returns:
    torch.Tensor: The denormalized tensor.
    """
    # Create tensors for min and max values for each channel
    min_vals = torch.tensor([min_r, min_g, min_b], device=x.device).view(-1, 1, 1)
    max_vals = torch.tensor([max_r, max_g, max_b], device=x.device).view(-1, 1, 1)
    
    # Reverse the normalization
    return (x * (max_vals - min_vals) + min_vals).float()


def asinh_stretch(tensor, a):
    """
    Apply the asinh stretch to the input tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to be stretched.
    a (float): The stretch parameter.
    
    Returns:
    torch.Tensor: The asinh stretched tensor.
    """
    return torch.asinh(tensor / a) / torch.asinh(torch.tensor(1.0 / a))

def inverse_asinh_stretch(tensor, a):
    """
    Apply the inverse of the asinh stretch to the input tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to be inversely stretched.
    a (float): The stretch parameter.
    
    Returns:
    torch.Tensor: The inverse asinh stretched tensor.
    """
    return a * torch.sinh(tensor * torch.asinh(torch.tensor(1.0 / a)))

# Operator A to pass to the inverse problem solver, a1 must be set to 0.01 to match the stretching used for training
def composition2(x, func1 = pre_process2, func2 = apply_psf_kernel, a1 = 0.01, a2 = 0.001):
    return normalize_min_max_func2(asinh_stretch(func2(func1(inverse_asinh_stretch(inverse_normalize_min_max_func2((x+1)/2), a1))), a2))*2 - 1

# Operator A to use to create the y observation
def composition(x, func1 = pre_process2, func2 = apply_psf_kernel):
    return func2(func1(x))


################################################################################################################


def main():

    cur_dir = os.getcwd()
    file = fits.open(cur_dir + 'TNG/shalo_089-1_v0_photo.fits')

    r = file[1].data
    g = file[2].data
    b = file[4].data

    interval = MinMaxInterval()
    transform = interval
    transform2 = AsinhStretch(a=0.001) + interval

    # Transform Magnitude to Janski

    r = 10 ** ((r - 8.9) / (-2.5))
    g = 10 ** ((g - 8.9) / (-2.5))
    b = 10 ** ((b - 8.9) / (-2.5))

    # Change pixel scaling

    unit_pixel = 1/4

    r = zoom(r, unit_pixel)
    g = zoom(g, unit_pixel)
    b = zoom(b, unit_pixel)

    # Compute old flux

    old_flux_r = np.sum(r)
    old_flux_g = np.sum(g)
    old_flux_b = np.sum(b)

    # compute x_true for testing purposes
    x_true = pilimg_to_tensor(np.dstack((transform2(crop_center(r, 256, 256)), transform2(crop_center(g, 256, 256)), transform2(crop_center(b, 256, 256)))))

    sizes = pre_process(x_true).shape[2:]

    # Apply pre-processing and PSF

    r = composition(torch.tensor(r).unsqueeze(0).unsqueeze(0)).squeeze(0)[0].numpy()
    g = composition(torch.tensor(g).unsqueeze(0).unsqueeze(0)).squeeze(0)[0].numpy()
    b = composition(torch.tensor(b).unsqueeze(0).unsqueeze(0)).squeeze(0)[0].numpy()

    # Normalize at a given magnitude (mag=18 in this example)

    mag_22 = 10 ** ((18 - 8.9) / (-2.5))

    r = r*0.167**2*sizes[0]**2*mag_22/old_flux_r
    g = g*0.167**2*sizes[0]**2*mag_22/old_flux_g
    b = b*0.167**2*sizes[0]**2*mag_22/old_flux_b

    r = crop_center(r, 512, 512)
    g = crop_center(g, 512, 512)
    b = crop_center(b, 512, 512)

    # Apply noise

    sigma_noise = 2.0302566460297157e-09

    r = r + sigma_noise * np.random.randn(*r.shape)
    g = g + sigma_noise * np.random.randn(*g.shape)
    b = b + sigma_noise * np.random.randn(*b.shape)

    # Stretch

    transform = AsinhStretch(a=0.01) + interval

    r = transform(r)
    g = transform(g)
    b = transform(b)

    # Stack arrays into a 3D array for RGB image
    rgb_image = np.dstack((r, g, b))

    x_noisy = pilimg_to_tensor(rgb_image)
    y = TF.center_crop(x_noisy, pre_process(x_true).shape[2:])

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop_inv if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            y,
            composition2,   # pass the operator A
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = (sample + 1)/2
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        cur_dir = os.getcwd()
        out_path = os.path.join(cur_dir, "samples", f"samples_{shape_str}_inv_mag_18.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=256,
        batch_size=4,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
