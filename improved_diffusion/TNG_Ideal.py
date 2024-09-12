import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from scipy.ndimage import zoom
import astropy
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.cosmology import Planck15
from astropy.visualization import MinMaxInterval, SqrtStretch, LogStretch, AsinhStretch, LinearStretch, PowerStretch, ImageNormalize

from random import shuffle
from PIL import Image
import blobfile as bf
from mpi4py import MPI

###########################################################################################
#   Script to extract and save images out of .fits files and save them as a .pth tensor   #
###########################################################################################


def load_data(
    *, batch_size, image_size, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # Put as input_dir your folder with all the TNG .fits file for training
    dataset = TNGDataset(input_dir='/gpfsstore/rech/tkc/uwa51yi/idealized', band_filters=['SUBARU_HSC.G', 'SUBARU_HSC.R', 'SUBARU_HSC.I'], split='train', device='cpu')

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

# We can both select the "TNG.pth" or the "TNG_276x276.pth"

class TNGDataset(Dataset):
    def __init__(self, input_dir, band_filters, split='train',
                 train_percent=0.9, val_percent=0.0, device='cuda', cache_file = os.getcwd()+"TNG_276x276.pth"):
        self.band_filters = band_filters
        self.device = device
        self.cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            self.loaded_data = torch.load(cache_file)
        else:
            self.loaded_data = []
            self._load_and_process_data(input_dir, split, train_percent, val_percent)
            if cache_file:
                self._save_processed_data()

    def random_shift_and_crop(self, image, crop_size=256, max_shift=10):
        """
        Shift the center of the image randomly within a [-10, 10] range for both x and y axes
        and then crop a 256x256 region out of the 276x276 image.

        Args:
            image (torch.Tensor): The input image tensor of shape (C, 276, 276).
            crop_size (int, optional): The size of the crop (default is 256).
            max_shift (int, optional): The maximum shift in pixels (default is 10).

        Returns:
            torch.Tensor: The cropped image tensor of shape (C, 256, 256).
        """
        # Generate random shifts for x and y within the range [-10, 10]
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        # Calculate the new center of the image
        center_x = 276 // 2 + shift_x
        center_y = 276 // 2 + shift_y
        
        # Calculate the top-left corner of the crop region
        x_start = center_x - crop_size // 2
        y_start = center_y - crop_size // 2
        
        # Crop the image
        cropped_image = image[:, y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        return cropped_image

    def _load_and_process_data(self, input_dir, split, train_percent, val_percent):
        # Populate list of FITS files
        all_files = []
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                if name.endswith("photo.fits"):
                    all_files.append(os.path.join(root, name))

        # Split data into training, validation, and test sets
        num_samples = len(all_files)
        print(num_samples)
        train_end = int(train_percent * num_samples)
        val_end = train_end + int(val_percent * num_samples)

        if split == 'train':
            self.data = all_files[:train_end]
        elif split == 'validation':
            self.data = all_files[train_end:val_end]
        elif split == 'test':
            self.data = all_files[val_end:]

        # Pre-load the data into memory
        for i, file_path in enumerate(self.data):
            try:
                filter_tensors = {filter_name: None for filter_name in self.band_filters}
                metadata = {}
                print("images loaded: ", i)

                with fits.open(file_path) as fitm:
                    for fit_elem in fitm:
                        extname = fit_elem.header.get("EXTNAME")
                        if extname in self.band_filters:
                            # Check the original image dimensions
                            original_height, original_width = fit_elem.data.shape

                            # Discard the image if it's smaller than 1104x1104 pixels, since then we'll zoom out
                            min_required_size = 276 * 4
                            if original_height < min_required_size or original_width < min_required_size:
                                print(f"Skipping image {extname} due to insufficient size: {fit_elem.data.shape}")
                                continue

                            # Process the image data
                            interval = MinMaxInterval()
                            transform = AsinhStretch(a=0.01) + interval

                            # magnitude to flux in janskies
                            img = 10 ** ((fit_elem.data - 8.9) / (-2.5))

                            # change pixel scale
                            unit_pixel = 1/4
                            img = zoom(img, unit_pixel)
                            img = transform(img)

                            # use 276 if we want to apply the random shift, else set to 256
                            img_tensor = torch.tensor(img, dtype=torch.float32, device=self.device)
                            image_desired_size = TF.center_crop(img_tensor, [276, 276])
                            
                            # Store tensor in the corresponding filter entry
                            filter_tensors[extname] = image_desired_size

                            # Merge metadata from all matched elements
                            metadata.update({
                                "ORIGIN": fit_elem.header.get("ORIGIN"),
                                "SIMTAG": fit_elem.header.get("SIMTAG"),
                                "SNAPNUM": fit_elem.header.get("SNAPNUM"),
                                "SUBHALO": fit_elem.header.get("SUBHALO"),
                                "CAMERA": fit_elem.header.get("CAMERA"),
                                "REDSHIFT": fit_elem.header.get("REDSHIFT")
                            })

                # Create RGB channels by combining specified filters
                red = filter_tensors['SUBARU_HSC.G']
                green = filter_tensors['SUBARU_HSC.R']
                blue = filter_tensors['SUBARU_HSC.I']
                rgb_image = torch.stack([red, green, blue], dim=0)  # Stack along color channel dimension

                self.loaded_data.append((rgb_image, {}))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    def _save_processed_data(self):
        torch.save(self.loaded_data, self.cache_file)

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        # Remove the random shift if we want to use the "TNG.pth" dataset
        return (self.random_shift_and_crop(self.loaded_data[idx][0])*2 - 1, self.loaded_data[idx][1])     #We want the images to be in the range [-1,1]