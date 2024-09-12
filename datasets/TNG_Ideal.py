import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from scipy.ndimage import zoom
import astropy
from astropy.visualization import astropy_mpl_style
from astropy.visualization import MinMaxInterval, SqrtStretch, LogStretch, AsinhStretch, LinearStretch, PowerStretch, ImageNormalize
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.cosmology import Planck15

from random import shuffle
from PIL import Image

##################################################################
#   Script to extract and save .png images out of .fits files    #
##################################################################


class TNGDataset(Dataset):
    def __init__(self, input_dir, band_filters, split='train', train_percent=0.9, val_percent=0.0, device='cpu'):
        self.band_filters = band_filters
        self.data = []
        self.device = device

        # Populate list of FITS files
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                if name.endswith("photo.fits"):
                    self.data.append(os.path.join(root, name))


        # Split data into training, validation, and test sets
        num_samples = len(self.data)
        train_end = int(train_percent * num_samples)
        val_end = train_end + int(val_percent * num_samples)

        if split == 'train':
            self.data = self.data[:train_end]
        elif split == 'validation':
            self.data = self.data[train_end:val_end]
        elif split == 'test':
            self.data = self.data[val_end:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        # Store tensors by filter type
        filter_tensors = {filter_name: None for filter_name in self.band_filters}
        metadata = {}
        try:
            with fits.open(file_path) as fitm:
                for fit_elem in fitm:
                    extname = fit_elem.header["EXTNAME"]
                    if extname in self.band_filters:
                        # Process the image data

                        interval = MinMaxInterval()
                        transform = AsinhStretch(a=0.01) + interval

                        # magnitude to flux in janskies
                        img = 10 ** ((fit_elem.data - 8.9) / (-2.5))

                        # change pixel scale
                        unit_pixel = 1/4
                        img = zoom(img, unit_pixel)
                        img = transform(img)

                        img_tensor = torch.tensor(img, dtype=torch.float32, device=self.device)
                        image_desired_size = TF.center_crop(img_tensor, [256, 256])

                        # Store tensor in the corresponding filter entry
                        filter_tensors[extname] = image_desired_size

                        # Merge metadata from all matched elements
                        metadata.update({
                            "ORIGIN": fit_elem.header["ORIGIN"],
                            "SIMTAG": fit_elem.header["SIMTAG"],
                            "SNAPNUM": fit_elem.header["SNAPNUM"],
                            "SUBHALO": fit_elem.header["SUBHALO"],
                            "CAMERA": fit_elem.header["CAMERA"],
                            "REDSHIFT": fit_elem.header["REDSHIFT"]
                        })

            # Create RGB channels by combining gri filters
            red = filter_tensors['SUBARU_HSC.G']
            green = filter_tensors['SUBARU_HSC.R']
            blue = filter_tensors['SUBARU_HSC.I']
            rgb_image = torch.stack([red, green, blue], dim=0).unsqueeze(0)  # Stack along color channel dimension
            return rgb_image, metadata
        except Exception as e:
                print(f"Error processing file {file_path}: {e}")

def main():
    for split in ["train", "test"]:
        cur_dir = os.getcwd()
        out_dir = cur_dir + f"datasets/TNG_{split}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("loading...")
        # Put as input_dir your folder with all the TNG .fits file for training
        dataset = TNGDataset(input_dir='/gpfsstore/rech/tkc/uwa51yi/idealized', band_filters=['SUBARU_HSC.G','SUBARU_HSC.R','SUBARU_HSC.I'], split=split, device='cuda')

        print("dumping images...")
        transform = T.ToPILImage()
        for i in tqdm(range(len(dataset))):
            try:
                if dataset[i] is not None:
                    rgb_image, _ = dataset[i]
                    if rgb_image is not None:
                        # Convert the tensor to a PIL Image and save it
                        img = transform(rgb_image.squeeze(0))  # Remove batch dimension
                        filename = os.path.join(out_dir, f"{i:05d}.png")
                        img.save(filename)
                    else:
                        print(f"Skipping index {i}, no valid image data.")
            except Exception as e:
                print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()