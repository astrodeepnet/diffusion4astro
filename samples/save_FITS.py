import os
import numpy as np
from astropy.io import fits

##########################################################
#   Script to save the sampled images into .fits files   #
##########################################################

# Function to save the images from a .npz file as FITS
def save_images_as_fits(npz_path, output_dir, m):
    # Load the data from .npz file
    data = np.load(npz_path)
    images = data['arr_0']

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(images.shape[0]):
        # Create the full path for each image
        fits_path = os.path.join(output_dir, f'image_{i:03d}.fits')

        # Create a FITS HDU object for each channel
        hdu_r = fits.PrimaryHDU(images[i,:,:,0])
        hdu_g = fits.ImageHDU(images[i,:,:,1])
        hdu_b = fits.ImageHDU(images[i,:,:,2])
        # extra dims
        hdu_y = fits.ImageHDU(images[i,:,:,3])
        hdu_z = fits.ImageHDU(images[i,:,:,4])

        # Create an HDU list and write to the FITS file
        hdul = fits.HDUList([hdu_r, hdu_g, hdu_b, hdu_y, hdu_z])
        hdul.writeto(fits_path, overwrite=True)

    print(f"{images.shape[0]} images have been saved in the directory: {output_dir}")

cur_dir = os.getcwd()
npz_path = cur_dir + 'samples/samples_256x256x256x3.npz'
output_dir = cur_dir + 'samples/samples_fits'

# Call the function
save_images_as_fits(npz_path, output_dir)