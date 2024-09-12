import os
import numpy as np
from PIL import Image

#########################################################
#   Script to save the sampled images into .png files   #
#########################################################

# Function to save the first m images from a .npz file as PNG
def save_images_as_png(npz_path, output_dir, m=1024):
    # Load the data from .npz file
    data = np.load(npz_path)
    images = data['arr_0']

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the first m images
    for i in range(min(images.shape[0],m)):
        # Create the full path for each image
        image_path = os.path.join(output_dir, f'image_{i:03d}.png')
        # Save the image 
        Image.fromarray(((images[i][...,0:3]) * 255).clip(0, 255).astype(np.uint8)).save(image_path)

    print(f"{images.shape[0]} images have been saved in the directory: {output_dir}")

cur_dir = os.getcwd()
npz_path = cur_dir + 'samples/samples_1024x256x256x3_450k.npz'
output_dir = cur_dir + 'samples/samples_fits_png_450k'

# Call the function
save_images_as_png(npz_path, output_dir, 1024)