import os
import numpy as np
from PIL import Image


# Function to save the first 100 images from a .npz file as PNG
def save_images_as_png(npz_path, output_dir):
    # Load the data from .npz file
    data = np.load(npz_path)
    images = data['arr_0']


    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the first 100 images
    for i in range(images.shape[0]):
        # Create the full path for each image
        image_path = os.path.join(output_dir, f'image_{i:03d}_.png')
        # Save the image
        Image.fromarray(images[i]).save(image_path)

    print(f"{images.shape[0]} images have been saved in the directory: {output_dir}")

# Assuming that the .npz file is in the current directory and named 'file.npz'
npz_path = '/gpfswork/rech/tkc/uwa51yi/DDPMv2/improved-diffusion/samples/samples_1024x128x128x3.npz'
output_dir = '/gpfswork/rech/tkc/uwa51yi/DDPMv2/improved-diffusion/samples/samples_color2'

# Call the function
save_images_as_png(npz_path, output_dir)