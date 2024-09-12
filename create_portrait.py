import os
import random
from PIL import Image

def create_grid_image(source_folder, output_file, grid_size=(16, 16), each_image_size=(256, 256)):
    # Get all image paths from the source folder
    image_paths = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Randomly select n images
    selected_images = random.sample(image_paths, 256) if len(image_paths) >= 256 else image_paths

    # Define the size of the grid
    grid_width = grid_size[0] * each_image_size[0]
    grid_height = grid_size[1] * each_image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste images into the grid
    for index, image_path in enumerate(selected_images):
        image = Image.open(image_path)
        image = image.resize(each_image_size)
        row = index // grid_size[0]
        col = index % grid_size[0]
        grid_image.paste(image, (col * each_image_size[0], row * each_image_size[1]))

    # Save the grid image
    grid_image.save(output_file)

# Usage
cur_dir = os.getcwd()
source_folder = cur_dir + 'samples/example_dir'
output_file = cur_dir + 'portrait.png'
create_grid_image(source_folder, output_file)