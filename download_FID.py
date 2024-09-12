import torch
import torchvision.models as models
import os

#################################################################################
#   Script to download and save the Inceptionv3 model for the FID computation   #
#################################################################################

def download_and_save_model(model_dir):
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Download Inception v3 model
    model = models.inception_v3(pretrained=True)

    # Save the model state
    model_path = os.path.join(model_dir, 'inception_v3.pth')
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    cur_dir = os.getcwd()
    model_dir = cur_dir + 'Inception'
    download_and_save_model(model_dir)