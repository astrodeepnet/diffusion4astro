import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import os

################################################################################################
#   Script to compute FID given a path with the Inceptionv3 model and two folders of images.    #
#   The folders with the images must contain the images inside a subfolder.                    #
################################################################################################

def load_pretrained_model(model_path):
    model = models.inception_v3(weights=None, init_weights=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_features(model, dataloader, device):
    model = model.to(device)
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.detach().cpu().numpy())
    return np.concatenate(features, axis=0)

def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1 @ sigma2, disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def main(model_path, path1, path2, device):
    model = load_pretrained_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset1 = datasets.ImageFolder(root=path1, transform=transform)
    dataset2 = datasets.ImageFolder(root=path2, transform=transform)
    dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False)
    features1 = calculate_features(model, dataloader1, device)
    features2 = calculate_features(model, dataloader2, device)
    fid_score = calculate_fid(features1, features2)
    print(f"FID score: {fid_score}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cur_dir = os.getcwd()
    model_path = cur_dir + 'Inception/inception_v3.pth'
    path1 = cur_dir + 'Inception/example_dir_train'
    path2 = cur_dir + 'samples/example_dir_samples'
    main(model_path, path1, path2, device)