# Diffusion4astro

This is the codebase for [Bayesian Deconvolution of Astronomical Images with Diffusion Models: Quantifying Prior-Driven Features in Reconstructions](https://arxiv.org/abs/2411.19158).

<p align="center">
    <img src="./Results/test1.png" width="400"/>
    <img src="./Results/test2.png" width="400"/>
</p>

<p align="center">
    <img src="./Results/Pipeline.png" width="400"/>
</p>

# Usage

This section of the README walks through how to train and sample from a model, then it focuses on the deployment through the .ipynb files.

## Installation

Clone this repository and navigate to it in your terminal. Then install the required packages through the `requirements.txt` file. This will be sufficient to run all the notebooks. To run the sampling and training script mpi4py is necessary, it can be installed following:

```
sudo apt install libmpich-dev
```

and then

```
pip install --no-binary=mpi4py mpi4py
```
## Training

To train your model, you should first decide some hyperparameters. We remind that the training code is an extended version of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), where the authors split up the hyperparameters into three groups: model architecture, diffusion process, and training flags. We can train a DDPM given a cached training set named `TNG.pth` or `TNG_276x276.pth`: the first case is for the standard dataset, the second one is if we want to activate the random 10px shift. To choose between the two, the `improved_diffusion/TNG_Ideal.py` class `TNGDataset` must be modified accordingly, together with the `__getitem__` function inside the same script (the default is to use the random shift version). To train we'll then run:

```
python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

where:

```
MODEL_FLAGS="--image_size 256 --num_channels 192 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --use_fp16 True" 
``` 

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples. We provide the model used to produce the experiments shown in the paper: [https://zenodo.org/records/14097251](https://zenodo.org/records/14097251).

Once you have a path to your model, you can generate a large batch of samples like so:

```
python image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

## Solving the inverse problem

Similarly, one could sample multiple times from the posterior distribution, i.e., use the DPS algorithm to solve the inverse problem and sample according to it. We provided two modified versions of the `image_sample.py` script, called `image_sample_inv.py` and `image_sample_inv_HSC.py`, that do this.

- `image_sample_inv.py`: takes an Idealized image from the TNG folder (contained in the `TNG.7z` file) to which artificially adds a certain level of noise after having fixed a certain magnitude. Once the image has been blurred the script samples `num_samples` samples from the posterior samples (the `num_samples` parameter has to be modified in the `create_argparser()` function inside the script).

- `image_sample_inv_HSC.py`: takes an actual HSC image from the `HSC PDR3` folder and then samples `num_samples` samples from the posterior samples (the `num_samples` parameter has to be modified in the `create_argparser()` function inside the script).

## Running the notebooks

The notebooks provided in the repository mean to give an easy way to deploy the DPS algorithm applied to different settings.

- The `DPS_TNG.ipynb` notebook takes a TNG-HSC image and tries to solve the deconvolution task given the redshift and PSF information contained in the TNG .fits file header. A comparison with the ground-truth TNG simulation is possible.
- The `DPS_HSC.ipynb` notebook takes a real HSC cropped image and the associated approximated PSF from the `HSC PDR3` folder, and tries to solve the deconvolution task. The redshift information is contained in the .csv files and must be manually set in the z variable inside the notebook.
- The `DPS_hallucinations.ipynb` notebook takes a Idealized TNG image, artificially adds a PSF and a Gaussian noise sampled from real HSC images, sets it to the HSC pixel scale, changes the magnitude to modify the SNR, and then tries to solve the deconvolution task given the redshift contained in the TNG .fits file header.
- The `DPS_variances.ipynb` notebook takes as input .npz files containing samples (like those we obtain as output of the sampling process) and computes the mean and variance. Similarly, it computes the Posterior Mean when it's run on samples obtained from the DPS algorithm.
- The `DPS_cat.ipynb` notebook shows an example of recostruction of a noisy and convolved image that does not belong to the training set manifold, such as the image of a cat. As the noise level increases we observe how the prior injects more information into the recostruction, ending up with a TNG-like cat-shaped galaxy.
- The `Deblending.ipynb` notebook shows a proof-of-concept of how to use the DPS algorithm and a DDPM model trained on the TNG dataset to address thet deblending task.
- The `UMAP.ipynb` notebook can be used to compute a UMAP projection of the training TNG images (given a folder containing them) and can then project sampled images, saving the representation as a .html file.

## Cite

To cite this work, please use the following reference:

```bibtex
@misc{spagnoletti2024bayesiandeconvolutionastronomicalimages,
      title={Bayesian Deconvolution of Astronomical Images with Diffusion Models: Quantifying Prior-Driven Features in Reconstructions}, 
      author={Alessio Spagnoletti and Alexandre Boucaud and Marc Huertas-Company and Wassim Kabalan and Biswajit Biswas},
      year={2024},
      eprint={2411.19158},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2411.19158}, 
}
```
If you find this repository useful for your research, please consider citing it to acknowledge our work.
