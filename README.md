# Diffusion4astro

This is the codebase for [Bayesian Deconvolution of Astronomical Images with Diffusion Models: Quantifying Prior-Driven Features in Reconstructions](ArXiv).

# Usage

This section of the README walks through how to train and sample from a model, then it focuses on the deployment through the .ipynb files.

## Installation

Clone this repository and navigate to it in your terminal. Then install the required packages through the `requirements.txt` file.

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

TODO: The training can also be done in parallel (Wassim please write this part).

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

TODO: The sampling can also be done in parallel (Wassim please write this part).

## Solving the inverse problem

Similarly, one could sample multiple times from the posterior distribution, i.e., use the DPS algorithm to solve the inverse problem and sample according to it. We provided two modified versions of the `image_sample.py` script, called `image_sample_inv.py` and `image_sample_inv_HSC.py`, that do this.

- `image_sample_inv.py`: takes an Idealized image from the TNG folder to which artificially adds a certain level of noise after having fixed a certain magnitude. Once the image has been blurred the script samples `num_samples` samples from the posterior samples (the `num_samples` parameter has to be modified in the `create_argparser()` function inside the script).

- `image_sample_inv_HSC.py`: takes an actual HSC image from the HSC PDR3 folder and then samples `num_samples` samples from the posterior samples (the `num_samples` parameter has to be modified in the `create_argparser()` function inside the script).

## Running the notebooks

TODO
