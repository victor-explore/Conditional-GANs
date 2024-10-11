# Conditional GANs for Animal Image Generation

This project implements a Conditional Generative Adversarial Network (CGAN) for generating images of animals based on class labels. The CGAN is trained on a dataset of animal images and can generate new, synthetic images of animals from specified classes.

## Features

- Conditional GAN architecture for class-specific image generation
- ResNet-style generator and discriminator models
- Wasserstein GAN with gradient penalty (WGAN-GP) training
- DiffAugment data augmentation technique
- TensorBoard integration for loss and gradient visualization
- Ability to generate images for multiple classes

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- PIL

## Usage

1. Prepare your dataset of animal images, organized in folders by class.
2. Update the `classes.txt` file with the names of the animal classes you want to use.
3. Run the Jupyter notebook `Conditional GANs.ipynb` to train the CGAN and generate images.

## Model Architecture

- **Generator**: ResNet-style architecture with conditional batch normalization
- **Discriminator**: ResNet-style architecture with spectral normalization

## Training

The model is trained using the WGAN-GP loss function with the following hyperparameters:

- Latent dimension: 100
- Number of classes: 20
- Image size: 128x128
- Learning rates: 0.0003 (Generator), 0.00031 (Discriminator)
- Batch size: 64
- Number of epochs: 1000

## Results

Generated images are saved in the `CGAN/generated_images/` directory. The notebook also includes functionality to display generated images for each class.

## Acknowledgements

This implementation is inspired by various GAN architectures and techniques, including:

- Wasserstein GAN with Gradient Penalty
- ResNet architecture for GANs
- DiffAugment for GAN training

## License

This project is open-source and available under the MIT License.
