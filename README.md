# SwinUNETR-Pix2Pix for Medical Image Translation

This repository contains the code for an image-to-image translation project between different medical imaging modalities, such as converting from Magnetic Resonance Imaging (MRI) to Computed Tomography (CT) and vice versa.

The architecture is based on the **Pix2Pix** framework, a type of conditional Generative Adversarial Network (GAN), with a key modification: the generator is implemented using a **SwinUNETR**, a state-of-the-art model that combines Swin Transformers with a U-Net architecture for medical image segmentation and reconstruction tasks.

## Repository Structure

- **/data**: Contains scripts for data loading, transformations, and dataloader creation.
- **/experiments**: Contains the configuration files (`.yaml`) for training and testing experiments, separated by translation type (e.g., `MR2CT`, `CT2MR`).
- **/models**: Defines the network architectures, including the generator (`generator.py`), the discriminator (`discriminator.py`), and the overall Pix2Pix model (`pix2pix.py`).
- **/utils**: Utility functions for analysis, visualization, and other support operations.
- `train.py`: Main script to start model training.
- `test.py`: Main script to run inference and model evaluation.

## Code Usage

### 1. Experiment Configuration

Before starting training or testing, you need to configure the experiment by modifying one of the `.yaml` files in the `experiments` folder. These files allow you to specify:
- Paths to training and validation data.
- Model hyperparameters (e.g., learning rate, batch size).
- Specific architecture options.

### 2. Training

To start the training process, run the `train.py` script, passing the chosen configuration file as an argument.

```bash
python train.py --config_file experiments/MR2CT/options_1.yaml
```

### 3. Testing and Inference

Once training is complete, you can use the trained model to generate images on a test dataset. Run the `test.py` script with the same configuration file used for training.

```bash
python test.py --options experiments/MR2CT/options_1.yaml
```

The results will be saved in the directories specified in the configuration file.


## License

This project is distributed under the terms of the license specified in the `LICENSE` file.