#  GAN-Based Image Completion using Easier Doodle Dataset

The model learns to reconstruct missing or incomplete parts of hand-drawn doodles using a U-Net-like Generator with residual blocks and self-attention, and a Convolutional Discriminator.

 **Dataset link:**  
https://www.kaggle.com/code/ashishjangra27/easier-doodle-dataset

## Model Architecture


1. **Input:** A partially masked doodle image (e.g. random patches removed).
2. **Generator:** Attempts to complete the missing parts of the doodle.
3. **Discriminator:** Tries to classify whether an image is real (from the dataset) or generated (from the Generator).
4. **Training:** The Generator and Discriminator train in a minimax game — the Generator gets better at fooling the Discriminator, and the Discriminator gets better at spotting fakes.


## How to RUN

Install the required libraries....

## 1. Prepare the dataset (load .npy files or pre-saved images)
## 2. Create Generator and Discriminator
generator = build_generator()
discriminator = build_discriminator()
## 3. Train using a GAN training loop
## 4. Visualize results

