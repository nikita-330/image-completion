import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian

# Configuration (MODIFIED)
class Config:
    IMG_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 200  # Reduced epochs for faster training
    LEARNING_RATE = 0.0002
    DATA_PATH = '/kaggle/input/quickdraw-doodle-recognition/train_simplified/'
    NUM_CATEGORIES = 2  # Kept 20 categories
    SAMPLES_PER_CATEGORY = 12000  # Reduced from 1000 to 500
    STROKE_WIDTH = 2
    MIN_STROKES_TO_REMOVE = 1
    MAX_STROKES_TO_REMOVE = 3
    SAVE_INTERVAL = 50  # More frequent saves for monitoring
    VALIDATION_SPLIT = 0.2
    GRADIENT_PENALTY_WEIGHT = 10.0

config = Config()

# Data Loading and Processing 
class QuickDrawLoader:
    def __init__(self):
        self.categories = self._get_categories()
        
    def _get_categories(self):
        files = [f for f in os.listdir(config.DATA_PATH) 
                if f.endswith('.csv')]
        return sorted(files)[:config.NUM_CATEGORIES]
    
    def strokes_to_image(self, strokes):
        img = np.ones((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
        for stroke in strokes:
            pts = np.array(stroke).T.astype(np.int32)
            for i in range(len(pts)-1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), 0, config.STROKE_WIDTH)
        return img
    
    def _remove_random_strokes(self, strokes):
        if len(strokes) <= 1:
            return strokes
        n_to_remove = np.random.randint(
            min(config.MIN_STROKES_TO_REMOVE, len(strokes)-1),
            min(config.MAX_STROKES_TO_REMOVE, len(strokes)-1) + 1
        )
        indices = list(range(len(strokes)))
        remove_indices = np.random.choice(indices, size=n_to_remove, replace=False)
        return [s for i, s in enumerate(strokes) if i not in remove_indices]
    
    def load_data(self):
        X_partial, X_complete = [], []
        
        for cat in tqdm(self.categories, desc="Loading Data"):
            try:
                df = pd.read_csv(os.path.join(config.DATA_PATH, cat), 
                               nrows=config.SAMPLES_PER_CATEGORY)
                for _, row in df.iterrows():
                    strokes = eval(row['drawing'])
                    complete = self.strokes_to_image(strokes)
                    partial = self.strokes_to_image(self._remove_random_strokes(strokes))
                    X_complete.append(complete)
                    X_partial.append(partial)
            except Exception as e:
                print(f"⚠️ Error loading {cat}: {str(e)}")
                continue
                
        X_partial = np.array(X_partial, dtype=np.float32)[..., np.newaxis]
        X_complete = np.array(X_complete, dtype=np.float32)[..., np.newaxis]
        X_partial = (X_partial * 2) - 1
        X_complete = (X_complete * 2) - 1
        
        split_idx = int(len(X_partial) * (1 - config.VALIDATION_SPLIT))
        return (X_partial[:split_idx], X_complete[:split_idx], 
                X_partial[split_idx:], X_complete[split_idx:])

# Model Architecture 
class SelfAttention(layers.Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters
        self.channels = filters // 8
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(1,), initializer='zeros')
        self.query_conv = layers.Conv2D(self.channels, 1, padding='same')
        self.key_conv = layers.Conv2D(self.channels, 1, padding='same')
        self.value_conv = layers.Conv2D(self.filters, 1, padding='same')
        self.out_conv = layers.Conv2D(self.filters, 1, padding='same')
        super().build(input_shape)
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        q_flat = tf.reshape(q, [batch_size, height * width, self.channels])
        k_flat = tf.reshape(k, [batch_size, height * width, self.channels])
        v_flat = tf.reshape(v, [batch_size, height * width, self.filters])
        
        attn_scores = tf.matmul(q_flat, k_flat, transpose_b=True)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)
        
        out = tf.matmul(attn_scores, v_flat)
        out = tf.reshape(out, [batch_size, height, width, self.filters])
        out = self.out_conv(out)
        
        return x + self.gamma * out

def _residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([shortcut, x])


def build_generator():
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1))
    # The function defines a U-Net-like generator with residual blocks and attention that takes in a grayscale image and generates an enhanced or restored  and complete version of the images.


def build_discriminator():
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1))
#This function builds a Discriminator model — whose job is to determine if an input image is real (from the dataset) or fake (generated by the Generator).
# It's essentially a binary classifier that takes in an image and outputs a single number (logit), indicating how likely the image is real.
    
    return models.Model(inputs,  name="Discriminator")

def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    
    gradients = tape.gradient(pred, interpolated)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return penalty

# Training System (UNCHANGED)
class GANTrainer:
    def __init__(self):
        self.history = {
            'd_loss': [], 'g_loss': [],
            'val_d_loss': [], 'val_g_loss': [],
            'epoch_times': []
        }
        
    def train(self, generator, discriminator, gan, X_partial, X_complete, X_val_partial, X_val_complete):
        d_optimizer = optimizers.Adam(config.LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        g_optimizer = optimizers.Adam(config.LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        
        @tf.function
        def train_step(partial_imgs, complete_imgs):
            with tf.GradientTape() as d_tape:
                gen_imgs = generator(partial_imgs, training=True)
                real_output = discriminator(complete_imgs, training=True)
                fake_output = discriminator(gen_imgs, training=True)
                
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = gradient_penalty(discriminator, complete_imgs, gen_imgs)
                d_loss_total = d_loss + config.GRADIENT_PENALTY_WEIGHT * gp
            
            d_gradients = d_tape.gradient(d_loss_total, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
            
            with tf.GradientTape() as g_tape:
                gen_imgs = generator(partial_imgs, training=True)
                fake_output = discriminator(gen_imgs, training=True)
                
                g_loss = -tf.reduce_mean(fake_output)
                l1_loss = tf.reduce_mean(tf.abs(gen_imgs - complete_imgs))
                g_loss_total = g_loss + 100 * l1_loss
            
            g_gradients = g_tape.gradient(g_loss_total, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
            
            return d_loss_total, g_loss_total
        
        print("\n{:<7} {:<12} {:<12} {:<12} {:<12} {:<10}".format(
            "Epoch", "D Loss", "G Loss", "Val D Loss", "Val G Loss", "Time (s)"))
        print("-" * 70)
        
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            batch_d_losses, batch_g_losses = [], []
            
            idx = np.random.permutation(len(X_partial))
            X_partial_shuffled = X_partial[idx]
            X_complete_shuffled = X_complete[idx]
            
            for batch in range(0, len(X_partial), config.BATCH_SIZE):
                batch_partial = X_partial_shuffled[batch:batch+config.BATCH_SIZE]
                batch_complete = X_complete_shuffled[batch:batch+config.BATCH_SIZE]
                d_loss, g_loss = train_step(batch_partial, batch_complete)
                batch_d_losses.append(float(d_loss))
                batch_g_losses.append(float(g_loss))
            
            avg_d_loss = np.mean(batch_d_losses)
            avg_g_loss = np.mean(batch_g_losses)
            
            val_idx = np.random.randint(0, len(X_val_partial), config.BATCH_SIZE)
            val_partial = X_val_partial[val_idx]
            val_complete = X_val_complete[val_idx]
            
            val_gen = generator(val_partial, training=False)
            val_d_loss = float(tf.reduce_mean(discriminator(val_gen)) - tf.reduce_mean(discriminator(val_complete)))
            val_g_loss = float(-tf.reduce_mean(discriminator(val_gen)))
            
            self.history['d_loss'].append(avg_d_loss)
            self.history['g_loss'].append(avg_g_loss)
            self.history['val_d_loss'].append(val_d_loss)
            self.history['val_g_loss'].append(val_g_loss)
            epoch_time = time.time() - start_time
            self.history['epoch_times'].append(epoch_time)
            
            print("{:<7} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<10.2f}".format(
                f"{epoch}/{config.EPOCHS}",
                avg_d_loss,
                avg_g_loss,
                val_d_loss,
                val_g_loss,
                epoch_time))
            
            if epoch % config.SAVE_INTERVAL == 0 or epoch == config.EPOCHS - 1:
                self._save_progress(generator, X_partial, epoch)
                print("\n=== Epoch {} Summary ===".format(epoch))
                print("Samples saved to progress/epoch_{}.png".format(epoch))
                print("Avg D Loss: {:.4f}".format(avg_d_loss))
                print("Avg G Loss: {:.4f}".format(avg_g_loss))
                print("Val D Loss: {:.4f}".format(val_d_loss))
                print("Val G Loss: {:.4f}".format(val_g_loss))
                print("Time: {:.2f}s".format(epoch_time))
                print("-" * 70)
    
    def _save_progress(self, generator, inputs, epoch):
        os.makedirs("progress", exist_ok=True)
        gen_imgs = generator.predict(inputs[:3])
        
        plt.figure(figsize=(12, 6))
        for i in range(3):
            input_img = (inputs[i].squeeze() + 1) / 2
            gen_img = (gen_imgs[i].squeeze() + 1) / 2
            
            plt.subplot(3, 3, i*3+1)
            plt.imshow(input_img, cmap='gray')
            plt.title("Input")
            plt.axis('off')
            
            plt.subplot(3, 3, i*3+2)
            plt.imshow(gen_img, cmap='gray')
            plt.title("Generated")
            plt.axis('off')
            
            plt.subplot(3, 3, i*3+3)
            processed = post_process(gen_imgs[i])
            plt.imshow(processed, cmap='gray')
            plt.title("Processed")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"progress/epoch_{epoch}.png")
        plt.close()

def post_process(generated_img):
    img = (127.5 * (generated_img.squeeze() + 1)).astype(np.uint8)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    smoothed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return gaussian(smoothed, sigma=0.8)

# Testing Function (UNCHANGED)
def test_model(model_path, test_images, num_samples=3):
    generator = tf.keras.models.load_model(model_path, compile=False)
    
    if len(test_images) < num_samples:
        num_samples = len(test_images)
    
    idx = np.random.choice(len(test_images), num_samples, replace=False)
    test_samples = test_images[idx]
    generated = generator.predict(test_samples)
    
    plt.figure(figsize=(15, 5*num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(test_samples[i].squeeze(), cmap='gray')
        plt.title("Partial Input")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(generated[i].squeeze(), cmap='gray')
        plt.title("Generated Output")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+3)
        processed = post_process(generated[i])
        plt.imshow(processed, cmap='gray')
        plt.title("Post-Processed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main Execution (UNCHANGED)
if __name__ == "__main__":
    print(" Starting QuickDraw Doodle Completion System")
    print("="*50)
    
    print("\n🔍 Loading and preprocessing data...")
    loader = QuickDrawLoader()
    X_partial, X_complete, X_val_partial, X_val_complete = loader.load_data()
    print(f" Loaded {len(X_partial)} training and {len(X_val_partial)} validation samples")
    
    print("\n Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    gan_input = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    
    print("\n Starting training...")
    print(f"Training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}")
    print(f"Saving progress every {config.SAVE_INTERVAL} epochs")
    print("="*50)
    
    trainer = GANTrainer()
    trainer.train(generator, discriminator, gan, X_partial, X_complete, X_val_partial, X_val_complete)
    
    generator.save("doodle_completer_final.h5")
    print("\n Training complete!")
    print("Saved model: doodle_completer_final.h5")


    
# 1. MODEL FILE VERIFICATION
model_name = "doodle_completer_final.h5"

# Check these locations for your model file
possible_locations = [
    model_name,  # Current directory
...............
]

# Find the model file
model_path = None
for path in possible_locations:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    print(" Model file not found. Searched in:")
    for path in possible_locations:
        print(f"- {path}")
    print("\nCurrent directory contents:")
    print(os.listdir())
    raise FileNotFoundError(f"Could not find {model_name}")

print(f"Found model at: {os.path.abspath(model_path)}")

# 2. COMPATIBLE SelfAttention LAYER
class SelfAttention(layers.Layer):
    def __init__(self, filters, **kwargs):  # Added **kwargs for compatibility
        super().__init__(**kwargs)
        self.filters = filters
        self.channels = filters // 8
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(1,), initializer='zeros')
        self.query_conv = layers.Conv2D(self.channels, 1, padding='same')
        self.key_conv = layers.Conv2D(self.channels, 1, padding='same')
        self.value_conv = layers.Conv2D(self.filters, 1, padding='same')
        self.out_conv = layers.Conv2D(self.filters, 1, padding='same')
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        q = tf.reshape(q, [batch_size, h*w, self.channels])
        k = tf.reshape(k, [batch_size, h*w, self.channels])
        v = tf.reshape(v, [batch_size, h*w, self.filters])
        
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.reshape(out, [batch_size, h, w, self.filters])
        out = self.out_conv(out)
        
        return x + self.gamma * out

# 3. MODEL TESTING FUNCTION
def test_model(model_path, num_samples=3):
    try:
        print("\nLoading model...")
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'SelfAttention': SelfAttention}
        )
        print(" Model loaded successfully")
        model.summary()
        
        # Create test data
        test_images = np.random.rand(num_samples, 128, 128, 1) * 2 - 1  # [-1, 1] range
        
        print("\n🔍 Testing model...")
        outputs = model.predict(test_images)
        
        # Display results
        print("\n Results:")
        for i in range(num_samples):
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(test_images[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
            plt.title(f"Input {i+1}")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(outputs[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
            plt.title("Generated")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow((outputs[i].squeeze() + 1) / 2, cmap='gray')  # Simple post-processing
            plt.title("Normalized")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'result_{i}.png')
            display(Image(f'result_{i}.png'))
            plt.close()
        
        print("\n Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Verify model file integrity")
        print("2. Check TensorFlow version compatibility")
        print(f"   - Current: {tf.__version__}")
        print("3. Try loading weights only:")
        print(f"   model = build_generator()")
        print(f"   model.load_weights('{model_name}')")
        return False

# 4. RUN THE TEST
print(f"\n Environment:")
print(f"TensorFlow: {tf.__version__}")
print(f"Python: {sys.version.split()[0]}")
print(f"Model: {model_name}")

test_model(model_path)

