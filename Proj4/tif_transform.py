from PIL import Image
import os
import numpy as np

from mnist_logreg_idx import load_idx_images, load_idx_labels

test_folder = "./mnist_tif_test"
os.makedirs(test_folder, exist_ok=True)

train_images = load_idx_images(r"c:\Users\natep\Downloads\MNIST\MNIST\t10k-images.idx3-ubyte")  # path to MNIST test images
train_labels = load_idx_labels(r"C:\Users\natep\Downloads\MNIST\MNIST\t10k-labels.idx1-ubyte")  # path to MNIST test labels

num_samples = 20  # how many images to test

for i in range(num_samples):
    img_array = train_images[i]  # shape (28,28), dtype=uint8
    img_array = 255 - img_array
    img = Image.fromarray(img_array)
    # Optionally invert colors to match your pipeline if needed
    img.save(os.path.join(test_folder, f"mnist_{i}.tif"))