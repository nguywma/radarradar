import cv2
import numpy as np

# --- Read the image in grayscale mode ---
image_path = "/media/manh/manh/oord_data/cen2018/Bellmouth_2_resize/1637844136778964.png"  # change this to your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# --- Normalize pixel values from 0–255 to 0–1 ---
normalized = img.astype(np.float32) / 255.0

# --- Flatten the 2D image into a 1D list ---
pixel_values = normalized.flatten().tolist()

# --- Print all normalized pixel values ---
for val in pixel_values:
    print(val)

# Optional: print basic info
print(f"\nImage shape: {img.shape}")
print(f"Total pixels: {len(pixel_values)}")
