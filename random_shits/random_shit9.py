import os
from PIL import Image
import numpy as np

def is_grayscale(img):
    """
    Check if an image is grayscale.
    A grayscale image has mode 'L', or if in RGB, all channels are identical.
    """
    if img.mode == "L":
        return True

    if img.mode == "RGB":
        arr = np.array(img)
        # Check if R == G == B for all pixels
        return np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2])

    return False


def is_all_black(img):
    """
    Check if an image is entirely black.
    Works for grayscale or RGB.
    """
    arr = np.array(img)
    return np.max(arr) == 0  # all pixels = 0


folder = "../../../oord_data/cen2019/Twolochs_2_resize"  # <-- change this
gray = 0 
for filename in os.listdir(folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        continue

    filepath = os.path.join(folder, filename)

    try:
        img = Image.open(filepath)

        if is_grayscale(img) and is_all_black(img):
            gray += 1

    except Exception as e:
        print("Error reading", filename, ":", e)
print("Total all-black grayscale images:", gray, "out of", len(os.listdir(folder)))