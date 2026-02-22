import cv2
import glob
from collections import Counter
import os

def analyze_image_sizes(folder):
    # collect all image paths (png, jpg, jpeg, bmp)
    imgs_path = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        imgs_path.extend(glob.glob(os.path.join(folder, ext)))

    if not imgs_path:
        print(f"[!] No image files found in {folder}")
        return [], {}

    sizes = []
    for i, path in enumerate(imgs_path):
        img = cv2.imread(path, 0)
        if img is None:
            print(f"[!] Failed to read image: {path}")
            continue

        h, w = img.shape
        sizes.append((h, w))

        # if i % 100 == 0:
        #     print(f"Processed {i}/{len(imgs_path)} images...")

    size_counts = Counter(sizes)

    print("\n📏 Unique image sizes found:")
    for (h, w), count in size_counts.items():
        print(f"  {h}x{w} — {count} images")

    if sizes:
        common_size = size_counts.most_common(1)[0][0]
        min_h, min_w = min(sizes)
        max_h, max_w = max(sizes)
        print(f"\nMost common size: {common_size}")
        print(f"Smallest size: {min_h}x{min_w}")
        print(f"Largest size: {max_h}x{max_w}")

    return sizes, size_counts

seq = os.listdir('/media/manh/manh/oord_data/cropped@75/')
for s in seq: 
    sizes, counts = analyze_image_sizes('/media/manh/manh/oord_data/cropped@75/' + s)