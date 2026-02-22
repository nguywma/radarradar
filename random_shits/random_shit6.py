import cv2
import os
from tqdm import tqdm

root_folder = "/media/manh/manh/oord_data/cropped"

def crop_images_recursively(root):
    # Collect all image file paths
    image_paths = []
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_paths.append(os.path.join(subdir, file))

    print(f"🔍 Found {len(image_paths)} images in {root}\n")

    cropped = 0

    for path in tqdm(image_paths, desc="Cropping images", unit="img"):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            tqdm.write(f"[!] Failed to read: {path}")
            continue

        h, w = img.shape[:2]
        if h == 400:
            img = img[:399, :]
            cv2.imwrite(path, img)
            cropped += 1

    # print(f"\n✅ Done! Cropped {cropped} images (400×3708 → 399×3708).")

# Run
if __name__ == "__main__":
    crop_images_recursively(root_folder)
