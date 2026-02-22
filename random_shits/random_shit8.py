import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load image ---
img_path = '/media/manh/manh/oord_data/cartesian/Bellmouth_1_resize/1637841702181266.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 256.0
img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
img = np.expand_dims(img, axis=0)  # (1, 3, H, W)
img = torch.from_numpy(img).to(device)

# --- Load ResNet34 ---
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
model.eval()

# --- Register hooks to capture feature maps ---
feature_maps = {}
def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

# Attach hooks to main convolutional/residual layers
model.conv1.register_forward_hook(get_activation("conv1"))
model.layer1.register_forward_hook(get_activation("layer1"))
model.layer2.register_forward_hook(get_activation("layer2"))
model.layer3.register_forward_hook(get_activation("layer3"))
model.layer4.register_forward_hook(get_activation("layer4"))

# --- Forward pass ---
with torch.no_grad():
    _ = model(img)

# --- Process feature maps ---
processed_feature_maps = {}
for name, fmap in feature_maps.items():
    fmap = fmap.squeeze(0)  # remove batch dim -> (C, H, W)
    mean_map = torch.mean(fmap, dim=0).cpu().numpy()  # average across channels
    processed_feature_maps[name] = mean_map

# --- Display info ---
print("\nCaptured feature maps:")
for k, v in feature_maps.items():
    print(f"{k}: {tuple(v.shape)}")

# --- Plot feature maps ---
fig = plt.figure(figsize=(20, 15))
for i, (name, fmap) in enumerate(processed_feature_maps.items(), 1):
    ax = fig.add_subplot(2, 3, i)
    ax.imshow(fmap, cmap='viridis')
    ax.axis('off')
    ax.set_title(name, fontsize=16)

plt.tight_layout()
plt.show()
