import cv2
import numpy as np
import os

os.makedirs("dataset", exist_ok=True)

classes = ["clean", "gaussian", "salt_pepper", "speckle", "motion_blur"]
for c in classes:
    os.makedirs(f"dataset/{c}", exist_ok=True)

def add_gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def add_salt_pepper(img):
    out = img.copy()

    prob = 0.08   # 🔥 increase from 0.02 → 0.08

    rnd = np.random.rand(*img.shape[:2])
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255

    return out

def add_speckle(img):
    noise = np.random.randn(*img.shape) * 0.1   # reduce from 0.2
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_motion_blur(img):
    kernel = np.zeros((9,9))
    kernel[4,:] = np.ones(9)/9
    return cv2.filter2D(img, -1, kernel)

# Use any clean images folder
SOURCE = "clean_images"   # 👉 put some normal images here

count = 0
for file in os.listdir(SOURCE):
    img = cv2.imread(os.path.join(SOURCE, file))
    img = cv2.resize(img, (128,128))

    cv2.imwrite(f"dataset/clean/{count}.png", img)
    cv2.imwrite(f"dataset/gaussian/{count}.png", add_gaussian_noise(img))
    cv2.imwrite(f"dataset/salt_pepper/{count}.png", add_salt_pepper(img))
    cv2.imwrite(f"dataset/speckle/{count}.png", add_speckle(img))
    cv2.imwrite(f"dataset/motion_blur/{count}.png", add_motion_blur(img))

    count += 1

print("✅ Dataset created!")