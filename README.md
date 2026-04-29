# 🔬 ImageCure – CNN-Based Noise Detection & Denoising

A smart web application that automatically detects noise in images using a CNN model and applies the best denoising technique, with additional enhancement and format conversion features.

---

## 🚀 Features

### 🧠 Noise Detection (CNN-Based)
- Detects:
  - Gaussian Noise
  - Salt & Pepper Noise
  - Speckle Noise
  - Motion Blur
  - Clean Image
- Shows confidence score

### 🧹 Automatic Denoising
- Applies best filter based on detected noise:
  - Median Filter (Salt & Pepper)
  - Non-Local Means (Gaussian)
  - Bilateral Filter (Speckle)
  - Sharpening (Motion Blur)

### 🎨 Image Enhancement
- CLAHE (Adaptive Contrast)
- Brightness & Contrast adjustment
- Gamma correction
- Sharpening

### 🔄 Image Format Conversion
- Convert final image to:
  - Grayscale
  - Binary
  - HSI

### ⬇️ Download Options
- Download final enhanced image
- Download converted image

---

## 🖥️ Tech Stack

- Python
- OpenCV
- NumPy
- TensorFlow / Keras (CNN Model)
- Streamlit (Web App UI)
- Matplotlib

---

## 📂 Project Structure
