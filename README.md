# ImageCure — Noise Detector & Denoiser
### Digital Image Processing · PBL Project

---

## What It Does

ImageCure is a Streamlit web app that:

1. **Accepts** an image via upload or live camera capture  
2. **Detects** the noise type automatically using statistical image analysis  
3. **Denoises** the image with the best-fit filter for that noise type  
4. **Enhances** the result with optional post-processing filters  
5. **Downloads** the final cleaned image as a PNG

---

## Noise Types Detected

| Noise Type    | Detection Method                          | Denoising Filter Applied        |
|---------------|-------------------------------------------|---------------------------------|
| Salt & Pepper | High % of extreme (black/white) pixels    | Median Filter (5×5)             |
| Gaussian      | Laplacian variance + std distribution     | Non-Local Means (NLM)           |
| Speckle       | High CVLV (local variance fluctuation)    | Bilateral Filter                |
| Motion Blur   | Low sharpness + directional gradient bias | Unsharp Masking                 |
| Poisson       | Variance ≈ Mean relationship              | Gaussian Blur (σ=1.0)           |
| Uniform       | Low skewness + moderate std               | Median + Bilateral combo        |

---

## Enhancement Filters (Optional)

Enable any of these from the sidebar after denoising:

- **CLAHE** — Contrast Limited Adaptive Histogram Equalization  
- **Contrast / Brightness** — Linear `α·pixel + β` adjustment  
- **Gamma Correction** — Nonlinear brightness curve  
- **Sharpening** — Laplacian-based edge enhancement

---

## Setup & Run Instructions

### 1. Prerequisites
- Python 3.9 or higher
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit opencv-python-headless numpy Pillow matplotlib
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### 4. Usage
1. Open the app in your browser
2. Choose **Upload Image** or **Camera Capture** in the sidebar
3. Upload/capture your noisy image
4. View the auto-detected noise type and confidence score
5. See the denoised result instantly
6. Optionally toggle enhancement filters in the sidebar
7. Click **Download Final Image** to save your result

---

## Project Structure

```
imagecure/
├── app.py           ← Main Streamlit application (single file)
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

### Code Modules (inside app.py)

| Module | Description |
|--------|-------------|
| Module 1 | Noise Detection (statistical analysis) |
| Module 2 | Denoising Filters (6 noise-specific filters) |
| Module 3 | Enhancement Filters (CLAHE, contrast, gamma, sharpening) |
| Module 4 | Utility Functions (conversions, PSNR, entropy) |
| Module 5 | UI Rendering Helpers (styled components) |

---

## Technical Notes

- **No deep learning** — pure classical image processing (OpenCV, NumPy)
- PSNR and Shannon entropy shown as quality metrics  
- Histogram comparison of original / denoised / enhanced images
- Images larger than 1200px are auto-resized for performance
- Camera capture works in Chrome/Edge (requires HTTPS or localhost)

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web app framework |
| opencv-python-headless | ≥4.9 | Image processing |
| numpy | ≥1.26 | Numerical computing |
| Pillow | ≥10.2 | Image I/O |
| matplotlib | ≥3.8 | Histogram plots |
