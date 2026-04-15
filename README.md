# Intelligent Noise Detection & Denoising

A Streamlit-based web app that:
- Detects noise in images automatically
- Identifies noise type (Gaussian, Speckle, Salt & Pepper, etc.)
- Applies appropriate denoising filters
- Preserves image clarity
- Allows enhancement and download

## Tech Stack
- Python
- OpenCV
- NumPy
- Streamlit

## Features
- Automatic noise detection
- Smart denoising
- Edge preservation (clarity maintained)
- Enhancement filters (CLAHE, Gamma, Sharpening)

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py