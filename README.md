# Diabetic-Retino-Project

A Streamlit web app to detect diabetic retinopathy using a VGG16 deep learning model and visualize predictions with Grad-CAM. Includes PDF medical report generation and history tracking via SQLite.

## Features
- Upload retinal image and predict DR stage
- Visual explanation via Grad-CAM overlay
- Downloadable PDF report with medical summary
- SQLite-based history tracking
- Hosted on Streamlit Cloud

## Setup

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3.Run locally: 
   streamlit run app.py
