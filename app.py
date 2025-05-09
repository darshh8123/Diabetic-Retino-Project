import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import sqlite3
import cv2
from fpdf import FPDF
import gdown
from datetime import datetime

# ────────────────────────────── Auto-download model ──────────────────────────────
model_path = "best_model.pth"
drive_file_id = "1hrRdV8601M6Khk95n0lw-0H9oNEPDV6g"
model_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# ────────────────────────────── Load Model ──────────────────────────────
@st.cache_resource
def load_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 5)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ────────────────────────────── Grad-CAM ──────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        x.requires_grad_()
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()
        gradients = self.gradients.detach()[0]
        activations = self.activations.detach()[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

# ────────────────────────────── Image Utilities ──────────────────────────────
def apply_colormap_on_image(org_img, activation_map, colormap_name=cv2.COLORMAP_JET):
    cam_resized = cv2.resize(activation_map, org_img.size, interpolation=cv2.INTER_CUBIC)
    cam_resized = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap_name)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.array(org_img.convert("RGB")) * 0.7 + heatmap * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap), cam_resized

def draw_detected_regions_on_cam(org_img, cam_mask, threshold=0.6):
    mask = (cam_mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_box = np.array(org_img.convert("RGB")).copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(img_with_box)

# ────────────────────────────── PDF Report ──────────────────────────────
def generate_pdf_report(name, age, date, prediction, probs, class_names, orig_img, region_img, overlay_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 10, "Diabetic Retinopathy Diagnostic Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, "Patient Information", border=1, ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(190, 8, f"Name: {name}", border=1, ln=True)
    pdf.cell(190, 8, f"Age: {age}", border=1, ln=True)
    pdf.cell(190, 8, f"Date: {date}", border=1, ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Prediction Result", border=1, ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(190, 8, f"Predicted Class: {prediction}", border=1, ln=True)
    pdf.cell(190, 8, "Reported Test Accuracy: 86.3%", border=1, ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Class Probabilities", border=1, ln=True)
    pdf.set_font("Arial", "", 11)
    for cls, p in zip(class_names, probs):
        pdf.cell(190, 8, f"{cls}: {p*100:.2f}%", border=1, ln=True)
    pdf.ln(4)

    orig_img.save("original.jpg")
    region_img.save("regions.jpg")
    overlay_img.save("overlay.jpg")

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Visual Analysis", border=1, ln=True)
    y = pdf.get_y()
    pdf.set_font("Arial", "I", 10)
    pdf.cell(63, 6, "Original", border=1, ln=False, align="C")
    pdf.cell(63, 6, "Detected Regions", border=1, ln=False, align="C")
    pdf.cell(63, 6, "Overlay", border=1, ln=True, align="C")
    pdf.image("original.jpg", x=10, y=y + 10, w=60)
    pdf.image("regions.jpg", x=75, y=y + 10, w=60)
    pdf.image("overlay.jpg", x=140, y=y + 10, w=60)

    pdf.set_y(y + 80)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 6)

    return io.BytesIO(pdf.output(dest="S").encode("latin1"))

# ────────────────────────────── Streamlit Setup ──────────────────────────────
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

conn = sqlite3.connect("retinopathy_predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        prediction TEXT,
        probs TEXT,
        date TEXT,
        image BLOB,
        heatmap BLOB
    )
''')
conn.commit()

# Sidebar Prediction History
st.sidebar.title("Patient Input")
st.sidebar.markdown("### Recent Predictions")
history = c.execute("SELECT name, age, prediction, date FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
if history:
    for row in history:
        st.sidebar.markdown(f"**{row[0]}, {row[1]}y** → *{row[2]}*  \n🕓 {row[3]}")
else:
    st.sidebar.write("No predictions saved yet.")

# Main Form
with st.form("predict_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    upload = st.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Run Prediction")

# Prediction & PDF Report
if submit:
    if not name or not upload:
        st.warning("Please enter name and upload an image.")
    else:
        image = Image.open(upload).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits[0], dim=0).cpu().numpy()

        top_idx = int(probs.argmax())
        prediction = class_names[top_idx]

        gradcam = GradCAM(model, model.features[26])
        cam = gradcam.generate(x, top_idx)

        overlay_img, heatmap_img, cam_mask = apply_colormap_on_image(image, cam)
        region_highlight_img = draw_detected_regions_on_cam(image, cam_mask)

        st.subheader(f"Prediction: *{prediction}* ({probs[top_idx]*100:.1f}%)")
        st.markdown("#### Class-wise Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"{class_names[i]}: {p*100:.2f}%")

        st.markdown("### Visual Output")
        col1, col2, col3 = st.columns(3)
        col1.image(image, caption="Original", width=300)
        col2.image(region_highlight_img, caption="Detected Regions", width=300)
        col3.image(overlay_img, caption=f"Overlay ({prediction})", width=300)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_buf = io.BytesIO(); image.save(img_buf, format="PNG")
        hm_buf = io.BytesIO(); overlay_img.save(hm_buf, format="PNG")

        c.execute("""
            INSERT INTO predictions (name, age, prediction, probs, date, image, heatmap)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            age,
            prediction,
            ",".join(f"{v:.4f}" for v in probs),
            now,
            img_buf.getvalue(),
            hm_buf.getvalue()
        ))
        conn.commit()

        pdf_buf = generate_pdf_report(name, age, now, prediction, probs, class_names, image, region_highlight_img, overlay_img)
        st.download_button(" Download PDF Report", data=pdf_buf.getvalue(), file_name=f"{name}_DR_Report.pdf", mime="application/pdf")














