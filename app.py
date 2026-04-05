# ============================================================
# Nombre:    Elian Desiderio Feliz Martinez
# Matricula: 24-EISN-2-041
# Archivo:   app.py
# ============================================================

import gradio as gr
import torch
import os
from PIL import Image

from model import PlantDiseaseModel
from utils import CLASS_NAMES, CLASS_INFO, preprocess_image, get_top_predictions, format_prediction_html

# Configuracion del dispositivo
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("models", "plant_disease_model.pth")

print(f"Dispositivo: {DEVICE}")

# Cargar modelo
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES), pretrained=True)

if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH, device=DEVICE)
else:
    print("No se encontro modelo entrenado, usando pesos de ImageNet")

model.to(DEVICE)
model.eval()