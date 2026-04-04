# ============================================================
# Nombre:    Elian Desiderio Feliz Martinez
# Matricula: 24-EISN-2-041
# Archivo:   utils.py
# ============================================================

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

# Lista de las 38 clases del dataset PlantVillage
# Formato: Planta___Enfermedad
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Valores de normalizacion estandar de ImageNet
# El modelo fue pre-entrenado con estos valores
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transformacion para cuando el modelo va a predecir
inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image):
    """
    Prepara una imagen PIL para pasarla al modelo.
    Convierte a RGB, redimensiona y normaliza.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = inference_transform(image)  # (3, 224, 224)
    tensor = tensor.unsqueeze(0)         # (1, 3, 224, 224)
    return tensor


def get_top_predictions(probs, class_names, top_k=5):
    """
    Retorna las top-k predicciones ordenadas
    de mayor a menor confianza.
    """
    indices = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in indices]


# Informacion detallada de cada enfermedad
CLASS_INFO = {
    "Tomato___Late_blight": {
        "nombre": "Tomate - Tizon tardio",
        "descripcion": "Phytophthora infestans. Lesiones grandes y acuosas que avanzan rapido.",
        "severidad": "Muy Alta",
        "color_severidad": "#c0392b",
        "tratamiento": "Fungicidas sistemicos (metalaxil + mancozeb). Destruir plantas infectadas.",
    },
    "Tomato___Early_blight": {
        "nombre": "Tomate - Tizon temprano",
        "descripcion": "Alternaria solani. Manchas concentricas en anillos en hojas inferiores.",
        "severidad": "Media",
        "color_severidad": "#f39c12",
        "tratamiento": "Fungicidas preventivos, eliminar hojas afectadas, rotacion de cultivos.",
    },
    "Tomato___healthy": {
        "nombre": "Tomate - Sano",
        "descripcion": "La planta no presenta sintomas de enfermedad detectables.",
        "severidad": "Ninguna",
        "color_severidad": "#27ae60",
        "tratamiento": "Fertilizacion equilibrada, riego por goteo y tutorado.",
    },
    "Potato___Late_blight": {
        "nombre": "Papa - Tizon tardio",
        "descripcion": "Phytophthora infestans. Lesiones acuosas que se expanden rapidamente.",
        "severidad": "Muy Alta",
        "color_severidad": "#c0392b",
        "tratamiento": "Fungicidas sistemicos, destruccion inmediata de plantas infectadas.",
    },
    "Potato___Early_blight": {
        "nombre": "Papa - Tizon temprano",
        "descripcion": "Alternaria solani. Manchas circulares con anillos concentricos.",
        "severidad": "Media",
        "color_severidad": "#f39c12",
        "tratamiento": "Fungicidas (clorotalonil, mancozeb), rotacion de cultivos.",
    },
    "Potato___healthy": {
        "nombre": "Papa - Sana",
        "descripcion": "La planta no presenta sintomas de enfermedad detectables.",
        "severidad": "Ninguna",
        "color_severidad": "#27ae60",
        "tratamiento": "Semilla certificada y buen drenaje del suelo.",
    },
    "Corn_(maize)___Common_rust": {
        "nombre": "Maiz - Roya comun",
        "descripcion": "Pustulas anaranjadas en ambas superficies de la hoja.",
        "severidad": "Media",
        "color_severidad": "#f39c12",
        "tratamiento": "Fungicidas preventivos, uso de hibridos resistentes.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "nombre": "Maiz - Tizon foliar del norte",
        "descripcion": "Lesiones elipticas largas de color verde grisaceo a marron.",
        "severidad": "Alta",
        "color_severidad": "#e74c3c",
        "tratamiento": "Rotacion de cultivos, fungicidas, variedades resistentes.",
    },
    "Corn_(maize)___healthy": {
        "nombre": "Maiz - Sano",
        "descripcion": "La planta no presenta sintomas de enfermedad detectables.",
        "severidad": "Ninguna",
        "color_severidad": "#27ae60",
        "tratamiento": "Fertilizacion nitrogenada adecuada y riego eficiente.",
    },
    "Apple___Apple_scab": {
        "nombre": "Manzana - Sarna del manzano",
        "descripcion": "Venturia inaequalis. Manchas oscuras y verrugosas en hojas y frutos.",
        "severidad": "Alta",
        "color_severidad": "#e74c3c",
        "tratamiento": "Fungicidas preventivos (captan, mancozeb). Eliminar hojas caidas.",
    },
    "Apple___healthy": {
        "nombre": "Manzana - Sana",
        "descripcion": "La planta no presenta sintomas de enfermedad detectables.",
        "severidad": "Ninguna",
        "color_severidad": "#27ae60",
        "tratamiento": "Mantener buenas practicas de riego, fertilizacion y poda.",
    },
}


def get_severity_label(severidad):
    """Retorna el nivel de severidad como texto."""
    mapping = {
        "Ninguna":  "Ninguna",
        "Media":    "Media",
        "Alta":     "Alta",
        "Muy Alta": "Muy Alta",
    }
    return mapping.get(severidad, "Desconocida")


def format_prediction_html(class_name, confidence, info, top5):
    """
    Genera el HTML para mostrar el resultado
    del diagnostico en la interfaz.
    """
    if not info:
        info = {
            "nombre": class_name.replace("_", " "),
            "descripcion": "Informacion no disponible.",
            "severidad": "Desconocida",
            "color_severidad": "#7f8c8d",
            "tratamiento": "Consulte a un agronomo.",
        }

    nombre    = info.get("nombre", class_name)
    desc      = info.get("descripcion", "")
    severidad = info.get("severidad", "Desconocida")
    color_sev = info.get("color_severidad", "#7f8c8d")
    trat      = info.get("tratamiento", "")

    conf_pct  = confidence * 100
    bar_color = "#27ae60" if conf_pct >= 70 else "#f39c12" if conf_pct >= 40 else "#e74c3c"

    html = f"""
    <div style="font-family:Arial,sans-serif; max-width:600px;">

      <div style="background:{color_sev}15; border-left:4px solid {color_sev};
                  border-radius:8px; padding:14px 18px; margin-bottom:14px;">
        <h2 style="margin:0 0 4px; color:{color_sev}; font-size:1.25em;">
          {nombre}
        </h2>
        <p style="margin:0; color:#555; font-size:0.92em;">{desc}</p>
      </div>

      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
          <span style="font-weight:600;">Confianza del diagnostico</span>
          <span style="font-weight:700; color:{bar_color};">{conf_pct:.1f}%</span>
        </div>
        <div style="background:#eee; border-radius:10px; height:12px;">
          <div style="width:{conf_pct:.1f}%; height:100%;
                      background:{bar_color}; border-radius:10px;"></div>
        </div>
      </div>

      <div style="background:#f0f8f0; border-radius:8px; padding:12px 16px;
                  border:1px solid #c8e6c9;">
        <strong style="color:#2e7d32;">Tratamiento recomendado:</strong>
        <p style="margin:6px 0 0; color:#333; font-size:0.93em;">{trat}</p>
      </div>

    </div>
    """

    return html