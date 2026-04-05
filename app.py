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


def predict(image):
    """
    Recibe una imagen, la procesa y retorna
    el diagnostico en formato HTML y las probabilidades.
    """
    if image is None:
        return "Por favor sube una imagen primero.", None

    # Preprocesar imagen y mover al dispositivo
    tensor = preprocess_image(image).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        outputs       = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    probs_np = probabilities.cpu().numpy()[0]

    # Obtener top 5 predicciones
    top5 = get_top_predictions(probs_np, CLASS_NAMES, top_k=5)

    # Resultado principal
    top_class, top_confidence = top5[0]
    info = CLASS_INFO.get(top_class, {})

    # Formatear resultado
    html_result = format_prediction_html(top_class, top_confidence, info, top5)
    label_dict  = {name: float(conf) for name, conf in top5}

    return html_result, label_dict


# Construir interfaz
with gr.Blocks(title="Detector de Enfermedades en Plantas") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px;">
      <h1>Detector de Enfermedades en Plantas</h1>
      <p style="color:#555;">
        Sube una foto de una hoja o parte afectada de la planta
        y el sistema identificara la enfermedad presente.
      </p>
      <p style="font-size:0.85em; color:#888;">
        Modelo: ResNet-50 | Dataset: PlantVillage | 38 clases
      </p>
    </div>
    """)

    with gr.Row():

        with gr.Column():
            input_image = gr.Image(type="pil", label="Imagen de la planta", height=320)
            btn_predict = gr.Button("Analizar", variant="primary", size="lg")
            btn_clear   = gr.Button("Limpiar", variant="secondary")

        with gr.Column():
            output_html  = gr.HTML(
                value="<p style='color:#aaa; text-align:center; padding:60px 0;'>"
                      "El diagnostico aparecera aqui...</p>"
            )
            output_label = gr.Label(label="Top 5 predicciones", num_top_classes=5)

    # Eventos
    btn_predict.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_html, output_label]
    )
    input_image.change(
        fn=predict,
        inputs=input_image,
        outputs=[output_html, output_label]
    )
    btn_clear.click(
        fn=lambda: (None,
                    "<p style='color:#aaa; text-align:center; padding:60px 0;'>"
                    "El diagnostico aparecera aqui...</p>",
                    None),
        inputs=[],
        outputs=[input_image, output_html, output_label]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)