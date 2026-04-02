# ============================================================
# Nombre:    Elian Desiderio Feliz Martinez
# Matricula: 24-EISN-2-041
# Archivo:   model.py
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class PlantDiseaseModel(nn.Module):
    """
    Modelo de clasificacion de enfermedades en plantas.
    Usa ResNet-50 como base con Transfer Learning.
    """

    def __init__(self, num_classes=38, pretrained=True):
        super(PlantDiseaseModel, self).__init__()

        # Cargar ResNet-50 preentrenado con ImageNet
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Congelar todas las capas para no perder lo que ya aprendio con ImageNet
        for param in backbone.parameters():
            param.requires_grad = False

        # Descongelar la ultima capa convolucional para que se adapte a hojas de plantas
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        in_features = backbone.fc.in_features
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Clasificador nuevo para las 38 clases de PlantVillage
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def load_weights(self, path, device=None):
        if device is None:
            device = torch.device("cpu")
        state_dict = torch.load(path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Pesos cargados desde: {path}")