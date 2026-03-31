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

        # Cargar ResNet-50 pre-entrenado con ImageNet
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)