# ============================================================
# Nombre:    Elian Desiderio Feliz Martinez
# Matricula: 24-EISN-2-041
# Archivo:   train.py
# ============================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

from model import PlantDiseaseModel
from utils import inference_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del detector de enfermedades en plantas"
    )
    parser.add_argument("--data_dir",   type=str,   default="data",
                        help="Carpeta con las imagenes de entrenamiento")
    parser.add_argument("--epochs",     type=int,   default=20,
                        help="Numero de epocas")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="Tamano del batch")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output",     type=str,
                        default="models/plant_disease_model.pth",
                        help="Ruta donde guardar el modelo entrenado")
    return parser.parse_args()