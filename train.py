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


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena el modelo por una epoca completa.
    Retorna el loss y accuracy promedio.
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds     = torch.max(outputs, 1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total * 100
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evalua el modelo en el conjunto de validacion.
    Retorna el loss y accuracy promedio.
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss  += loss.item() * images.size(0)
            _, preds       = torch.max(outputs, 1)
            correct       += (preds == labels).sum().item()
            total         += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total * 100
    return epoch_loss, epoch_acc


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Verificar que existan las carpetas del dataset
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir):
        print(f"No se encontro la carpeta: {train_dir}")
        print("Descarga el dataset PlantVillage y organizalo en data/train y data/val")
        return

    # Cargar dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=inference_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=inference_transform)

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size,
                               shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    print(f"Clases encontradas: {num_classes}")
    print(f"Imagenes de entrenamiento: {len(train_dataset)}")
    print(f"Imagenes de validacion: {len(val_dataset)}")

    # Crear modelo, criterio y optimizador
    model     = PlantDiseaseModel(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_val_acc = 0.0

    # Bucle de entrenamiento
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoca {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch,
                 "val_acc": best_val_acc},
                args.output,
            )
            print(f"  Mejor modelo guardado (Val Acc: {best_val_acc:.2f}%)")

    print(f"\nEntrenamiento completo. Mejor Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()