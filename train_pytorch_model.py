# ===============================
# Ensemble Training Script
# Cervical Cancer Detection
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json

# -------------------------------
# Configuration
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"Using device: {device}")
print("Model Type: Ensemble (MobileNetV2 + ResNet18)")
print("=" * 60)

# -------------------------------
# Data transforms
# -------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Load datasets
# -------------------------------
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("dataset/validation", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print("=" * 60)

# -------------------------------
# Ensemble Model
# -------------------------------
class CervicalCancerEnsemble(nn.Module):
    def __init__(self):
        super().__init__()

        # MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights="DEFAULT")
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        mob_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(mob_features, 1),
            nn.Sigmoid()
        )

        # ResNet18
        self.resnet = models.resnet18(weights="DEFAULT")
        for param in self.resnet.parameters():
            param.requires_grad = False

        res_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(res_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.mobilenet(x)
        out2 = self.resnet(x)
        return (out1 + out2) / 2  # Ensemble average


# -------------------------------
# Model, loss, optimizer
# -------------------------------
model = CervicalCancerEnsemble().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model initialized successfully")
print("=" * 60)

# -------------------------------
# Training function
# -------------------------------
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(loader), acc


# -------------------------------
# Validation function
# -------------------------------
def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(loader), acc


# -------------------------------
# Training loop
# -------------------------------
print("Training started...")
print("=" * 60)

best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "cervical_cancer_ensemble_model.pth")
        print(" ✅ Best model saved")

    print("-" * 60)

print("Training completed")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# -------------------------------
# Save class indices
# -------------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_dataset.class_to_idx, f)

print("Model saved as: cervical_cancer_ensemble_model.pth")
print("Class indices saved as: class_indices.json")
print("Next step → python app.py")
print("=" * 60)
