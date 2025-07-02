import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import joblib

#Model architecture
def get_model(num_classes=14, freeze_backbone=False):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

     # Freeze base layers (optional: for faster training initially)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False # Set to False if you want to freeze

    model.fc = nn.Linear(model.fc.in_features, num_classes) # Replace final FC layer for multilabel classification

    return model

#Load model weights
def load_model(model_path: str, num_classes: int) -> torch.nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=num_classes).to(device)
    model.to(device)

    # load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

#Load saved MultiLabelBinarizer
def load_label_binarizer(pkl_path: str):
    mlb = joblib.load(pkl_path)
    return mlb

#Predict from image tensor
def predict_image(model, image_tensor, threshold=0.25, device=None):
    """
    Predict labels for a single image tensor using a multi-label classifier.

    Args:
        model: Trained PyTorch model.
        image_tensor: Tensor of shape [3, H, W] (single image).
        threshold: Sigmoid threshold for binary label assignment.
        device: torch.device object or None (auto-detect if None).

    Returns:
        prediction_labels: Tensor of binary predictions (0 or 1).
        prediction_probs: Tensor of probabilities after sigmoid.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    input_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        output = model(input_tensor)  # image at index 0
        prediction_probs = torch.sigmoid(output)
        prediction_labels = (prediction_probs > threshold).int()

    return prediction_labels.cpu(), prediction_probs[0].cpu()