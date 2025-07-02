import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def denormalize(tensor, mean, std):
    # Clone so we don't change original tensor
    result = tensor.clone()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result

def cam_img(model, image_tensor, target_class_idx, use_cuda=True, image_weight=0.6):
    """
    Generate a Grad-CAM heatmap for a single image and class index.

    Args:
        model: The trained PyTorch model (e.g., ResNet50).
        image_tensor: A single image tensor [3, H, W] (not batched).
        target_class_idx: Integer index of the target class.
        use_cuda: Whether to use GPU if available.

    Returns:
        cam_image: Numpy array of CAM heatmap overlay (RGB).
        target_class_idx: The class index used.
    """

    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    model.to(device).eval()

    input_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim

    # Pick the layer to visualize
    target_layer = model.layer4[-1]  # Last conv layer in ResNet50

    # Initialize GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Choose class index
    targets = [ClassifierOutputTarget(target_class_idx)]

    # Run Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # Remove batch dimension

    img_denorm = denormalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_rgb = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_rgb = np.clip(img_rgb, 0, 1)

    # Create CAM overlay
    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True, image_weight=image_weight)

    return cam_image, target_class_idx
