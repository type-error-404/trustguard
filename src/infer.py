import os
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from dataset import BASE_TRANSFORM
from model import build_resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = os.path.join(os.path.dirname(__file__), "..")
CKPT_PATH = os.path.join(ROOT, "models", "resnet18_deepfake.pt")

_model = None


def load_model():
    global _model

    if _model is None:
        _model = build_resnet18(pretrained=False).to(DEVICE)

        ckpt = torch.load(
            CKPT_PATH,
            map_location=DEVICE
        )

        _model.load_state_dict(ckpt["model_state"])
        _model.eval()

    return _model


def generate_gradcam(model, tensor, original_image, output_path, target_class):
    """
    Generate an actual Grad-CAM visualization from ResNet-18.
    """

    activations = []
    gradients = []

    # Last convolution layer of ResNet-18
    target_layer = model.layer4[-1].conv2

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()

        # Forward pass
        logits = model(tensor)

        # Backpropagate predicted class
        score = logits[:, target_class]
        score.backward()

        activation = activations[0]
        gradient = gradients[0]

        # Global average pooling of gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Grad-CAM
        cam = (weights * activation).sum(dim=1).squeeze()

        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.detach().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        width, height = original_image.size

        cam = cv2.resize(
            cam,
            (width, height)
        )

        # Create heatmap
        heatmap = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_TURBO
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Original image
        base = np.array(original_image)

        # Overlay
        overlay = (
            0.55 * base +
            0.45 * heatmap
        )

        overlay = np.uint8(
            np.clip(overlay, 0, 255)
        )

        # Save
        Image.fromarray(overlay).save(
            output_path,
            quality=92
        )

    finally:
        forward_handle.remove()
        backward_handle.remove()


def predict_image(image_path, heatmap_path=None):

    model = load_model()

    original_image = Image.open(
        image_path
    ).convert("RGB")

    tensor = BASE_TRANSFORM(
        original_image
    ).unsqueeze(0).to(DEVICE)

    # Grad-CAM needs gradients
    with torch.enable_grad():

        logits = model(tensor)

        probs = F.softmax(
            logits,
            dim=1
        )[0]

        real_probability = float(probs[0])
        fake_probability = float(probs[1])

        target_class = int(
            torch.argmax(probs).item()
        )

        label = (
            "fake"
            if target_class == 1
            else "real"
        )

        confidence = max(
            real_probability,
            fake_probability
        )

        # Generate actual Grad-CAM
        if heatmap_path:

            os.makedirs(
                os.path.dirname(heatmap_path),
                exist_ok=True
            )

            generate_gradcam(
                model,
                tensor,
                original_image,
                heatmap_path,
                target_class
            )

    # Honest AI explanation
    if label == "fake":

        reasons = [
            "The model found visual patterns closer to its learned manipulated-image examples.",
            f"Fake probability ({fake_probability * 100:.1f}%) is higher than real probability ({real_probability * 100:.1f}%).",
            "The AI Focus Map highlights regions that contributed more strongly to this prediction."
        ]

    else:

        reasons = [
            "The model found visual patterns closer to its learned real-image examples.",
            f"Real probability ({real_probability * 100:.1f}%) is higher than fake probability ({fake_probability * 100:.1f}%).",
            "The AI Focus Map highlights regions that contributed more strongly to this prediction."
        ]

    return {
        "label": label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": confidence,
        "reasons": reasons
    }


def predict_video(video_path, tmp_dir=None):

    from frame_extraction import extract_frames

    if tmp_dir is None:
        tmp_dir = os.path.join(
            ROOT,
            "temp_frames"
        )

    frames = extract_frames(
        video_path,
        tmp_dir,
        every_n_frames=10,
        max_frames=15,
        face_crop=True
    )

    if not frames:
        raise RuntimeError(
            "No frames could be extracted or no face was detected."
        )

    fake_probs = []

    for frame in frames:

        result = predict_image(frame)

        fake_probs.append(
            result["fake_probability"]
        )

    avg_fake_probability = (
        sum(fake_probs) / len(fake_probs)
    )

    label = (
        "fake"
        if avg_fake_probability > 0.5
        else "real"
    )

    real_probability = (
        1 - avg_fake_probability
    )

    confidence = max(
        real_probability,
        avg_fake_probability
    )

    if label == "fake":

        reasons = [
            "The model found manipulated-image patterns across the analyzed video frames.",
            f"The average fake probability was {avg_fake_probability * 100:.1f}%.",
            f"{len(frames)} video frames were analyzed by the detection model."
        ]

    else:

        reasons = [
            "The model found patterns closer to its learned real-image examples across the analyzed frames.",
            f"The average real probability was {real_probability * 100:.1f}%.",
            f"{len(frames)} video frames were analyzed by the detection model."
        ]

    return {
        "label": label,
        "fake_probability": avg_fake_probability,
        "real_probability": real_probability,
        "confidence": confidence,
        "frames_analyzed": len(frames),
        "per_frame_fake_probability": fake_probs,
        "reasons": reasons
    }