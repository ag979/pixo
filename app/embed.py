import cv2
import numpy as np
import torch
from PIL import Image


def image_to_segmentations(model, img_path, conf_threshold=0.25, iou_threshold=0.1):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    results = model.predict(img_path, conf=conf_threshold, iou=iou_threshold)
    masks = results[0].masks.data.cpu().numpy()  # Shape: (N, h, w)

    segmentations = []

    for binary_mask in masks:
        if binary_mask.shape != (height, width):
            binary_mask = cv2.resize(
                binary_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )

        binary_mask = (binary_mask > 0).astype(np.uint8)

        masked = image_rgb * binary_mask[:, :, None]

        segmentations.append(Image.fromarray(masked))

    return segmentations


def segmentations_to_embeddings(model, transform, segmentations):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    for pil_crop in segmentations:
        input_tensor = transform(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings.append(model.backbone(input_tensor).cpu().numpy().flatten())
    return embeddings


def image_to_embeddings(src_img, segmenter, encoder):
    encoder, preprocess = encoder
    segmentations = image_to_segmentations(segmenter, src_img)
    embeddings = segmentations_to_embeddings(encoder, preprocess, segmentations)
    return embeddings


def encode_image(src_img, encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, preprocess = encoder
    pil_img = Image.open(src_img).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = encoder.backbone(input_tensor).cpu().numpy().flatten()
    return embedding
