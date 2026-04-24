from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM

import matplotlib.pyplot as plt

from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

import pandas as pd
from tabulate import tabulate

import json
import torch

def cam_prerequisites():
    weights = get_model_weights("resnet18").DEFAULT
    model = get_model("resnet18", weights=weights).eval()
    preprocess = weights.transforms()
    return model, preprocess

def Preprocess(model, preprocess, image_path, target_class=None):

    img = decode_image(image_path)
    input_tensor = preprocess(img)

    with LayerCAM(model) as cam_extractor:
        out = model(input_tensor.unsqueeze(0))
        if target_class is None:
            target_class = out.squeeze(0).argmax().item()
        activation_map = cam_extractor(target_class, out)
    return img, out, activation_map


def ShowOverlayMask(img, activation_map):
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def predict_class_top5(output_tensor: torch.Tensor, class_index_path: str) -> dict:
    """
    Maps the top 5 logit in a softmax output tensor to an ImageNet class name.

    Args:
        output_tensor:    1D or 2D tensor of shape (1000,) or (1, 1000),
                          typically the output of a softmax layer from ResNet18.
        class_index_path: Path to the imagenet_class_index.json file.

    Returns:
        A dict with keys:
            - 'class_index'  (int)   : index of the predicted class (0–999)
            - 'class_id'     (str)   : WordNet synset ID, e.g. "n01440764"
            - 'class_name'   (str)   : human-readable label, e.g. "tench"
            - 'confidence'   (float) : softmax probability of the top class
    """
    with open(class_index_path, "r") as f:
        class_index = json.load(f)  # keys are str "0".."999"

    # Flatten to 1-D in case the tensor has a batch dimension
    probs = output_tensor.squeeze()          # (1000,)
    if probs.ndim != 1 or probs.shape[0] != 1000:
        raise ValueError(
            f"Expected a tensor of 1000 values, got shape {tuple(output_tensor.shape)}"
        )

    top5 = torch.topk(probs, 5)
    probability_list =[]
    
    for index in top5.indices:
        index = int(index)
        synset_id, class_name = class_index[str(index)]
        probability_list.append({
            "class_index": index,
            "class_id":    synset_id,
            "class_name":  class_name,
            "confidence":  round(float(probs[index]),6)
            })
    probability_df = pd.DataFrame(probability_list)
    return probability_df


def ImageClassificator(image_path, target_class=None):
    model, preprocess = cam_prerequisites()
    img, out, activation_map = Preprocess(model, preprocess, image_path, target_class)
    ShowOverlayMask(img, activation_map)

    prediction = out.squeeze(0).softmax(0)
    df = predict_class_top5(prediction.detach(), "data/imagenet_class_index.json")
    print(tabulate(df, headers='keys', tablefmt='psql')) #Making nice print in terminal


