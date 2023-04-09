import logging
import os
import random
import tarfile
from typing import Tuple

import dill
import gdown
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

logger = logging.getLogger(__file__)

to_tensor = ToTensor()


def preprocess_image(
    image: Image, resize_shape: Tuple[int, int] = (256, 256), center_crop=True
):
    processed_image = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        processed_image = image.crop((left, top, right, bottom))

    processed_image = processed_image.resize(resize_shape)

    image = to_tensor(processed_image)
    image = image.unsqueeze(0) * 2 - 1

    return processed_image, image


def download_artifacts(output_path: str):
    logger.error("Downloading the model artifacts...")
    if not os.path.exists(output_path):
        gdown.download(id=os.environ["GDRIVE_ID"], output=output_path, quiet=True)


def extract_artifacts(path: str):
    logger.error("Extracting the model artifacts...")
    if not os.path.exists("model.pkl"):
        with tarfile.open(path) as tar:
            tar.extractall()


def setup_environment():
    os.environ["PYTHONPATH"] = os.getcwd()

    artifacts_path = "artifacts.tar.gz"

    download_artifacts(output_path=artifacts_path)

    extract_artifacts(path=artifacts_path)


def get_predictor():
    logger.error("Loading the predictor...")
    with open("model.pkl", "rb") as fp:
        return dill.load(fp)


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
