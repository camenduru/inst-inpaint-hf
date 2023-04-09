import gradio as gr
import numpy as np
import torch
from PIL import Image

import constants
import utils

PREDICTOR = None


def inference(image: np.ndarray, text: str, center_crop: bool):
    num_steps = 10
    if not text.lower().startswith("remove the"):
        raise gr.Error("Instruction should start with 'Remove the' !")

    image = Image.fromarray(image)
    cropped_image, image = utils.preprocess_image(image, center_crop=center_crop)

    utils.seed_everything()
    prediction = PREDICTOR.predict(image, text, num_steps)

    print("Num steps:", num_steps)

    return cropped_image, prediction


if __name__ == "__main__":
    utils.setup_environment()

    if not PREDICTOR:
        PREDICTOR = utils.get_predictor()

    sample_image, sample_instruction, sample_step = constants.EXAMPLES[3]

    gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(type="numpy", value=sample_image, label="Source Image").style(
                height=256
            ),
            gr.Textbox(
                label="Instruction",
                lines=1,
                value=sample_instruction,
            ),
            gr.Checkbox(value=True, label="Center Crop", interactive=False),
        ],
        outputs=[
            gr.Image(type="pil", label="Cropped Image").style(height=256),
            gr.Image(type="pil", label="Output Image").style(height=256),
        ],
        allow_flagging="never",
        examples=constants.EXAMPLES,
        cache_examples=True,
        title=constants.TITLE,
        description=constants.DESCRIPTION,
    ).launch()
