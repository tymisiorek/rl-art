# src/utils.py
import imageio
import os
from PIL import Image
import numpy as np

def create_animation(folder_path, output_file="animation.gif"):
    images = []
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 9999,
    )

    base_size = None

    for filename in files:
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            if base_size is None:
                # Set the base size from the first image
                base_size = img.size  # (width, height) in pixels
            else:
                # If this image differs in size, resize it to match the base size
                if img.size != base_size:
                    img = img.resize(base_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))

    if images:
        imageio.mimsave(output_file, images, fps=2)
