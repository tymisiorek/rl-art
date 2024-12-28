import imageio
import os

def create_animation(folder_path, output_file="animation.gif"):
    """Creates a GIF from all PNG images in `folder_path`."""
    images = []
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 9999,
    )
    for filename in files:
        image_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(image_path))
    imageio.mimsave(output_file, images, fps=2)
