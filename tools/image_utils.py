"""image_utils.py: practical tool for image manipulation"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import colorsys
import numpy as np
import os
import png as pypng  # pip install pypng
import PIL.Image as Image


# ================================================
# Apply mask to a single image
# ================================================
# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
    """Generate random colors.
    To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6,
            10]
    colors = [colors[idx] for idx in perm]
    return colors


def store_masks_to_png(masks, outdir: str):
    """Combine a list of input masks to a single numpy array,
    and store as png file with conserved pixel values strictly consistend with
    the mask's annotation. (single pixel values can be e.g. 2033)

    To read the mask annotations with conserved pixel values, use:
    ```
        import PIL.Image as Image
        mask = np.asarray(Image.open(FILE_PATH))
    ```

    Args:
        masks (:obj:`list` of :obj:`numpy.ndarray`):  Each np.ndarray should have exactly the same shape.
        outdir (str): Output directory of the png file.
    """
    img_size = masks[0].shape
    png = np.zeros(img_size).astype(np.uint16)
    name = "examplePNG.png"

    with open(os.path.join(outdir, name), 'wb') as f:
        # writer = pypng.Writer(width=img_size[1], height=img_size[0], bitdepth=16, greyscale=True)
        writer = pypng.Writer(width=img_size[1], height=img_size[0], bitdepth=16)
        png2list = png.tolist()
        writer.write(f, png2list)