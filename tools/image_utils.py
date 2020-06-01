"""image_utils.py: practical tool for image manipulation"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import numpy as np
import os
import png as pypng  # pip install pypng
import PIL.Image as Image


def store_masks_as_png(masks, outdir: str):
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