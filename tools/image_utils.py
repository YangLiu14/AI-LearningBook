"""image_utils.py: practical tool for image manipulation"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import colorsys
import cv2
import numpy as np
import os
import png as pypng  # pip install pypng
import PIL.Image as Image


def image_stitching(image_paths, rows, cols, out_path):
    """
    Stitch list of images into (rows x cols) image-tiles.
    Args:
        image_paths: List of image paths, ordered in the fashion that, the 1st image with be placed at (0,0)
                     in the resulting image tile, the 2nd image at(0,1), 3rd at (0,2) and so on.
        rows: Int, number of rows in the resulting image tile.
        cols: Int, number of columns in the resulting image tile.
        out_path: Str, output path and file name.
    """
    assert rows * cols == len(image_paths)
    # Read images as numpy arrays
    img_list = list()
    # for path in image_paths:
    #     img = Image.open(path)
    #     img_np = np.array(img.getdata())
    #     img_list.append(img_np)
    img_list = [cv2.imread(filename) for filename in image_paths]

    # combine images vertically
    img_vert = list()

    while img_list:
        curr_row = img_list[:3]
        img_list = img_list[3:]
        combined_img = np.hstack(curr_row)
        img_vert.append(combined_img)
    # combine images horizontally
    all_combined = np.vstack(img_vert)

    # Save image
    cv2.imwrite(out_path, all_combined)



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


if __name__ == "__main__":
    image_paths = ["COCOunknown_score.png", "COCOunknown_bg.png", "COCOunknown_(1-bg).png",
                   "COCOneighbor_score.png", "COCOneighbor_bg.png", "COCOneighbor_(1-bg).png",
                   "COCOknown_score.png", "COCOknown_bg.png", "COCOknown_(1-bg).png"]
    root_dir = ""
    for i in range(len(image_paths)):
        image_paths[i] = root_dir + image_paths[i]

    output_path = "combined.png"
    image_stitching(image_paths, 3, 3, output_path)