"""common.py: common tools"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import glob
import os
import shutil
import tqdm


# Template function
def list_files_in_dir(root_dir: str, file_type=".jpg"):
    # List all the folders under the root_dir
    # Only the folder names that doesn't contain the full path
    folders = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]

    for idx, folder in enumerate(tqdm.tqdm(folders)):
        fpath = os.path.join(root_dir, folder)
        # List all files in the current folder
        files = sorted(glob.glob(fpath + '/*' + file_type))
        # =================================
        # Do the rest stuffs from here
        # =================================


# Delete all txt file in the given folder
def delete_files(target_dir: str, suffix='.txt'):
    """
    Delete existing files of certain format
    """
    if os.path.exists(target_dir):
        all_files = os.listdir(target_dir)
        if len(all_files):
            print(target_dir, "is not empty, will delete existing txt files.")
            input("Press Enter to confirm DELETE and continue ...")
            for item in all_files:
                if item.endswith(suffix):
                    os.remove(os.path.join(target_dir, item))


def train_val_split(image_dir: str, split_ratio=0.9):
    """
    Split the images in a folder to train and validation set.
    Specifically designed for ImageNet-like folder structures:

    - image_dir
        - train
            - classA
            - classB
                - image1.jpg
                - image2.jpg

    :param image_dir: the root directory of different classes of images.
    :param split_ratio: train_set = num_images * ratio, val_set = num_images * (1-ratio)
    """
    folders = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(image_dir, 'train', '*')))]

    val_dir = image_dir + "/val"
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for idx, folder in enumerate(tqdm.tqdm(folders)):
        fpath = os.path.join(image_dir, 'train', folder)
        # List all files in the current folder
        files = glob.glob(fpath + '/*' + '.jpg')
        num_train = int(len(files) * split_ratio)
        num_val = len(files) - num_train

        # move
        val_folder = val_dir + '/' + folder
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        for i in range(num_train, len(files)):
            src = files[i]
            fname = src.split('/')[-1]
            dest = os.path.join(val_dir, folder , fname)
            shutil.move(src, dest)


if __name__ == "__main__":
    image_dir = "/Users/lander14/Desktop/ImageNet/"
    train_val_split(image_dir)