"""common.py: common tools"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import glob
import os
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


