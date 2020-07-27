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
