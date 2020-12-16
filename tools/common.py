"""common.py: common tools"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import glob
import os
import shutil
import pycocotools.mask as rletools
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
            dest = os.path.join(val_dir, folder, fname)
            shutil.move(src, dest)


# =========================================
# Code from mots-tools
# https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py
# ========================================
class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_sequences(path, seqmap):
    objects_per_frame_per_sequence = {}
    for seq in seqmap:
        print("Loading sequence", seq)
        seq_path_folder = os.path.join(path, seq)
        seq_path_txt = os.path.join(path, seq + ".txt")
        if os.path.isdir(seq_path_folder):
            pass
            # objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
        elif os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
        else:
            assert False, "Can't find data in directory " + path

    return objects_per_frame_per_sequence


def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def load_seqmap(seqmap_filename):
    print("Loading seqmap...")
    seqmap = []
    max_frames = {}
    with open(seqmap_filename, "r") as fh:
        for i, l in enumerate(fh):
            fields = l.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)
            max_frames[seq] = int(fields[3])
    return seqmap, max_frames
# ======================================================


if __name__ == "__main__":
    image_dir = "/Users/lander14/Desktop/ImageNet/"
    train_val_split(image_dir)
