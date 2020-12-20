import argparse
import colorsys
import glob
import json
import os
import shutil
import sys
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as rletools

from PIL import Image
from multiprocessing import Pool
# from mots_common.io import load_sequences, load_seqmap
from functools import partial
from subprocess import call

from tools.visualize import generate_colors
from tools.video_utils import frames2video

# =======================================================
# Global variables
# =======================================================
"""
known_tao_ids: set of tao ids that can be mapped exactly to coco ids.
neighbor_classes: tao classes that are similar to coco_classes.
unknown_tao_ids: all_tao_ids that exclude known_tao_ids and neighbor_classes..
"""

IOU_THRESHOLD = 0.5
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/'.join(ROOT_DIR.split('/')[:-1])
all_ids = set([i for i in range(1, 1231)])

# Category IDs in TAO that are known (appeared in COCO)
with open(ROOT_DIR + "/datasets/tao/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])
# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)

# neighbor classes
with open(ROOT_DIR + "/datasets/tao/neighbor_classes.json") as f:
    coco2neighbor_classes = json.load(f)
# Gather tao_ids that can be categorized in the neighbor_classes
neighbor_classes = set()
for coco_id, neighbor_ids in coco2neighbor_classes.items():
    neighbor_classes = neighbor_classes.union(set(neighbor_ids))

# Exclude neighbor classes from unknown_tao_ids
unknown_tao_ids = unknown_tao_ids.difference(neighbor_classes)


# =======================================================
# =======================================================



# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def process_one_frame(frame_path, next_frame_name, img_path, output_folder, topN_proposals):
    dpi = 100.0
    colors = generate_colors(30)

    # Load image and schrink it
    img = Image.open(img_path)

    size = 1024, 1024
    img.thumbnail(size, Image.ANTIALIAS)  # Downsize the image
    img = np.array(img).astype(np.uint8)
    img = img / 255.0

    img_sizes = img.shape
    fig = plt.figure()
    fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()

    # Load proposals
    with open(frame_path, 'r') as f:
        proposals = json.load(f)
    # Sort proposals by score
    proposals.sort(key=lambda p: p['objectness'], reverse=True)
    proposals = proposals[:topN_proposals]

    for idx, prop in enumerate(proposals):
        color = colors[idx % len(colors)]
        binary_mask = rletools.decode(prop['forward_segmentation'])
        apply_mask(img, binary_mask, color)

    ax.imshow(img)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fig.savefig(output_folder + "/" + next_frame_name + '.jpg')
    plt.close(fig)

    print("DONE")


def main():

    parser = argparse.ArgumentParser(description='Visualization script for tracking result')
    parser.add_argument("--props_folder", type=str,
                        default="/home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/opt_flow_output002/")
    parser.add_argument("--img_folder", type=str, default="/home/kloping/OpenSet_MOT/data/TAO/frames/val/")
    parser.add_argument("--datasrc", type=str, default='LaSOT')
    parser.add_argument("--phase", default="objectness", help="objectness, score or one_minus_bg_score", type=str)
    parser.add_argument("--topN_proposals", default="30",
                        help="for each frame, only display top N proposals (according to their scores)", type=int)
    args = parser.parse_args()

    curr_data_src = args.datasrc

    props_folder = args.props_folder + '/_' + args.phase + '/' + curr_data_src
    img_folder = args.img_folder + '/' + curr_data_src
    output_folder = args.props_folder + "/viz_" + args.phase + str(args.topN_proposals) + '/' + curr_data_src
    topN_proposals = args.topN_proposals
    print("For each frame, display top {} proposals".format(topN_proposals))

    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_folder, '*')))]
    for idx, video in enumerate(tqdm.tqdm(videos)):
        fpath = os.path.join(props_folder, video)
        frames = sorted(glob.glob(fpath + '/*' + '.json'))

        for i, frame_path in enumerate(frames):
            if i == len(frames):
                break
            next_frame_name = frames[i+1].split('/')[-1].replace(".json", "")
            img_path = os.path.join(img_folder, video, next_frame_name + '.jpg')
            outdir = os.path.join(output_folder, video)
            process_one_frame(frame_path, next_frame_name, img_path, outdir, topN_proposals)


if __name__ == "__main__":
    main()
