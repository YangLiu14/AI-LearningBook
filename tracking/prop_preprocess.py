"""prop_preprocess.py
This script is to preprocess the 1000 proposals come from detectron2-model.
Among those 1000 propsals, there could be following unwanted cases:

    1. invalid bbox: x2 == x1 or y2 == y1
    2. invalid mask:
        - area(mask) == 0
        - one or more isolated `1` in the mask that don't belong to the object blob
    3. loose bbox prediction: the regressed bbox directly from the model is more loose than
                              the bbox computed from the mask.

This script aims to provide a sanity check of the generated proposals from detectron2,
and solve the above mentioned problem along the way.
"""

import argparse
import glob
import json
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import scipy.ndimage as ndimage
import tqdm

from PIL import Image
from pycocotools.mask import encode, decode, toBbox, area
from typing import List, Dict
from tools.visualize import apply_mask, generate_colors

class SegmentedObject:
    def __init__(self, bbox, mask, class_id, track_id):
        self.bbox = bbox
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def mask2bbox(mask_rle):
    """
    Convert mask (in RLE form) to bbox [x1,y1,x2,y2]
    """
    mask = decode(mask_rle)
    np.save("test.npy", mask)
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]


def mask_smooth(mask_rle):
    """
    Return RLE mask with completely isolated single cells removed
    https://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

    Returns:
        smoothed mask in RLE format
    """
    mask = decode(mask_rle)
    struct = np.ones((3, 3))
    smoothed = np.copy(mask)
    id_regions, num_ids = ndimage.label(smoothed, structure=struct)
    id_sizes = np.array(ndimage.sum(mask, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    smoothed[area_mask[id_regions]] = 0

    smoothed_rle = encode(np.array(smoothed[:, :, np.newaxis], order='F'))[0]
    smoothed_rle['counts'] = smoothed_rle['counts'].decode(encoding="utf-8")

    return smoothed_rle


def vis_one_proposal(img_fpath: str, bbox: List, mask: Dict, draw_boxes=True):
    """
    Visualized bbox and mask of one proposal.
    Args:
        img_fpath: the image file path.
        bbox: [x, y, w, h]
        mask: RLE format
    """
    img_name = img_fpath.split('/')[-1].replace(".jpg", "")
    colors = generate_colors()
    dpi = 100.0
    img = np.array(Image.open(img_fpath), dtype="float32") / 255
    img_sizes = mask["size"]

    fig = plt.figure()
    fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()

    color = colors[0]
    x, y, w, h = toBbox(mask)
    rect = patches.Rectangle((x, y), w, h, linewidth=1, linestyle='-.',
                             edgecolor=color, facecolor='none', alpha=1.0)
    ax.add_patch(rect)

    if draw_boxes:
        xb, yb, wb, hb = bbox
        rect = patches.Rectangle((xb, yb), wb, hb, linewidth=1,
                                 edgecolor=colors[-1], facecolor='none', alpha=1.0)
        ax.add_patch(rect)

    category_name = "object"
    ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                fontsize=7, ha='center', va='center', alpha=1.0)
    binary_mask = decode(mask)
    apply_mask(img, binary_mask, color)

    ax.imshow(img)
    fig.savefig("plots/" + img_name + ".jpg")
    plt.close(fig)


def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            # if int(fields[1]) in track_ids_per_frame[frame]:
            #     assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            # else:
            track_ids_per_frame[frame].add(int(fields[1]))

            # class_id = int(fields[2])
            class_id = 1
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            fields[12] = fields[12].strip()
            mask = {'size': [int(fields[10]), int(fields[11])], 'counts': fields[12].encode(encoding='UTF-8')}
            bbox = [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])]  # [x, y, w, h]
            # if frame not in combined_mask_per_frame:
            #   combined_mask_per_frame[frame] = mask
            # elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
            #   assert False, "Objects with overlapping masks in frame " + fields[0]
            # else:
            #   combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)

            objects_per_frame[frame].append(SegmentedObject(
                bbox,
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def load_sequences(seq_paths):
    objects_per_frame_per_sequence = {}
    print("Loading Sequences")
    for seq_path_txt in tqdm.tqdm(seq_paths[:5]):
        seq = seq_path_txt.split("/")[-1][:-4]
        if os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
        else:
            assert False, "Can't find data in directory " + seq_path_txt

    return objects_per_frame_per_sequence


def process_all_sequences_mot(input_dir: str, image_dir: str, datasrc: str):
    """
    The proposals loaded from `mot` format, has bbox in [x,y,w,h]
    """
    root_dir = os.path.join(input_dir, datasrc)
    files = sorted(glob.glob(root_dir + '/*' + '.txt'))
    for txt_file in files:
        all_frames = load_txt(txt_file)
        for frame_id, dets_per_frame in all_frames.items():
            for det in dets_per_frame:
                bbox = np.array(det.bbox)
                bbox_from_mask = toBbox(det.mask)
                diff = np.abs(bbox - bbox_from_mask)
                if not (diff < 20).all():
                    print(txt_file)
                    print("frame id:", frame_id)
                    print("bbox: {}".format(bbox.tolist()))
                    print("mask: {}".format(bbox_from_mask.tolist()))


def process_all_sequences_unovost(input_dir: str, outdir: str, image_dir: str, datasrc: str, frames_format=".npz"):
    """
    The proposals loaded from `unovost` format, has bbox in [x1,y1,x2,y2]
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    log_path = os.path.join(outdir, datasrc + '.log')
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    root_dir = os.path.join(input_dir, datasrc)
    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
    for v_idx, video in enumerate(videos):
        print("{}/{} Processing: {}".format(v_idx, len(videos), video))
        video_dir = os.path.join(root_dir, video)
        frames = sorted(glob.glob(video_dir + '/*' + frames_format))
        for frame_id, fpath in enumerate(tqdm.tqdm(frames)):
            if frames_format == ".json":
                with open(fpath, 'r') as f:
                    proposals = json.load(f)
            elif frames_format == ".npz":
                proposals = np.load(fpath, allow_pickle=True)['arr_0']
                proposals = proposals.tolist()

            processed = list()
            for prop in proposals:
                if area(prop["instance_mask"]) == 0:
                    continue
                bbox = np.array(prop["bbox"])  # [x1,y1,x2.y2]
                if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                    continue
                x, y, w, h = toBbox(prop["instance_mask"])
                # Convert [x,y,w,h] to [x1,y1,x2,y2]
                bbox_from_mask = np.array([x, y, x+w, y+h])

                is_wrong = False
                # The mask is clearly wrong, if its box is outside of regressed-bbox
                tolerance = 5
                if int(bbox[0]) > int(bbox_from_mask[0]) + tolerance:
                    is_wrong = True
                if int(bbox[1]) > int(bbox_from_mask[1]) + tolerance:
                    is_wrong = True
                if math.ceil(bbox[2]) + tolerance < int(bbox_from_mask[2]):
                    is_wrong = True
                if math.ceil(bbox[3]) + tolerance < int(bbox_from_mask[3]):
                    is_wrong = True

                # Log and Fix isolated bboxes
                if is_wrong:
                    logging.error("bbox from mask outside of regressed-bbox [x1,y1.x2,y2]")
                    logging.info(fpath)
                    logging.info("frame id:".format(frame_id))
                    logging.info("regressed-bbox: {}".format(bbox.tolist()))
                    logging.info("bbox from mask: {}".format(bbox_from_mask.tolist()))
                    logging.info("score: {}".format(prop["score"]))
                    logging.info("--- After fix ---")
                    # Consider the case of isolated '1`
                    smoothed_rle = mask_smooth(prop["instance_mask"])
                    xs, ys, ws, hs = toBbox(smoothed_rle)
                    bbox_from_smoothed = [xs, ys, xs+ws, ys+hs]
                    logging.info("smoothed bbox:  {}".format(bbox_from_smoothed))

                    prop["instance_mask"] = smoothed_rle
                    bbox_from_mask = np.array(bbox_from_smoothed)

                # Log and Fix loose regressed-bbox
                diff = np.abs(bbox - bbox_from_mask)
                if (diff > 50).any():
                    logging.debug("Loose regressed-bbox [x1,y1.x2,y2], bbox converted from mask will be used instead.")
                    logging.info(fpath)
                    logging.info("frame id:".format(frame_id))
                    logging.info("regressed-bbox: {}".format(bbox.tolist()))
                    logging.info("bbox from mask: {}".format(bbox_from_mask.tolist()))
                    logging.info("score: {}".format(prop["score"]))

                    # # Visualize the mask
                    # img_name = fpath.split('/')[-1].replace(frames_format, ".jpg")
                    # img_fpath = os.path.join(image_dir, video, img_name)
                    # x1, y1, x2, y2 = prop["bbox"]
                    # bbox_to_draw = [x1, y1, x2-x1, y2-y1]
                    # mask = {"size": prop["instance_mask"]["size"],
                    #         "counts": prop["instance_mask"]["counts"].encode(encoding='UTF-8')}
                    # test = mask2bbox(mask)
                    # vis_one_proposal(img_fpath, bbox_to_draw, mask)

                    prop["bbox"] = bbox_from_mask
                processed.append(prop)

            # Store processed proposals per frame as new npz file
            out_folder = os.path.join(outdir, datasrc, video)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            frame_name = fpath.split('/')[-1].replace(frames_format, ".npz")
            out_fpath = out_folder + '/' + frame_name
            np.savez_compressed(out_fpath, processed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', type=str, help="input directory of the proposals")
    parser.add_argument('--outdir', type=str, help="output directory of the processed propsals")
    parser.add_argument('--image_dir', type=str, help="image directory of the dataset")
    parser.add_argument('--datasrc', default="Charades", type=str,
                        help="[ArgoVerse, BDD, Charades, LaSOT, YFCC100M]")
    parser.add_argument('--track_result_format', type=str, help='"mot" or "coco" or "unovost"')

    args = parser.parse_args()
    image_dir = os.path.join(args.image_dir, args.datasrc)

    if args.track_result_format == "mot":
        process_all_sequences_mot(args.input_dir, args.outdir, image_dir, args.datasrc)
    elif args.track_result_format == "unovost":
        process_all_sequences_unovost(args.input_dir, args.outdir, image_dir, args.datasrc, ".npz")


