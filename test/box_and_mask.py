"""box_and_mask.py
Test if the predicted bbox and the bbox convert from mask are (more of less) the same.
"""
import argparse
import glob
import json
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


def process_all_sequences_mot(input_dir: str, image_dir: str, track_result_format: str, datasrc: str):
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


def process_all_sequences_unovost(input_dir: str, image_dir: str,
                                  track_result_format: str, datasrc: str, frames_format=".npz"):
    """
    The proposals loaded from `unovost` format, has bbox in [x1,y1,x2,y2]
    """
    root_dir = os.path.join(input_dir, datasrc)
    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
    for v_idx, video in enumerate(videos):
        print("{}/{} Processing: {}".format(v_idx+1, len(videos), video))
        video_dir = os.path.join(root_dir, video)
        frames = sorted(glob.glob(video_dir + '/*' + frames_format))
        for frame_id, fpath in enumerate(tqdm.tqdm(frames)):
            if frames_format == ".json":
                with open(fpath, 'r') as f:
                    proposals = json.load(f)
            elif frames_format == ".npz":
                proposals = np.load(fpath, allow_pickle=True)['arr_0']
                proposals = proposals.tolist()

            for prop in proposals:
                if area(prop["instance_mask"]) == 0:
                    continue
                bbox = np.array(prop["bbox"])  # [x1,y1,x2.y2]
                x, y, w, h = toBbox(prop["instance_mask"])
                # Convert [x,y,w,h] to [x1,y1,x2,y2]
                bbox_from_mask = np.array([x, y, x+w, y+h])
                diff = np.abs(bbox - bbox_from_mask)

                is_wrong = False
                # The mask is clearly wrong, if its box is outside of regressed-bbox
                if int(bbox[0]) > int(bbox_from_mask[0]):
                    is_wrong = True
                if int(bbox[1]) > int(bbox_from_mask[1]):
                    is_wrong = True
                if math.ceil(bbox[2]) + 1 < int(bbox_from_mask[2]):
                    is_wrong = True
                if math.ceil(bbox[3]) + 1 < int(bbox_from_mask[3]):
                    is_wrong = True

                # if (diff > 50).any() and is_wrong:
                if (diff > 50).any():
                    print(fpath)
                    print("frame id:", frame_id)
                    print("bbox: {}".format(bbox.tolist()))
                    print("mask: {}".format(bbox_from_mask.tolist()))
                    print("score: {}".format(prop["objectness"]))

                # else:
                    # Visualize the mask
                    img_name = fpath.split('/')[-1].replace(frames_format, ".jpg")
                    img_fpath = os.path.join(image_dir, video, img_name)
                    x1, y1, x2, y2 = prop["bbox"]
                    bbox_to_draw = [x1, y1, x2-x1, y2-y1]
                    mask = {"size": prop["instance_mask"]["size"],
                            "counts": prop["instance_mask"]["counts"].encode(encoding='UTF-8')}
                    test = mask2bbox(mask)

                    # TEST
                    smoothed = mask_smooth(prop["instance_mask"])
                    xs, ys, ws, hs = toBbox(smoothed)
                    bbox_from_smoothed = [xs, ys, xs+ws, ys+hs]
                    print("smoo: {}".format(bbox_from_smoothed ))
                    # END of TEST

                    vis_one_proposal(img_fpath, bbox_to_draw, mask)


if __name__ == "__main__":
    # ===============================
    # History of args:
    # ===============================
    # --input_dir
    # /home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/forSORT_masks/_objectness/
    # --track_result_format mot

    # --input_dir
    # /home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS/_objectness/
    # --track_result_format
    # unovost

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', type=str, help="input directory of the tracking result")
    parser.add_argument('--image_dir', type=str, help="image directory of the dataset")
    parser.add_argument('--datasrc', default="BDD", type=str,
                        help="[ArgoVerse, BDD, Charades, LaSOT, YFCC100M]")
    parser.add_argument('--track_result_format', type=str, help='"mot" or "coco" or "unovost"')

    args = parser.parse_args()
    image_dir = os.path.join(args.image_dir, args.datasrc)

    # # TEST of the stability of mask encode & decode
    # test_arr = np.load("test.npy")
    # new_arr = np.copy(test_arr)
    # if test_arr[203][1525] == 1:
    #     print("The test_arr is crooked")
    #     new_arr[203][1525] = 0
    #
    # # Encode
    # wrong_rle = encode(np.array(test_arr[:, :, np.newaxis], order='F'))[0]
    # wrong_rle['counts'] = wrong_rle['counts'].decode(encoding="utf-8")
    #
    # right_rle = encode(np.array(new_arr[:, :, np.newaxis], order='F'))[0]
    # right_rle['counts'] = right_rle['counts'].decode(encoding="utf-8")
    #
    # orginal_str = 'bXfo0:TU1<E8I4L2N2N0001O0O10000000010O02OO1O010O2N2N2N1O1N3N1O2M4LhkQg0@^e[G'
    # print(orginal_str)
    # print(wrong_rle["counts"])
    # print(right_rle["counts"])
    #
    # # Decode
    # decoded = decode(mask_rle)
    # if decoded[203][1525] == 1:
    #     print("Your decode method is wrong")
    #
    # import pdb; pdb.set_trace()
    # # END of TEST

    if args.track_result_format == "mot":
        process_all_sequences_mot(args.input_dir, image_dir, args.track_result_format, args.datasrc)
    elif args.track_result_format == "unovost":
        process_all_sequences_unovost(args.input_dir, image_dir, args.track_result_format, args.datasrc, ".json")


