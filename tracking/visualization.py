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

from tools.visualize import generate_colors, colormap
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


class SegmentedObject:
    def __init__(self, bbox, mask, score, class_id, track_id):
        self.bbox = bbox
        self.mask = mask
        self.score = score
        self.class_id = class_id
        self.track_id = track_id


def box_IoU_xywh(boxA, boxB):
    """input box: [x,y,w,h]"""
    # convert [x,y,w,h] to [x1,y1,x2,y2]
    xA, yA, wA, hA = boxA
    boxA = [xA, yA, xA + wA, yA + hA]
    xB, yB, wB, hB = boxB
    boxB = [xB, yB, xB + wB, yB + hB]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


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
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            # class_id = int(fields[2])
            class_id = 1
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            fields[12] = fields[12].strip()
            score = float(fields[6])
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
                score,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def load_sequences(seq_paths):
    objects_per_frame_per_sequence = {}
    print("Loading Sequences")
    for seq_path_txt in tqdm.tqdm(seq_paths):
        seq = seq_path_txt.split("/")[-1].replace(".txt", "")
        if os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
            # Sort proposals in each frame by score
            for frame_id, props in objects_per_frame_per_sequence[seq].items():
                props.sort(key=lambda p: p.score, reverse=True)
        else:
            assert False, "Can't find data in directory " + seq_path_txt

    return objects_per_frame_per_sequence


def process_sequence(seq_fpaths, tracks_folder, img_folder, output_folder, max_frames, all_frames_dict,
                     annot_frames_dict,
                     topN_proposals, gt_frame2anns, tao_id2name, only_annotated, draw_boxes=True, create_video=True):
    folder_name = tracks_folder.split("/")[-1]
    # print("Processing sequence", seq_name)
    os.makedirs(output_folder, exist_ok=True)
    # tracks = load_sequences(seq_fpaths)
    for seq_fpath in seq_fpaths:
        tracks = load_sequences([seq_fpath])
        seq_id = seq_fpath.split('/')[-1].replace(".txt", "")
        max_frames_seq = max_frames[seq_id]
        all_frames = all_frames_dict[seq_id]
        if seq_id in annot_frames_dict.keys():
            annot_frames = annot_frames_dict[seq_id]

            visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, all_frames, annot_frames, output_folder,
                                topN_proposals, gt_frame2anns, tao_id2name, 'unknown', only_annotated, draw_boxes,
                                create_video)
            visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, all_frames, annot_frames, output_folder,
                                topN_proposals, gt_frame2anns, tao_id2name, 'known', only_annotated, draw_boxes,
                                create_video)
            visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, all_frames, annot_frames, output_folder,
                                topN_proposals, gt_frame2anns, tao_id2name, 'neighbor', only_annotated, draw_boxes,
                                create_video)
        else:
            visualize_all_sequences(seq_id, tracks, max_frames_seq, all_frames, output_folder, topN_proposals, draw_boxes=False,
                                    create_video=True)


def process_sequence_coco(track_result_map, img_id2name, datasrc, img_folder, output_folder, max_frames,
                          annot_frames_dict,
                          topN_proposals, gt_frame2anns, tao_id2name, draw_boxes=True, create_video=True):
    """
    Args:
        track_result_map: Dict { img_id: List[detections] }. tracking result
        img_id2name: a Dict, that maps img_id to img_name (datasrc/video_name/frame_name)
        img_folder:
        output_folder:
        max_frames:
        annot_frames_dict:
        draw_boxes:
        create_video:

    Returns:

    """

    # print("Processing sequence", seq_name)
    os.makedirs(output_folder, exist_ok=True)

    split = img_folder.split('/')
    if "train" in split:
        gt_path = os.path.join(BASE_DIR, "data/TAO/annotations/train.json")
    elif "val" in split:
        gt_path = os.path.join(BASE_DIR, "data/TAO/annotations/validation.json")
    else:
        raise Exception("invalid img_folder")

    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)
    gt_images = gt_dict["images"]
    img_name2id = dict()
    for img in gt_images:
        img_name = img["file_name"]
        if img["id"] not in img_id2name.keys():
            img_name2id[img_name] = -1
        img_name2id[img_name] = img["id"]

    # tracks: List of
    #   { video_name: {frame_id: List[SegmentedObject]}, sorted in the order of frame }
    tracks = dict()
    for video, frames in annot_frames_dict.items():
        frames.sort()
        if video not in tracks.keys():
            tracks[video] = dict()
        for idx, frame in enumerate(frames):
            frame_id = idx + 1
            tracks[video][frame_id] = list()
            frame_name = '/'.join(frame.split('/')[7:])
            img_id = img_name2id[frame_name]
            if img_id not in track_result_map.keys():
                # There are not detections in the current frame
                dets = list()
            else:
                dets = track_result_map[img_id]
            # Convert each dets to SegmentedObject
            dets.sort(key=lambda k: k['score'])
            for det in dets:
                tracks[video][frame_id].append(SegmentedObject(bbox=det["bbox"],
                                                               class_id=det["category_id"],
                                                               track_id=det["track_id"]))
            # only keep detections with top 10 highest scores.
            tracks[video][frame_id] = tracks[video][frame_id][:10]

    # Now we have tracks
    # Get the seq_id, which is just the video name
    # Get the max_frames

    for seq_id in annot_frames_dict.keys():
        annot_frames = annot_frames_dict[seq_id]
        max_frames_seq = len(annot_frames)
        visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, annot_frames, output_folder,
                            topN_proposals, draw_boxes, create_video)


def visualize_all_sequences(seq_id, tracks, max_frames_seq, all_frames, output_folder, topN_proposals, draw_boxes=False,
                            create_video=True):
    colors = colormap()
    dpi = 100.0
    frames_with_annotations = [frame.split('/')[-1] for frame in all_frames]
    all_frame_names = [frame.split('/')[-1] for frame in all_frames]
    for t in range(max_frames_seq):
        print("Processing frame", all_frames[t])
        filename_t = all_frames[t]
        img = np.array(Image.open(filename_t), dtype="float32") / 255
        img_sizes = img.shape
        fig = plt.figure()
        fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.subplots()
        ax.set_axis_off()

        if t + 1 in tracks[seq_id].keys():
            for obj in tracks[seq_id][t + 1][:topN_proposals]:
                color = colors[obj.track_id % len(colors)]
                if obj.class_id == 1:
                    category_name = ""
                elif obj.class_id == 2:
                    category_name = "Pedestrian"
                else:
                    category_name = "Ignore"
                    color = (0.7, 0.7, 0.7)

                if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
                    x, y, w, h = rletools.toBbox(obj.mask)
                    if draw_boxes:
                        import matplotlib.patches as patches
                        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                                 edgecolor=color, facecolor='none', alpha=1.0)
                        ax.add_patch(rect)
                    category_name += ":" + str(obj.track_id)
                    ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                                fontsize=7, ha='center', va='center', alpha=1.0)
                binary_mask = rletools.decode(obj.mask)
                apply_mask(img, binary_mask, color)

        ax.imshow(img)
        if not os.path.exists(os.path.join(output_folder + "/" + seq_id)):
            os.makedirs(os.path.join(output_folder + "/" + seq_id))
        fig.savefig(output_folder + "/" + seq_id + "/" + all_frame_names[t])
        plt.close(fig)
    if create_video:
        fps = 10
        frames2video(pathIn=output_folder + "/" + seq_id + '/',
                     pathOut=output_folder + "/" + seq_id + ".mp4", fps=fps)

        # Delete the frames
        shutil.rmtree(output_folder + "/" + seq_id)


def visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, all_frames, annot_frames, output_folder,
                        topN_proposals, gt_frame2anns, tao_id2name, split, only_annotated=True, draw_boxes=True,
                        create_video=True):
    # colors = generate_colors(min(60, topN_proposals))
    colors = colormap()
    dpi = 100.0
    # frames_with_annotations = [frame for frame in tracks.keys() if len(tracks[frame]) > 0]
    # img_sizes = next(iter(tracks[frames_with_annotations[0]])).mask["size"]
    frames_with_annotations = [frame.split('/')[-1] for frame in annot_frames]
    all_frame_names = [frame.split('/')[-1] for frame in all_frames]
    # img_sizes = next(iter(tracks[seq_id])).bbox
    if only_annotated:
        frame_idx_with_annotations = list()
        for frame_name in frames_with_annotations:
            frame_idx = all_frame_names.index(frame_name)
            frame_idx_with_annotations.append(frame_idx)
        annot_frame_idx = 0
    for t in range(max_frames_seq):
        if only_annotated:
            if t != frame_idx_with_annotations[annot_frame_idx]:
                continue
            else:
                annot_frame_idx += 1
        print("Processing frame", all_frames[t])
        # filename_t = img_folder + "/" + seq_id + "/" + frames_with_annotations[t]
        filename_t = all_frames[t]
        img = np.array(Image.open(filename_t), dtype="float32") / 255
        img_sizes = img.shape
        fig = plt.figure()
        fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.subplots()
        ax.set_axis_off()

        # GT bbox and masks
        valid_cat_ids = list()
        if split == 'known':
            valid_cat_ids = known_tao_ids
        elif split == "neighbor":
            valid_cat_ids = neighbor_classes
        elif split == "unknown":
            valid_cat_ids = unknown_tao_ids
        # Filter out gt-annotations that don't belong to current split
        gt_anns_filtered = list()
        k = '/'.join(filename_t.split('/')[-2:]).replace(".jpg", "")
        for ann in gt_frame2anns[k]:
            if ann["category_id"] in valid_cat_ids:
                gt_anns_filtered.append(ann)

        # Apply bbox and maks from Proposals to image
        if t + 1 in tracks[seq_id].keys():
            for obj in tracks[seq_id][t + 1][:topN_proposals]:
                # Compare obj.bbox with every GT-bbox
                # bbox from GT and proposal is in the form of [x,y,w,h]
                matched_gt = None
                max_IoU_score = 0
                for ann in gt_anns_filtered:
                    iou = box_IoU_xywh(obj.bbox, ann['bbox'])
                    if iou >= IOU_THRESHOLD and iou > max_IoU_score:
                        matched_gt = ann
                        max_IoU_score = max(max_IoU_score, iou)
                if not matched_gt:
                    continue

                color = colors[obj.track_id % len(colors)]
                if obj.class_id == 1:
                    category_name = ""
                elif obj.class_id == 2:
                    category_name = "Pedestrian"
                else:
                    category_name = "Ignore"
                    color = (0.7, 0.7, 0.7)

                if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
                    x, y, w, h = obj.bbox
                    if draw_boxes:
                        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                                 edgecolor=color, facecolor='none', alpha=1.0)
                        ax.add_patch(rect)
                    category_name += ":" + str(obj.track_id)
                    # ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                    #             fontsize=7, ha='center', va='center', alpha=1.0)
                binary_mask = rletools.decode(obj.mask)
                apply_mask(img, binary_mask, color)
            # Draw GT bbox
            color_gt = (1.0, 0.0, 0.0)
            for ann in gt_anns_filtered:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=5, linestyle='-.',
                                         edgecolor=color_gt, facecolor='none', alpha=0.5)
                ax.add_patch(rect)
                category_name = tao_id2name[ann["category_id"]]
                ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color_gt, weight='bold',
                            fontsize=7, ha='center', va='center', alpha=1.0)

        ax.imshow(img)
        if not os.path.exists(os.path.join(output_folder + "/" + seq_id)):
            os.makedirs(os.path.join(output_folder + "/" + seq_id))
        fig.savefig(output_folder + "/" + seq_id + "/" + all_frame_names[t])
        plt.close(fig)

    if create_video:
        fps = 1 if only_annotated else 10
        frames2video(pathIn=output_folder + "/" + seq_id + '/',
                     pathOut=output_folder + '/' + split + "/" + seq_id + ".mp4", fps=fps)

        # Delete the frames
        shutil.rmtree(output_folder + "/" + seq_id)

        # os.chdir(output_folder + "/" + seq_id)
        # call(["ffmpeg", "-framerate", "10", "-y", "-i", "%06d.jpg", "-c:v", "libx264", "-profile:v", "high", "-crf",
        #       "20",
        #       "-pix_fmt", "yuv420p", "-vf", "pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'", "output.mp4"])


def load_and_preprocessing_gt(gt_path, datasrc):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    # Get map: image_id --> frame_name
    img_id2frame_name = dict()
    frame_name2ima_id = dict()
    for image in gt["images"]:
        img_id2frame_name[image["id"]] = image["file_name"]
        frame_name2ima_id[image["file_name"]] = image["id"]

    # Get map: frame_name --> annotations
    frame_name2anns = dict()
    for ann in gt["annotations"]:
        frame_path = img_id2frame_name[ann["image_id"]]
        curr_datasrc = frame_path.split('/')[1]
        if curr_datasrc == datasrc:
            video_name = frame_path.split('/')[-2]
            frame_name = frame_path.split('/')[-1].replace(".jpg", "")
            k = video_name + '/' + frame_name
            if k not in frame_name2anns.keys():
                frame_name2anns[k] = list()
            frame_name2anns[k].append(ann)

    tao_id2name = dict()
    for cat in gt["categories"]:
        tao_id2name[cat["id"]] = cat["name"]

    # There are soem frames although listed as annotated frames, but have no annotations
    with open(ROOT_DIR + '/datasets/tao/val_annotated_{}.txt'.format(datasrc), 'r') as f:
        content = f.readlines()
    for frame in content:
        frame = frame.strip()
        frame_name = '/'.join(frame.split('/')[2:]).replace(".jpg", "")
        if frame_name not in frame_name2anns.keys():
            # print(datasrc + '/' + frame_name)
            frame_name2anns[frame_name] = list()

    return frame_name2anns, tao_id2name


def main():
    # if len(sys.argv) != 5:
    #     print("Usage: python visualize_mots.py tracks_folder(gt or tracker results) img_folder output_folder seqmap")
    #     sys.exit(1)

    # tracks_folder = sys.argv[1]
    # img_folder = sys.argv[2]
    # output_folder = sys.argv[3]
    # seqmap_filename = sys.argv[4]

    parser = argparse.ArgumentParser(description='Visualization script for tracking result')
    parser.add_argument("--tracks_folder", type=str, default="/home/kloping/OpenSet_MOT/Tracking/unovost_RAFT_noReID/")
    parser.add_argument("--gt_path", type=str, default="/home/kloping/OpenSet_MOT/data/TAO/annotations/validation.json")
    parser.add_argument("--img_folder", type=str, default="/home/kloping/OpenSet_MOT/data/TAO/frames/val/")
    parser.add_argument("--datasrc", type=str, default='LaSOT')
    parser.add_argument("--phase", default="objectness", help="objectness, score or one_minus_bg_score", type=str)
    parser.add_argument("--only_annotated", action="store_true")
    parser.add_argument("--tao_subset", action="store_true", help="if only process fixed tao-subsets")
    parser.add_argument("--topN_proposals", default="1000",
                        help="for each frame, only display top N proposals (according to their scores)", type=int)
    parser.add_argument("--track_format", help="The file format of the tracking result", type=str, default='mot')
    args = parser.parse_args()

    if args.track_format == "mot":
        """
        The tracking result is stored in the mot-format.
            https://motchallenge.net/instructions/
        """
        curr_data_src = args.datasrc

        tracks_folder = args.tracks_folder + '/_' + args.phase + '/' + curr_data_src
        img_folder = args.img_folder + '/' + curr_data_src
        output_folder = args.tracks_folder + "/viz_" + args.phase + str(args.topN_proposals) + '/' + curr_data_src
        topN_proposals = args.topN_proposals
        print("For each frame, display top {} proposals".format(topN_proposals))

        if args.tao_subset:
            with open('../datasets/tao/tao_val_subset.txt', 'r') as f:
                content = f.readlines()
            content = [c.strip() for c in content]
            seqmap_filenames = [
                os.path.join(args.tracks_folder, '_' + args.phase, curr_data_src, c.split('/')[-1] + '.txt')
                for c in content if c.split('/')[0] == curr_data_src]
        else:
            seqmap_filenames = sorted(glob.glob(tracks_folder + '/*' + '.txt'))

        # Pre-processing GT
        gt_frame_name2anns, tao_id2name = load_and_preprocessing_gt(args.gt_path, curr_data_src)

        # Image path in all sequences
        all_frames = dict()  # {seq_name: List[frame_paths]}
        sequences = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(img_folder, '*')))]
        for video in sequences:
            fpath = os.path.join(img_folder, video)
            frames = sorted(glob.glob(fpath + '/*' + '.jpg'))
            all_frames[video] = frames

        annot_frames = dict()  # annotated frames for each sequence
        if args.only_annotated:
            # Get the annotated frames in the current sequence.
            txt_fname = ROOT_DIR + "/datasets/tao/val_annotated_{}.txt".format(curr_data_src)
            with open(txt_fname) as f:
                content = f.readlines()
            content = ['/'.join(c.split('/')[2:]) for c in content]
            annot_seq_paths = [os.path.join(img_folder, x.strip()) for x in content]

            for s in annot_seq_paths:
                seq_name = s.split('/')[-2]
                if seq_name not in annot_frames.keys():
                    annot_frames[seq_name] = []
                annot_frames[seq_name].append(s)

        max_frames = dict()
        for seq_name, frames in annot_frames.items():
            max_frames[seq_name] = len(frames)

        for seq_fpath in seqmap_filenames:
            seq_name = seq_fpath.split('/')[-1][:-4]
            max_frames[seq_name] = len(all_frames[seq_name])

        process_sequence_part = partial(process_sequence, max_frames=max_frames, all_frames_dict=all_frames,
                                        annot_frames_dict=annot_frames,
                                        tracks_folder=tracks_folder, img_folder=img_folder,
                                        gt_frame2anns=gt_frame_name2anns, tao_id2name=tao_id2name,
                                        only_annotated=args.only_annotated,
                                        output_folder=output_folder, topN_proposals=topN_proposals)
        process_sequence_part(seqmap_filenames)



    elif args.track_format == "coco":
        """
        The tracking result is stored in the coco-format.
            https://github.com/TAO-Dataset/tao/blob/master/docs/evaluation.md
        """
        max_frames = dict()  # dummy variable here that is not used.
        datasrc = "Charades"  # ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M"]

        tracks_folder = "/home/kloping/OpenSet_MOT/Tracking/tao_track_results/SORT_val/sort_tao_val_results.json"
        img_folder = "/home/kloping/OpenSet_MOT/data/TAO/frames/val/"
        output_folder = "/home/kloping/OpenSet_MOT/Tracking/tao_track_results/viz/SORT_val/" + datasrc

        annot_frames = dict()  # annotated frames for each sequence
        # Get the annotated frames in the current sequence.
        txt_fname = "../data/tao/val_annotated_{}.txt".format(datasrc)
        with open(txt_fname) as f:
            content = f.readlines()
        content = ['/'.join(c.split('/')[1:]) for c in content]
        annot_seq_paths = [os.path.join(img_folder, datasrc, x.strip()) for x in content]

        for s in annot_seq_paths:
            seq_name = s.split('/')[-2]
            if seq_name not in annot_frames.keys():
                annot_frames[seq_name] = []
            annot_frames[seq_name].append(s)
            annot_frames[seq_name].sort()

        # Load tracking result json file
        with open(tracks_folder, 'r') as f:
            track_results = json.load(f)

        # Get the (train/val) GT annotation json from img_folder
        #   expected img_folder: /home/kloping/OpenSet_MOT/data/TAO/frames/val/
        split = img_folder.split('/')
        if "train" in split:
            gt_path = os.path.join(BASE_DIR, "data/TAO/annotations/train.json")
        elif "val" in split:
            gt_path = os.path.join(BASE_DIR, "data/TAO/annotations/validation.json")
        else:
            raise Exception("invalid img_folder")

        with open(gt_path, 'r') as f:
            gt_dict = json.load(f)
        gt_images = gt_dict["images"]
        img_id2name = dict()
        for img in gt_images:
            img_name = img["file_name"][4:-4]
            if img["id"] not in img_id2name.keys():
                img_id2name[img["id"]] = ''
            img_id2name[img["id"]] = img_name

        # Filter, let the images in the current datasrc remain.
        track_result_map = dict()  # {img_id: List[detections]}
        for det in track_results:
            img_name = img_id2name[det["image_id"]]
            video_name = '/'.join(img_name.split('/')[:2])  # datasrc/video_name
            curr_d = img_name.split('/')[0]
            if curr_d == datasrc:
                if det["image_id"] not in track_result_map.keys():
                    track_result_map[det["image_id"]] = list()
                track_result_map[det["image_id"]].append(det)

        process_sequence_part = partial(process_sequence_coco, max_frames=max_frames, datasrc=datasrc,
                                        annot_frames_dict=annot_frames,
                                        img_id2name=img_id2name, img_folder=img_folder, output_folder=output_folder)
        process_sequence_part(track_result_map)

    # with Pool(10) as pool:
    #     pool.map(process_sequence_part, seqmap)
    # for seq in seqmap:
    #  process_sequence_part(seq)


if __name__ == "__main__":
    main()
