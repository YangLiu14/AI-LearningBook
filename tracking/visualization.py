import argparse
import colorsys
import glob
import json
import os
import sys
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as rletools

from PIL import Image
from multiprocessing import Pool
# from mots_common.io import load_sequences, load_seqmap
from functools import partial
from subprocess import call

from tools.visualize import generate_colors
from tools.video_utils import frames2video

BASE_DIR = "/home/kloping/OpenSet_MOT/"
class SegmentedObject:
    def __init__(self, bbox, mask, score, class_id, track_id):
        self.bbox = bbox
        self.mask = mask
        self.score = score
        self.class_id = class_id
        self.track_id = track_id


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


def process_sequence(seq_fpaths, tracks_folder, img_folder, output_folder, max_frames, annot_frames_dict,
                     topN_proposals, draw_boxes=True, create_video=True):
    folder_name = tracks_folder.split("/")[-1]
    # print("Processing sequence", seq_name)
    os.makedirs(output_folder, exist_ok=True)
    tracks = load_sequences(seq_fpaths)
    for seq_fpath in seq_fpaths:
        seq_id = seq_fpath.split('/')[-1].replace(".txt", "")
        max_frames_seq = max_frames[seq_id]
        annot_frames = annot_frames_dict[seq_id]
        visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, annot_frames, output_folder,
                            topN_proposals, draw_boxes, create_video)


def process_sequence_coco(track_result_map, img_id2name, datasrc, img_folder, output_folder, max_frames, annot_frames_dict,
                     topN_proposals, draw_boxes=True, create_video=True):
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


def visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, annot_frames, output_folder, topN_proposals, draw_boxes=True, create_video=True):
    colors = generate_colors(min(60, topN_proposals))
    dpi = 100.0
    # frames_with_annotations = [frame for frame in tracks.keys() if len(tracks[frame]) > 0]
    # img_sizes = next(iter(tracks[frames_with_annotations[0]])).mask["size"]
    frames_with_annotations = [frame.split('/')[-1] for frame in annot_frames]
    # img_sizes = next(iter(tracks[seq_id])).bbox
    for t in range(max_frames_seq):
        print("Processing frame", annot_frames[t])
        # filename_t = img_folder + "/" + seq_id + "/" + frames_with_annotations[t]
        filename_t = annot_frames[t]
        img = np.array(Image.open(filename_t), dtype="float32") / 255
        img_sizes = img.shape
        fig = plt.figure()
        fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.subplots()
        ax.set_axis_off()

        if t+1 in tracks[seq_id].keys():
            for obj in tracks[seq_id][t+1][:topN_proposals]:
                color = colors[obj.track_id % len(colors)]
                if obj.class_id == 1:
                    category_name = "obj"
                elif obj.class_id == 2:
                    category_name = "Pedestrian"
                else:
                    category_name = "Ignore"
                    color = (0.7, 0.7, 0.7)
                if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
                    x, y, w, h = obj.bbox
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
        fig.savefig(output_folder + "/" + seq_id + "/" + frames_with_annotations[t])
        plt.close(fig)

    if create_video:
        frames2video(pathIn=output_folder + "/" + seq_id , pathOut=output_folder + "/" + seq_id + ".mp4", fps=10)

        # os.chdir(output_folder + "/" + seq_id)
        # call(["ffmpeg", "-framerate", "10", "-y", "-i", "%06d.jpg", "-c:v", "libx264", "-profile:v", "high", "-crf",
        #       "20",
        #       "-pix_fmt", "yuv420p", "-vf", "pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'", "output.mp4"])


def main():
    # if len(sys.argv) != 5:
    #     print("Usage: python visualize_mots.py tracks_folder(gt or tracker results) img_folder output_folder seqmap")
    #     sys.exit(1)

    # tracks_folder = sys.argv[1]
    # img_folder = sys.argv[2]
    # output_folder = sys.argv[3]
    # seqmap_filename = sys.argv[4]

    parser = argparse.ArgumentParser(description='Visualization script for tracking result')
    parser.add_argument("--tracks_folder", type=str, default="/home/kloping/OpenSet_MOT/Tracking/SORT_results/")
    parser.add_argument("--img_folder", type=str, default="/home/kloping/OpenSet_MOT/data/TAO/frames/val/")
    parser.add_argument("--datasrc", type=str, default='ArgoVerse')
    parser.add_argument("--phase", default="objectness", help="objectness, score or one_minus_bg_score", type=str)
    parser.add_argument("--topN_proposals", default="30",
                        help="for each frame, only display top N proposals (according to their scores)", type=int)
    parser.add_argument("--track_format", help="The file format of the tracking result", type=str, default='mot')
    args = parser.parse_args()

    if args.track_format == "mot":
        """
        The tracking result is stored in the mot-format.
            https://motchallenge.net/instructions/
        """
        curr_data_src = args.datasrc

        with open('../datasets/tao/tao_val_subset.txt', 'r') as f:
            content = f.readlines()
        content = [c.strip() for c in content]

        tracks_folder = args.tracks_folder + '/_' + args.phase + '/' + curr_data_src
        img_folder = args.img_folder + '/' + curr_data_src
        output_folder = args.tracks_folder + "/viz_" + args.phase + '/' + curr_data_src
        topN_proposals = args.topN_proposals
        print("For each frame, display top {} proposals".format(topN_proposals))
        # seqmap_filenames = sorted(glob.glob(tracks_folder + '/*' + '.txt'))
        seqmap_filenames = [os.path.join(args.tracks_folder, '_' + args.phase, curr_data_src, c.split('/')[-1] + '.txt')
                            for c in content if c.split('/')[0] == curr_data_src]
        seqmap_filenames = seqmap_filenames[:1]  # TODO: delete

        # Image path in all sequences
        all_frames = dict()  # {seq_name: List[frame_paths]}
        sequences = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(img_folder, '*')))]
        for video in sequences:
            fpath = os.path.join(img_folder, video)
            frames = sorted(glob.glob(fpath + '/*' + '.jpg'))
            all_frames[video] = frames

        max_frames = dict()

        for seq_fpath in seqmap_filenames:
            seq_name = seq_fpath.split('/')[-1][:-4]
            max_frames[seq_name] = len(all_frames[seq_name])

            # Get the max number of frames of the current sequence
            # num_frames = 0
            # with open(seq_fpath, 'r') as f:
            #     for line in f:
            #         line = line.strip()
            #         if not line:
            #             continue
            #         fields = line.split(",")
            #         num_frames = int(fields[0])
            # max_frames[seq_name] = num_frames

        # annot_frames = dict()  # annotated frames for each sequence
        # # Get the annotated frames in the current sequence.
        # txt_fname = "../datasets/tao/val_annotated_{}.txt".format(curr_data_src)
        # with open(txt_fname) as f:
        #     content = f.readlines()
        # content = ['/'.join(c.split('/')[1:]) for c in content]
        # annot_seq_paths = [os.path.join(img_folder, x.strip()) for x in content]
        #
        # for s in annot_seq_paths:
        #     seq_name = s.split('/')[-2]
        #     if seq_name not in annot_frames.keys():
        #         annot_frames[seq_name] = []
        #     annot_frames[seq_name].append(s)

        process_sequence_part = partial(process_sequence, max_frames=max_frames, annot_frames_dict=all_frames,
                                        tracks_folder=tracks_folder, img_folder=img_folder,
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

        process_sequence_part = partial(process_sequence_coco, max_frames=max_frames, datasrc=datasrc, annot_frames_dict=annot_frames,
                                        img_id2name=img_id2name, img_folder=img_folder, output_folder=output_folder)
        process_sequence_part(track_result_map)

    # with Pool(10) as pool:
    #     pool.map(process_sequence_part, seqmap)
    # for seq in seqmap:
    #  process_sequence_part(seq)


if __name__ == "__main__":
    main()
