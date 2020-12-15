"""box_and_mask.py
Test if the predicted bbox and the bbox convert from mask are (more of less) the same.
"""
import argparse
import glob
import json
import numpy as np
import os
import tqdm

from pycocotools.mask import toBbox


class SegmentedObject:
    def __init__(self, bbox, mask, class_id, track_id):
        self.bbox = bbox
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


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


def process_all_sequences(input_dir: str, track_result_format: str, datasrc: str):
    root_dir = os.path.join(input_dir, datasrc)
    files = sorted(glob.glob(root_dir + '/*' + '.txt'))
    for txt_file in files:
        all_frames = load_txt(txt_file)
        for framd_id, dets_per_frame in all_frames.items():
            for det in dets_per_frame:
                bbox = np.array(det.bbox)
                bbox_from_mask = toBbox(det.mask)
                diff = np.abs(bbox - bbox_from_mask)
                if not (diff < 20).all():
                    print(txt_file)
                    print("frame id:", framd_id)
                    print("bbox: {}".format(bbox.tolist()))
                    print("mask: {}".format(bbox_from_mask.tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', type=str, help="input directory of the tracking result")
    parser.add_argument('--datasrc', default="ArgoVerse", type=str,
                        help="[ArgoVerse, BDD, Charades, LaSOT, YFCC100M]")
    parser.add_argument('--track_result_format', type=str, help='"mot" or "coco"')

    args = parser.parse_args()

    process_all_sequences(args.input_dir, args.track_result_format, args.datasrc)