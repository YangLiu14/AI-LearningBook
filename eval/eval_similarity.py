import glob
import json
import numpy as np
import os
import tqdm

from eval_utils import open_flow_png_file, warp_flow, bbox_iou
from similarity_funcs import similarity_optical_flow


def map_image_id2fname(annot_dict: str):
    """
    Map the image_id in annotation['images'] to its index.
    Args:
        annot_dict: The annotation file (loaded from json)

    Returns:
        Dict
    """
    images = annot_dict['images']
    res = dict()
    for i, img in enumerate(images):
        res[img['id']] = img['file_name']

    return res


def load_gt(gt_path: str, datasrc: str):
    print("Loading GT")
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)

    image_id2fname = map_image_id2fname(gt_dict)

    res = dict()
    for ann in tqdm.tqdm(gt_dict['annotations']):
        cat_id = ann['category_id']
        fname = image_id2fname[ann['image_id']]
        if fname.split('/')[1] == datasrc:
            video_name = fname.split('/')[2]
            frame_name = fname.split('/')[-1].replace('.jpg', '').replace('.png', '')
            detection = {'bbox': ann['bbox'],   # [x,y,w,h]
                         'category_id': cat_id,
                         'track_id': ann['track_id']}
            if video_name not in res.keys():
                res[video_name] = dict()
            if frame_name not in res[video_name].keys():
                res[video_name][frame_name] = list()
            res[video_name][frame_name].append(detection)

    return res


def load_proposals(prop_dir, curr_video):
    datasrc = prop_dir.split('/')[-1]
    with open('../datasets/tao/val_annotated_{}.txt'.format(datasrc), 'r') as f:
        txt_data = f.readlines()

    print("Loading proposals in", datasrc)
    video2annot_frames = dict()
    for line in tqdm.tqdm(txt_data):
        line = line.strip()
        video_name = line.split('/')[-2]
        if video_name == curr_video:
            frame_name = line.split('/')[-1].replace(".jpg", "").replace(".png", "")
            if video_name not in video2annot_frames.keys():
                video2annot_frames[video_name] = dict()

            # Load proposals in current frame
            frame_path = os.path.join(prop_dir, video_name, frame_name + '.npz')
            proposals = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
            video2annot_frames[video_name][frame_name] = proposals

    return video2annot_frames


def match_prop_to_gt(frame_path, gt_objects):
    """
    Compare IoU of each propals in current frame with gt_objects, return the proposals that
    have highest IoU match with each gt_objects.
    If the highest IoU score < 0.5, the proposal with not be added to the returning results

    Args:
        frame_path: the file path of current frame with contains all the proposals. (.npz file)
        gt_objects: list of dict.

    Returns:
        list of proposals.
    """
    proposals = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
    # TODO: use regressed-bbox from model or use bbox converted from mask?
    # Plan1: use regressed-box directly from model
    prop_bboxes = [prop['bbox'] for prop in proposals]
    # # Plan2: use bbox converted from mask
    # prop_bboxes = [toBbox(prop['instance_mask'] for prop in proposals)]  # [x,y,w,h]
    # prop_bboxes = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in prop_bboxes]  [x1,y1,x2,y2]

    picked_props = list()
    valid_track_ids = list()
    for gt_obj in gt_objects:
        x, y, w, h = gt_obj['bbox']
        # convert [x,y,w,h] to [x1,y1,x2,y2]
        gt_box = [x, y, x+w, y+h]
        ious = np.array([bbox_iou(gt_box, box) for box in prop_bboxes])
        if np.max(ious) > 0.5:
            chosen_idx = int(np.argmax(ious))
            proposals[chosen_idx]['gt_track_id'] = gt_obj['track_id']
            picked_props.append(proposals[chosen_idx])
            valid_track_ids.append(gt_obj['track_id'])

    return picked_props, set(valid_track_ids)


def find_objects_in_both_frames(gt, prop_dir:str, video: str, frameL: str, frameR: str):
    """
    1. Ensure that the gt_bbox in both frames can find at least one proposals with
    IoU(gt_bbox, prop_bbox) > 0.5. Otherwise we ignore the gt_bbox.

    2. If frameL contains objects {A, B, C} and frameR contains objects {B, C, D, E},
    return objects {B, C}.
    Two objects are the same in two frames, when their `track_id` matches.
    """
    objects_L = gt[video][frameL]
    objects_R = gt[video][frameR]

    _, track_ids_L = match_prop_to_gt(os.path.join(prop_dir, video, frameL + '.npz'), objects_L)
    _, track_ids_R = match_prop_to_gt(os.path.join(prop_dir, video, frameR + '.npz'), objects_R)
    common_ids = track_ids_L.intersection(track_ids_R)

    common_objects = list()
    for obj in objects_L:
        if obj['track_id'] in common_ids:
            common_objects.append(obj)

    return common_objects


def eval_similarity(datasrc: str, gt_path: str, prop_dir: str, opt_flow_dir: str, image_dir: str):
    # Only load gt and proposals relevant to current datasrc
    gt = load_gt(gt_path, datasrc)

    num_correct = 0
    num_evaled = 0
    for vidx, video in enumerate(sorted(gt.keys())):
        print("{}/{} Process Videos {}/{}".format(vidx, len(gt.keys()), datasrc, video))
        proposals_per_video = load_proposals(prop_dir, video)
        annot_frames = sorted(list(proposals_per_video[video]))
        pairs = [(frame1, frame2) for frame1, frame2 in zip(annot_frames[:-1], annot_frames[1:])]

        for frameL, frameR in tqdm.tqdm(pairs):
            frameL_path = os.path.join(prop_dir, video, frameL + '.npz')
            # gt_objects = gt[video][frameL]
            gt_objects = find_objects_in_both_frames(gt, prop_dir, video, frameL, frameR)

            props_L, _ = match_prop_to_gt(frameL_path, gt_objects)
            props_R = np.load(os.path.join(prop_dir, video, frameR + '.npz'),
                              allow_pickle=True)['arr_0'].tolist()

            # ================================================
            # Similarity match
            # ================================================
            for propL in props_L:
                match = similarity_optical_flow(propL, props_R, frameL, frameR,
                                   os.path.join(image_dir, video),
                                   os.path.join(prop_dir, video),
                                   os.path.join(opt_flow_dir, video), use_frames_in_between=True)
                num_correct += match
                num_evaled += 1
        print("Current accuracy: {}/{}".format(num_correct, num_evaled))
    print("Top 1 accuracy = {}/{} = {}".format(num_correct, num_evaled, num_correct/num_evaled) )



if __name__ == "__main__":
    # datasrcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M", "AVA", "HACS"]
    datasrc = "ArgoVerse"
    image_dir = os.path.join("/storage/slurm/liuyang/data/TAO/TAO_VAL/val/", datasrc)
    prop_dir = os.path.join("/storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/"
                            "Panoptic_Cas_R101_NMSoff_forTracking_Embed/preprocessed/", datasrc)
    opt_flow_dir = os.path.join("/storage/slurm/liuyang/Optical_Flow/pwc_net/", datasrc)
    gt_path = "/storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json"

    eval_similarity(datasrc, gt_path, prop_dir, opt_flow_dir, image_dir)
