import glob
import json
import numpy as np
import os
import tqdm

from pycocotools.mask import encode, decode, toBbox

from eval_utils import open_flow_png_file, warp_flow, bbox_iou


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
    have highes IoU match with each gt_objects.
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
    for gt_obj in gt_objects:
        x, y, w, h = gt_obj['bbox']
        # convert [x,y,w,h] to [x1,y1,x2,y2]
        gt_box = [x, y, x+w, y+h]
        ious = np.array([bbox_iou(gt_box, box) for box in prop_bboxes])
        chosen_idx = int(np.argmax(ious))
        picked_props.append(proposals[chosen_idx])

    return picked_props


def eval_similarity(datasrc: str, gt_path: str, prop_dir: str):
    # Only load gt and proposals relevant to current datasrc
    gt = load_gt(gt_path, datasrc)

    for video in gt.keys():
        proposals_per_video = load_proposals(prop_dir, video)
        annot_frames = sorted(list(proposals_per_video[video]))
        pairs = [(frame1, frame2) for frame1, frame2 in zip(annot_frames[:-1], annot_frames[1:])]

        for frameL, frameR in pairs:
            frameL_path = os.path.join(prop_dir, video, frameL + '.npz')
            gt_objects = gt[video][frameL]
            props_L = match_prop_to_gt(frameL_path, gt_objects)
            props_R = np.load(os.path.join(prop_dir, video, frameR + '.npz'),
                              allow_pickle=True)['arr_0'].tolist()

            # ================================================
            # TODO: not implemented yet
            # ================================================
            # similarities = similarity_function(pair.first_proposal, pair.all_second_proposals)
            # top_similarity = argmax(similarities)
            # correct = compare(top_similarity, pair.correct_first_proposals)
            # num_correct += correct
            # num_evaled += 1


# ================================================================
# Switch the method here for different similarity method
# ================================================================
def optical_flow_match(prop_L, props_R, image_dir, frame_L, frame_R, opt_flow_dir, use_frames_in_between=False):
    """
    Compare the similarities of one proposals (in frame_L) with N proposals (in frame_R).
    Between frame_L and frame_R, there could be k continous frames.
    Args:
        prop_L: Dict. A single proposal.
        props_R: list of N proposals.
        image_dir: str. Directory contains the images in the current video sequence.
        frame_L: str. Frame name of the left image.
        frame_R: str. Frame name of the right image.
        opt_flow_dir: str. forward optical flow dir.

    Returns:
        numpy.array, shape=(1, N), similarity matrix
    """
    # Extract information from prop_L
    mask_L = decode(prop_L['instance_mask'])

    if use_frames_in_between:
        image_paths = sorted(glob.glob(image_dir + '/*' + '.jpg'))
        idx_L = image_paths.index(os.path.join(image_dir, frame_L))
        idx_R = image_paths.index(os.path.join(image_dir, frame_R))
        assert idx_L < idx_R
        image_idxs = np.arange(idx_L, idx_R + 1)

        assert opt_flow_dir, "if using frames in between, then optical flow must be pre-computed."
        flow_fpaths = sorted(glob.glob(opt_flow_dir + '/*' + '.png'))
        for idx in image_idxs:
            flow_fn1 = flow_fpaths[idx * 2]
            flow_fn2 = flow_fpaths[idx * 2 + 1]
            # Check the consistence or file name.
            image_fn = image_paths[idx].replace('.jpg', '')
            import pdb; pdb.set_trace()

            flow = open_flow_png_file([flow_fpaths[idx * 2], flow_fpaths[idx * 2 + 1]])
            mask_L = warp_flow(mask_L, flow)  # warp flow to next frame
# ================================================================
# different similarity method
# ================================================================


if __name__ == "__main__":
    # datasrcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M", "AVA", "HACS"]
    datasrc = "ArgoVerse"
    image_dir = os.path.join("/storage/slurm/liuyang/data/TAO/TAO_VAL/val/", datasrc)
    prop_dir = os.path.join("/storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/"
                            "Panoptic_Cas_R101_NMSoff_forTracking_Embed/preprocessed/", datasrc)
    opt_flow_dir = os.path.join("/storage/slurm/liuyang/Optical_Flow/pwc_net/", datasrc)
    gt_path = "/storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json"

    eval_similarity(datasrc, gt_path, prop_dir)
