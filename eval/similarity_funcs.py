import glob
import numpy as np
import os

from filterpy.kalman import KalmanFilter
from pycocotools.mask import encode, decode, toBbox
from pycocotools.mask import iou as mask_iou

from eval_utils import open_flow_png_file, warp_flow, bbox_iou

np.random.seed(0)


# ================================================================
# Kalman Filter - adapted from SORT
# https://github.com/abewley/sort
# ================================================================
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


# ================================================================
# Optical Flow
# ================================================================
def match_warped_mask_with_props(warped_mask, masks):
    """
    Match the warped_mask with proposals by comparing the IoU score
    Args:
        warped_mask: rle_mask,
            using optical flow vector to warp mask from previous frame to current frame.
        masks: list of rle_masks. Masks in current frame.

    Returns:
        index of the picked mask
    """
    iou_scores = mask_iou([warped_mask], masks, np.array([0], np.uint8))
    return int(np.argmax(iou_scores)), np.max(iou_scores)


def similarity_optical_flow(gt_per_video, prop_L, props_R, frameL, frameR, image_dir, prop_dir, opt_flow_dir, use_frames_in_between=False):
    """
    Compare the similarities of one proposals (in frame_L) with N proposals (in frame_R).
    Between frame_L and frame_R, there could be k continous frames.
    Args:
        gt_per_video: all the ground truth in currect video, organized into corresponding frames.
        prop_L: Dict. A single proposal at frame L
        props_R: list of N proposals at frame R
        image_dir: str. Directory contains the images in the current video sequence.
        frameL: str. Frame name of the left image.
        frameR: str. Frame name of the right image.
        opt_flow_dir: str. forward optical flow dir.

    Returns:
        bool. Whether prop_L finds a match among any proposals in props_R
    """
    # Extract information from prop_L
    mask_L = decode(prop_L['instance_mask'])
    gt_track_id = prop_L['gt_track_id']

    if use_frames_in_between:
        image_paths = sorted(glob.glob(image_dir + '/*' + '.jpg'))
        prop_paths = sorted(glob.glob(prop_dir + '/*' + '.npz'))
        idx_L = image_paths.index(os.path.join(image_dir, frameL + '.jpg'))
        idx_R = image_paths.index(os.path.join(image_dir, frameR + '.jpg'))
        assert idx_L < idx_R
        image_idxs = np.arange(idx_L, idx_R + 1)

        assert opt_flow_dir, "if using frames in between, then optical flow must be pre-computed."
        flow_fpaths = sorted(glob.glob(opt_flow_dir + '/*' + '.png'))
        for idx in image_idxs[:-1]:  # exclude the last frame, which is frameR
            flow_fn1 = flow_fpaths[idx * 2]
            flow_fn2 = flow_fpaths[idx * 2 + 1]
            # Check the consistence of file names
            image_fn = image_paths[idx].replace('.jpg', '')
            image_fn = image_fn.split('/')[-1]
            assert image_fn in flow_fn1 and image_fn in flow_fn2

            flow = open_flow_png_file([flow_fpaths[idx * 2], flow_fpaths[idx * 2 + 1]])
            warped_mask = warp_flow(mask_L, flow)  # warp flow to next frame
            warp_enc = encode(np.array(warped_mask[:, :, np.newaxis], order='F'))[0]
            warp_enc['counts'] = warp_enc['counts'].decode(encoding="utf-8")

            # Load proposals of next_frame (idx+1)
            next_proposals = np.load(prop_paths[idx + 1], allow_pickle=True)['arr_0'].tolist()
            next_masks = [prop['instance_mask'] for prop in next_proposals]
            mask_idx, _ = match_warped_mask_with_props(warp_enc, next_masks)
            # Update mask_L
            mask_L = decode(next_proposals[mask_idx]['instance_mask'])

        # Match warped-mask with the proposals in last frame
        masks_R = [prop['instance_mask'] for prop in props_R]
        mask_L = encode(np.array(mask_L[:, :, np.newaxis], order='F'))[0]
        mask_L['counts'] = mask_L['counts'].decode(encoding="utf-8")
        match, top_iou = match_warped_mask_with_props(mask_L, masks_R)
        if top_iou <= 0.5:
            return 0
        # Check if the picked prposals match with corresponding gt in frameR
        prop_R = props_R[match]
        gt_objects_R = gt_per_video[frameR]
        for obj in gt_objects_R:
            if obj['track_id'] == gt_track_id:
                x, y, w, h = obj['bbox']
                gt_box = [x, y, x+w, y+h]
                return 1 if bbox_iou(gt_box, prop_R['bbox']) > 0.5 else 0

