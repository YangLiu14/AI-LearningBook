import glob
import numpy as np
import os

from pycocotools.mask import encode, decode, toBbox
from pycocotools.mask import iou as mask_iou

from eval_utils import open_flow_png_file, warp_flow, bbox_iou

# ================================================================
# Switch the method here for different similarity method
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


def similarity_optical_flow(prop_L, props_R, frameL, frameR, image_dir, prop_dir, opt_flow_dir, use_frames_in_between=False):
    """
    Compare the similarities of one proposals (in frame_L) with N proposals (in frame_R).
    Between frame_L and frame_R, there could be k continous frames.
    Args:
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
        return 1 if top_iou > 0.5 else 0
