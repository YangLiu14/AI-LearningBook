import cv2
import numpy as np
import os
import png


def bbox_iou(boxA, boxB):
    """
    bbox in the form of [x1,y1,x2,y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def open_flow_png_file(file_path_list):
    # Decode the information stored in the filename
    flow_png_info = {}
    for file_path in file_path_list:
        file_token_list = os.path.splitext(file_path)[0].split("_")
        minimal_value = int(file_token_list[-1].replace("minimal", ""))
        flow_axis = file_token_list[-2]
        flow_png_info[flow_axis] = {'path': file_path,
                                    'minimal_value': minimal_value}

    # Open both files and add back the minimal value
    for axis, flow_info in flow_png_info.items():
        png_reader = png.Reader(filename=flow_info['path'])
        flow_2d = np.vstack(list(map(np.float32, png_reader.asDirect()[2])))

        # Add the minimal value back
        flow_2d = flow_2d.astype(np.float32) + flow_info['minimal_value']

        flow_png_info[axis]['flow'] = flow_2d

    # Combine the flows
    flow_x = flow_png_info['x']['flow']
    flow_y = flow_png_info['y']['flow']
    flow = np.stack([flow_x, flow_y], 2)

    return flow


def warp_flow(img, flow, binarize=True):
    """
    Use the given optical-flow vector to warp the input image/mask in frame t-1,
    to estimate its shape in frame t.
    :param img: (H, W, C) numpy array, if C=1, then it's omissible. The image/mask in previous frame.
    :param flow: (H, W, 2) numpy array. The optical-flow vector.
    :param binarize:
    :return: (H, W, C) numpy array. The warped image/mask.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res