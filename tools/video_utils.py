"""video_utils.py: practical tools for video manipulation"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import cv2
import numpy as np
import os
import tqdm

from typing import List


def video2frames(video_path: str, outdir: str):
    """ Split video into frames
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        # cv2.imwrite("0002/frame%d.jpg" % count, image)     # save frame as JPEG file
        cv2.imwrite(outdir + str(count).zfill(6) + ".jpg", image)
        success, image = vidcap.read()
        print('Read a new frame {}: {}'.format(count, success))
        count += 1


def frames2video(pathIn, pathOut, fps):
    """Convert continous single frames to a video.
    Args:
        pathIn (str): Example: "/home/yang/frames/"
        pathOut (str): Example: "/home/yang/frames/output_video.mp4"
        fps (int): frame rate per second of the output video
    """
    outdir = "/".join(pathOut.split("/")[:-1])
    if not os.path.exists(outdir):
        print(outdir, "does not exist, creating it...")
        os.makedirs(outdir)

    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f)) and f != '._.DS_Store']

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    files.sort()

    for i in tqdm.tqdm(range(len(files))):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # print(filename)

        # inserting the frames into an image array
        # frame_array.append(img)
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def combine_multi_frames2video(in_paths: List[str], out_path, fps: int):
    """ Combine multiple frames (at the same time step) vertically and output one video.
    Args:
        in_paths: List of the input paths. Each element is the folder path that contains the frames.
        out_path: Example: "/home/yang/videos/combined_video.mp4"
        fps: frame rate per second of the output video
    """
    outdir = "/".join(out_path.split("/")[:-1])
    if not os.path.exists(outdir):
        print(outdir, "does not exist, creating it...")
        os.makedirs(outdir)

    frame_array = []
    files_list = []
    num_imgs = len(in_paths)
    for pathIn in in_paths:
        files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f)) and f != '._.DS_Store' and f != ".DS_Store"]
        files.sort()
        files_list.append(files)

    for i in tqdm.tqdm(range(len(files_list[0]))):
        filenames = [pathIn + files[i] for pathIn, files in zip(in_paths, files_list)]
        # reading each files
        imgs = [cv2.imread(filename) for filename in filenames]
        if None in imgs:
            print("Warning !!!!! missing frames.")
            continue
        height, width, layers = imgs[0].shape
        # print(filename)

        # combine frames vertically
        combined_img = np.vstack(imgs)
        size = (width, num_imgs*height)

        # inserting the frames into an image array
        frame_array.append(combined_img)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
