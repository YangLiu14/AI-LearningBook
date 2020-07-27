"""visualize.py: tools for visualizing bboxes, masks and trackings.

code are adapted from:
https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

"""
import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os



