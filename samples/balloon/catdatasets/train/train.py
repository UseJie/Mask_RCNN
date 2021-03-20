import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Configurations
class CatConfig(Config):
    """Configuration for training on the cat dataset
    """

    NAME = "cat"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

# Dataset
# override three functions
# load_image
# load_mask
# image_reference
class CatDataset(utils.Dataset):

    def load_cat(self, dataset_dir, subset):
        """Load a subset of the cat dataset.
        dataset_dir: Rott directory of the dataset.
        subset: Subset to load: train or val
        """
        # 添加类型.
        self.add_class("cat", 1, "cat")

        # Train or validation dataset?
        # Directory structure is
        # dataset_dir/catdatasets/train/......
        # dataset_dir/catdatasets/val/......
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # 我们使用lableme标注的图片并且生成json文件
        # 读取一个文件夹下所有的json文件
        if subset is 'train':
            filenames = os.listdir(dataset_dir)
            print(filenames)
            annotations = []
            for filename in filenames:
                # 结果是一个字典包含：label，point，group_id，shape_type，flags
                # filename_annotation = list(json.load(open(dataset_dir.join(filename)))['shapes'])[0]
                # print(filename_annotation)
                pass
        elif subset is 'val':
            print('current directory is val')
            pass

    def load_mask(self, image_id):
        pass

    def image_reference(self, image_id):
        pass

if __name__ == '__main__':
    cat = CatDataset()
    cat.load_cat('/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/', 'train')
