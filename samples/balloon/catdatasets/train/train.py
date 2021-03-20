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

    STEPS_PER_EPOCH = 10

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
        image_source = os.path.join(dataset_dir, subset)
        #dataset_dir = os.path.join(dataset_dir, subset)+'/json/'
        dataset_dir = image_source+'/json/'
        #dataset_dir = dataset_dir.join('json')
        # Load annotations
        # 我们使用lableme标注的图片并且生成json文件
        # 读取一个文件夹下所有的json文件
        if subset is 'train':
            filenames = os.listdir(dataset_dir)
            for filename in filenames:
                file_attribute = json.load(open(dataset_dir+filename))  # 读取整个json文件
                filename_shapes = list(file_attribute['shapes'])[0]     # 读取json中shapes关键字
                file_height = file_attribute['imageHeight']             # 读取height
                file_weidth = file_attribute['imageWidth']              # 读取weidth
                image_path = os.path.join(image_source, file_attribute['imagePath'])
                #polygons = filename_shapes['points']                    # 读取x，y的坐标
                # 读取x，y的坐标
                all_points_x = []
                all_points_y = []
                #print(filename_shapes['points'])
                for x_y in filename_shapes['points']:
                    all_points_x.append(x_y[0])
                    all_points_y.append(x_y[1])
                polygons = [{'all_points_x' : all_points_x,
                            'all_points_y' : all_points_y}
                            ]
                self.add_image(
                    "cat",
                    image_id=file_attribute['imagePath'],
                    path=image_path,
                    width=file_weidth, height=file_height,
                    polygons=polygons
                )
                print(self.image_info['image_id'==file_attribute['imagePath']])
                #print("info[\"polygons\"]", self.image_info)
        elif subset is 'val':
            print('current directory is val')
            pass

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        :param image_id:
        :return:
        masks: A bool array of shape [height, width, instance count] with
                one mask per instace
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "cat":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cat":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#Training
dataset_train = CatDataset()
dataset_train.load_cat('/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/', 'train')
dataset_train.prepare()
config = CatConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir='/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/')
model.train(dataset_train, dataset_train,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# def train(model):
#     """Train the model."""
#     # Training dataset.
#     dataset_train = CatDataset
#     dataset_train.load_cat(args.dataset, "train")

if __name__ == '__main__':
    cat = CatDataset()
    cat.load_cat('/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/', 'train')
