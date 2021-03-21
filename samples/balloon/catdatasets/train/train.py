"""
Mask R-CNN
Train on the cat dataset and implement color splash effect.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.color
import skimage.io
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

def color_splash(image, mask):
    """
    Apply color splash effect.
    :param image: RGB image [height, width, 3]
    :param mask: instance segmentation mask [height, width, instance count]
    :return: result image
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)

    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        pass
    elif video_path:
        pass


# #Training
# dataset_train = CatDataset()
# dataset_train.load_cat('/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/', 'train')
# dataset_train.prepare()
# config = CatConfig()
# model = modellib.MaskRCNN(mode="training", config=config,
#                           model_dir='/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/')
# model.train(dataset_train, dataset_train,
#             learning_rate=config.LEARNING_RATE,
#             epochs=10,
#             layers='heads')

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CatDataset()
    dataset_train.load_cat(args.dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CatDataset()
    dataset_val.load_cat(args.dataset, 'val')
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

if __name__ == '__main__':
    # cat = CatDataset()
    # cat.load_cat('/Users/JIE/GitHub/Mask_RCNN/samples/balloon/catdatasets/', 'train')
    import argparse

    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cat.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cat/dataset/",
                        help='Directory of the Cat dataset')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (defalut=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to vido",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is requried for training"
    elif args.command == "splash":
        assert  args.image or args.video,\
        "Provide --image or --video to apply color splash"

    print("Weight: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations:
    if args.command == "train":
        config = CatConfig()
    else:
        class InferenceConfig(CatConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weight.lower() == "imagenet":
        # Start from ImageNet trained weight
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized."
              "Use 'train' or 'splash'".format(args.command))
