import numpy as np
import os
import spines
from mrcnn.model import data_generator
import tensorflow as tf

config = spines.SpinesConfig()
config.display()

# Training dataset
dataset_train = spines.SpinesDataset()
dataset_train.load_spines(200,config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],'train')
dataset_train.prepare()
#
# Validation dataset
dataset_val = spines.SpinesDataset()
dataset_val.load_spines(30, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],'val')
dataset_val.prepare()

def to_ndarrays(dataset, file_name):
    train_generator = data_generator(dataset_train, config, shuffle=False,
                                     batch_size=1)

    dataset_length = len(dataset_train.image_ids)
    # dataset_length = 10

    images = []
    image_meta = []
    rpn_match = []
    rpn_bbox = []
    gt_class_ids = []
    gt_boxes = []
    gt_masks = []

    for i in range(dataset_length):
        (i, i_m, r_m, r_b, gt_c_id, gt_b, gt_m), _ = next(train_generator)
        images.append(i)
        image_meta.append(i_m)
        rpn_match.append(r_m)
        rpn_bbox.append(r_b)
        gt_class_ids.append(gt_c_id)
        gt_boxes.append(gt_b)
        gt_masks.append(gt_m.astype(np.bool))

    images_nd = np.concatenate(images)
    image_meta_nd = np.concatenate(image_meta)
    rpn_match_nd = np.concatenate(rpn_match)
    rpn_bbox_nd = np.concatenate(rpn_bbox)
    gt_class_ids_nd = np.concatenate(gt_class_ids)
    gt_boxes_nd = np.concatenate(gt_boxes)
    gt_masks_nd = np.concatenate(gt_masks)

    data = {
        "images": images_nd,
        "image_meta": image_meta_nd,
        "rpn_match": rpn_match_nd,
        "rpn_bbox": rpn_bbox_nd,
        "gt_class_ids": gt_class_ids_nd,
        "gt_boxes_nd": gt_boxes_nd,
        "gt_masks_nd": gt_masks_nd
    }

    save_path = os.path.join(os.path.dirname(__file__), "../../data" ,file_name)

    np.save(save_path, data)


to_ndarrays(dataset_train, "train")
to_ndarrays(dataset_val, "val")

