from utils import dataset_helpers as ds_helpers
from utils import heatmap_helpers as hm_helpers
from utils import image_helpers as im_helpers
from utils import augmentation_helpers as aug_helpers


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import cv2
import pandas as pd
import time
import albumentations as A



class LoadLoop(Dataset):
    def __init__(self, labeled_data_path,
                 image_training_resW = 640, image_training_resH = 480,
                 hm_training_resW = 80, hm_training_resH = 64,
                 aug_shape_x=640, aug_shape_y=480,
                 augmentation_pipeline=False, train=False, norm_mode='norm'):

        self.labeled_data_path = labeled_data_path
        self.image_training_resW = image_training_resW
        self.image_training_resH = image_training_resH
        self.hm_training_resW = hm_training_resW
        self.hm_training_resH = hm_training_resH
        self.aug_shape_x = aug_shape_x
        self.aug_shape_y = aug_shape_y
        self.augmentation_pipeline = augmentation_pipeline
        self.train = train

        self.sample_image = np.zeros((self.aug_shape_y, self.aug_shape_x, 3))
        if self.augmentation_pipeline:
            self.both_augs = aug_helpers.main_augmentation(self.sample_image, norm_mode='norm')
            if self.train:
                self.transform = self.both_augs['train']

            else:
                self.transform = self.both_augs['val']

        else:
            self.transform = ''

        #print(self.augmentation_pipeline)


        ## image helpers
        self.images = os.listdir(os.path.join(self.labeled_data_path, 'images'))

        ## label helpers
        self.labels = pd.read_csv(os.path.join(self.labeled_data_path, 'labels', 'labels.csv'), header=None)
        self.labels.columns = ['behavior', 'x', 'y', 'image', 'width', 'height']
        self.nClasses = self.labels.behavior.nunique()
        self.label_idx = {}
        for b_count, b in enumerate(self.labels.behavior.unique().tolist()):
            self.label_idx[b] = b_count


    def __len__(self):
        return len(self.images)



    def __getitem__(self, idx):
        img_batch = self.images[idx]
        labels_batch = ds_helpers.read_labels(self.labels, img_batch).reset_index(drop=True)

        output_image = ds_helpers.read_image(os.path.join(self.labeled_data_path, 'images', img_batch))
        output_label = ds_helpers.pd_reader(df=labels_batch, idx=self.label_idx,
                                            resW=output_image.shape[1], resH=output_image.shape[0])


        if self.transform:
            mask_list = [m for m in output_label]
            augmented = self.transform(image=output_image, masks=mask_list)
            output_image = augmented['image']
            output_label = augmented['masks']

            output_image = im_helpers.letterbox(output_image, (self.image_training_resH, self.image_training_resW))[0]
            output_label = im_helpers.batch_letterbox(output_label, self.hm_training_resH, self.hm_training_resW)

        else:
            output_image = im_helpers.letterbox(output_image, (self.image_training_resH, self.image_training_resW))[0]
            output_label = im_helpers.batch_letterbox(output_label, self.hm_training_resH, self.hm_training_resW)


        output_image = np.transpose(output_image, (2,0,1))
        return output_image, output_label


class LoadLoopInference(LoadLoop):
    def __init__(self, labeled_data_path,
                 image_training_resW = 640, image_training_resH = 480,
                 hm_training_resW = 80, hm_training_resH = 64,
                 aug_shape_x=640, aug_shape_y=480,
                 augmentation_pipeline=False, train=False, norm_mode='norm'):

        super().__init__(labeled_data_path,
                         image_training_resW, image_training_resH,
                         hm_training_resW, hm_training_resH,
                         aug_shape_x, aug_shape_y,
                         augmentation_pipeline, train, norm_mode)

    def __getitem__(self, idx):
        img_batch = self.images[idx]
        labels_batch = ds_helpers.read_labels(self.labels, img_batch).reset_index(drop=True)
        output_image = ds_helpers.read_image(os.path.join(self.labeled_data_path, 'images', img_batch))
        output_image = im_helpers.letterbox(output_image, (self.image_training_resH, self.image_training_resW))[0]
        output_image = np.transpose(output_image, (2,0,1))

        x, y, behavior_channel = ds_helpers.pd_reader_inference(df=labels_batch,
                                                                idx=self.label_idx,
                                                                resW=output_image.shape[1],
                                                                resH=output_image.shape[0])

        output = {'output_image': output_image, 'xybc': [x, y, behavior_channel], 'name': img_batch}
        return output


def data_loader(labeled_data_path,
                image_training_resW = 640, image_training_resH = 480,
                hm_training_resW = 80, hm_training_resH = 64,
                aug_shape_x=640, aug_shape_y=480,
                augmentation_pipeline=False, train=False, norm_mode='norm',
                batch_size=16,
                num_workers=0, persistent_workers=False):


    dataset = LoadLoop(labeled_data_path = labeled_data_path,
                       image_training_resW = image_training_resW,
                       image_training_resH = image_training_resH,
                       hm_training_resW = hm_training_resW,
                       hm_training_resH = hm_training_resH,
                       aug_shape_x = aug_shape_x,
                       aug_shape_y = aug_shape_y,
                       augmentation_pipeline = augmentation_pipeline,
                       train = train,
                       norm_mode = norm_mode)

    if num_workers == 0:
        PW =  False
    else:
        PW = True

    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=PW, pin_memory=True, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=32, num_workers=num_workers, persistent_workers=PW, pin_memory=True, shuffle=False)

    return dataloader


def data_loader_after_split(dataset, batch_size=16, num_workers=0, persistent_workers=False):
    train_ds = dataset[0]
    test_ds = dataset[1]

    if num_workers == 0:
        PW =  False
    else:
        PW = True


    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, persistent_workers=PW, pin_memory=True, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=num_workers, pin_memory=True, shuffle=False)

    return train_dl, test_dl
