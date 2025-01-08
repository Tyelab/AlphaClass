import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import kornia
from typing import Tuple, Union

import numpy as np
import os
import cv2


def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 256))
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    img = torch.from_numpy(img).float() / 255.0
    return {'torch': img, 'cv2': cv2.imread(img_path)}

def pred_to_im(pred, concat=True):
    pred = pred.numpy()

    if concat:
        pred = pred.sum(axis=1)
    return pred

def plot_im_and_pred(im, first='cv2', second='pred'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(im[first])
    ax[1].imshow(pred_to_im(im[second])[0])

    pred = im['pred']
    print(f'max conf is {pred.max(): .5f}')


def plot_pred_and_nms(im):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(pred_to_im(im['pred'])[0])
    ax[1].imshow(pred_to_im(im['nms'])[0])

    nms = im['nms']
    print(f'max conf is {nms.max(): .5f}')


#### peak-extraction nms

class NonMaximaSuppression2d(nn.Module):
    r"""From Kornia: Applies non maxima suppression to filter.
    """
    def __init__(self, kernel_size: Tuple[int, int], mask_only: bool = False):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int, int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = kornia.feature.nms._get_nms_kernel2d(*kernel_size)
        self.mask_only = mask_only

    @staticmethod
    def _compute_zero_padding2d(kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size  # we assume a cubic kernel
        return (pad(ky), pad(ky), pad(kx), pad(kx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.size()
        # find local maximum values
        max_non_center = F.conv2d(
            F.pad(x, list(self.padding)[::-1], mode='replicate'),
            self.kernel.repeat(CH, 1, 1, 1).to(x.device, x.dtype),
            stride=1,
            groups=CH
        ).view(B, CH, -1, H, W).max(dim=2)[0]
        mask = x > max_non_center
        if self.mask_only:
            return mask
        return x * (mask.to(x.dtype))


class NMS(nn.Module):
    def __init__(self, conf_threshold, kernel_size=(3, 3)):
        super(NMS, self).__init__()

        self.conf_threshold = conf_threshold
        self.f = NonMaximaSuppression2d(kernel_size, mask_only=True)

    def forward(self, x):
        nms = self.f(x)
        binary_array = nms * (x > self.conf_threshold)
        out = torch.nonzero(binary_array)
        confidence_values = x[binary_array.nonzero(as_tuple=True)]
        return nms, out, confidence_values




#### post-peak nms

def coord_to_box(t, buffer=3):
    t_box = torch.zeros(len(t), 6)
    t_box[:,[0,1]] = t[:,[0,1]]

    i = t[:,[2,3]]
    t_box[:,[2,3]] = i-buffer
    t_box[:,[4,5]] = i+buffer

    return t_box


def box_to_coord(b, buffer=3):
    c = torch.zeros(len(b), 4)
    c[:,[0,1]] = b[:,[0,1]]

    x = b[:,[2,4]].mean(1)
    y = b[:,[3,5]].mean(1)

    c[:,2] = x
    c[:,3] = y

    return c


def post_peak_nms(t, c, buffer=3, thresh=0.7):

    t_box = coord_to_box(t, buffer=buffer)
    unique_indices = torch.unique(t[:,0])
    unique_locations = [torch.nonzero(t[:,0] == i, as_tuple=True)[0].numpy().tolist()
                        for i in unique_indices]

    all_boxes = []
    all_confs = []
    for ii in unique_locations:
        tt=t[ii]
        tt_box=t_box[ii]
        tc=c[ii]
        unique_channels = torch.unique(tt[:,1])
        unique_ch_locs = [torch.nonzero(tt[:,1] == i, as_tuple=True)[0].numpy().tolist()
                          for i in unique_channels]

        no_nms_inds = [i_count for i_count, i in enumerate(unique_ch_locs) if len(i)==1]
        no_nms_inds = [unique_ch_locs[i][0] for i in no_nms_inds]
        send_to_nms_inds = [i_count for i_count, i in enumerate(unique_ch_locs) if len(i)>1]
        boxes_to_keep = []
        confs_to_keep = []

        if no_nms_inds:
            boxes_to_keep.append(tt_box[no_nms_inds].cpu().numpy().tolist())
            confs_to_keep.append(tc[no_nms_inds].cpu().numpy().tolist())

        if send_to_nms_inds:
            send_to_nms_ch = [unique_ch_locs[i] for i in send_to_nms_inds]

            for ch in send_to_nms_ch:
                obj_box = tt_box[ch]
                obj_conf = tc[ch]
                nms_inds = torchvision.ops.nms(obj_box[:,2:].float().cpu(), obj_conf.float().cpu(), thresh)
                keep_box = obj_box[nms_inds].cpu().numpy().tolist()
                keep_conf = obj_conf[nms_inds].cpu().numpy().tolist()

                boxes_to_keep.append(keep_box)
                confs_to_keep.append(keep_conf)
                #print(keep_box, keep_conf)

        #print(no_nms_inds)
        #print(send_to_nms_inds)
        boxes_to_keep = [x for y in boxes_to_keep for x in y]
        confs_to_keep = [x for y in confs_to_keep for x in y]
        all_boxes.append(boxes_to_keep)
        all_confs.append(confs_to_keep)

    all_boxes = torch.tensor([x for y in all_boxes for x in y])
    all_boxes = box_to_coord(all_boxes, buffer=buffer)

    all_confs = torch.tensor([x for y in all_confs for x in y])

    return all_boxes, all_confs



def agnostic_nms(processed_peaks, processed_confs, buffer, thresh):

    processed_peaks_box = coord_to_box(processed_peaks, buffer=buffer)
    unique_indices = torch.unique(processed_peaks[:,0])
    unique_locations = [torch.nonzero(processed_peaks[:,0] == i, as_tuple=True)[0].numpy().tolist()
                        for i in unique_indices]

    all_boxes = []
    all_confs = []
    boxes_to_keep = []
    confs_to_keep = []
    for ii in unique_locations:
        if len(ii) > 1:
            tt=processed_peaks[ii]
            tt_box=processed_peaks_box[ii]
            tc=processed_confs[ii]

            nms_inds = torchvision.ops.nms(tt_box[:,2:], tc, thresh)
            keep_box = tt_box[nms_inds].cpu().numpy().tolist()
            keep_conf = tc[nms_inds].cpu().numpy().tolist()
            boxes_to_keep.append(keep_box)
            confs_to_keep.append(keep_conf)

        else:
            boxes_to_keep.append(processed_peaks_box[ii].cpu().numpy().tolist())
            confs_to_keep.append(processed_confs[ii].cpu().numpy().tolist())

    boxes_to_keep = torch.tensor([x for y in boxes_to_keep for x in y])
    confs_to_keep = torch.tensor([x for y in confs_to_keep for x in y])

    boxes_to_keep = box_to_coord(boxes_to_keep, buffer=buffer)

    return boxes_to_keep, confs_to_keep
