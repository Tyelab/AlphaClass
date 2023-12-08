import torch
import torchvision
from torch import nn

import numpy as np
import os
import json
import cv2
import pickle
import h5py
import time

import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self, method='dist'):
        self.method = method
        self.current_frame_aligned_matrix = None ## empty value for ID-adjusted current frame


    def _generate_cost_matrix(self, t1, t2):
        return scipy.spatial.distance.cdist(t1[:,2:], t2[:,2:]) ## return pairwise-distances between previous and current frame predictions


    def _generate_matching_ids(self, cost):
        return linear_sum_assignment(cost, maximize=False)[1] ## return matching ID indices from cost matrix


    def step(self, entry, conf, timestep):
        if timestep == 0:
            self.previous_frame_matrix = np.array(entry).copy() ## if first timestep, "previous" frame is really just the current frame
            self.current_frame_matrix = self.previous_frame_matrix.copy()
            self.current_frame_aligned_matrix = self.previous_frame_matrix.copy()
            self.conf = np.array(conf).copy()
            return self.current_frame_aligned_matrix, self.conf ## no alignment needed since only one frame is read

        self.current_frame_matrix = np.array(entry).copy() ## assign current frame
        self.cost = self._generate_cost_matrix(self.previous_frame_matrix, self.current_frame_matrix) ## generate cost matrix between previous and current frame
        self.ids = self._generate_matching_ids(self.cost) ## generate aligned IDs
        self.current_frame_aligned_matrix = np.array([self.current_frame_matrix[ii] for ii in self.ids]) ## re-order current frame matrix to match ID matrix
        self.conf = np.array([conf[ii] for ii in self.ids]) ## re-order current conf matrix to match ID matrix
        return self.current_frame_aligned_matrix, self.conf
