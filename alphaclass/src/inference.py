import torch

torch.backends.cudnn.benchmark = True ## to speed up constant-size inference

from torch import nn
import torchvision

import numpy as np
import os
import json
import cv2
import pickle
import h5py
import time

import loop
from models import constructor
from utils.inference_helpers import NonMaximaSuppression2d, NMS, post_peak_nms, agnostic_nms
from utils.stream_helpers import Streams, StreamViewer
from utils.image_helpers import letterbox, batch_letterbox

from argparse import ArgumentParser


def inference(configs):
    if type(configs) == str:
        configs_path = configs
        with open(configs, 'r') as f:
            configs = json.load(f)


    ## choose device based on availability
    compute = 'cuda' if torch.cuda.is_available() else 'cpu' ## use CUDA (gpu) if available, else use cpu
    ds = loop.LoadLoopInference(labeled_data_path = configs['labeled_data_path'])
    nClasses = ds.nClasses ## get total number of target classes (behaviors)


    ## model loading
    model = constructor.return_model(configs, nClasses) ## load model from constructor
    weights = torch.load(configs['weights_path'], map_location=torch.device('cpu')) ## load weights
    model.load_state_dict(weights) ## match weights with model structure
    model.to(compute).eval() ## set to eval mode for inference
    if torch.cuda.is_available():
        model = model.cuda().half().eval() ## half-precision inference if on CUDA


    ## initialize tracker
    #tracker = Tracker()


    ## starting streams ##
    cap = StreamViewer(configs['streams'], size=(configs['image_training_width'], configs['image_training_height']),
                       buffer=configs['stream_buffer'],
                       letterbox=configs['letterbox']) ## start streams, frames already resized via letterbox
    nms = NMS(configs['conf']).to(compute) ## initialize NMS class


    ## save videos and initialize
    out_saves = []
    test_frames = cap.update()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for st, sz in zip(configs['streams'], test_frames):
        stream_savepath = os.path.join(configs['run_savepath'], f'stream_{st}.mp4')
        video = cv2.VideoWriter(stream_savepath, fourcc, configs['fps'], (sz.shape[1], sz.shape[0]))
        out_saves.append(video)


    ## open savepath file
    h5savepath = os.path.join(configs['run_savepath'], 'data.h5')
    h5File = h5py.File(h5savepath, 'w')
    type32 = h5py.special_dtype(vlen=np.dtype('float32'))
    h5_dataset1 = h5File.create_dataset('points', shape=(1,4,), maxshape=(None, 4,), dtype=type32)
    h5_dataset2 = h5File.create_dataset('confs', shape=(1,1), maxshape=(None,1), dtype=type32)
    h5_dataset3 = h5File.create_dataset('time', shape=(1,1), maxshape=(None,1), dtype='f')


    ## main inference loop
    inference_counter = 0
    peaks_out = []
    confs_out = []
    while 1:
        frames = cap.update() ## read frame
        for i_count, i in enumerate(frames):
            out_saves[i_count].write(i) ## write to video
            cv2.imshow(f'{i_count}', i) ## display frames from all streams
            k = cv2.waitKey(1) & 0xFF ## 1 ms delay when displaying frame


        ## start timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time1 = time.time() ## clock in


        ## preprocessing
        frames = np.array(frames) ## convert frames from list to numpy
        frames = np.transpose(frames, (0, 3, 1, 2)) ## (B, H, W, C) to (B, C, H, W) format
        frames = torch.from_numpy(frames).to(compute)/255. ## normalze from 0 to 1
        if torch.cuda.is_available():
            frames = frames.cuda().half() ## half-precision inference if on CUDA


        ## prediction
        with torch.no_grad():
            pred = model(frames) ## main prediction step, generates heatmap output: 'pred'
            nms_bool, nms_result, nms_confidences = nms(pred) ## peak detection

            nms_bool = nms_bool.cpu().float() ## to cpu and float, update to CUDA eventually
            nms_result = nms_result.cpu().float() ## to cpu and float, update to CUDA eventually
            nms_confidences = nms_confidences.cpu().float() ## to cpu and float, update to CUDA eventually

            if len(nms_result) > 0:
                processed_peaks, processed_confs = post_peak_nms(nms_result.float(), nms_confidences,
                                                                 buffer=configs['box_buffer'], thresh=configs['box_thresh']) ## if peaks, then suppress low-confidence peaks

                agnostic_peaks, agnostic_confs = agnostic_nms(processed_peaks, processed_confs,
                                                              buffer=configs['box_buffer'], thresh=configs['agnostic_thresh']) ## if multiple classes/behavior for same peak, suppress low-confidence predictions

                agnostic_peaks = agnostic_peaks.cpu().numpy().tolist()
                agnostic_confs = agnostic_confs.cpu().numpy().tolist()
            else:
                ## if no peaks found, send empty list
                agnostic_peaks = []
                agnostic_confs = []


            ## identity tracking
            #agnostic_peaks, agnostic_confs = tracker.step(agnostic_peaks, agnostic_confs) ## use cdist (cost) and hungarian matching (linear assignment) to track IDs


            ## end timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time2 = time.time() ## clock out


            #### don't forget, the peaks are in the following format: (image_index, channel/behavior, y_coordinate, x_coordinate)
            print(f'{(time2-time1): .6f} s, {i.shape} shape,', f'behavior: {agnostic_peaks}')


            ## save results to file
            h5_points_entry = np.array(agnostic_peaks)
            h5_confs_entry = np.array(agnostic_confs)

            h5_dataset1.resize(inference_counter+1, 0) ## resize dataset during loop
            h5_dataset2.resize(inference_counter+1, 0) ## resize dataset during loop
            h5_dataset3.resize(inference_counter+1, 0) ## resize dataset during loop

            h5_dataset1[inference_counter] = h5_points_entry.T.copy() ## append point locations
            h5_dataset2[inference_counter] = h5_confs_entry.copy() ## append confidence values
            h5_dataset3[inference_counter] = (time2-time1) ## append inference times


        ## if ESC, then break the loop, stop recording, and save files
        if k==27:
            for vid_stream in out_saves:
                vid_stream.release() ## close video stream once ESC key pressed
            #rf.close()
            h5File.close() ## close and save data file once ESC key pressed
            break


        ## increment counter
        inference_counter += 1


    cap.stop() ## close cameras once loop is broken
    print('finished')
    configs_run_savepath = configs['run_savepath']
    print(f'video and results saved at {configs_run_savepath}')
    print(f'results saved at {h5savepath}')






## command line interface
if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--options', type=str, default='', help='experiment options json path')
    parse.add_argument('--streams', nargs='+', default=[], help='list of streams')
    parse.add_argument('--stream_buffer', action='store_true', help='buffer streams to avoid dropped frames')
    parse.add_argument('--weights_path', type=str, default='', help='optional path for weights')
    parse.add_argument('--conf', type=float, default=0.2, help='nms confidence threshold')
    parse.add_argument('--box_buffer', type=int, default=15, help='box width/height for post-peak-nms')
    parse.add_argument('--box_thresh', type=float, default=0.6, help='nms threshold for filtering out overlapping peaks')
    parse.add_argument('--agnostic_thresh', type=float, default=0.7, help='agnostic nms iou threshold')
    parse.add_argument('--fps', type=int, default=30, help='target FPS to save recorded videos')
    parse.add_argument('--letterbox', type=str, default='yes', help='letterbox during inference')


    options = parse.parse_args()
    with open(options.options, 'r') as f:
        c = json.load(f)

    options.streams = [int(i) for i in options.streams]
    split = options.options.split(os.sep)[:-1]
    c['exp_path'] = os.path.join(*split)

    ## increment path
    i = 0
    while os.path.exists(os.path.join(c['exp_path'], 'inference_runs', 'run%02d' % i)):
        i+=1
    c['run_savepath'] = os.path.join(c['exp_path'], 'inference_runs', f'run{i:02d}')
    os.makedirs(c['run_savepath'])

    c['streams'] = options.streams
    c['weights_path'] = options.weights_path
    c['stream_buffer'] = True if options.stream_buffer else False
    c['conf'] = options.conf
    c['box_buffer'] = options.box_buffer
    c['box_thresh'] = options.box_thresh
    c['agnostic_thresh'] = options.agnostic_thresh
    c['fps'] = options.fps
    c['letterbox'] = options.letterbox

    with open(os.path.join(c['run_savepath'], 'inference_options.json'), 'w') as f:
        json.dump(c, f, indent=4)

    inference(c)
