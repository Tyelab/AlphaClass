import h5py
import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from time import sleep 
from argparse import ArgumentParser

def gen_colors():
    return np.random.randint(0, 255)


def plot(configs):
    colors = [(gen_colors(), gen_colors(), gen_colors()) for i in range(50)]

    options_path = configs.options
    root = os.path.split(options_path)[0]
    data_path = os.path.join(root, 'data.h5')
    video_path = configs.streams

    ## load data
    data = h5py.File(data_path, 'r')
    points = data['points']
    confs = data['confs']

    ## get sample frame shape
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    height, width, _ = frame.shape
    cap.release()

    ## load options to get scaling factors
    with open(options_path, 'r') as f:
        options = json.load(f)
    training_height, training_width = options['image_training_height'], options['image_training_width']
    x_factor = training_width / width
    y_factor = training_height / height

    ## load label_idx
    with open(os.path.join(root, 'label_idx.json'), 'r') as f:
        label_idx = json.load(f)
    mapper = {v: k for k, v in label_idx.items()}

    ## begin main loop
    cap = cv2.VideoCapture(video_path)
    radius = configs.radius
    height = configs.height
    width = configs.width
    #size = (width, height)
    size = (training_width, training_height)

    ## generate video saver
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #stream_savepath = os.path.join(configs['run_savepath'], f'stream_{st}.mp4')
    vid_i = 0
    while os.path.exists(os.path.join(root, 'plotted_video%s.mp4' % vid_i)):
        vid_i+=1
    stream_savepath = os.path.join(root, f'plotted_video{vid_i}.mp4')
    video_save = cv2.VideoWriter(stream_savepath, fourcc, 30, size)


    ## main plottin loop
    for index in tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            continue

        ## main plotting code
        point = np.stack(points[index]).T
        if point.shape[0] > 0:
            behaviors = []
            for p in point:
                b_ind = int(p[1])
                behavior = mapper[p[1]]
                x, y = p[3], p[2]
                x_, y_ = int(x*4), int(y*4)
                #x_, y_ = int(x), int(y)
                #print("This is x:", x_)
                #print("This is y:",y_)
                #print(behavior)
                cv2.circle(frame, (x_, y_), radius, colors[b_ind], -1)   

                
        else:
            behavior = 'none'


        frame = cv2.resize(frame, size)
        video_save.write(frame)
        if configs.show:
            cv2.imshow('frame', frame)
            cv2.waitKey(configs.delay)

    cap.release()
    video_save.release()
    cv2.destroyAllWindows()




## command line interface
if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--options', type=str, default='', help='experiment options json path')
    parse.add_argument('--streams', type=str, default='', help='video path')
    parse.add_argument('--radius', type=int, default=5, help='radius of circle')
    parse.add_argument('--width', type=int, default=640, help='output video width')
    parse.add_argument('--height', type=int, default=480, help='output video height')
    parse.add_argument('--show', action='store_true', help='show frames as they are being processed')
    parse.add_argument('--delay', type=int, default=30, help='millisecond delay if showing images')


    configs = parse.parse_args()
    plot(configs)
