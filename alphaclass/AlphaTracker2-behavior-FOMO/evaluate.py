import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import os
import copy
from tqdm import tqdm
import cv2
from argparse import ArgumentParser

import loop
from models import constructor
from utils.inference_helpers import NonMaximaSuppression2d, NMS
from utils.inference_helpers import coord_to_box, box_to_coord, post_peak_nms, agnostic_nms
from utils.lib.BoundingBox import BoundingBox
from utils.lib.BoundingBoxes import BoundingBoxes
from utils.lib.utils import CoordinatesType, BBType, BBFormat
from utils.lib.getbox import getBoundingBoxes
from utils.lib.Evaluator import Evaluator


def custom_collate_fn(batch):
    o_image = [b['output_image'] for b in batch]
    im_name = [b['name'] for b in batch]
    batch = [b['xybc'] for b in batch]

    collate = []
    for o_count, o in enumerate(batch):
        o = np.array(o)

        x = o[0].tolist()
        y = o[1].tolist()
        bc = o[2].tolist()

        for x_, y_, bc_ in zip(x, y, bc):
            if x_:
                collate.append([o_count, bc_, y_, x_])

    o_image = torch.tensor(o_image)
    collate_tensor = torch.tensor(collate)
    return o_image, collate_tensor, im_name


def evaluate(configs):
    if type(configs) == str:
        configs_path = configs
        with open(configs, 'r') as f:
            configs = json.load(f)


    compute = 'cuda' if torch.cuda.is_available() else 'cpu'

    im_training_resW = configs['image_training_width']
    im_training_resH = configs['image_training_height']

    if configs['model_type'] == 'mobilenet':
        hm_width = int(im_training_resW/4)
        hm_height = int(im_training_resH/4)
    elif configs['model_type'] == 'unet_double':
        hm_width = im_training_resW
        hm_height = im_training_resH
    elif configs['model_type'] == 'unet_single':
        hm_width = im_training_resW
        hm_height = im_training_resH
    elif configs['model_type'] == 'shufflenet':
        hm_width = int(im_training_resW/4)
        hm_height = int(im_training_resH/4)
    elif configs['model_type'] == 'resnet':
        hm_width = int(im_training_resW/4)
        hm_height = int(im_training_resH/4)

    ds = loop.LoadLoopInference(labeled_data_path = configs['labeled_data_path'],
                                image_training_resW = im_training_resW,
                                image_training_resH = im_training_resH,
                                hm_training_resW = hm_width,
                                hm_training_resH = hm_height,
                                augmentation_pipeline = True, train = True)

    random_seed  = configs['random_seed']
    train_len = int(len(ds)*configs['train_test_split'])
    ranks = [train_len, int(len(ds)-train_len)]

    train_ds, test_ds = torch.utils.data.random_split(ds, ranks, generator=torch.Generator().manual_seed(random_seed))

    train_ds_copy = copy.deepcopy(train_ds)
    test_ds_copy = copy.deepcopy(test_ds)

    train_ds_copy.dataset.transform = train_ds_copy.dataset.both_augs['train']
    test_ds_copy.dataset.transform = ''


    test_dl = DataLoader(test_ds_copy, batch_size=1, num_workers=0,
                         shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)



    model = constructor.return_model(configs, ds.nClasses)
    weights = torch.load(configs['weights_path'], map_location=torch.device('cpu'))
    print(ds.nClasses, configs['weights_path'])
    model.load_state_dict(weights)
    model.to(compute).eval()
    if torch.cuda.is_available():
        model = model.half()


    #test_dl = tqdm(test_dl)
    nms = NMS(configs['conf']).to(compute)
    rz_factor = 4 if configs['model_type'] != 'unet' else 1
    mapper = {v: k for k, v in ds.label_idx.items()}

    test_dl_pbar = tqdm(test_dl)
    for i_count, (output_image, collate_tensor, image_name) in enumerate(test_dl_pbar):
        output_image = output_image.to(compute)/255.
        if torch.cuda.is_available():
            output_image = output_image.half()

        with torch.no_grad():
            pred = model(output_image)
            nms_bool, nms_result, nms_confidences = nms(pred)
            nms_result = nms_result.cpu()
            nms_confidences = nms_confidences.cpu()

            if len(nms_result) > 0:
                nms_result, nms_confidences = post_peak_nms(nms_result.float(), nms_confidences,
                                                            buffer=configs['box_buffer'], thresh=configs['iou_thresh'])
                nms_result, nms_confidences = agnostic_nms(nms_result, nms_confidences,
                                                           buffer=configs['box_buffer'], thresh=configs['agnostic_thresh'])

            else:
                nms_result = []
                nms_confidences = []


            ## write preds to file
            boxes = []
            if len(nms_result) > 0:

                nms_result = coord_to_box(nms_result, configs['box_buffer'])
                for box_c, b in enumerate(nms_result):
                    b_x1 = b[3].item()*rz_factor
                    b_y1 = b[2].item()*rz_factor
                    b_x2 = b[5].item()*rz_factor
                    b_y2 = b[4].item()*rz_factor
                    cls = b[1].item()
                    entry = [str(mapper[int(cls)]), nms_confidences[box_c].item(), b_x1, b_y1, b_x2, b_y2]
                    boxes.append(entry)

                fsp = os.path.join(configs['run_savepath'], 'preds', f'{image_name[0]}.txt')
                with open(fsp, 'w') as f:
                    for item in boxes:
                        for t in item:
                            f.write("%s " % str(t))
                        f.write("\n")
                f.close()

            else:
                fsp = os.path.join(configs['run_savepath'], 'preds', f'{image_name[0]}.txt')
                with open(fsp, 'w') as f:
                    f.close()



            ## write ground truth to file
            boxes = []
            if len(collate_tensor) > 0:

                collate_tensor[:,2:] = collate_tensor[:,2:]/rz_factor
                collate_tensor = coord_to_box(collate_tensor.float(), configs['box_buffer'])
                for box_c, b in enumerate(collate_tensor):
                    b_x1 = b[3].item()*rz_factor
                    b_y1 = b[2].item()*rz_factor
                    b_x2 = b[5].item()*rz_factor
                    b_y2 = b[4].item()*rz_factor
                    cls = b[1].item()
                    entry = [str(mapper[int(cls)]), b_x1, b_y1, b_x2, b_y2]
                    #entry = [str(int(cls)), nms_confidences[box_c].item(), b_x1, b_y1, b_x2, b_y2]
                    boxes.append(entry)

                fsp = os.path.join(configs['run_savepath'], 'gt', f'{image_name[0]}.txt')
                with open(fsp, 'w') as f:
                    for item in boxes:
                        for t in item:
                            f.write("%s " % str(t))
                        f.write("\n")
                f.close()

            else:
                fsp = os.path.join(configs['run_savepath'], 'gt', f'{image_name[0]}.txt')
                with open(fsp, 'w') as f:
                    f.close()



            #collate_tensor[:,2:] = collate_tensor[:,2:]/rz_factor


        #print(collate_tensor, output_image.shape)
        #print()
        #print()
    test_dl_pbar.close()


    ## calculate metrics
    gt_path = os.path.join(configs['run_savepath'], 'gt')
    pred_path = os.path.join(configs['run_savepath'], 'preds')
    boundingBoxes = getBoundingBoxes(gt_path, pred_path)
    evaluator = Evaluator()
    mpc = evaluator.GetPascalVOCMetrics(boundingBoxes, IOUThreshold=0.5)
    print(mpc)
    print()
    print('finished')






if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--options', type=str, default='', help='experiment options json path')
    parse.add_argument('--weights_path', type=str, default='', help='optional path for weights')
    parse.add_argument('--conf', type=float, default=0.2, help='nms confidence threshold value')
    parse.add_argument('--box_buffer', type=int, default=15, help='box width/height for post-peak-nms')
    parse.add_argument('--iou_thresh', type=float, default=0.7, help='nms iou threshold value')
    parse.add_argument('--agnostic_thresh', type=float, default=0.7, help='agnostic iou threshold value')

    #parse.add_argument('--eval_box_buffer', type=int, default=6, help='buffer for bounding box coords for evaluation')
    #parse.add_argument('--dist', type=float, default=10, help='minimum distance threshold between peaks')


    options = parse.parse_args()
    with open(options.options, 'r') as f:
        c = json.load(f)


    split = options.options.split(os.sep)[:-1]
    c['exp_path'] = os.path.join(*split)


    ## increment path
    i = 0
    while os.path.exists(os.path.join(c['exp_path'], 'evaluation_runs', 'run%s' % i)):
        i+=1
    c['run_savepath'] = os.path.join(c['exp_path'], 'evaluation_runs', f'run{i}')
    os.makedirs(c['run_savepath'])
    os.makedirs(os.path.join(c['run_savepath'], 'preds'))
    os.makedirs(os.path.join(c['run_savepath'], 'gt'))


    ## write configurations to dictionary
    c['weights_path'] = options.weights_path
    c['conf'] = options.conf
    c['box_buffer'] = options.box_buffer
    c['iou_thresh'] = options.iou_thresh
    c['agnostic_thresh'] = options.agnostic_thresh

    #c['eval_box_buffer'] = options.eval_box_buffer
    #c['dist'] = options.dist


    ## write dictionary to json file
    with open(os.path.join(c['run_savepath'], 'eval_options.json'), 'w') as f:
        json.dump(c, f, indent=4)


    ## run eval
    evaluate(c)
