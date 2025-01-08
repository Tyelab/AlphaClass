import torchvision
import torch

torch.backends.cudnn.benchmark = True # use only if images are of same size

import numpy as np
import os
import json
import cv2
from argparse import ArgumentParser
import copy
import time
from tqdm import tqdm
import albumentations as A

#from data import loop
import loop
from models import constructor


from torch import nn
import torchvision
from models.layers_helper import DUC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class ReusableLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def train(configs):

    ## config loading
    if type(configs) == str:
        with open(configs, 'r') as f:
            configs = json.load(f)


    ## device assignment
    compute = 'cuda' if torch.cuda.is_available() else 'cpu'


    ## data loading
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
        print('scaled image size: ',hm_width,hm_height)

    ds = loop.LoadLoop(labeled_data_path = configs['labeled_data_path'],
                       image_training_resW = im_training_resW,
                       image_training_resH = im_training_resH,
                       hm_training_resW = hm_width,
                       hm_training_resH = hm_height,
                       augmentation_pipeline = True, train = True)
    #print(ds)              

    train_len = int(len(ds)*configs['train_test_split'])
    ranks = [train_len, int(len(ds)-train_len)]

    random_seed = configs['random_seed']
    train_ds, test_ds = torch.utils.data.random_split(ds, ranks, generator=torch.Generator().manual_seed(random_seed))

    train_ds_copy = copy.deepcopy(train_ds)
    test_ds_copy = copy.deepcopy(test_ds)

    train_ds_copy.dataset.transform = train_ds_copy.dataset.both_augs['train']
    test_ds_copy.dataset.transform = ''

    print(f'training with following augmentations: {train_ds_copy.dataset.transform, test_ds_copy.dataset.transform}')
    print()
    print(f'{ds.nClasses} target labels')

    ## save augmentation to file
    A.save(train_ds_copy.dataset.transform, os.path.join(configs['exp_path'], 'augmentation.json'))

    #if configs['safe_minibatches']:
    #    if configs['batch_size'] >= 32:
    #        batch_size=16
    #    else:
    #        batch_size = configs['batch_size']

    #else:
    #    batch_size=configs['batch_size']
    batch_size = configs['batch_size']
    print(batch_size)


    train_dl = ReusableLoader(train_ds_copy, batch_size=batch_size, num_workers=configs['num_workers'],
                              shuffle=True, pin_memory=True)

    test_dl = ReusableLoader(test_ds_copy, batch_size=16, num_workers=configs['num_workers'],
                             shuffle=False, pin_memory=True)


    ## model construction
    print(f'using {compute}')
    model = constructor.return_model(configs, ds.nClasses)
    model = model.to(compute)
    if configs['pretrained']:
        model_weights = model.state_dict()
        pretrained_weights = torch.load(configs['pretrained'], map_location='cpu')

        weights_changed = {}
        for w in pretrained_weights:
            if 'conv_out.weight' in w or 'conv_out.bias' in w:
                continue
            weights_changed[w] = pretrained_weights[w]
        model_weights.update(weights_changed)
        model.load_state_dict(model_weights)
        print()
        print('using pretrained model from {}'.format(configs['pretrained']))
        print()

    model = torch.nn.DataParallel(model) ## for future
    test_output = model(torch.randn(1, 3, configs['image_training_height'], configs['image_training_width']).to(compute))
    print('model shapes successful')
    print(f'using {count_parameters(model)} parameters')


    ## optimizer initializer from YOLOv5: don't decay certain types
    if configs['cycler']:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay


        if configs['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(pg0, lr=configs['learning_rate'], momentum=0.9, nesterov=True)
            optimizer.add_param_group({'params': pg1, 'weight_decay': configs['weight_decay']})
            optimizer.add_param_group({'params': pg2})
            print()
            print('using SGD with decay')
        elif configs['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(pg0, lr=configs['learning_rate'], betas=(0.9, 0.99), amsgrad=True)
            optimizer.add_param_group({'params': pg1, 'weight_decay': configs['weight_decay']})
            optimizer.add_param_group({'params': pg2})
            print()
            print('using Adam with decay')
            print('weight decay: {}'.format(configs['weight_decay']))



    else:
        if configs['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'], momentum=0.9, nesterov=True)
            print()
            print('using SGD without decay')

        elif configs['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], betas=(0.9, 0.99), amsgrad=True)
            print()
            print('using Adam without decay')


    ## MSE loss
    criterion = torch.nn.MSELoss().to(compute)


    ## lr scheduler
    if configs['cycler'] == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr = configs['maximum_lr'],
                                                        steps_per_epoch = len(train_dl),
                                                        epochs = configs['epochs'],
                                                        pct_start = configs['percent_lr_up'])
        print('using one cycle learning rate')

    elif configs['cycler'] == 'warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0 = 2,
                                                                         T_mult = 2,
                                                                         eta_min = 1e-7)
        print('using warm restart learning rate')

    elif configs['cycler'] == 'cyclic':
        num_batches = len(train_dl)
        num_peaks = 20
        up_iters = int(configs['epochs'] / (num_peaks*10))*num_batches
        down_iters = int(configs['epochs'] / num_peaks)*num_batches
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=1e-6,
                                                      max_lr=0.05,
                                                      step_size_up=up_iters,
                                                      step_size_down=down_iters,
                                                      mode="triangular2")
        print('using cyclic learning rate')


    elif configs['cycler'] == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.5,
                                                               patience=5,
                                                               threshold=1e-6, # check this again
                                                               threshold_mode='abs',
                                                               cooldown=3)
        print('using lr reduction on plateau learning rate')

    ## training loop
    best_index = []
    latest_lr = []
    #print('configs: ',configs)
    #print('')
    for e in range(configs['epochs']):

        #### training
        model.train()
        optimizer.zero_grad()

        train_dl_pbar = tqdm(train_dl)
        train_accumulated_loss = 0
        for batch_count, (image, label) in enumerate(train_dl_pbar):

            ## put variables on compute device
            image = image.to(compute, non_blocking=True).float() / 255.0 ## color range 
            label = label.to(compute, non_blocking=True).float()
            
            ## make predictions
            #print('')
            #print('raw image=',image.size())
            #print(model)
            prediction = model(image)
            #print('prediction (image run through model)=',prediction.size())
            #print('label=',label.size())
            
            loss = criterion(prediction, label)
            #print(loss)
            train_accumulated_loss += loss.item()


            ## step backwards
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if configs['cycler'] == 'one_cycle' or configs['cycler'] == 'warm_restarts' or configs['cycler'] == 'cyclic':
                scheduler.step()


            ## loggers
            if configs['cycler'] == 'reduce':
                latest_lr.append(optimizer.param_groups[0]['lr'])
            elif configs['cycler'] and configs['cycler'] != 'reduce':
                latest_lr.append(scheduler.get_last_lr())
            else:
                latest_lr.append(configs['learning_rate'])


        #### testing
        model.eval()

        test_dl_pbar = tqdm(test_dl)
        test_loss_accumulated = 0
        for test_batch, (test_image, test_label) in enumerate(test_dl_pbar):
            test_image = test_image.to(compute, non_blocking=True).float() / 255.0
            test_label = test_label.to(compute, non_blocking=True).float()

            with torch.no_grad():
                test_pred = model(test_image)
                test_loss_accumulated += criterion(test_pred, test_label).item()# / len(test_dl)

        show = test_pred.cpu().detach().numpy()
        for shc, sh in enumerate(show):
            cv2.imshow(f'{shc}', sh.sum(axis=0))
        cv2.waitKey(1)
        train_dl_pbar.close()
        test_dl_pbar.close()


        ## save and log results
        averaged_train_loss = train_accumulated_loss / len(train_dl)
        averaged_test_loss = test_loss_accumulated / len(test_dl)

        if configs['cycler'] == 'reduce':
            scheduler.step(averaged_test_loss)


        if configs['cycler'] == 'reduce':
            recent_lr = [optimizer.param_groups[0]['lr']]
        elif configs['cycler'] and configs['cycler'] != 'reduce':
            recent_lr = scheduler.get_last_lr()
        else:
            recent_lr = [configs['learning_rate'], configs['learning_rate'], configs['learning_rate']]

        results = (e, averaged_train_loss, averaged_test_loss,
                   recent_lr[0], )#recent_lr[1], recent_lr[2])

        results = ' '.join(tuple([str(f) for f in results]))
        results = results + '\n'

        with open(os.path.join(configs['exp_path'], 'metrics.txt'), 'a') as f:
            #f.write('%s\n' % results)
            f.write(results)


        best_index.append(averaged_test_loss)
        if np.argmin(best_index)+1 == len(best_index):
            model_best = model.module
            torch.save(copy.deepcopy(model_best.state_dict()), os.path.join(configs['exp_path'], '{}.best.pt'.format(configs['model_type'])))


        model_last = model.module
        torch.save(copy.deepcopy(model_last.state_dict()), os.path.join(configs['exp_path'], '{}.last.pt'.format(configs['model_type'])))


        print('epoch: {}'.format(e), 'train_loss:{0:.6f}, test_loss:{1:.6f}'.format(averaged_train_loss, averaged_test_loss),
              'lr:{0:.8f}'.format(recent_lr[0]), 'max_conf:{0:.6f}'.format(test_pred.max().item()))
        print()



if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--options', type=str, default='', help='path to options file')
    parse.add_argument('--labeled_data_path', type=str, default='', help='path to labeled data')
    parse.add_argument('--model_type', type=str, default='', help='model type')
    parse.add_argument('--model_weight', type=str, default='', help='model weights')
    parse.add_argument('--image_training_width', type=int, help='image width during training')
    parse.add_argument('--image_training_height', type=int, help='image height during training')
    parse.add_argument('--heatmap_width', type=int, help='label width')
    parse.add_argument('--heatmap_height', type=int, help='label height')
    parse.add_argument('--train_test_split', type=float, help='train test split during training')
    parse.add_argument('--batch_size', type=int, help='batch size during training')
    parse.add_argument('--epochs', type=int, help='epochs during training')
    parse.add_argument('--learning_rate', type=float, help='lr during training')
    parse.add_argument('--safe_minibatches', action='store_true', help='ceiling of max batch size')
    parse.add_argument('--optimizer', type=str, default='sgd', help='optimizer during training')
    parse.add_argument('--cycler', type=str, default='one_cycle', help='lr rate scheduler during training')
    parse.add_argument('--percent_lr_up', type=float, help='% of lr going up if cycler==one_cycle')
    parse.add_argument('--maximum_lr', type=float, help='highest lr if one_cycle')
    parse.add_argument('--num_workers', type=int, help='num workers to use during training')
    parse.add_argument('--persistent_workers', action='store_true', help='reuse workers, recommended True for Windows')

    opt = parse.parse_args()

    with open(opt.options, 'r') as f:
        options = json.load(f)

    ## increment save path
    i = 0
    while os.path.exists(os.path.join('..', 'Results', 'run%s' % i)):
        i+=1
    options['exp_path'] = os.path.join('..', 'Results', 'run{}'.format(i))
    os.makedirs(options['exp_path'])

    ## save all configs to file
    with open(os.path.join(options['exp_path'], 'options.json'), 'w') as f:
        json.dump(options, f, indent=4)


    ## start training
    print('#### starting training ####')
    print()
    t1 = time.time()
    train(options)
    t_finished = time.time()
    print('seconds elapsed: {}'.format((t_finished-t1)))
