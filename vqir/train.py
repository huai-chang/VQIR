import os.path as osp
import math
import argparse
import random
import logging
import torch, torchvision
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import build_model
import os
import numpy as np
from tqdm import tqdm

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='vqir/options/train/train_vqir_stage1.yml',
                        help='Path to option YAML file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # load resume states if necessary
    resume_state = util.load_resume_state(opt)

    # train from scratch OR resume training
    if resume_state is None:
        util.make_exp_dirs(opt)
    
    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
    
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['logger']['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=opt['path']['log'] + '/tb_logger/' + opt['name'])
    else:
        tb_logger = None
    
    # print('')

    # random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    model = build_model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 1

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train'].get('warmup_iter', -1))   

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            
            # validation
            if opt.get('val') is not None and (current_step % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_step, tb_logger, opt['val']['save_img'])

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_step)

    logger.info('Saving the final model.')
    model.save(epoch, current_iter='latest')
    logger.info('End of training.')
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    main()


    


