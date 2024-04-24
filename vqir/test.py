import logging
import torch
import os
import argparse
from os import path as osp
import utils.util as util

from models import build_model
import options.options as option
from data import create_dataset, create_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='vqir/options/test/test_vqir_stage2.yml', help='Path to options YAML file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    util.make_exp_dirs(opt)
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_loaders = []

    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    main()