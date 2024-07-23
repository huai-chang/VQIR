import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from utils import util
import logging
from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel

logger = logging.getLogger('base')

@MODEL_REGISTRY.register()
class VQIRModel(BaseModel):
    def __init__(self, opt):
        super(VQIRModel, self).__init__(opt)
        
        self.net_ir = build_network(opt['network_ir'])
        self.net_ir = self.model_to_device(self.net_ir)
        self.print_network(self.net_ir)
        
        self.load()

        if self.is_train:
            self.init_training_settings()

    def load(self):
        load_path = self.opt['path'].get('pretrain_network_ir', None)
        if load_path is not None:
            logger.info(f'Loading net_ir from {load_path}')
            self.load_network(self.net_ir, load_path, self.opt['path'].get('strict_load_ir', True))

    def setup_optimizers(self):
        train_opt = self.opt['train']
        encoder_params = []
        for k, v in self.net_ir.named_parameters():
            if 'icrm' in k:
                encoder_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_ir_enc'].pop('type')
        self.optimizer_ir = self.get_optimizer(optim_type, encoder_params, **train_opt['optim_ir_enc'])
        self.optimizers.append(self.optimizer_ir)
    
    def init_training_settings(self):
        self.net_ir.train()
        train_opt = self.opt['train']

        if train_opt.get('guide_opt'):
            self.cri_guide = build_loss(train_opt['guide_opt']).to(self.device)
        else:
            self.cri_guide = None

        if train_opt.get('feature_opt'):
            self.cri_quant_feature = build_loss(train_opt['feature_opt']).to(self.device)
        else:
            self.cri_quant_feature = None

        if train_opt.get('gram_opt'):
            self.cri_quant_gram = build_loss(train_opt['gram_opt']).to(self.device)
        else:
            self.cri_quant_gram = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.hr = data['HR'].to(self.device)
        self.lr = data['LR'].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_ir.zero_grad()

        sr_res = self.net_ir(self.hr, is_train=True)

        l_total_g = 0.0

        if self.cri_guide:
            l_guide = self.cri_guide(sr_res['LR'], self.lr)
            loss_dict['l_guide'] = l_guide.detach().mean()
            l_total_g += l_guide

        if self.cri_quant_feature:
            l_quant_feature = self.cri_quant_feature(sr_res['re_quant_feat'], sr_res['quant_feat'])
            loss_dict['l_feat'] = l_quant_feature.detach().mean()
            l_total_g += l_quant_feature

        if self.cri_quant_gram:
            l_quant_gram = self.cri_quant_gram(sr_res['re_quant_feat'], sr_res['quant_feat'])
            loss_dict['l_gram'] = l_quant_gram.detach().mean()
            l_total_g += l_quant_gram

        loss_dict['l_total_g'] = l_total_g.detach().mean()
        l_total_g.backward()
        self.optimizer_ir.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self, input=None):
        self.net_ir.eval()
        with torch.no_grad():
            if input is None:
                input = self.hr
            
            self.output = self.net_ir(x_hr=input, is_train=False)
            self.sr = self.output['SR']
            self.dr = self.output['LR']
        self.net_ir.train()

    def test_tile(self, tile_num=[4,4] ,tile_pad=32):
        self.net_ir.eval()
        with torch.no_grad():
            SR_output_shape = self.hr.shape
            batch, channel, height, width = SR_output_shape
            scale = self.opt['ir_scale']

            DR_output_height = height // scale
            DR_output_width = width // scale
            DR_output_shape = (batch, channel, DR_output_height, DR_output_width)

            # start with black image
            SR_output = self.hr.new_zeros(SR_output_shape)
            DR_output = self.hr.new_zeros(DR_output_shape)
            length_x, ofs_x = util.split_image(width / scale, tile_num[0])
            length_y, ofs_y = util.split_image(height / scale, tile_num[1])

            # loop over all tiles
            for y in range(len(ofs_y)):
                for x in range(len(ofs_x)):
                    # input tile area on total image
                    start_x = ofs_x[x] * scale
                    end_x = (ofs_x[x]+length_x[x]) * scale
                    start_y = ofs_y[y] * scale
                    end_y = (ofs_y[y]+length_y[y]) * scale

                    # input tile area on total image with padding
                    start_x_pad = max(start_x - tile_pad, 0)
                    end_x_pad = min(end_x + tile_pad, width)
                    start_y_pad = max(start_y - tile_pad, 0)
                    end_y_pad = min(end_y + tile_pad, height)

                    # input tile dimensions
                    tile_width = end_x - start_x
                    tile_height = end_y - start_y
                    input_tile = self.hr[:, :, start_y_pad:end_y_pad, start_x_pad:end_x_pad]

                    # output tile area without padding
                    start_x_tile = (start_x - start_x_pad)
                    end_x_tile = start_x_tile + tile_width
                    start_y_tile = (start_y - start_y_pad)
                    end_y_tile = start_y_tile + tile_height

                    self.test(input_tile)
                    SR_output[:, :, start_y:end_y, start_x:end_x] = self.sr[:, :, start_y_tile:end_y_tile, start_x_tile:end_x_tile]
                    DR_output[:, :, start_y//scale:end_y//scale, start_x//scale:end_x//scale] = self.dr[:, :, start_y_tile//scale:end_y_tile//scale,
                                                                  start_x_tile//scale:end_x_tile//scale]
            self.sr = SR_output
            self.dr = DR_output
        self.net_ir.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['HR'] = self.hr.detach().cpu()
        out_dict['SR'] = self.sr.detach().cpu()
        out_dict['LR'] = self.lr.detach().cpu()
        out_dict['DR'] = self.dr.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_ir, 'net_ir', current_iter)
        if hasattr(self, 'net_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        name_lpips = None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        hr_path = dataloader.dataset.opt['dataroot_HR']

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['LR_path'][0]))[0]  # LR_Path

            self.feed_data(val_data)
            _, _, width, height = self.hr.shape
            if height*width < 2048*1024:
                self.test()
            else:
                self.test_tile([2,1],128)

            visuals = self.get_current_visuals()
            range = self.opt['val'].get('range', [-1, 1])

            sr_img = util.tensor2img(visuals['SR'], min_max=range)  # uint8
            hr_img = util.tensor2img(visuals['HR'], min_max=range)  # uint8
            dr_img = util.tensor2img(visuals['DR'], min_max=range)  # uint8
            lr_img = util.tensor2img(visuals['LR'], min_max=range)  # uint8

            # tentative for out of GPU memory
            del self.sr
            del self.hr
            del self.dr
            del self.lr
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_hr_path = osp.join(self.opt['path']['val_images'], f'{current_iter}', 'HR' ,f'{img_name}.png')
                    save_sr_path = osp.join(self.opt['path']['val_images'], f'{current_iter}', 'SR' ,f'{img_name}.png')
                    save_lr_path = osp.join(self.opt['path']['val_images'], f'{current_iter}', 'LR' ,f'{img_name}.png')
                    save_dr_path = osp.join(self.opt['path']['val_images'], f'{current_iter}', 'DR' ,f'{img_name}.png')

                    util.imwrite(hr_img, save_hr_path)
                    util.imwrite(sr_img, save_sr_path)
                    util.imwrite(lr_img, save_lr_path)
                    util.imwrite(dr_img, save_dr_path)
                else:
                    if self.opt['val']['suffix']:
                        save_sr_path = osp.join(self.opt['path']['results_root'], dataset_name, 'SR', f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_dr_path = osp.join(self.opt['path']['results_root'], dataset_name, 'DR', f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_sr_path = osp.join(self.opt['path']['results_root'], dataset_name, 'SR', f'{img_name}.png')
                        save_dr_path = osp.join(self.opt['path']['results_root'], dataset_name, 'DR', f'{img_name}.png')
                    util.imwrite(sr_img, save_sr_path)
                    util.imwrite(dr_img, save_dr_path)
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in name:
                        name_lpips = name
                        opt_lpips  = opt_
                        continue
                    elif 'dr' in name:
                        metric_data = dict(img=dr_img, img2=lr_img)
                    else:
                        metric_data = dict(img=sr_img, img2=hr_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if name_lpips:
                self.metric_results[name_lpips] = calculate_metric(dict(folder_gt=hr_path, folder_restored=osp.split(save_sr_path)[0]), opt_lpips)

            log_str = f'# Validation {dataset_name} ||'
            for metric, value in self.metric_results.items():
                log_str += f' {metric}: {value:.4f} ||'
            if self.opt['is_train']:
                logger_val = logging.getLogger('val')
            else: 
                logger_val = logging.getLogger('base')
            logger_val.info(log_str)
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
            