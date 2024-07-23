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
class VQIRRFModel(BaseModel):
    def __init__(self, opt):
        super(VQIRRFModel, self).__init__(opt)
        
        self.net_ir = build_network(opt['network_ir'])
        self.net_ir = self.model_to_device(self.net_ir)
        self.print_network(self.net_ir)

        # define network net_d
        if 'network_d' in self.opt:
            self.net_d = build_network(self.opt['network_d'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d)  # to avoid broadcast buffer error
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
        
        self.load()

        if self.is_train:
            self.init_training_settings()

    def load(self):
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, True)

        load_path = self.opt['path'].get('pretrain_network_ir', None)
        if load_path is not None:
            logger.info(f'Loading net_ir from {load_path}')
            self.load_network(self.net_ir, load_path, self.opt['path'].get('strict_load_ir', True))

    def setup_optimizers(self):
        train_opt = self.opt['train']
        encoder_params = []
        for k, v in self.net_ir.named_parameters():
            if 'refine' in k:
                encoder_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_ir_enc'].pop('type')
        self.optimizer_ir = self.get_optimizer(optim_type, encoder_params, **train_opt['optim_ir_enc'])
        self.optimizers.append(self.optimizer_ir)

        # optimizer d
        if 'optim_d' in train_opt:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)
    
    def init_training_settings(self):
        self.net_ir.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.hr = {}
        self.scale = self.opt['scale']
        for i in self.scale:
            self.hr['Level_%d'%i] = data['Level_%d'%i].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_ir.zero_grad()

        sr_res = self.net_ir(self.hr['Level_1'], is_train=True)

        l_total_g = 0.0

        # pixel reconstruction loss
        if self.cri_pix:
            loss_dict['l_rec'] = 0.0
            for i in self.scale:
                l_rec = self.cri_pix(sr_res['Level_%d'%i], self.hr['Level_%d'%i])
                loss_dict['l_rec'] += l_rec.detach().mean()
                l_total_g += l_rec
        
        if self.cri_edge:
            loss_dict['l_edge'] = 0.0
            for i in self.scale:
                l_edge = self.cri_edge(sr_res['Level_%d'%i], self.hr['Level_%d'%i])
                loss_dict['l_edge'] += l_edge.detach().mean()
                l_total_g += l_edge

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(sr_res['Level_1'], self.hr['Level_1'])

            if l_g_percep is not None:
                l_total_g += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep.detach().mean()
            if l_g_style is not None:
                l_total_g += l_g_style
                loss_dict['l_g_style'] = l_g_style.detach().mean()

        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            for p in self.net_d.parameters():
                p.requires_grad = False
            fake_pred = self.net_d(sr_res['Level_1'])
            l_g_gan = self.cri_gan(fake_pred, True, is_disc=False)
            loss_dict['l_g_gan'] = l_g_gan.detach().mean()
            l_total_g += l_g_gan

        loss_dict['l_total_g'] = l_total_g.detach().mean()
        l_total_g.backward()
        self.optimizer_ir.step()

        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            fake_pred = self.net_d(sr_res['Level_1'].detach())
            real_pred = self.net_d(self.hr['Level_1'])
            l_d_real = self.cri_gan(real_pred, True, is_disc=True)
            l_d_fake = self.cri_gan(fake_pred, False, is_disc=True)
            l_d = l_d_real + l_d_fake

            loss_dict['l_d'] = l_d.detach().mean()
            loss_dict['l_d_real'] = l_d_real.detach().mean()
            loss_dict['l_d_fake'] = l_d_fake.detach().mean()
            l_d.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self, input=None):
        self.net_ir.eval()
        with torch.no_grad():
            if input is None:
                input = self.hr['Level_1']
            self.output = self.net_ir(x_hr=input, is_train=False)
            self.dr = self.output['LR']
        self.net_ir.train()

    def test_tile(self, tile_num=[4,4] ,tile_pad=32):
        self.net_ir.eval()
        with torch.no_grad():
            batch, channel, height, width = self.hr['Level_1'].shape
            scale = self.opt['scale']
            ir_scale = self.opt['ir_scale']

            SR_output = {}
            for i in scale: 
                SR_output_height = height // i
                SR_output_width = width // i
                SR_output_shape = (batch, channel, SR_output_height, SR_output_width)

                # start with black image
                SR_output['Level_%d'%i] = self.hr['Level_1'].new_zeros(SR_output_shape)

            DR_output_height = height // ir_scale
            DR_output_width = width // ir_scale
            DR_output_shape = (batch, channel, DR_output_height, DR_output_width)
            DR_output = self.hr['Level_1'].new_zeros(DR_output_shape)

            length_x, ofs_x = util.split_image(width / ir_scale, tile_num[0])
            length_y, ofs_y = util.split_image(height / ir_scale, tile_num[1])

            # loop over all tiles
            for y in range(len(ofs_y)):
                for x in range(len(ofs_x)):
                    # input tile area on total image
                    start_x = ofs_x[x] * ir_scale
                    end_x = (ofs_x[x]+length_x[x]) * ir_scale
                    start_y = ofs_y[y] * ir_scale
                    end_y = (ofs_y[y]+length_y[y]) * ir_scale

                    # input tile area on total image with padding
                    start_x_pad = max(start_x - tile_pad, 0)
                    end_x_pad = min(end_x + tile_pad, width)
                    start_y_pad = max(start_y - tile_pad, 0)
                    end_y_pad = min(end_y + tile_pad, height)

                    # input tile dimensions
                    tile_width = end_x - start_x
                    tile_height = end_y - start_y
                    input_tile = self.hr['Level_1'][:, :, start_y_pad:end_y_pad, start_x_pad:end_x_pad]

                    # output tile area without padding
                    start_x_tile = (start_x - start_x_pad)
                    end_x_tile = start_x_tile + tile_width
                    start_y_tile = (start_y - start_y_pad)
                    end_y_tile = start_y_tile + tile_height

                    self.test(input_tile)
                    for i in scale:
                        SR_output['Level_%d'%i][:, :, start_y//i:end_y//i, start_x//i:end_x//i] = self.output['Level_%d'%i][:, :, start_y_tile//i:end_y_tile//i,
                                                                    start_x_tile//i:end_x_tile//i]
                    DR_output[:, :, start_y//ir_scale:end_y//ir_scale, start_x//ir_scale:end_x//ir_scale] = self.output['LR'][:, :, start_y_tile//ir_scale:end_y_tile//ir_scale,
                                                                  start_x_tile//ir_scale:end_x_tile//ir_scale]
            self.output = SR_output
            self.dr = DR_output
        self.net_ir.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        for i in self.hr:
            out_dict[f'HR_{i}'] = self.hr[i].detach().cpu()
            out_dict[i] = self.output[i].detach().cpu()
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
            img_name = osp.splitext(osp.basename(val_data['HR_path'][0]))[0]  # HR_Path

            self.feed_data(val_data)
            _, _, width, height = self.hr['Level_1'].shape
            if height*width < 2048*1024:
                self.test()
            else:
                self.test_tile([2,1],128)


            visuals = self.get_current_visuals()
            range = self.opt['val'].get('range', [-1, 1])

            sr_img = {}
            hr_img = {}
            for i in self.hr:
                sr_img[i] = util.tensor2img(visuals[i], min_max=range)  # uint8
                hr_img[i] = util.tensor2img(visuals[f'HR_{i}'], min_max=range)  # uint8
            dr_img = util.tensor2img(visuals['DR'], min_max=range)  # uint8

            # tentative for out of GPU memory
            del self.hr
            del self.output
            del self.dr
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_path = {}
                    for i in sr_img:
                        save_path[i] = osp.join(self.opt['path']['val_images'], f'{current_iter}', i ,f'{img_name}.png')
                        util.imwrite(sr_img[i], save_path[i])

                    save_path_dr = osp.join(self.opt['path']['val_images'], f'{current_iter}', 'DR' ,f'{img_name}.png')
                    util.imwrite(dr_img, save_path_dr)
                   
                else:
                    if self.opt['val']['suffix']:
                        save_path = {}
                        for i in sr_img:
                            save_path[i] = osp.join(self.opt['path']['results_root'], dataset_name, i ,f'{img_name}_{self.opt["val"]["suffix"]}.png')
                            util.imwrite(sr_img[i], save_path[i])
                        save_path_dr = osp.join(self.opt['path']['results_root'], dataset_name, 'DR' ,f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        util.imwrite(dr_img, save_path_dr)
                    else:
                        save_path = {}
                        for i in sr_img:
                            save_path[i] = osp.join(self.opt['path']['results_root'], dataset_name, i ,f'{img_name}.png')
                            util.imwrite(sr_img[i], save_path[i])
                        save_path_dr = osp.join(self.opt['path']['results_root'], dataset_name, 'DR' ,f'{img_name}.png')
                        util.imwrite(dr_img, save_path_dr)
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in name:
                        name_lpips = name
                        opt_lpips  = opt_
                        continue
                    else:
                        for i in  sr_img:
                            if i in name:
                                metric_data = dict(img=sr_img[i], img2=hr_img[i])
                                self.metric_results[name] += calculate_metric(metric_data, opt_)
                                break
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if name_lpips:
                self.metric_results[name_lpips] = calculate_metric(dict(folder_gt=hr_path, folder_restored=osp.split(save_path['Level_1'])[0]), opt_lpips)

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
            