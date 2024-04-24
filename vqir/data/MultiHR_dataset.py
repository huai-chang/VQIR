import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class MultiHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(MultiHRDataset, self).__init__()
        self.opt = opt
        self.paths_HR = None
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])

        assert self.paths_HR, 'Error: HR path is empty.'

        self.random_scale_list = [1,1/2]

    def __getitem__(self, index):
        HR_path = None
        scale = self.opt['scale']
        ir_scale = self.opt['ir_scale']
        HR_size = self.opt['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, ir_scale)
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]

        # randomly scale during training
        if self.opt['phase'] == 'train':
            random_scale = random.choice(self.random_scale_list)
            H_s, W_s, _ = img_HR.shape

            def _mod(n, random_scale, scale, thres):
                rlt = int(n * random_scale)
                rlt = (rlt // scale) * scale
                return thres if rlt < thres else rlt

            H_s = _mod(H_s, random_scale, ir_scale, HR_size)
            W_s = _mod(W_s, random_scale, ir_scale, HR_size)
            img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
            # force to 3 channels
            if img_HR.ndim == 2:
                img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

        res = {}
        H, W, _ = img_HR.shape
        # using matlab imresize
        for i in scale:
            img = util.imresize_np(img_HR, 1 / i, True)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            res['Level_%d' % i] = img

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                for i in scale:
                    img = util.imresize_np(img_HR, 1 / i, True)
                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=2)
                    res['Level_%d' % i] = img
            
            img = util.imresize_np(img_HR, 1 / 16, True)
            img = util.imresize_np(img, 16, True)

            H, W, C = res['Level_%d' % scale[-1]].shape
            LR_size = HR_size // scale[-1]

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            for i in scale:
                img_size = int(HR_size / i)
                rnd_h_, rnd_w_ = int(rnd_h * scale[-1] / i), int(rnd_w * scale[-1] / i)
                res['Level_%d' % i] = res['Level_%d' % i][rnd_h_:rnd_h_ + img_size, rnd_w_:rnd_w_ + img_size, :]

            # augmentation - flip, rotate
            img_list = util.augment([res[i] for i in res], self.opt['use_flip'], \
                                          self.opt['use_rot'])
            
            for idx, name in enumerate(res):
                res[name] = img_list[idx]

        for i in res:
            # change color space if necessary
            if self.opt['color']:
                res[i] = util.channel_convert(res[i].shape[2], self.opt['color'], [res[i]])[0]  # TODO during val no definetion

            # BGR to RGB, HWC to CHW, numpy to tensor
            if res[i].shape[2] == 3:
                res[i] = res[i][:, :, [2, 1, 0]]
            # [0,1] to [-0.5, 0.5] to [-1, 1]
            res[i] = (res[i] - 0.5) / 0.5
            res[i] = torch.from_numpy(np.ascontiguousarray(np.transpose(res[i], (2, 0, 1)))).float()


        res['HR_path'] = HR_path
        return res

    def __len__(self):
        return len(self.paths_HR)
