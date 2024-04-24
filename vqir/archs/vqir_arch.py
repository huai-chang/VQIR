import torch
import torch.nn as nn

from utils.registry import ARCH_REGISTRY
from archs.modules.attn_arch import Inter_SA, Intra_SA
from archs.modules.vqgan_arch import VQGANDecoder, VQGANEncoder, ResnetBlock
from archs.modules.quantize_arch import VectorQuantizer2 as VectorQuantizer
from archs.inv_arch import ICRM
from timm.models.layers import trunc_normal_

class Quan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, train=True):
        x = (x + 1.) / 2.
        x = x * 255.0
        if train:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5).cuda()
            output = x + noise
            output = torch.clamp(output, 0, 255.)
        else:
            output = x.round() * 1.0
            output = torch.clamp(output, 0, 255.)
        return (output / 255.0 - 0.5) / 0.5

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
        
class Refine_branch(nn.Module):
    def __init__(self, in_channels, num_attn_blocks, out_channels=None, head_num=4, return_map=False):
        super().__init__()

        self.return_map = return_map
        
        rfb = []
        for i in range(num_attn_blocks):
            rfb.append(Intra_SA(in_channels, head_num))
            rfb.append(Inter_SA(in_channels, head_num))
        self.rfb = nn.Sequential(*rfb)

        self.img_out = nn.Conv2d(in_channels, 3, 3, 1, 1)

        if return_map:
            self.attn_out = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, 1),
                                            Upsample(out_channels,with_conv=True),
                                            nn.Sigmoid()])
    
    def forward(self, in_feat, attn_map=None):
        res = {}

        if attn_map is not None:
            in_feat = in_feat * attn_map
        in_feat = self.rfb(in_feat)
        img = self.img_out(in_feat)
        res['img'] = img

        if self.return_map:
            attn_map = self.attn_out(in_feat)
            res['attn_map'] = attn_map
        
        return res
    
class Refine(nn.Module):
    def __init__(self, ch, ch_mult, attn_mult=[3,3]):
        super().__init__()
        self.num_branch = len(attn_mult) + 1
        ch_mult = ch_mult[:self.num_branch][::-1]

        self.refine = nn.ModuleList()
        in_ch = ch*ch_mult[0]
        for i in range(self.num_branch-1):
            out_ch = ch*ch_mult[i+1]
            self.refine.append(Refine_branch(in_channels=in_ch,num_attn_blocks=attn_mult[i],out_channels=out_ch,head_num=in_ch//64,return_map=True))
            in_ch = out_ch

        self.refine_out = nn.Sequential(ResnetBlock(in_channels=in_ch,out_channels=in_ch,dropout=0.0),
                                        ResnetBlock(in_channels=in_ch, out_channels=in_ch,dropout=0.0),
                                        nn.Conv2d(in_ch,3,3,1,1))

    
    def forward(self, feat_dict):
        results = {}
        attn_map = None
        for i in range(self.num_branch-1):
            res = self.refine[i](feat_dict['Level_%d' % 2**(self.num_branch-i-1)], attn_map)
            results['Level_%d' % 2**(self.num_branch-i-1)] = res['img']

            if 'attn_map' in res:
                attn_map = res['attn_map']
        
        if attn_map is not None:
            feat = feat_dict['Level_1'] * attn_map
        else:
            feat = feat_dict['Level_1']
        results['Level_1'] = self.refine_out(feat)
            
        return results
        

@ARCH_REGISTRY.register()
class VQIR(nn.Module):
    def __init__(self, stage, ir_scale, in_channels, out_channels, ch, ch_mult, double_z,
                 num_res_blocks, z_channels, n_embed, embed_dim, attn_mult=None,
                 beta=0.25, remap=None, sane_index_shape=False):
        super().__init__()

        self.stage1 = True if stage==1 else False

        self.encoder = VQGANEncoder(ch=ch,ch_mult=ch_mult,num_res_blocks=num_res_blocks,
                                    in_channels=in_channels,z_channels=z_channels,double_z=double_z)

        self.decoder = VQGANDecoder(ch=ch, out_channels=out_channels,ch_mult=ch_mult, 
                                    num_res_blocks=num_res_blocks,in_channels=in_channels,
                                    z_channels=z_channels)
        
        self.quan = Quan()

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta, remap, sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if not self.stage1:
            self.refine = Refine(ch=ch,ch_mult=ch_mult,attn_mult=attn_mult)

        self.apply(self._init_weights)

        self.icrm = ICRM(in_channels=embed_dim,out_channels=out_channels, ir_scale=ir_scale)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def forward(self, x_hr, is_train=True):
        res = {}
        
        
        with torch.no_grad():
            enc_feat = self.encoder(x_hr)
            enc_feat = self.quant_conv(enc_feat)
            quant_feat, _, _ = self.quantize(enc_feat)
        

        if self.stage1:    
            res['quant_feat'] = quant_feat

            x_lr = self.quan(self.icrm(quant_feat, rev=False),train=is_train)
            res['LR'] = x_lr
            re_quant_feat = self.icrm(x_lr, rev=True)
            res['re_quant_feat'] = re_quant_feat

            if not is_train:
                with torch.no_grad():
                    quant_feat, _, _ = self.quantize(re_quant_feat)
                    quant_feat = self.post_quant_conv(quant_feat)
                    dec = self.decoder(quant_feat)
                    res['SR'] = dec

            return res

        else:
            with torch.no_grad():
                x_lr = self.quan(self.icrm(quant_feat, rev=False),train=False)
                res['LR'] = x_lr
                re_quant_feat = self.icrm(x_lr, rev=True)
                quant_feat, _, _ = self.quantize(re_quant_feat)
                quant_feat = self.post_quant_conv(quant_feat)
                dec, feat_dict = self.decoder(quant_feat, return_feat=True)
                
            results = self.refine(feat_dict)
            res.update(results)

            return res



if __name__ == '__main__':
    x = torch.randn(2,3,256,256)
    model = VQIR(stage=1,
                 in_channels=3, 
                 out_channels=3, 
                 ch=128, 
                 ch_mult=[1,1,2,2,4], 
                 attn_mult=[4,4,2],
                 num_res_blocks=2, 
                 z_channels=256, 
                 n_embed=1024, 
                 embed_dim=256, 
                 double_z=False)

    res = model(x,is_train=False)
    for i in res:
        print(i,res[i].shape)

    
