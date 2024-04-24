# Towards Extreme Image Rescaling with Generative Prior and Invertible Prior

> Hao Wei, Chenyang Ge, Zhiyuan Li, Xin Qiao, Pengchao Deng <br>
>
> This paper explore the implicit rich and diverse generative prior embedded in the pretrained VQGAN to reduce the ambiguity of extreme upscaling (16$\times$ and 32$\times$) and improve the high-resolution reconstruction. To achieve a mutual inverse between semantically meaningful low-resolution images and quantized features, we develop an invertible feature recovery module that achieves better feature matching accuracy and converges more stable than conventional encoder-decoder networks. In addition, a multi-scale refinement module is proposed to help alleviate color distortions and artifacts.

## :wrench: Requirements

```bash
# Build VQIR with extension
pip install -r requirements.txt
```

## :zap: Inference

- Download pre-trained **VQIR 16x/32x models** [[Google Drive](https://drive.google.com/drive/folders/1NRJfaSShSXs2SZ0O0lzCoP4pl6mYKWun?usp=share_link)] into `vqir/pretrained`.

- Modify the configuration file `vqir/options/test/test_vqir_stage2.yml` accordingly.

- Inference on your own datasets.

```bash
python vqir/test.py -opt vqir/options/test/test_vqir_stage2.yml
```

## :computer: Training

We provide the training codes for VQIR (used in our paper).

- Dataset preparation: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Download [VQGAN](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/?p=%2F) weights into `vqir/pretrained/vqgan`

**Stage 1: IFRM Training**

- Modify the configuration file `vqir/options/train/train_vqir_stage1.yml` accordingly.

- Train IFRM on DIV2K datasets.
```bash
 python vqir/options/train/train_vqir_stage1.yml
```

**Stage 2: MSRM Training**

- Modify the configuration file `vqir/options/train/train_vqir_stage2.yml` accordingly.

- Train MSRM on DIV2K datasets.
```bash
 python vqir/options/train/train_vqir_stage2.yml
```

## :heart: Acknowledgement

Thanks to the following open-source projects:

- [Taming-transformers](https://github.com/CompVis/taming-transformers)
  
- [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling)
  
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
  
## :clipboard: Citation

    @article{wei2024towards,
        title={Towards Extreme Image Rescaling with Generative Prior and Invertible Prior},
        author={Wei, Hao and Ge, Chenyang and Li, Zhiyuan and Qiao, Xin and Deng, Pengchao},
        journal={IEEE Transactions on Circuits and Systems for Video Technology},
        year={2024},
        publisher={IEEE}
    }
