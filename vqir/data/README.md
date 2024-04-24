
## Dataloader

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`preprocess`](https://github.com/Maclory/SPSR/tree/master/preprocess).
    
- can downsample images using `matlab bicubic` function. However, the speed is a bit slow. Implemented in [`util.py`](https://github.com/Maclory/SPSR/tree/master/code/data/util.py). More about [`matlab bicubic`](https://github.com/xinntao/BasicSR/wiki/Matlab-bicubic-imresize) function.


## Contents

- `LRHR_dataset`: reads LR and HR pairs from image folders or lmdb files. If only HR images are provided, downsample the images on-the-fly. Used in stage 1 training and validation.
- `MultiR_dataset`: reads HR and multi-scale downsampling images from image folders or lmdb files. If only HR images are provided, downsample the images on-the-fly. Used in stage 2 training and validation.

## How To Prepare Data
Please refer to [Dataset Preparation](https://github.com/Maclory/SPSR/tree/master).
