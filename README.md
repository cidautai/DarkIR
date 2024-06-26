# Low-Light Image Deblurring

## Introduction
The repository is structured in the next way:

```
/archs:
    /fourlie_archs
    /nafnet_utils
    __init__.py
    network.py
/data:
    datapipeline.py
    dataset.py
    dataset_LOLBlur.py
    dataset_NBDN.py
    __init__.py
/losses:
    loss.py
    loss_utils.py
    vgg_arch.py
/options:
    /train
    options.py
train.py
```
At this moment what can be done with this repo is train the net defined in /archs/network.py. To design the hyperparameters of the training you may change the /options/train/config.yaml file that you want to run, and when running train.py select the config.yaml desired.

## Datasets
The datasets used in the paper, LOL-Blur and NBDN have the next properties:

|Dataset     | Sets of images | URL  |
| -----------| :---------------:|------|
|LOL-Blur    | 10200 training pairs / 1800 testing pairs| [Google Drive](/https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX) |
|NBDN        | 500 training images / 100 test images | [Google Drive](/https://drive.google.com/file/d/1C7J9rn2xbeJ4-Aom4KEQJdpFyBd2M4Zv/view) |
|GOPRO       | 2103 training pairs / 1111 test pairs | [Official Site](/https://cg.postech.ac.kr/research/realblur/)  |
|RealBlur-J  | 3758 training pairs / 980 image pairs | [Official Site](/https://seungjunnah.github.io/Datasets/gopro)  |

## Gallery
|                          LOLBlur-Low                          |                          LOLBlur-Normal                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![add](/examples/inputs/0010.png)                          |                          ![add](/examples/results/0010.png) |
| ![add](/examples/inputs/0088.png)                          |                          ![add](/examples/results/0088.png) |
| ![<img src="/examples/inputs/0087.png" height="250px"/>](https://imgsli.com/Mjc0NzIz) | ![<img src="/examples/inputs/0060.png" height="250px"/>](https://imgsli.com/Mjc0NzI0) |

&nbsp;

## Contact

If you have any questions, please contact danfei@cidaut.es.