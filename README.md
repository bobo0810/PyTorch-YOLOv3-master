
## 说明：
###### 该仓库为 原作者的内容，非常感谢大佬的无私奉献，本人仅阅读并在代码中加入大量中文注释，以便理解。

- ### 第一次更新：新增本仓库，代码未加完注释

- ### 第二次更新： 整个训练过程、损失计算等加完注释，测试和验证代码量少、简单易懂且与训练代码大致相同，暂时不加。

## 接下来工作：

 - ## [重构代码](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov3_pytorch)，并加入visdom可视化等，敬请期待~

## 参考文献：

推荐配合阅读，效果更佳~

- [从0到1实现YOLOv3（part one）](https://blog.csdn.net/qq_25737169/article/details/80530579)

- [从0到1实现YOLO v3（part two）](https://blog.csdn.net/qq_25737169/article/details/80634360)

- [yolo v3 译文](https://zhuanlan.zhihu.com/p/34945787)

- [YOLO v3网络结构分析](https://blog.csdn.net/qq_37541097/article/details/81214953)

 ## 环境：

 | python版本 | pytorch版本 |
|------------|-------------|
| 3.5        | 0.4.1       |


---

# [原仓库传送门](https://github.com/eriklindernoren/PyTorch-YOLOv3)
# PyTorch-YOLOv3
Minimal implementation of YOLOv3 in PyTorch.

## Table of Contents
- [PyTorch-YOLOv3](#pytorch-yolov3)
  * [Table of Contents](#table-of-contents)
  * [Paper](#paper)
  * [Installation](#installation)
  * [Inference](#inference)
  * [Test](#test)
  * [Train](#train)
  * [Credit](#credit)

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh

## Inference
Uses pretrained weights to make predictions on images. Below table displays the inference times when using as inputs images scaled to 256x256. The ResNet backbone measurements are taken from the YOLOv3 paper. The Darknet-53 measurement marked shows the inference time of this implementation on my 1080ti card.

| Backbone                | GPU      | FPS      |
| ----------------------- |:--------:|:--------:|
| ResNet-101              | Titan X  | 53       |
| ResNet-152              | Titan X  | 37       |
| Darknet-53 (paper)      | Titan X  | 76       |
| Darknet-53 (this impl.) | 1080ti   | 74       |

    $ python3 detect.py --image_folder /data/samples

<p align="center"><img src="assets/giraffe.png" width="480"\></p>
<p align="center"><img src="assets/dog.png" width="480"\></p>
<p align="center"><img src="assets/traffic.png" width="480"\></p>
<p align="center"><img src="assets/messi.png" width="480"\></p>

## Test
Evaluates the model on COCO test.

    $ python3 test.py --weights_path weights/yolov3.weights

| Model                   | mAP (min. 50 IoU) |
| ----------------------- |:----------------:|
| YOLOv3 (paper)          | 57.9             |
| YOLOv3 (this impl.)     | 58.2             |

## Train
Model does not converge yet during training. Data augmentation as well as additional training tricks remains to be implemented. PRs are welcomed!
```
    train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR]
```

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
