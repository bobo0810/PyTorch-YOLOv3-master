from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    根据module_defs（list形式）中的模块配置 来构造 网络模块list
    """
    # 第一行存放的是 超参数，所以需要pop出来
    hyperparams = module_defs.pop(0)
    # 输入图像的通道数为3
    output_filters = [int(hyperparams['channels'])]
    # 保存yolov3网络模型
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        # 解析cfg网络结构，转化为pytorch网络结构
        if module_def['type'] == 'convolutional':
            '''
            每个卷积层后都会跟一个BN层和一个LeakyReLU，算作list中的一行
            pad = 1 表示 使用pad,但是具体pad值时按照kernel_size计算的
            bn=1 也表示 使用bn,具体值为 输出通道数
            '''
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            # // 表示先做除法，然后向下取整
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                # 值为 输出通道数
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                # 激活函数
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            '''
            上采样与rount搭配使用
            
            上采样将feature map变大，然后与 之前的较大feature map在深度上合并
            '''

            # nearest 使用最邻近 nrighbours 对输入进行采样 像素值.
            upsample = nn.Upsample( scale_factor=int(module_def['stride']),
                                    mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            '''
            route 指 按照列来合并tensor,即扩展深度
            filters为该层输出，保存到output_filters
            '''
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            '''
            shortcut 指  残差结构，卷积的跨层连接，即 将不同两层输出（即输出+残差块）相加 为 最后结果
            filters为该层输出，保存到output_filters
            '''
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            '''
            对于YOLOLayer层：
            训练阶段返回 各loss
            预测阶段返回  预测结果
            '''
            # mask为 即从 anchor集合中选用哪几个anchor
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors  提取anchor
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            # 只拿 该层挑选之后的anchor
            anchors = [anchors[i] for i in anchor_idxs]
            # 数据集共多少类别。coco数据集80类别
            num_classes = int(module_def['classes'])
            # 输入的训练图像大小416
            img_height = int(hyperparams['height'])
            # Define detection layer 定义检测层
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        # 注册模块列表和输出过滤器的数量
        #保存 模型结构list
        module_list.append(modules)
        # 保存每层的输出结果list
        output_filters.append(filters)

    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    '''
    “route”和“shortcut”层的占位符
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""
    '''
    检测层
    训练时计算损失
    预测时输出预测结果
    '''
    def __init__(self, anchors, num_classes, img_dim):
        '''
        :param anchors: 该检测层 挑选的几个anchor
        :param num_classes: 数据集类别，coco数据集共80类
        :param img_dim: 输入图像大小416
        '''
        super(YOLOLayer, self).__init__()
        self.anchors = anchors    #该检测层 挑选的几个anchor
        self.num_anchors = len(anchors)
        self.num_classes = num_classes  #数据集类别，coco数据集共80类
        self.bbox_attrs = 5 + num_classes  #一个 网格需要预测的值个数
        self.img_dim = img_dim   # 输入训练图像的大小
        self.ignore_thres = 0.5  #阈值
        self.lambda_coord = 1  #计算损失时的lambda，一般默认为1

        self.mse_loss = nn.MSELoss()   #均方误差 损失函数
        self.bce_loss = nn.BCELoss()  #计算目标和输出之间的二进制交叉熵  损失函数

    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.img_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training 训练阶段
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                            targets.cpu().data,
                                                                            scaled_anchors,
                                                                            self.num_anchors,
                                                                            self.num_classes,
                                                                            g_dim,
                                                                            self.ignore_thres,
                                                                            self.img_dim)

            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor))
            conf_mask = Variable(conf_mask.type(FloatTensor))

            # Handle target variables
            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls.type(FloatTensor), requires_grad=False)

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall

        else:
            # If not in training phase return predictions
            # 预测阶段，返回 预测结果
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    '''
    YOLOv3物体检测模型
    '''
    def __init__(self, config_path, img_size=416):
        '''
        输入通常为416（32的倍数）
        理由：参与预测层的最小特征图为13x13,为原图缩小32倍
        '''
        super(Darknet, self).__init__()
        # 将cfg配置文件转化为list,每一行 为网络的一部分
        self.module_defs = parse_model_config(config_path)
        # 解析list，返回 pytorch模型结构
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                '''
                route 指 按照列来合并tensor,即扩展深度
                
                当属性只有一个值时，它会输出由该值索引的网络层的特征图。
                在我们的示例中，它是−4，因此这个层将从Route层向后输出第4层的特征图。

                当图层有两个值时，它会返回由其值所索引的图层的连接特征图。 
                在我们的例子中，它是−1,61，并且该图层将输出来自上一层（-1）和第61层的特征图，并沿深度的维度连接。
                
                '''
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                '''
                shortcut 指  残差结构，卷积的跨层连接，即 将不同两层输出（即输出+残差块）相加 为 最后结果
                参数from是−3，意思是shortcut的输出是通过与先前的倒数第三层网络相加而得到。
                '''
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                # 训练阶段：获得损失
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                # 测试阶段：获取检测
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses['recall'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        '''
        解析并加载存储在'weights_path中的权重
        '''

        #Open the weights file
        fp = open(weights_path, "rb")
        # First five are header values  前五个为标题信息
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)

        # Needed to write header when saving weights
        # 保存权重时需要写头
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    # 如果设置的是False，只需加载卷积层的偏置即可
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                # 最终，加载卷积层参数：
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


    def save_weights(self, path, cutoff=-1):
        """
            保存模型权重(仅保存卷积层conv、BN层batch_normalize的权重参数信息，其余参数如shortcut、rount等为定值，无需保存)
            权重文件是包含以串行方式存储的权重的二进制文件
            当BN层出现在卷积块中时，不存在偏差。 但是，当没有BN layer 时，偏差“权重”必须从文件中读取

            @:param path    - path of the new weights file  （保存路径）
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
            当cutoff=-1时：保存全部网络参数
            当cutoff不为-1时，保存指定的部分网络参数
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        # tofile 将数组中的数据以二进制格式写进文件。文件路径为fp
        self.header_info.tofile(fp)

        # Iterate through layers 遍历网络层
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
