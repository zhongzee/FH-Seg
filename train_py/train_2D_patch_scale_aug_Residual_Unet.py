import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
sys.path.append("/Data/DoDNet/")
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from unet2D_Dodnet_scale import UNet2D as UNet2D_scale
#from unet2D_Dodnet_scale_v2 import UNet2D as UNet2D_scale

# from model.concave_dps_w import ResUNet


import os.path as osp
from MOTSDataset_2D_Patch_normal import MOTSDataSet as MOTSDataSet_normal
from MOTSDataset_2D_Patch_normal import MOTSValDataSet, my_collate
from MOTSDataset_2D_Patch_scale import MOTSDataSet as MOTSDataSet_scale

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from network import deepv3
from util.image_pool import ImagePool
#from network import deepv3
from monai.networks.nets.swin_unetr import SwinUNETR
#from torchvision.models.segmentation import deeplabv3
from unet2D_residual import UNet2D
def one_hot_3D(targets,C = 2):
    targets_extend=targets.clone()
    if targets_extend.dim() == 3:
        targets_extend = targets_extend.unsqueeze(1)  # 添加一个轴，使其变为 Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")
    #parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_trainingset_patch')
    #parser.add_argument("--trainset_dir", type=str, default='/Data2/Demo_KI_data_trainingset_patch')
    parser.add_argument("--trainset_dir", type=str, default='./data/omniseg-sampled/train')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_trainingset_patch')
    parser.add_argument("--valset_dir", type=str, default='./data/omniseg-sampled/val')
    #parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    #parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--train_list", type=str, default='./data/omniseg-sampled/train/data_list.csv')
    parser.add_argument("--val_list", type=str, default='./data/omniseg-sampled/val/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.2)

    parser.add_argument("--scale", type=str2bool, default=False)
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/Residual_Unet/fold1_with_white_scale_normalwhole_1217/')
    parser.add_argument("--reload_path", type=str, default='')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    #parser.add_argument("--input_size", type=str, default='256,256')
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))

    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        try:
            rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
            cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]
        except:
            rmin[ki], rmax[ki] = 0, 255
            cmin[ki], cmax[ki] = 0, 255

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)

def get_scale_tensor(pred, rmin, rmax, cmin, cmax):
    if len(pred.shape) == 3:
        return pred[:,rmin:rmax,cmin:cmax].unsqueeze(0)
    else:
        return pred[rmin:rmax, cmin:cmax].unsqueeze(0)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        # preds1 = sum(preds,[])
        # labels1 = sum(labels,[])
        try:
            Val_DICE += dice_score(pred, label)
            # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
            preds1 = pred[1, ...].flatten().detach().cpu().numpy()
            # labels1 = labels[:,1,...].view(-1).cpu().numpy()
            labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

            cnf_matrix = confusion_matrix(preds1, labels1)

            FP = cnf_matrix[1,0]
            FN = cnf_matrix[0,1]
            TP = cnf_matrix[1,1]
            TN = cnf_matrix[0,0]

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            Val_TPR += TP / (TP + FN)
            Val_PPV += TP / (TP + FP)

            Val_F1 += f1_score(preds1, labels1, average='macro')

        except:

            Val_DICE += 1.
            Val_F1 += 1.
            Val_TPR += 1.
            Val_PPV += 1.

    return Val_F1/cnt, Val_DICE/cnt, Val_TPR/cnt, Val_PPV/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE):

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_all = 0
    # rmin, rmax, cmin, cmax = mask_to_box(images)
    # for li in range(len(images)):
    #     pred = get_scale_tensor(preds[li], rmin[li], rmax[li], cmin[li], cmax[li])
    #     label = get_scale_tensor(labels[li], rmin[li], rmax[li], cmin[li], cmax[li])
    #     wgt = get_scale_tensor(weight[li], rmin[li], rmax[li], cmin[li], cmax[li])
    #     cnt += 1
    #     try:

    term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
    term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
    term_all += (term_seg_Dice + term_seg_BCE)

    #
    # term_seg_Dice = term_seg_Dice / cnt
    # term_seg_BCE = term_seg_BCE / cnt
    # term_all = term_all / cnt

    return term_seg_Dice, term_seg_BCE, term_all

def TAL_pred(preds, task_id):
    """
    if task_id == 0:
        preds_p2 = preds[:, 0:1, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 1:
        preds_p2 = preds[:, 1:2, :, :].clone()
        preds_p1 = preds[:, 0:1, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 2:
        preds_p2 = preds[:, 2:3, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 3:
        preds_p2 = preds[:, 3:4, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 4:
        preds_p2 = preds[:, 4:5, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    else:
        preds_p2 = preds[:, 5:6, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 0:1, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)
    """
    #####################################################################################################
    if task_id == 0:
        preds_p2 = preds[:, 0:1, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 1:
        preds_p2 = preds[:, 1:2, :, :].clone()
        preds_p1 = preds[:, 0:1, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 2:
        preds_p2 = preds[:, 2:3, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 3:
        preds_p2 = preds[:, 3:4, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 4:
        preds_p2 = preds[:, 4:5, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    else:
        preds_p2 = preds[:, 5:6, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 0:1, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)
    #####################################################################################################

    return new_preds

def main():
    """Create the model and start the training."""
    print("hello!")
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        #model = UNet2D_scale(num_classes=args.num_classes, weight_std = False)
        #model = SwinUNETR(img_size=512, in_channels=3, out_channels=7, spatial_dims=2)

        model = UNet2D(num_classes=6)
        check_wo_gpu = 0

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...a
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights = [1., 5.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=1).to(device)
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=1, ignore_index=255).to(device)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights).to(device)
            #criterion2 = FocalLoss2d(weight=class_weights).to(device)

        else:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=1)
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights)
            #criterion2 = FocalLoss2d(weight=class_weights)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        trainloader = DataLoader(
            MOTSDataSet_normal(args.trainset_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                        edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=4)

        valloader = DataLoader(
            MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=4)
        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999

        for epoch in range(0,args.num_epochs):
            model.train()
            # create multi-task image pool
            task0_pool_image = ImagePool(8)
            task0_pool_mask = ImagePool(8)
            task0_pool_weight = ImagePool(8)
            task0_scale = []
            task1_pool_image = ImagePool(8)
            task1_pool_mask = ImagePool(8)
            task1_pool_weight = ImagePool(8)
            task1_scale = []
            task2_pool_image = ImagePool(8)
            task2_pool_mask = ImagePool(8)
            task2_pool_weight = ImagePool(8)
            task2_scale = []
            task3_pool_image = ImagePool(8)
            task3_pool_mask = ImagePool(8)
            task3_pool_weight = ImagePool(8)
            task3_scale = []
            task4_pool_image = ImagePool(8)
            task4_pool_mask = ImagePool(8)
            task4_pool_weight = ImagePool(8)
            task4_scale = []
            task5_pool_image = ImagePool(8)
            task5_pool_mask = ImagePool(8)
            task5_pool_weight = ImagePool(8)
            task5_scale = []

            ###########################################
            task6_pool_image = ImagePool(8)
            task6_pool_mask = ImagePool(8)
            task6_pool_weight = ImagePool(8)
            task6_scale = []


            ############################################

            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            batch_size = args.batch_size
            task_num = 6
            #############################################
            # task_num = 7
            #############################################
            each_loss = torch.zeros((task_num)).cuda()
            count_batch = torch.zeros((task_num)).cuda()
            loss_weight = torch.ones((task_num)).cuda()

            # loss_weight[0] = 10.
            # loss_weight[4] = 10.
            # loss_weight[5] = 10.

            for iter, batch in enumerate(trainloader):

                # imgs = torch.from_numpy(batch['image']).cuda()
                # lbls = torch.from_numpy(batch['label']).cuda()
                # volumeName = batch['name']
                # wt = torch.from_numpy(batch['weight']).cuda().float()
                # t_ids = torch.from_numpy(batch['task_id']).cuda()
                # #s_ids = torch.from_numpy(batch['scale_id']).cuda()
                # s_ids = batch['scale_id']
                # # print(task_ids)
                # # print(batch['name'])

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                sum_loss = 0

                # plt.imshow(imgs[0].permute([1,2,0]).cpu().numpy())
                # plt.show()

                for ki in range(len(imgs)):
                    now_task = t_ids[ki]
                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_pool_weight.add(wt[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_pool_weight.add(wt[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_pool_weight.add(wt[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                    elif now_task == 3:
                        task3_pool_image.add(imgs[ki].unsqueeze(0))
                        task3_pool_mask.add(lbls[ki].unsqueeze(0))
                        task3_pool_weight.add(wt[ki].unsqueeze(0))
                        task3_scale.append((s_ids[ki]))
                    elif now_task == 4:
                        task4_pool_image.add(imgs[ki].unsqueeze(0))
                        task4_pool_mask.add(lbls[ki].unsqueeze(0))
                        task4_pool_weight.add(wt[ki].unsqueeze(0))
                        task4_scale.append((s_ids[ki]))
                    elif now_task == 5:
                        task5_pool_image.add(imgs[ki].unsqueeze(0))
                        task5_pool_mask.add(lbls[ki].unsqueeze(0))
                        task5_pool_weight.add(wt[ki].unsqueeze(0))
                        task5_scale.append((s_ids[ki]))

                    ##########################################################
                    elif now_task == 6:
                        task6_pool_image.add(imgs[ki].unsqueeze(0))
                        task6_pool_mask.add(lbls[ki].unsqueeze(0))
                        task6_pool_weight.add(wt[ki].unsqueeze(0))
                        task6_scale.append((s_ids[ki]))
                    ##########################################################

                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    wts = task0_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task0_scale.pop(0)

                    #preds = model(images, torch.ones(batch_size).cuda()*0, scales)
                    preds = model(images)
                    #print(preds.shape)
                    ##################################################################################
                    #preds = preds[0]
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    # preds = (preds[:,1,...] > preds[:,0,...]).type(torch.FloatTensor).cuda()

                    #loss1 = criterion1(preds, labels.squeeze(1).long())

                    #labels = one_hot(labels.squeeze(1).long())
                    #loss2 = criterion2(preds, labels)
                    weight = edge_weight**wts

                    now_task = 0
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[0] += term_all
                    count_batch[0] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # reduce_all = reduce_all + loss1 + loss2

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    wts = task1_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task1_scale.pop(0)
                    weight = edge_weight**wts
                    #optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*1, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################

                    #preds = model(images)
                    labels = one_hot_3D(labels.long())

                    now_task = 1
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[1] += term_all
                    count_batch[1] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    wts = task2_pool_weight.query(batch_size)
                    weight = edge_weight ** wts
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task2_scale.pop(0)
                    #optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*2, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 2
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[2] += term_all
                    count_batch[2] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                if task3_pool_image.num_imgs >= batch_size:
                    images = task3_pool_image.query(batch_size)
                    labels = task3_pool_mask.query(batch_size)
                    wts = task3_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task3_scale.pop(0)
                    #optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*3, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 3
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)


                    #sum_loss += term_all
                    each_loss[3] += term_all
                    count_batch[3] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                if task4_pool_image.num_imgs >= batch_size:
                    images = task4_pool_image.query(batch_size)
                    labels = task4_pool_mask.query(batch_size)
                    wts = task4_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task4_scale.pop(0)
                    #optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*4, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 4
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    #sum_loss += term_all
                    each_loss[4] += term_all
                    count_batch[4] += 1

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                if task5_pool_image.num_imgs >= batch_size:
                    images = task5_pool_image.query(batch_size)
                    labels = task5_pool_mask.query(batch_size)
                    wts = task5_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task5_scale.pop(0)
                    # optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*5, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################

                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts
                    now_task = 5



                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:, now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)


                    #sum_loss += term_all
                    each_loss[5] += term_all
                    count_batch[5] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

                #####################################################################################
                if task6_pool_image.num_imgs >= batch_size:
                    images = task6_pool_image.query(batch_size)
                    labels = task6_pool_mask.query(batch_size)
                    wts = task6_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task6_scale.pop(0)
                    # optimizer.zero_grad()
                    #preds = model(images, torch.ones(batch_size).cuda()*6, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 6
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[6] += term_all
                    count_batch[6] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))



                #####################################################################################
                #if sum_loss > 0:
                #if sum(each_loss > 0) == task_num:
                # avg_loss = 0
                # if sum(each_loss) > 0:
                #     for ei in range(len(each_loss)):
                #         if each_loss[ei] > 0:
                #             avg_loss += each_loss[ei] * loss_weight[ei] # / count_batch[ei].detach()
                #     optimizer.zero_grad()
                #
                #     if args.FP16:
                #         with amp.scale_loss(avg_loss, optimizer) as scaled_loss:
                #             scaled_loss.backward()
                #     else:
                #         avg_loss.backward()
                #     optimizer.step()
                #
                #     each_loss = torch.zeros((task_num)).cuda()
                #     count_batch = torch.zeros((task_num)).cuda()

            ###############################################################################################
            if (task0_pool_image.num_imgs < batch_size) & (task0_pool_image.num_imgs > 0):
                    left_size = task0_pool_image.num_imgs
                    images = task0_pool_image.query(left_size)
                    labels = task0_pool_mask.query(left_size)
                    wts = task0_pool_weight.query(left_size)
                    scales = torch.ones(left_size).cuda()

                    if(left_size == 1):
                        images = images.repeat(4,1,1,1)
                        labels = labels.repeat(4,1,1,1)
                        wts = wts.repeat(4,1,1,1)

                    if(left_size == 2):
                        images = images.repeat(2,1,1,1)
                        labels = labels.repeat(2,1,1,1)
                        wts = wts.repeat(2,1,1,1)

                    if(left_size == 3):
                        last_image = torch.unsqueeze(images[-1], dim=0)
                        images = torch.cat((images, last_image), dim=0)

                        last_label = torch.unsqueeze(labels[-1], dim=0)
                        labels = torch.cat((labels, last_label), dim=0)

                        last_wts = torch.unsqueeze(wts[-1], dim=0)
                        wts = torch.cat((wts, last_wts), dim=0)









                    for bi in range(len(scales)):
                        scales[bi] = task0_scale.pop(0)

                    #preds = model(images, torch.ones(left_size).cuda()*0, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    # preds = (preds[:,1,...] > preds[:,0,...]).type(torch.FloatTensor).cuda()

                    #loss1 = criterion1(preds, labels.squeeze(1).long())

                    #labels = one_hot(labels.squeeze(1).long())
                    #loss2 = criterion2(preds, labels)
                    weight = edge_weight**wts

                    now_task = 0
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[0] += term_all
                    count_batch[0] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # reduce_all = reduce_all + loss1 + loss2

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

            if (task1_pool_image.num_imgs < batch_size) & (task1_pool_image.num_imgs > 0):
                    left_size = task1_pool_image.num_imgs
                    images = task1_pool_image.query(left_size)
                    labels = task1_pool_mask.query(left_size)
                    wts = task1_pool_weight.query(left_size)
                    scales = torch.ones(batch_size).cuda()

                    if (left_size == 1):
                        images = images.repeat(4, 1, 1, 1)
                        labels = labels.repeat(4, 1, 1)
                        wts = wts.repeat(4, 1, 1)

                    if (left_size == 2):
                        images = images.repeat(2, 1, 1, 1)
                        labels = labels.repeat(2, 1, 1)
                        wts = wts.repeat(2, 1, 1)

                    if (left_size == 3):
                        last_image = torch.unsqueeze(images[-1], dim=0)
                        images = torch.cat((images, last_image), dim=0)

                        last_label = torch.unsqueeze(labels[-1], dim=0)
                        labels = torch.cat((labels, last_label), dim=0)

                        last_wts = torch.unsqueeze(wts[-1], dim=0)
                        wts = torch.cat((wts, last_wts), dim=0)

                    # for bi in range(len(scales)):
                    #     scales[bi] = task1_scale.pop(0)
                    weight = edge_weight ** wts
                    # optimizer.zero_grad()
                    #preds = model(images, torch.ones(left_size).cuda() * 1, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################

                    # preds = model(images)
                    labels = one_hot_3D(labels.long())

                    now_task = 1
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)


                    # sum_loss += term_all
                    each_loss[1] += term_all
                    count_batch[1] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))
            if (task2_pool_image.num_imgs < batch_size) & (task2_pool_image.num_imgs > 0):
                left_size = task2_pool_image.num_imgs
                images = task2_pool_image.query(left_size)
                labels = task2_pool_mask.query(left_size)
                wts = task2_pool_weight.query(left_size)
                weight = edge_weight ** wts
                scales = torch.ones(batch_size).cuda()

                if (left_size == 1):
                    images = images.repeat(4, 1, 1, 1)
                    labels = labels.repeat(4, 1, 1)
                    wts = wts.repeat(4, 1, 1)

                if (left_size == 2):
                    images = images.repeat(2, 1, 1, 1)
                    labels = labels.repeat(2, 1, 1)
                    wts = wts.repeat(2, 1, 1)

                if (left_size == 3):
                    last_image = torch.unsqueeze(images[-1], dim=0)
                    images = torch.cat((images, last_image), dim=0)

                    last_label = torch.unsqueeze(labels[-1], dim=0)
                    labels = torch.cat((labels, last_label), dim=0)

                    last_wts = torch.unsqueeze(wts[-1], dim=0)
                    wts = torch.cat((wts, last_wts), dim=0)


                # for bi in range(len(scales)):
                #     scales[bi] = task2_scale.pop(0)
                # optimizer.zero_grad()
                #preds = model(images, torch.ones(left_size).cuda() * 2, scales)
                ##################################################################################
                #preds = preds[0]
                preds = model(images)
                ##################################################################################
                labels = one_hot_3D(labels.long())
                weight = edge_weight ** wts

                now_task = 2
                term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:, now_task].unsqueeze(1),
                                                                 labels[:, 1].unsqueeze(1), weight, loss_seg_DICE,
                                                                 loss_seg_CE)

                # sum_loss += term_all
                each_loss[2] += term_all
                count_batch[2] += 1

                reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                reduce_all = engine.all_reduce_tensor(term_all)

                optimizer.zero_grad()
                reduce_all.backward()
                optimizer.step()

                # if args.FP16:
                #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     term_all.backward()
                # optimizer.step()

                epoch_loss.append(float(reduce_all))

                if (args.local_rank == 0):
                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))

            if (task3_pool_image.num_imgs < batch_size) & (task3_pool_image.num_imgs > 0):
                    left_size = task3_pool_image.num_imgs
                    images = task3_pool_image.query(left_size)
                    labels = task3_pool_mask.query(left_size)
                    wts = task3_pool_weight.query(left_size)
                    scales = torch.ones(batch_size).cuda()

                    if (left_size == 1):
                        images = images.repeat(4, 1, 1, 1)
                        labels = labels.repeat(4, 1, 1)
                        wts = wts.repeat(4, 1, 1)

                    if (left_size == 2):
                        images = images.repeat(2, 1, 1, 1)
                        labels = labels.repeat(2, 1, 1)
                        wts = wts.repeat(2, 1, 1)

                    if (left_size == 3):
                        last_image = torch.unsqueeze(images[-1], dim=0)
                        images = torch.cat((images, last_image), dim=0)

                        last_label = torch.unsqueeze(labels[-1], dim=0)
                        labels = torch.cat((labels, last_label), dim=0)

                        last_wts = torch.unsqueeze(wts[-1], dim=0)
                        wts = torch.cat((wts, last_wts), dim=0)
                    # for bi in range(len(scales)):
                    #     scales[bi] = task3_scale.pop(0)
                    #optimizer.zero_grad()
                    #preds = model(images, torch.ones(left_size).cuda()*3, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 3
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)


                    #sum_loss += term_all
                    each_loss[3] += term_all
                    count_batch[3] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

            if (task4_pool_image.num_imgs < batch_size) & (task4_pool_image.num_imgs > 0):
                left_size = task4_pool_image.num_imgs
                images = task4_pool_image.query(left_size)
                labels = task4_pool_mask.query(left_size)
                wts = task4_pool_weight.query(left_size)
                scales = torch.ones(batch_size).cuda()

                if (left_size == 1):
                    images = images.repeat(4, 1, 1, 1)
                    labels = labels.repeat(4, 1, 1)
                    wts = wts.repeat(4, 1, 1)

                if (left_size == 2):
                    images = images.repeat(2, 1, 1, 1)
                    labels = labels.repeat(2, 1, 1)
                    wts = wts.repeat(2, 1, 1)

                if (left_size == 3):
                    last_image = torch.unsqueeze(images[-1], dim=0)
                    images = torch.cat((images, last_image), dim=0)

                    last_label = torch.unsqueeze(labels[-1], dim=0)
                    labels = torch.cat((labels, last_label), dim=0)

                    last_wts = torch.unsqueeze(wts[-1], dim=0)
                    wts = torch.cat((wts, last_wts), dim=0)
                # for bi in range(len(scales)):
                #     scales[bi] = task4_scale.pop(0)
                # optimizer.zero_grad()
                #preds = model(images, torch.ones(left_size).cuda() * 4, scales)
                ##################################################################################
                #preds = preds[0]
                preds = model(images)
                ##################################################################################
                labels = one_hot_3D(labels.long())
                weight = edge_weight ** wts

                now_task = 4
                term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:, now_task].unsqueeze(1),
                                                                 labels[:, 1].unsqueeze(1), weight, loss_seg_DICE,
                                                                 loss_seg_CE)

                reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                reduce_all = engine.all_reduce_tensor(term_all)

                optimizer.zero_grad()
                reduce_all.backward()
                optimizer.step()

                # sum_loss += term_all
                each_loss[4] += term_all
                count_batch[4] += 1

                # if args.FP16:
                #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     term_all.backward()
                # optimizer.step()

                epoch_loss.append(float(reduce_all))

                if (args.local_rank == 0):
                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))

            if (task5_pool_image.num_imgs < batch_size) & (task5_pool_image.num_imgs > 0):
                    left_size = task5_pool_image.num_imgs
                    images = task5_pool_image.query(left_size)
                    labels = task5_pool_mask.query(left_size)
                    wts = task5_pool_weight.query(left_size)
                    scales = torch.ones(batch_size).cuda()

                    if (left_size == 1):
                        images = images.repeat(4, 1, 1, 1)
                        labels = labels.repeat(4, 1, 1)
                        wts = wts.repeat(4, 1, 1)

                    if (left_size == 2):
                        images = images.repeat(2, 1, 1, 1)
                        labels = labels.repeat(2, 1, 1)
                        wts = wts.repeat(2, 1, 1)

                    if (left_size == 3):
                        last_image = torch.unsqueeze(images[-1], dim=0)
                        images = torch.cat((images, last_image), dim=0)

                        last_label = torch.unsqueeze(labels[-1], dim=0)
                        labels = torch.cat((labels, last_label), dim=0)

                        last_wts = torch.unsqueeze(wts[-1], dim=0)
                        wts = torch.cat((wts, last_wts), dim=0)

                    # for bi in range(len(scales)):
                    #     scales[bi] = task5_scale.pop(0)
                    # optimizer.zero_grad()
                    #preds = model(images, torch.ones(left_size).cuda()*5, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################

                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 5
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)




                    #sum_loss += term_all
                    each_loss[5] += term_all
                    count_batch[5] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))

            if (task6_pool_image.num_imgs < batch_size) & (task6_pool_image.num_imgs > 0):
                    left_size = task6_pool_image.num_imgs
                    images = task6_pool_image.query(left_size)
                    labels = task6_pool_mask.query(left_size)
                    wts = task6_pool_weight.query(left_size)
                    scales = torch.ones(batch_size).cuda()

                    if (left_size == 1):
                        images = images.repeat(4, 1, 1, 1)
                        labels = labels.repeat(4, 1, 1)
                        wts = wts.repeat(4, 1, 1)

                    if (left_size == 2):
                        images = images.repeat(2, 1, 1, 1)
                        labels = labels.repeat(2, 1, 1)
                        wts = wts.repeat(2, 1, 1)

                    if (left_size == 3):
                        last_image = torch.unsqueeze(images[-1], dim=0)
                        images = torch.cat((images, last_image), dim=0)

                        last_label = torch.unsqueeze(labels[-1], dim=0)
                        labels = torch.cat((labels, last_label), dim=0)

                        last_wts = torch.unsqueeze(wts[-1], dim=0)
                        wts = torch.cat((wts, last_wts), dim=0)

                    # for bi in range(len(scales)):
                    #     scales[bi] = task6_scale.pop(0)
                    # optimizer.zero_grad()
                    #preds = model(images, torch.ones(left_size).cuda()*6, scales)
                    ##################################################################################
                    #preds = preds[0]
                    preds = model(images)
                    ##################################################################################
                    labels = one_hot_3D(labels.long())
                    weight = edge_weight**wts

                    now_task = 6
                    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)



                    #sum_loss += term_all
                    each_loss[6] += term_all
                    count_batch[6] += 1

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    # if args.FP16:
                    #     with amp.scale_loss(term_all, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     term_all.backward()
                    # optimizer.step()

                    epoch_loss.append(float(reduce_all))

                    if (args.local_rank == 0):
                        print(
                            'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                                epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                                reduce_BCE.item(), reduce_all.item()))


            ###############################################################################################

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

            if (epoch >= 0) and (args.local_rank == 0) and (((epoch % 10 == 0) and (epoch >= 800)) or (epoch % 1 == 0)):
                print('save validation image ...')

                model.eval()
                task0_pool_image = ImagePool(8)
                task0_pool_mask = ImagePool(8)
                task0_scale = []
                task1_pool_image = ImagePool(8)
                task1_pool_mask = ImagePool(8)
                task1_scale = []
                task2_pool_image = ImagePool(8)
                task2_pool_mask = ImagePool(8)
                task2_scale = []
                task3_pool_image = ImagePool(8)
                task3_pool_mask = ImagePool(8)
                task3_scale = []
                task4_pool_image = ImagePool(8)
                task4_pool_mask = ImagePool(8)
                task4_scale = []
                task5_pool_image = ImagePool(8)
                task5_pool_mask = ImagePool(8)
                task5_scale = []

                #########################################################
                task6_pool_image = ImagePool(8)
                task6_pool_mask = ImagePool(8)
                task6_scale = []
                ###########################################################

                val_loss = np.zeros((6))
                val_F1 = np.zeros((6))
                val_Dice = np.zeros((6))
                val_TPR = np.zeros((6))
                val_PPV = np.zeros((6))
                cnt = np.zeros((6))
                ##########################################################
                val_loss = np.zeros((7))
                val_F1 = np.zeros((7))
                val_Dice = np.zeros((7))
                val_TPR = np.zeros((7))
                val_PPV = np.zeros((7))
                cnt = np.zeros((7))
                ##########################################################

                with torch.no_grad():
                    for iter, batch in enumerate(valloader):
                        #
                        # if iter > 50:
                        #     break

                        # imgs = torch.from_numpy(batch['image']).cuda()
                        # lbls = torch.from_numpy(batch['label']).cuda()
                        # volumeName = batch['name']
                        # t_ids = torch.from_numpy(batch['task_id']).cuda()
                        # # s_ids = torch.from_numpy(batch['scale_id']).cuda()
                        # s_ids = batch['scale_id']
                        # # print(task_ids)
                        # # print(batch['name'])

                        'dataloader'
                        imgs = batch[0].cuda()
                        lbls = batch[1].cuda()
                        wt = batch[2].cuda().float()
                        volumeName = batch[3]
                        t_ids = batch[4].cuda()
                        s_ids = batch[5]

                        for ki in range(len(imgs)):
                            now_task = t_ids[ki]
                            if now_task == 0:
                                task0_pool_image.add(imgs[ki].unsqueeze(0))
                                task0_pool_mask.add(lbls[ki].unsqueeze(0))
                                task0_scale.append((s_ids[ki]))
                            elif now_task == 1:
                                task1_pool_image.add(imgs[ki].unsqueeze(0))
                                task1_pool_mask.add(lbls[ki].unsqueeze(0))
                                task1_scale.append((s_ids[ki]))
                            elif now_task == 2:
                                task2_pool_image.add(imgs[ki].unsqueeze(0))
                                task2_pool_mask.add(lbls[ki].unsqueeze(0))
                                task2_scale.append((s_ids[ki]))
                            elif now_task == 3:
                                task3_pool_image.add(imgs[ki].unsqueeze(0))
                                task3_pool_mask.add(lbls[ki].unsqueeze(0))
                                task3_scale.append((s_ids[ki]))
                            elif now_task == 4:
                                task4_pool_image.add(imgs[ki].unsqueeze(0))
                                task4_pool_mask.add(lbls[ki].unsqueeze(0))
                                task4_scale.append((s_ids[ki]))
                            elif now_task == 5:
                                task5_pool_image.add(imgs[ki].unsqueeze(0))
                                task5_pool_mask.add(lbls[ki].unsqueeze(0))
                                task5_scale.append((s_ids[ki]))
                            ##################################################
                            elif now_task == 6:
                                task6_pool_image.add(imgs[ki].unsqueeze(0))
                                task6_pool_mask.add(lbls[ki].unsqueeze(0))
                                task6_scale.append((s_ids[ki]))



                            ####################################################
                        #parser.add_argument("--snapshot_dir", type=str,
                         #                   default='snapshots_2D/fold1_with_white_5layer_0903/')

                        #output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/fold1_with_white', '/Data/DoDNet/MIDL/OmniSeg_2D/validation'), str(epoch))
                        output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/fold1_with_white',  '/data2/DoDNet/MIDL/Residual_Unet/HC_validation'), str(epoch))
                        #output_folder = os.path.join('/Data/DoDNet/a_DynConv/validation_noscale_0829', str(epoch))
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        optimizer.zero_grad()

                        if task0_pool_image.num_imgs >= batch_size:
                            images = task0_pool_image.query(batch_size)
                            labels = task0_pool_mask.query(batch_size)
                            now_task = torch.tensor(0)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task0_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*0, scales)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################
                            #now_preds = preds[:,1,...] > preds[:,0,...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = mask_to_box(images)

                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[0] += F1
                            val_Dice[0] += DICE
                            val_TPR[0] += TPR
                            val_PPV[0] += PPV
                            cnt[0] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())

                        if task1_pool_image.num_imgs >= batch_size:
                            images = task1_pool_image.query(batch_size)
                            labels = task1_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task1_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*1, scales)

                            now_task = torch.tensor(1)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[1] += F1
                            val_Dice[1] += DICE
                            val_TPR[1] += TPR
                            val_PPV[1] += PPV
                            cnt[1] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())

                        if task2_pool_image.num_imgs >= batch_size:
                            images = task2_pool_image.query(batch_size)
                            labels = task2_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task2_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*2, scales)

                            now_task = torch.tensor(2)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[2] += F1
                            val_Dice[2] += DICE
                            val_TPR[2] += TPR
                            val_PPV[2] += PPV
                            cnt[2] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        if task3_pool_image.num_imgs >= batch_size:
                            images = task3_pool_image.query(batch_size)
                            labels = task3_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task3_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*3, scales)

                            now_task = torch.tensor(3)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[3] += F1
                            val_Dice[3] += DICE
                            val_TPR[3] += TPR
                            val_PPV[3] += PPV
                            cnt[3] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        if task4_pool_image.num_imgs >= batch_size:
                            images = task4_pool_image.query(batch_size)
                            labels = task4_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task4_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*4, scales)

                            now_task = torch.tensor(4)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[4] += F1
                            val_Dice[4] += DICE
                            val_TPR[4] += TPR
                            val_PPV[4] += PPV
                            cnt[4] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        if task5_pool_image.num_imgs >= batch_size:
                            images = task5_pool_image.query(batch_size)
                            labels = task5_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task5_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*5, scales)

                            now_task = torch.tensor(5)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[5] += F1
                            val_Dice[5] += DICE
                            val_TPR[5] += TPR
                            val_PPV[5] += PPV
                            cnt[5] += 1


                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        ################################################################
                        if task6_pool_image.num_imgs >= batch_size:
                            images = task6_pool_image.query(batch_size)
                            labels = task6_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task6_scale.pop(0)
                            #preds = model(images, torch.ones(batch_size).cuda()*6, scales)

                            now_task = torch.tensor(6)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[6] += F1
                            val_Dice[6] += DICE
                            val_TPR[6] += TPR
                            val_PPV[6] += PPV
                            cnt[6] += 1


                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        ################################################################

                    ####################################################################

                    if (task0_pool_image.num_imgs < batch_size) & (task0_pool_image.num_imgs > 0):
                            left_size = task0_pool_image.num_imgs
                            images = task0_pool_image.query(left_size)
                            labels = task0_pool_mask.query(left_size)
                            now_task = torch.tensor(0)
                            scales = torch.ones(batch_size).cuda()

                            if (left_size == 1):
                                images = images.repeat(4, 1, 1, 1)
                                labels = labels.repeat(4, 1, 1)
                                wts = wts.repeat(4, 1, 1)

                            if (left_size == 2):
                                images = images.repeat(2, 1, 1, 1)
                                labels = labels.repeat(2, 1, 1)
                                wts = wts.repeat(2, 1, 1)

                            if (left_size == 3):
                                last_image = torch.unsqueeze(images[-1], dim=0)
                                images = torch.cat((images, last_image), dim=0)

                                last_label = torch.unsqueeze(labels[-1], dim=0)
                                labels = torch.cat((labels, last_label), dim=0)

                                last_wts = torch.unsqueeze(wts[-1], dim=0)
                                wts = torch.cat((wts, last_wts), dim=0)
                            # for bi in range(len(scales)):
                            #     scales[bi] = task0_scale.pop(0)
                            #preds = model(images, torch.ones(left_size).cuda()*0, scales)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################
                            #now_preds = preds[:,1,...] > preds[:,0,...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())


                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = mask_to_box(images)

                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[0] += F1
                            val_Dice[0] += DICE
                            val_TPR[0] += TPR
                            val_PPV[0] += PPV
                            cnt[0] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())

                    if (task1_pool_image.num_imgs < batch_size) & (task1_pool_image.num_imgs > 0):
                            left_size = task1_pool_image.num_imgs
                            images = task1_pool_image.query(left_size)
                            labels = task1_pool_mask.query(left_size)
                            scales = torch.ones(batch_size).cuda()

                            if (left_size == 1):
                                images = images.repeat(4, 1, 1, 1)
                                labels = labels.repeat(4, 1, 1)
                                wts = wts.repeat(4, 1, 1)

                            if (left_size == 2):
                                images = images.repeat(2, 1, 1, 1)
                                labels = labels.repeat(2, 1, 1)
                                wts = wts.repeat(2, 1, 1)

                            if (left_size == 3):
                                last_image = torch.unsqueeze(images[-1], dim=0)
                                images = torch.cat((images, last_image), dim=0)

                                last_label = torch.unsqueeze(labels[-1], dim=0)
                                labels = torch.cat((labels, last_label), dim=0)

                                last_wts = torch.unsqueeze(wts[-1], dim=0)
                                wts = torch.cat((wts, last_wts), dim=0)
                            # for bi in range(len(scales)):
                            #     scales[bi] = task1_scale.pop(0)
                            #preds = model(images, torch.ones(left_size).cuda()*1, scales)

                            now_task = torch.tensor(1)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())


                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[1] += F1
                            val_Dice[1] += DICE
                            val_TPR[1] += TPR
                            val_PPV[1] += PPV
                            cnt[1] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())

                    if (task2_pool_image.num_imgs < batch_size) & (task2_pool_image.num_imgs > 0):
                            left_size = task2_pool_image.num_imgs
                            images = task2_pool_image.query(left_size)
                            labels = task2_pool_mask.query(left_size)
                            scales = torch.ones(batch_size).cuda()

                            if (left_size == 1):
                                images = images.repeat(4, 1, 1, 1)
                                labels = labels.repeat(4, 1, 1)
                                wts = wts.repeat(4, 1, 1)

                            if (left_size == 2):
                                images = images.repeat(2, 1, 1, 1)
                                labels = labels.repeat(2, 1, 1)
                                wts = wts.repeat(2, 1, 1)

                            if (left_size == 3):
                                last_image = torch.unsqueeze(images[-1], dim=0)
                                images = torch.cat((images, last_image), dim=0)

                                last_label = torch.unsqueeze(labels[-1], dim=0)
                                labels = torch.cat((labels, last_label), dim=0)

                                last_wts = torch.unsqueeze(wts[-1], dim=0)
                                wts = torch.cat((wts, last_wts), dim=0)
                            # for bi in range(len(scales)):
                            #     scales[bi] = task2_scale.pop(0)
                            #preds = model(images, torch.ones(left_size).cuda()*2, scales)

                            now_task = torch.tensor(2)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[2] += F1
                            val_Dice[2] += DICE
                            val_TPR[2] += TPR
                            val_PPV[2] += PPV
                            cnt[2] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                    if (task3_pool_image.num_imgs < batch_size) & (task3_pool_image.num_imgs > 0):
                            left_size = task3_pool_image.num_imgs
                            images = task3_pool_image.query(left_size)
                            labels = task3_pool_mask.query(left_size)
                            scales = torch.ones(batch_size).cuda()

                            if (left_size == 1):
                                images = images.repeat(4, 1, 1, 1)
                                labels = labels.repeat(4, 1, 1)
                                wts = wts.repeat(4, 1, 1)

                            if (left_size == 2):
                                images = images.repeat(2, 1, 1, 1)
                                labels = labels.repeat(2, 1, 1)
                                wts = wts.repeat(2, 1, 1)

                            if (left_size == 3):
                                last_image = torch.unsqueeze(images[-1], dim=0)
                                images = torch.cat((images, last_image), dim=0)

                                last_label = torch.unsqueeze(labels[-1], dim=0)
                                labels = torch.cat((labels, last_label), dim=0)

                                last_wts = torch.unsqueeze(wts[-1], dim=0)
                                wts = torch.cat((wts, last_wts), dim=0)
                            # for bi in range(len(scales)):
                            #     scales[bi] = task3_scale.pop(0)
                            #preds = model(images, torch.ones(left_size).cuda()*3, scales)

                            now_task = torch.tensor(3)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[3] += F1
                            val_Dice[3] += DICE
                            val_TPR[3] += TPR
                            val_PPV[3] += PPV
                            cnt[3] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                    if (task4_pool_image.num_imgs < batch_size) & (task4_pool_image.num_imgs > 0):
                        left_size = task4_pool_image.num_imgs
                        images = task4_pool_image.query(left_size)
                        labels = task4_pool_mask.query(left_size)
                        scales = torch.ones(batch_size).cuda()

                        if (left_size == 1):
                            images = images.repeat(4, 1, 1, 1)
                            labels = labels.repeat(4, 1, 1)
                            wts = wts.repeat(4, 1, 1)

                        if (left_size == 2):
                            images = images.repeat(2, 1, 1, 1)
                            labels = labels.repeat(2, 1, 1)
                            wts = wts.repeat(2, 1, 1)

                        if (left_size == 3):
                            last_image = torch.unsqueeze(images[-1], dim=0)
                            images = torch.cat((images, last_image), dim=0)

                            last_label = torch.unsqueeze(labels[-1], dim=0)
                            labels = torch.cat((labels, last_label), dim=0)

                            last_wts = torch.unsqueeze(wts[-1], dim=0)
                            wts = torch.cat((wts, last_wts), dim=0)
                        # for bi in range(len(scales)):
                        #     scales[bi] = task4_scale.pop(0)
                        #preds = model(images, torch.ones(left_size).cuda() * 4, scales)

                        now_task = torch.tensor(4)
                        ##################################################################################
                        #preds = preds[0]
                        preds = model(images)
                        ##################################################################################

                        #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds = torch.argmax(preds, 1) == now_task
                        now_preds_onehot = one_hot_3D(now_preds.long())

                        labels_onehot = one_hot_3D(labels.long())
                        rmin, rmax, cmin, cmax = mask_to_box(images)
                        F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                        val_F1[4] += F1
                        val_Dice[4] += DICE
                        val_TPR[4] += TPR
                        val_PPV[4] += PPV
                        cnt[4] += 1

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy())
                            plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy())

                    if (task5_pool_image.num_imgs < batch_size) & (task5_pool_image.num_imgs > 0):
                            left_size = task5_pool_image.num_imgs
                            images = task5_pool_image.query(left_size)
                            labels = task5_pool_mask.query(left_size)
                            scales = torch.ones(batch_size).cuda()

                            if (left_size == 1):
                                images = images.repeat(4, 1, 1, 1)
                                labels = labels.repeat(4, 1, 1)
                                wts = wts.repeat(4, 1, 1)

                            if (left_size == 2):
                                images = images.repeat(2, 1, 1, 1)
                                labels = labels.repeat(2, 1, 1)
                                wts = wts.repeat(2, 1, 1)

                            if (left_size == 3):
                                last_image = torch.unsqueeze(images[-1], dim=0)
                                images = torch.cat((images, last_image), dim=0)

                                last_label = torch.unsqueeze(labels[-1], dim=0)
                                labels = torch.cat((labels, last_label), dim=0)

                                last_wts = torch.unsqueeze(wts[-1], dim=0)
                                wts = torch.cat((wts, last_wts), dim=0)
                            # for bi in range(len(scales)):
                            #     scales[bi] = task5_scale.pop(0)
                            #preds = model(images, torch.ones(left_size).cuda()*5, scales)

                            now_task = torch.tensor(5)
                            ##################################################################################
                            #preds = preds[0]
                            preds = model(images)
                            ##################################################################################

                            #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = mask_to_box(images)
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[5] += F1
                            val_Dice[5] += DICE
                            val_TPR[5] += TPR
                            val_PPV[5] += PPV
                            cnt[5] += 1


                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                    if (task6_pool_image.num_imgs < batch_size) & (task6_pool_image.num_imgs > 0):
                        left_size = task6_pool_image.num_imgs
                        images = task6_pool_image.query(left_size)
                        labels = task6_pool_mask.query(left_size)
                        scales = torch.ones(batch_size).cuda()

                        if (left_size == 1):
                            images = images.repeat(4, 1, 1, 1)
                            labels = labels.repeat(4, 1, 1)
                            wts = wts.repeat(4, 1, 1)

                        if (left_size == 2):
                            images = images.repeat(2, 1, 1, 1)
                            labels = labels.repeat(2, 1, 1)
                            wts = wts.repeat(2, 1, 1)

                        if (left_size == 3):
                            last_image = torch.unsqueeze(images[-1], dim=0)
                            images = torch.cat((images, last_image), dim=0)

                            last_label = torch.unsqueeze(labels[-1], dim=0)
                            labels = torch.cat((labels, last_label), dim=0)

                            last_wts = torch.unsqueeze(wts[-1], dim=0)
                            wts = torch.cat((wts, last_wts), dim=0)
                        # for bi in range(len(scales)):
                        #     scales[bi] = task6_scale.pop(0)
                        #preds = model(images, torch.ones(left_size).cuda() * 6, scales)

                        now_task = torch.tensor(6)
                        ##################################################################################
                        #preds = preds[0]
                        preds = model(images)
                        ##################################################################################

                        #now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds = torch.argmax(preds, 1) == now_task
                        now_preds_onehot = one_hot_3D(now_preds.long())

                        labels_onehot = one_hot_3D(labels.long())
                        rmin, rmax, cmin, cmax = mask_to_box(images)
                        F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                        val_F1[6] += F1
                        val_Dice[6] += DICE
                        val_TPR[6] += TPR
                        val_PPV[6] += PPV
                        cnt[6] += 1

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy())
                            plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy())
                    ####################################################################
                    avg_val_F1 = val_F1 / cnt
                    avg_val_Dice = val_Dice / cnt
                    avg_val_TPR = val_TPR / cnt
                    avg_val_PPV = val_PPV / cnt

                    print('Validate \n 0dt_f1={:.4} 0dt_dsc={:.4} 0dt_tpr={:.4} 0dt_ppv={:.4}'
                          ' \n 1pt_f1={:.4} 1pt_dsc={:.4} 1pt_tpr={:.4} 1pt_ppv={:.4}\n'
                          ' \n 2cps_f1={:.4} 2cps_dsc={:.4} 2cps_tpr={:.4} 2cps_ppv={:.4}\n'
                          ' \n 3tf_f1={:.4} 3tf_dsc={:.4} 3tf_tpr={:.4} 3tf_ppv={:.4}\n'
                          ' \n 4vs_f1={:.4} 4vs_dsc={:.4} 4vs_tpr={:.4} 4vs_ppv={:.4}\n'
                          ' \n 5ptc_f1={:.4} 5ptc_dsc={:.4} 5ptc_tpr={:.4} 5ptc_ppv={:.4}\n'
                          ##########################################################################
                          ' \n 6ptc_f1={:.4} 6ptc_dsc={:.4} 6ptc_tpr={:.4} 6ptc_ppv={:.4}\n'
                          ######################################################################
                          .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item(),
                                  avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item(),
                                  avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item(),
                                  avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item(),
                                  avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item(),
                                  avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()
                                  #############################################################################################
                                  ,avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item()
                                  #############################################################################################

                                  ))

                df = pd.DataFrame(columns = ['task','F1','Dice','TPR','PPV'])
                df.loc[0] = ['0dt', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item()]
                df.loc[1] = ['1pt', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item()]
                df.loc[2] = ['2capsule', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item()]
                df.loc[3] = ['3tuft', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item()]
                df.loc[4] = ['4vessel', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item()]
                df.loc[5] = ['5ptc', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()]
                ################################################################################################################
                df.loc[6] = ['5ptc', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item()]
                ################################################################################################################
                df.to_csv(os.path.join(output_folder,'validation_result.csv'))


                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
