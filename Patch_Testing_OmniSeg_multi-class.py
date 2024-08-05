import argparse
import os, sys
import pandas as pd

sys.path.append("/Data/DoDNet/")
from skimage.transform import rescale, resize
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

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm
from torchvision import transforms

import skimage
import re

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
from MOTSDataset_2D_Patch_joint_csv_PTC import MOTSValDataSet as MOTSValDataSet_joint

#from unet2D_ns import UNet2D as UNet2D_ns

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool

# from unet import UNet

from model import Custom

def one_hot_2D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
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


def get_arguments(img, output_folder):

    parser = argparse.ArgumentParser(description="DeepLabV3")
    parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_train_scale_aug_patch')

    parser.add_argument("--valset_dir", type=str, default=img + '/data_list.csv')
    parser.add_argument("--output_dir", type=str, default=output_folder)

    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white_scale_multi-class_normalwhole_0907')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_multi-class_normalwhole_0907/MOTS_DynConv_fold1_with_white_scale_multi-class_normalwhole_0907_e85.pth')
    parser.add_argument("--best_epoch", type=int, default=85)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='256,256')
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

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.


def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        try:

            hausdorff, meansurfaceDistance = surfd(preds0, labels0)
            Val_HD += hausdorff
            Val_MSD += meansurfaceDistance

            Val_F1 += f1_score(preds1, labels1, average='macro')

        except:
            Val_DICE += 1.
            Val_F1 += 1.
            Val_HD += 0.
            Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


# def testing_40X(imgs_40X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
#     stride = int(patch_size / 4)
#     img_batch = torch.zeros((169, 3, patch_size, patch_size))
#     cnt = 0
#     for ki in range(13):
#         for kj in range(13):
#             start_x = ki * stride
#             start_y = kj * stride
#             end_x = start_x + patch_size
#             end_y = start_y + patch_size
#
#             img_batch[cnt] = imgs_40X[0, :, start_x:end_x, start_y:end_y]
#             cnt += 1
#
#     batch_num = int(169 / batch_size) + 1
#
#     preds_batch = torch.zeros((169, 2, patch_size, patch_size))
#     for bi in range(batch_num):
#         if bi != batch_num - 1:
#             preds_batch[bi * batch_size : (bi + 1) * batch_size] = model(img_batch[bi * batch_size : (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
#         else:
#             preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
#
#     big_preds = torch.zeros(2, imgs_40X.shape[2], imgs_40X.shape[3])
#
#     cnt = 0
#     for ki in range(13):
#         for kj in range(13):
#             start_x = ki * stride
#             start_y = kj * stride
#             end_x = start_x + patch_size
#             end_y = start_y + patch_size
#
#             big_preds[:, start_x:end_x, start_y:end_y] = big_preds[:, start_x:end_x, start_y:end_y] + preds_batch[cnt]
#             cnt += 1
#
#     big_img_resize = imgs[0]
#
#     prediction = (big_preds[1, ...] > big_preds[0, ...]).detach().numpy().astype(np.float32)
#
#     out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
#     img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
#
#     plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
#     plt.imsave(os.path.join(output_folder, 'Big_pred.png'), prediction, cmap=cm.gray)
#
#
# def testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
#
#     stride = int(patch_size / 4)
#     img_batch = torch.zeros((25, 3, patch_size, patch_size))
#     cnt = 0
#     for ki in range(5):
#         for kj in range(5):
#             start_x = ki * stride
#             start_y = kj * stride
#             end_x = start_x + patch_size
#             end_y = start_y + patch_size
#
#             img_batch[cnt] = imgs_10X[0, :, start_x:end_x, start_y:end_y]
#             cnt += 1
#
#     batch_num = int(25 / batch_size) + 1
#
#     preds_batch = torch.zeros((25, 2, patch_size, patch_size))
#     for bi in range(batch_num):
#         if bi != batch_num - 1:
#             preds_batch[bi * batch_size : (bi + 1) * batch_size] = model(img_batch[bi * batch_size : (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
#         else:
#             preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
#
#     big_preds = torch.zeros(2, imgs_10X.shape[2], imgs_10X.shape[3])
#
#     cnt = 0
#     for ki in range(5):
#         for kj in range(5):
#             start_x = ki * stride
#             start_y = kj * stride
#             end_x = start_x + patch_size
#             end_y = start_y + patch_size
#
#             big_preds[:, start_x:end_x, start_y:end_y] = big_preds[:, start_x:end_x, start_y:end_y] + preds_batch[cnt]
#             cnt += 1
#
#     resize_function = transforms.Resize(1024)
#     big_preds_resize = resize_function(big_preds)
#     big_img_resize = imgs[0]
#
#     prediction = (big_preds_resize[1, ...] > big_preds_resize[0, ...]).detach().numpy().astype(np.float32)
#
#     out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
#     img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
#
#     plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
#     plt.imsave(os.path.join(output_folder, 'Big_pred.png'),prediction, cmap=cm.gray)


def testing_40X(imgs_40X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
    stride = int(patch_size / 4)
    img_batch = torch.zeros((169, 3, patch_size, patch_size))
    cnt = 0
    for ki in range(13):
        for kj in range(13):
            start_x = ki * stride
            start_y = kj * stride
            end_x = start_x + patch_size
            end_y = start_y + patch_size

            img_batch[cnt] = imgs_40X[0, :, start_x:end_x, start_y:end_y]
            cnt += 1

    batch_num = int(169 / batch_size) + 1

    preds_batch = torch.zeros((169, 2, patch_size, patch_size))
    for bi in range(batch_num):
        if bi != batch_num - 1:
            preds_batch[bi * batch_size : (bi + 1) * batch_size] = model(img_batch[bi * batch_size : (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
        else:
            preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)


    prediction = torch.zeros(imgs_40X.shape[2], imgs_40X.shape[3])

    cnt = 0
    for ki in range(13):
        for kj in range(13):
            start_x = ki * stride
            start_y = kj * stride
            end_x = start_x + patch_size
            end_y = start_y + patch_size

            if ki !=0 and ki != 12 and kj !=0 and kj != 12:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x+64:end_x-64, start_y+64:end_y-64] = prediction[start_x+64:end_x-64, start_y+64:end_y-64] + now_prediction[64:-64,64:-64]
                cnt += 1

            else:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x:end_x, start_y:end_y] = prediction[start_x:end_x, start_y:end_y] + now_prediction
                cnt += 1


    prediction[256:-256, 256:-256] = prediction[256:-256, 256:-256] * (prediction[256:-256, 256:-256] > 1)
    prediction = prediction > 0

    big_img_resize = imgs[0]

    # prediction = (big_preds[1, ...] > big_preds[0, ...]).detach().numpy().astype(np.float32)

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'), prediction, cmap=cm.gray)


def testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):

    stride = int(patch_size / 4)
    img_batch = torch.zeros((25, 3, patch_size, patch_size))
    cnt = 0
    for ki in range(5):
        for kj in range(5):
            start_x = ki * stride
            start_y = kj * stride
            end_x = start_x + patch_size
            end_y = start_y + patch_size

            img_batch[cnt] = imgs_10X[0, :, start_x:end_x, start_y:end_y]
            cnt += 1

    batch_num = int(25 / batch_size) + 1

    preds_batch = torch.zeros((25, 2, patch_size, patch_size))
    for bi in range(batch_num):
        if bi != batch_num - 1:
            preds_batch[bi * batch_size : (bi + 1) * batch_size] = model(img_batch[bi * batch_size : (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
        else:
            preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)


    prediction = torch.zeros(imgs_10X.shape[2], imgs_10X.shape[3])

    cnt = 0
    for ki in range(5):
        for kj in range(5):
            start_x = ki * stride
            start_y = kj * stride
            end_x = start_x + patch_size
            end_y = start_y + patch_size

            if ki != 0 and ki != 4 and kj != 0 and kj != 4:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x + 64:end_x - 64, start_y + 64:end_y - 64] = prediction[start_x + 64:end_x - 64,
                                                                               start_y + 64:end_y - 64] + now_prediction[
                                                                                                          64:-64,
                                                                                                          64:-64]
                cnt += 1

            else:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x:end_x, start_y:end_y] = prediction[start_x:end_x, start_y:end_y] + now_prediction
                cnt += 1

    prediction[64:-64, 64:-64] = prediction[64:-64, 64:-64] * (prediction[64:-64, 64:-64] > 1)
    prediction = prediction > 0

    resize_function = transforms.Resize(1024)
    prediction = resize_function(prediction.unsqueeze(0)).squeeze(0)
    big_img_resize = imgs[0]

    # prediction[256:-256, 256:-256] = prediction[256:-256, 256:-256] * (prediction[256:-256, 256:-256] > 1)
    # prediction = prediction > 0


    # prediction = (big_preds_resize[1, ...] > big_preds_resize[0, ...]).detach().numpy().astype(np.float32)

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'),prediction, cmap=cm.gray)

def testing_5X(imgs_5X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
    batch = imgs_5X.repeat(4,1,1,1)
    preds_batch = model(batch.cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)

    big_preds = preds_batch
    resize_function = transforms.Resize(1024)
    big_preds_resize = resize_function(big_preds)
    big_img_resize = imgs[0]

    prediction = (big_preds_resize[0,1, ...] > big_preds_resize[0,0, ...]).detach().cpu().numpy().astype(np.float32)

    if prediction.sum() > 0.2 * prediction.shape[0] * prediction.shape[1]:
        prediction = np.zeros((prediction.shape))

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'),prediction, cmap=cm.gray)

def main(img, output_dir, case_name):
    """Create the model and start the training."""

    output_folder = os.path.join(output_dir, case_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    parser = get_arguments(img, output_folder)
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
        model = Custom(input_ch = 3, output_ch = 7, modelDim=2)
        check_wo_gpu = 0

        print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

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

        # load checkpoint...
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
            weights = [1., 10.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights).to(device)
            #criterion2 = FocalLoss2d(weight=class_weights).to(device)

        else:
            weights = [1., 10.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights)
            #criterion2 = FocalLoss2d(weight=class_weights)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        valloader = DataLoader(
            MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=1,shuffle=True,num_workers=num_worker)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        batch_size = args.batch_size

        model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()

        task1_pool_image = ImagePool(8)
        task1_pool_mask = ImagePool(8)
        task1_scale = []
        task1_name = []
        task2_pool_image = ImagePool(8)
        task2_pool_mask = ImagePool(8)
        task2_scale = []
        task2_name = []


        with torch.no_grad():
            for iter, batch in enumerate(valloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = [batch[3][0].split('-')[-2] + '-' + batch[3][0].split('-')[-1]]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                for ki in range(len(imgs)):
                    now_task = t_ids[ki]
                    if now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                        task1_name.append((volumeName[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                        task2_name.append((volumeName[ki]))

                optimizer.zero_grad()

                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                        filename.append(task1_name.pop(0))

                    transform = transforms.Resize(512)
                    images_resize = transform(images)

                    preds_resize = (torch.argmax(model(images_resize, torch.ones(batch_size).cuda() * 2, scales)[0], 1) == 1).float()

                    transform_back = transforms.Resize(1024)
                    preds = transform_back(preds_resize)

                    for pi in range(len(images)):
                        prediction = preds[pi]
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds.png'),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)


                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
                        filename.append(task2_name.pop(0))

                    transform = transforms.Resize(256)
                    images_resize = transform(images)

                    preds_resize = (torch.argmax(model(images_resize, torch.ones(batch_size).cuda() * 2, scales)[0], 1) == 2).float()

                    transform_back = transforms.Resize(256)
                    preds = transform_back(preds_resize)

                    for pi in range(len(images)):
                        prediction = preds[pi]
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds.png'),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

            if task1_pool_image.num_imgs != 0:
                now_batch_size = task1_pool_image.num_imgs
                images = task1_pool_image.query(now_batch_size)
                labels = task1_pool_mask.query(now_batch_size)
                scales = torch.ones(now_batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                    filename.append(task1_name.pop(0))

                transform = transforms.Resize(256)
                images_resize = transform(images)

                preds_resize = (torch.argmax(model(images_resize, torch.ones(batch_size).cuda() * 2, scales)[0], 1) == 1).float()

                transform_back = transforms.Resize(256)
                preds = transform_back(preds_resize)

                for pi in range(len(images)):
                    prediction = preds[pi]
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_preds.png'),
                               prediction.detach().cpu().numpy(), cmap=cm.gray)


            if task2_pool_image.num_imgs != 0:
                now_batch_size = task2_pool_image.num_imgs
                images = task2_pool_image.query(now_batch_size)
                labels = task2_pool_mask.query(now_batch_size)
                scales = torch.ones(now_batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
                    filename.append(task2_name.pop(0))

                transform = transforms.Resize(512)
                preds_resize = transform(images)

                preds_resize = (torch.argmax(model(images_resize, torch.ones(batch_size).cuda() * 2, scales)[0], 1) == 2).float()

                transform_back = transforms.Resize(1024)
                preds = transform_back(preds_resize)

                for pi in range(len(images)):
                    prediction = preds[pi]
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_preds.png'),
                               prediction.detach().cpu().numpy(), cmap=cm.gray)


    end = timeit.default_timer()
    print(end - start, 'seconds')


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == '__main__':

    data_dir = '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_scalecap/V11M25-279'
    output_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_Multi-Kidney_cap256to256/V11M25-279'

    cases = glob.glob(os.path.join(data_dir,'*'))
    cases.sort(key=natural_keys)

    for now_case in cases:
        case_name = os.path.basename(now_case)
        # for img in images:
        main(now_case, output_dir, case_name)