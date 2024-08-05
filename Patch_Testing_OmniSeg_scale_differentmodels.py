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

from unet2D_Dodnet_scale import UNet2D as unet2D_scale

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

    'scale no2'
    # parser.add_argument("--snapshot_dir", type=str, default='/Data/DoDNet/MIDL/MICCAI_psuedo/snapshots_2D/fold1_with_white_scale_psuedo_all_withsemi_0.25_0.25_sublabel_normalwhole_0217')
    # parser.add_argument("--reload_path", type=str, default='/Data/DoDNet/MIDL/MICCAI_psuedo/snapshots_2D/fold1_with_white_scale_psuedo_all_withsemi_0.25_0.25_sublabel_normalwhole_0217/MOTS_DynConv_fold1_with_white_scale_psuedo_all_withsemi_0.25_0.25_sublabel_normalwhole_0217_e87.pth')
    # parser.add_argument("--best_epoch", type=int, default=87)

    'scale no4'
    # parser.add_argument("--snapshot_dir", type=str, default='/Data/DoDNet/MIDL/MICCAI_supervise/snapshots_2D/fold1_with_white_scale_supervised_fullydata_normalwhole_0208')
    # parser.add_argument("--reload_path", type=str, default='/Data/DoDNet/MIDL/MICCAI_supervise/snapshots_2D/fold1_with_white_scale_supervised_fullydata_normalwhole_0208/MOTS_DynConv_fold1_with_white_scale_supervised_fullydata_normalwhole_0208_e90.pth')
    # parser.add_argument("--best_epoch", type=int, default=90)

    # 'scale no5'
    # parser.add_argument("--snapshot_dir", type=str, default='/Data/DoDNet/MIDL/MICCAI_psuedo/snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_withsemi_0.1_0.1_normalwhole_0410')
    # parser.add_argument("--reload_path", type=str, default='/Data/DoDNet/MIDL/MICCAI_psuedo/snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_withsemi_0.1_0.1_normalwhole_0410/MOTS_DynConv_fold1_with_white_scale_allpsuedo_allMatching_withsemi_0.1_0.1_normalwhole_0410_e89.pth')
    # parser.add_argument("--best_epoch", type=int, default=89)

    'scale no6'
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.1_0.1_normalwhole_0907/')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.1_0.1_normalwhole_0907/MOTS_DynConv_fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.1_0.1_normalwhole_0907_e95.pth')
    parser.add_argument("--best_epoch", type=int, default=95)

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
        model = unet2D_scale(num_classes=args.num_classes, weight_std = False)
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
                        scales[bi] = task1_scale.pop(0)
                        filename.append(task1_name.pop(0))

                    transform = transforms.Resize(512)
                    images_resize = transform(images)

                    preds_resize,_ = model(images_resize, torch.ones(batch_size).cuda() * 1, scales)

                    transform_back = transforms.Resize(1024)
                    preds = transform_back(preds_resize)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
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
                        scales[bi] = task2_scale.pop(0)
                        filename.append(task2_name.pop(0))

                    transform = transforms.Resize(256)
                    images_resize = transform(images)

                    preds_resize,_ = model(images_resize, torch.ones(batch_size).cuda() * 2, scales)

                    transform_back = transforms.Resize(256)
                    preds = transform_back(preds_resize)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
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
                    scales[bi] = task1_scale.pop(0)
                    filename.append(task1_name.pop(0))

                transform = transforms.Resize(512)
                images_resize = transform(images)

                preds_resize,_ = model(images_resize, torch.ones(now_batch_size).cuda() * 1, scales)

                transform_back = transforms.Resize(1024)
                preds = transform_back(preds_resize)

                for pi in range(len(images)):
                    prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
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
                    scales[bi] = task2_scale.pop(0)
                    filename.append(task2_name.pop(0))

                transform = transforms.Resize(256)
                images_resize = transform(images)

                preds_resize,_ = model(images_resize, torch.ones(now_batch_size).cuda() * 2, scales)

                transform_back = transforms.Resize(256)
                preds = transform_back(preds_resize)

                for pi in range(len(images)):
                    prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
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

    # data_dir = '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_pt/V11M25-279'
    # output_dir = '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_OmniSeg_scale_No6_cap256to256/V11M25-279'

    # data_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_scalecap/V11M25-279'
    # output_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279'

    data_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_scalecap/V11M25-279'
    output_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279'

    cases = glob.glob(os.path.join(data_dir,'*'))
    cases.sort(key=natural_keys)

    for now_case in cases:
        case_name = os.path.basename(now_case)
        # for img in images:
        main(now_case, output_dir, case_name)