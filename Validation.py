import argparse
import os, sys
import pandas as pd

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
from AttentionUnet3Plus import AttentionUnet3Plus
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# from unet2D_Dodnet_scale_Attention_fullscale import UNet2DAttentionFullscale as UNet2D_scaleAttentionFullscale
# from unet2D_Dodnet_scale_Attention import UNet2DAttention as UNet2D_scaleAttention
from network import deepv3
from scipy.ndimage import distance_transform_edt,binary_erosion,generate_binary_structure
from matplotlib import cm

import skimage

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
#from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet_joint
# from MOTSDataset_2D_Patch_supervise_normal_csv_512 import MOTSValDataSet as MOTSValDataSet_joint
from MOTSDataset_2D_Patch_supervise_normal_csv_512 import MOTSValDataSet_512_center as MOTSValDataSet_joint
from unet2D_ns import UNet2D as UNet2D_ns
from unet2D_Dodnet_scale_for_10_class import UNet2D as UNet2D_scale
#from unet2D_Dodnet_scale_v2 import UNet2D as UNet2D_scale
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
from unet_v0 import UNet
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from util.image_pool import ImagePool


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



def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")

    #parser.add_argument("--valset_dir", type=str, default='KI_data_testingset_demo/data_list.csv')
    #parser.add_argument("--valset_dir", type=str, default='./data/HC_data_patch/ddd/data_list.csv')
    #parser.add_argument("--valset_dir", type=str, default='./data/Human_Patches/val/data_list.csv')
    parser.add_argument("--valset_dir", type=str, default='./data/omniseg-separated9/test/data_list.csv')
    #parser.add_argument("--valset_dir", type=str, default='./data/Mice_Patches/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/Omniseg-true/our_model_Deeplab_for_6class_2GPU/')
    #parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/MOTS_DynConv_fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907_e74.pth')
    #parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white/')
    #parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_final_e199.pth')
    #parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/For_Hum/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e136.pth')
    parser.add_argument("--reload_path", type=str,
                        default="/root/autodl-tmp/Omni-Seg_revision/snapshots_2D/Omniseg-true/our_model_Deeplab_for_6class_2GPU/MOTS_DynConv_our_model_Deeplab_for_6class_2GPU_e199.pth")
    #parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/For_Mice/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e176.pth')

    ##################################################################################################################################
    #parser.add_argument("--reload_path", type=str,
    #                    default='snapshots_2D/final/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e50.pth')

    ###################################################################################################################################
    #parser.add_argument("--best_epoch", type=int, default=74)
    #parser.add_argument("--best_epoch", type=int, default=51)
    parser.add_argument("--best_epoch", type=int, default=199)
    #parser.add_argument("--best_epoch", type=int, default=176)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='./data/omniseg-separated9/train/data_list.csv')
    #parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    #parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--val_list", type=str, default='./data/omniseg-separated9/train/test/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=2)
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
    conn = generate_binary_structure(input1.ndim, connectivity)

    S = input1 - binary_erosion(input1, conn)
    Sprime = input2 - binary_erosion(input2, conn)

    print("S:", np.sum(S), "Sprime:", np.sum(Sprime))  # 查看S和Sprime中的非零元素数

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))

    dta = distance_transform_edt(~S, sampling)
    dtb = distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    print("sds:", sds)

    if sds.size == 0:
        return float('nan'), float('nan')  # 返回NaN或其他合适的值表示无法计算

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

        # try:
        ###############################################################################
        hausdorff, meansurfaceDistance = surfd(preds0, labels0)
        print(hausdorff)
        Val_HD += hausdorff
        print(meansurfaceDistance)
        Val_MSD += meansurfaceDistance
        ##############################################################################3

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # except:
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_HD += 0.
        #     Val_MSD += 0.

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


def main():
    """Create the model and start the training."""
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
        #model = UNet2D_ns(num_classes=args.num_classes, weight_std = False)
        # model = UNet2D_scale(num_classes=args.num_classes, weight_std=False)
        # model = UNet(n_classes=6, in_channels= 3, padding=True, depth=5, wf = 4, up_mode='upconv',
        #              batch_norm=True)
        # model = UNet2D_scaleAttention(num_classes=6, weight_std=False)  # 消融实验
        model = deepv3.DeepV3R50(2, criterion=criterion)
        # model = UNet2D_scaleAttentionFullscale(n_classes=6, weight_std=False)
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
        base_path = args.reload_path
        for epoch_i in range(199, 200):
            # load checkpoint...
            args.reload_path = base_path.replace('_e199.pth', '_e%s.pth' % (str(epoch_i)))
            args.best_epoch = epoch_i
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


            if not os.path.exists(args.snapshot_dir):
                os.makedirs(args.snapshot_dir)

            edge_weight = args.edge_weight

            num_worker = 8

            valloader = DataLoader(
                MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                               crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                               edge_weight=edge_weight),batch_size=1,shuffle=False,num_workers=num_worker)

            all_tr_loss = []
            all_va_loss = []
            train_loss_MA = None
            val_loss_MA = None

            val_best_loss = 999999
            batch_size = args.batch_size
            # for epoch in range(0,args.num_epochs):

            model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

            model.eval()
            task0_pool_image = ImagePool(8)
            task0_pool_mask = ImagePool(8)
            task0_scale = []
            task0_name = []
            task1_pool_image = ImagePool(8)
            task1_pool_mask = ImagePool(8)
            task1_scale = []
            task1_name = []
            task2_pool_image = ImagePool(8)
            task2_pool_mask = ImagePool(8)
            task2_scale = []
            task2_name = []
            task3_pool_image = ImagePool(8)
            task3_pool_mask = ImagePool(8)
            task3_scale = []
            task3_name = []
            task4_pool_image = ImagePool(8)
            task4_pool_mask = ImagePool(8)
            task4_scale = []
            task4_name = []
            task5_pool_image = ImagePool(8)
            task5_pool_mask = ImagePool(8)
            task5_scale = []
            task5_name = []
            ###################################################################
            # task6_pool_image = ImagePool(8)
            # task6_pool_mask = ImagePool(8)
            # task6_scale = []
            # task6_name = []
            #
            # task7_pool_image = ImagePool(8)
            # task7_pool_mask = ImagePool(8)
            # task7_scale = []
            # task7_name = []
            #
            # task8_pool_image = ImagePool(8)
            # task8_pool_mask = ImagePool(8)
            # task8_scale = []
            # task8_name = []
            #
            # task9_pool_image = ImagePool(8)
            # task9_pool_mask = ImagePool(8)
            # task9_scale = []
            # task9_name = []
            ###################################################################

            # val_loss = np.zeros((6))
            # val_F1 = np.zeros((6))
            # val_Dice = np.zeros((6))
            # val_HD = np.zeros((6))
            # val_MSD = np.zeros((6))
            # cnt = np.zeros((6))
            ####################################################################
            val_loss = np.zeros((6))
            val_F1 = np.zeros((6))
            val_Dice = np.zeros((6))
            val_HD = np.zeros((6))
            val_MSD = np.zeros((6))
            cnt = np.zeros((6))
            opt = np.zeros((6))

            ####################################################################
            single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            single_df_2 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            single_df_3 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            single_df_4 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            single_df_5 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

            #########################################################################
            # single_df_6 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            # single_df_7 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            # single_df_8 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            # single_df_9 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
            ################################################################################

            with torch.no_grad():
                for iter, batch in enumerate(valloader):

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
                            task0_name.append((volumeName[ki]))
                        elif now_task == 1:
                            task1_pool_image.add(imgs[ki].unsqueeze(0))
                            task1_pool_mask.add(lbls[ki].unsqueeze(0))
                            task1_scale.append((s_ids[ki]))
                            task1_name.append((volumeName[ki]))
                        elif now_task == 2:
                            task2_pool_image.add(imgs[ki].unsqueeze(0))
                            task2_pool_mask.add(lbls[ki].unsqueeze(0))
                            task2_scale.append((s_ids[ki]))
                            task2_name.append((volumeName[ki]))
                        elif now_task == 3:
                            task3_pool_image.add(imgs[ki].unsqueeze(0))
                            task3_pool_mask.add(lbls[ki].unsqueeze(0))
                            task3_scale.append((s_ids[ki]))
                            task3_name.append((volumeName[ki]))
                        elif now_task == 4:
                            task4_pool_image.add(imgs[ki].unsqueeze(0))
                            task4_pool_mask.add(lbls[ki].unsqueeze(0))
                            task4_scale.append((s_ids[ki]))
                            task4_name.append((volumeName[ki]))
                        elif now_task == 5:
                            task5_pool_image.add(imgs[ki].unsqueeze(0))
                            task5_pool_mask.add(lbls[ki].unsqueeze(0))
                            task5_scale.append((s_ids[ki]))
                            task5_name.append((volumeName[ki]))

                        #########################################################
                        elif now_task == 6:
                            task6_pool_image.add(imgs[ki].unsqueeze(0))
                            task6_pool_mask.add(lbls[ki].unsqueeze(0))
                            task6_scale.append((s_ids[ki]))
                            task6_name.append((volumeName[ki]))

                        elif now_task == 7:
                            task7_pool_image.add(imgs[ki].unsqueeze(0))
                            task7_pool_mask.add(lbls[ki].unsqueeze(0))
                            task7_scale.append((s_ids[ki]))
                            task7_name.append((volumeName[ki]))

                        elif now_task == 8:
                            task8_pool_image.add(imgs[ki].unsqueeze(0))
                            task8_pool_mask.add(lbls[ki].unsqueeze(0))
                            task8_scale.append((s_ids[ki]))
                            task8_name.append((volumeName[ki]))

                        elif now_task == 9:
                            task9_pool_image.add(imgs[ki].unsqueeze(0))
                            task9_pool_mask.add(lbls[ki].unsqueeze(0))
                            task9_scale.append((s_ids[ki]))
                            task9_name.append((volumeName[ki]))

                        ########################################################3


                    #output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/fold1_with_white','/Data/DoDNet/MIDL/MIDL_github/testing_%s' % (args.validsetname)), str(args.best_epoch))

                    ###For Human
                    output_folder = os.path.join('/data2/DoDNet/MIDL/Test_for_MH2H_Omniseg_10class/Validation_%s' % (args.validsetname), str(args.best_epoch))
                    ###For Mice
                    # output_folder = os.path.join('/data2/DoDNet/MIDL/MIDL_github/For_Mice/testing_%s' % (args.validsetname),
                    #                              str(args.best_epoch))
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    optimizer.zero_grad()

                    #print("task0", task0_pool_image.num_imgs)




                    if task0_pool_image.num_imgs >= batch_size:
                        images = task0_pool_image.query(batch_size)
                        labels = task0_pool_mask.query(batch_size)
                        now_task = torch.tensor(0)
                        scales = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                            filename.append(task0_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 0, scales)
                        # preds= model(images, torch.ones(batch_size).cuda() * 0, scales)
                        preds = model(images)

                        now_preds = preds[:,1,...] > preds[:,0,...]
                        now_preds_onehot = one_hot_2D(now_preds.long())

                        labels_onehot = one_hot_2D(labels.long())

                        rmin, rmax, cmin, cmax = mask_to_box(images)

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_0)
                            single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[0] += F1
                                val_Dice[0] += DICE
                                cnt[0] += 1 
                                continue
                            val_F1[0] += F1
                            val_Dice[0] += DICE
                            val_HD[0] += HD
                            val_MSD[0] += MSD
                            cnt[0] += 1
                            opt[0] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task0_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)
                    if task1_pool_image.num_imgs >= batch_size:
                        images = task1_pool_image.query(batch_size)
                        labels = task1_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                            filename.append(task1_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 1, scales)
                        preds = model(images)
                        now_task = torch.tensor(1)

                        now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds_onehot = one_hot_2D(now_preds.long())
                        labels_onehot = one_hot_2D(labels.long())
                        rmin, rmax, cmin, cmax = mask_to_box(images)


                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_1)
                            single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[1] += F1
                                val_Dice[1] += DICE
                                cnt[1] += 1 
                                continue
                            val_F1[1] += F1
                            val_Dice[1] += DICE
                            val_HD[1] += HD
                            val_MSD[1] += MSD
                            cnt[1] += 1
                            opt[1] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task1_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)

                    if task2_pool_image.num_imgs >= batch_size:
                        images = task2_pool_image.query(batch_size)
                        labels = task2_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
                            filename.append(task2_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 2, scales)
                        preds = model(images)
                        now_task = torch.tensor(2)

                        now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds_onehot = one_hot_2D(now_preds.long())

                        labels_onehot = one_hot_2D(labels.long())
                        rmin, rmax, cmin, cmax = mask_to_box(images)

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_2)
                            single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[2] += F1
                                val_Dice[2] += DICE
                                cnt[2] += 1 
                                continue
                            val_F1[2] += F1
                            val_Dice[2] += DICE
                            val_HD[2] += HD
                            val_MSD[2] += MSD
                            cnt[2] += 1
                            opt[2] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task2_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)

                    if task3_pool_image.num_imgs >= batch_size:
                        images = task3_pool_image.query(batch_size)
                        labels = task3_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task3_scale.pop(0)
                            filename.append(task3_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 3, scales)
                        preds = model(images)
                        now_task = torch.tensor(3)

                        now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds_onehot = one_hot_2D(now_preds.long())

                        labels_onehot = one_hot_2D(labels.long())

                        rmin, rmax, cmin, cmax = mask_to_box(images)

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_3)
                            single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[3] += F1
                                val_Dice[3] += DICE
                                cnt[3] += 1 
                                continue
                            val_F1[3] += F1
                            val_Dice[3] += DICE
                            val_HD[3] += HD
                            val_MSD[3] += MSD
                            cnt[3] += 1
                            opt[3] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task3_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)
                    if task4_pool_image.num_imgs >= batch_size:
                        images = task4_pool_image.query(batch_size)
                        labels = task4_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        filename = []

                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task4_scale.pop(0)
                            filename.append(task4_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 4, scales)
                        preds = model(images)
                        now_task = torch.tensor(4)

                        now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds_onehot = one_hot_2D(now_preds.long())

                        labels_onehot = one_hot_2D(labels.long())

                        rmin, rmax, cmin, cmax = mask_to_box(images)

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_4)
                            single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[4] += F1
                                val_Dice[4] += DICE
                                cnt[4] += 1 
                                continue
                            val_F1[4] += F1
                            val_Dice[4] += DICE
                            val_HD[4] += HD
                            val_MSD[4] += MSD
                            cnt[4] += 1
                            opt[4] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task4_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)
                    if task5_pool_image.num_imgs >= batch_size:
                        images = task5_pool_image.query(batch_size)
                        labels = task5_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[int(len(scales) - 1 - bi)] = task5_scale.pop(0)
                            filename.append(task5_name.pop(0))

                        # preds = model(images, torch.ones(batch_size).cuda() * 5, scales)
                        preds = model(images)
                        now_task = torch.tensor(5)

                        now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        now_preds_onehot = one_hot_2D(now_preds.long())

                        labels_onehot = one_hot_2D(labels.long())

                        rmin, rmax, cmin, cmax = mask_to_box(images)

                        for pi in range(len(images)):
                            prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                       img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                       prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                             rmin, rmax, cmin, cmax)
                            row = len(single_df_5)
                            single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                            if np.isnan(HD) or np.isnan(MSD):
                                val_F1[5] += F1
                                val_Dice[5] += DICE
                                cnt[5] += 1 
                                continue
                            val_F1[5] += F1
                            val_Dice[5] += DICE
                            val_HD[5] += HD
                            val_MSD[5] += MSD
                            cnt[5] += 1
                            opt[5] += 1
                            if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                                print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task5_pool_image.num_imgs >= batch_size")
                                print("val_F1:", val_F1)
                                print("val_Dice:", val_Dice)
                                print("val_HD:", val_HD)
                                print("val_MSD:", val_MSD)


                    ###############################################################################


                if (task1_pool_image.num_imgs < batch_size) & (task1_pool_image.num_imgs >0):
                    left_size = task1_pool_image.num_imgs
                    images = task1_pool_image.query(left_size)
                    labels = task1_pool_mask.query(left_size)
                    scales = torch.ones(left_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                        filename.append(task1_name.pop(0))

                    # preds = model(images, torch.ones(left_size).cuda() * 1, scales)
                    preds = model(images)
                    now_task = torch.tensor(1)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())
                    labels_onehot = one_hot_2D(labels.long())
                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)
                        row = len(single_df_1)
                        single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        if np.isnan(HD) or np.isnan(MSD):
                            val_F1[1] += F1
                            val_Dice[1] += DICE
                            cnt[1] += 1 
                            continue
                        val_F1[1] += F1
                        val_Dice[1] += DICE
                        val_HD[1] += HD
                        val_MSD[1] += MSD
                        cnt[1] += 1
                        opt[1] += 1
                        if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                            print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task1_pool_image.num_imgs < batch_size")
                            print("val_F1:", val_F1)
                            print("val_Dice:", val_Dice)
                            print("val_HD:", val_HD)
                            print("val_MSD:", val_MSD)
                if (task2_pool_image.num_imgs < batch_size) & (task2_pool_image.num_imgs >0):
                    left_size = task2_pool_image.num_imgs
                    images = task2_pool_image.query(left_size)
                    labels = task2_pool_mask.query(left_size)
                    scales = torch.ones(left_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
                        filename.append(task2_name.pop(0))

                    # preds = model(images, torch.ones(left_size).cuda() * 2, scales)
                    preds = model(images)
                    now_task = torch.tensor(2)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())

                    labels_onehot = one_hot_2D(labels.long())
                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)
                        row = len(single_df_2)
                        single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        if np.isnan(HD) or np.isnan(MSD):
                            val_F1[2] += F1
                            val_Dice[2] += DICE
                            cnt[2] += 1 
                            continue
                        val_F1[2] += F1
                        val_Dice[2] += DICE
                        val_HD[2] += HD
                        val_MSD[2] += MSD
                        cnt[2] += 1
                        opt[2] += 1
                        if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                            print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task2_pool_image.num_imgs < batch_size")
                            print("val_F1:", val_F1)
                            print("val_Dice:", val_Dice)
                            print("val_HD:", val_HD)
                            print("val_MSD:", val_MSD)
                if (task3_pool_image.num_imgs < batch_size) & (task3_pool_image.num_imgs >0):
                    left_size = task3_pool_image.num_imgs
                    images = task3_pool_image.query(left_size)
                    labels = task3_pool_mask.query(left_size)
                    scales = torch.ones(left_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task3_scale.pop(0)
                        filename.append(task3_name.pop(0))

                    # preds = model(images, torch.ones(left_size).cuda() * 3, scales)
                    preds = model(images)
                    now_task = torch.tensor(3)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)
                        row = len(single_df_3)
                        single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        if np.isnan(HD) or np.isnan(MSD):
                            val_F1[3] += F1
                            val_Dice[3] += DICE
                            cnt[3] += 1 
                            continue
                        val_F1[3] += F1
                        val_Dice[3] += DICE
                        val_HD[3] += HD
                        val_MSD[3] += MSD
                        cnt[3] += 1
                        opt[3] += 1
                        if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                            print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task3_pool_image.num_imgs < batch_size")
                            print("val_F1:", val_F1)
                            print("val_Dice:", val_Dice)
                            print("val_HD:", val_HD)
                            print("val_MSD:", val_MSD)
                if (task4_pool_image.num_imgs < batch_size) & (task4_pool_image.num_imgs >0):
                    left_size = task4_pool_image.num_imgs
                    images = task4_pool_image.query(left_size)
                    labels = task4_pool_mask.query(left_size)
                    scales = torch.ones(left_size).cuda()
                    filename = []

                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task4_scale.pop(0)
                        filename.append(task4_name.pop(0))

                    # preds = model(images, torch.ones(left_size).cuda() * 4, scales)
                    preds = model(images)
                    now_task = torch.tensor(4)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)
                        row = len(single_df_4)
                        single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        if np.isnan(HD) or np.isnan(MSD):
                            val_F1[4] += F1
                            val_Dice[4] += DICE
                            cnt[4] += 1 
                            continue
                        val_F1[4] += F1
                        val_Dice[4] += DICE
                        val_HD[4] += HD
                        val_MSD[4] += MSD
                        cnt[4] += 1
                        opt[4] += 1
                        if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                            print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task4_pool_image.num_imgs < batch_size")
                            print("val_F1:", val_F1)
                            print("val_Dice:", val_Dice)
                            print("val_HD:", val_HD)
                            print("val_MSD:", val_MSD)
                if (task5_pool_image.num_imgs < batch_size) & (task5_pool_image.num_imgs >0):
                    left_size = task5_pool_image.num_imgs
                    images = task5_pool_image.query(left_size)
                    labels = task5_pool_mask.query(left_size)
                    scales = torch.ones(left_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task5_scale.pop(0)
                        filename.append(task5_name.pop(0))

                    # preds = model(images, torch.ones(left_size).cuda() * 5, scales)
                    preds = model(images)
                    now_task = torch.tensor(5)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)
                        row = len(single_df_5)
                        single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        if np.isnan(HD) or np.isnan(MSD):
                            val_F1[5] += F1
                            val_Dice[5] += DICE
                            cnt[5] += 1

                            continue
                        val_F1[5] += F1
                        val_Dice[5] += DICE
                        val_HD[5] += HD
                        val_MSD[5] += MSD
                        cnt[5] += 1
                        opt[5] += 1
                        if np.isnan(val_HD[:6]).all() and np.isnan(val_MSD[:6]).all():
                            print(f"所有 val_HD 和 val_MSD 的值都是 NaN, task5_pool_image.num_imgs < batch_size")
                            print("val_F1:", val_F1)
                            print("val_Dice:", val_Dice)
                            print("val_HD:", val_HD)
                            print("val_MSD:", val_MSD)
            print(val_F1,val_Dice,val_HD,val_MSD)
            avg_val_F1 = val_F1 / cnt
            avg_val_Dice = val_Dice / cnt
            avg_val_HD = val_HD / opt
            avg_val_MSD = val_MSD / opt

            print('Validate \n 0Luman_f1={:.4} 0Luman_dsc={:.4} 0Luman_hd={:.4} 0Luman_msd={:.4}'
                  ' \n 1_0_Tunica_intima_f1={:.4} 1_0_Tunica_intima_dsc={:.4} 1_0_Tunica_intima_hd={:.4} 1_0_Tunica_intima_msd={:.4}\n'
                  ' \n 2_0_Tunica_media_f1={:.4} 2_0_Tunica_media_dsc={:.4} 2_0_Tunica_media_hd={:.4} 2_0_Tunica_media_msd={:.4}\n'
                  ' \n 3_0_Artery_f1={:.4} 3_0_Artery_dsc={:.4} 3_0_Artery_hd={:.4} 3_0_Artery_msd={:.4}\n'
                  ' \n 4_0_Artery_wall_f1={:.4} 4_0_Artery_wall_dsc={:.4} 4_0_Artery_wall_hd={:.4} 4_0_Artery_wall_msd={:.4}\n'
                  ' \n 5_0_Hyline_f1={:.4} 5_0_Hyline_dsc={:.4} 5_0_Hyline_hd={:.4} 5_0_Hyline_msd={:.4}\n'
                  ########################################################################
                  # ' \n 6ptc_f1={:.4} 6ptc_dsc={:.4} 6ptc_hd={:.4} 6ptc_msd={:.4}\n'
                  # ' \n 7ptc_f1={:.4} 7ptc_dsc={:.4} 7ptc_hd={:.4} 7ptc_msd={:.4}\n'
                  # ' \n 8ptc_f1={:.4} 8ptc_dsc={:.4} 8ptc_hd={:.4} 8ptc_msd={:.4}\n'
                  # ' \n 9ptc_f1={:.4} 9ptc_dsc={:.4} 9ptc_hd={:.4} 9ptc_msd={:.4}\n'
                  # #######################################################################
                  .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
                          avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item(),
                          avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item(),
                          avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item(),
                          avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item(),
                          avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()
                          #############################################################################################
                          # , avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item()
                          # , avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()
                          # , avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(), avg_val_MSD[8].item()
                          # , avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item()
                          # #############################################################################################
                          ))

            df = pd.DataFrame(columns = ['task','F1','Dice','HD','MSD'])
            df.loc[0] = ['0_0_Lumen', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item()]
            df.loc[1] = ['1_0_Tunica_intima', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item()]
            df.loc[2] = ['2_0_Tunica_media', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item()]
            df.loc[3] = ['3_0_Artery', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item()]
            df.loc[4] = ['4_0_Artery_wall', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item()]
            df.loc[5] = ['5_0_Hyline', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()]
            ##########################################################################
            # df.loc[6] = ['6mely', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item()]
            # df.loc[7] = ['7micro', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()]
            # df.loc[8] = ['8nos', avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(), avg_val_MSD[8].item()]
            # df.loc[9] = ['9segs', avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item()]
            ##########################################################################
            df.to_csv(os.path.join(output_folder,'testing_result.csv'))

            single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
            single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))
            single_df_2.to_csv(os.path.join(output_folder,'testing_result_2.csv'))
            single_df_3.to_csv(os.path.join(output_folder,'testing_result_3.csv'))
            single_df_4.to_csv(os.path.join(output_folder,'testing_result_4.csv'))
            single_df_5.to_csv(os.path.join(output_folder,'testing_result_5.csv'))
            ###############################################################################
            # single_df_6.to_csv(os.path.join(output_folder, 'testing_result_6.csv'))
            # single_df_7.to_csv(os.path.join(output_folder, 'testing_result_7.csv'))
            # single_df_8.to_csv(os.path.join(output_folder, 'testing_result_8.csv'))
            # single_df_9.to_csv(os.path.join(output_folder, 'testing_result_9.csv'))
            ###############################################################################


    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
