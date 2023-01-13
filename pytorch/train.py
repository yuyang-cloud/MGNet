import sys
sys.path.append('.')
sys.path.append('../')

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from data_loader import DBGDataSet, gen_mask
from model import DBG
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from test import test_one_epoch
from post_processing import post_processing
from eval import eval
import utils
import pdb

""" Load config
"""
from config_loader import dbg_config

checkpoint_dir = dbg_config.checkpoint_dir
batch_size = dbg_config.batch_size
learning_rate = dbg_config.learning_rate
tscale = dbg_config.tscale
feature_dim = dbg_config.feature_dim
epoch_num = dbg_config.epoch_num

""" Initialize map mask
"""
mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()
tmp_mask = mask.repeat(batch_size, 1, 1, 1).requires_grad_(False)  # tmp_mask = bs,1,L,L  右上角为1
tmp_mask = tmp_mask > 0                                             # 右上角为True


def binary_logistic_loss(gt_scores, pred_anchors):
    """
    Calculate weighted binary logistic loss
    :param gt_scores: gt scores tensor
    :param pred_anchors: prediction score tensor
    :return: loss output tensor
    """
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)

    pmask = (gt_scores > 0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = pmask.size()[0]

    ratio = num_entries / max(num_positive, 1)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    neg_pred_anchors = 1.0 - pred_anchors + epsilon
    pred_anchors = pred_anchors + epsilon

    loss = coef_1 * pmask * torch.log(pred_anchors) + coef_0 * (1.0 - pmask) * torch.log(
        neg_pred_anchors)
    loss = -1.0 * torch.mean(loss)
    return loss


def IoU_loss(gt_iou, pred_iou):
    """
    Calculate IoU loss
    :param gt_iou: gt IoU tensor
    :param pred_iou: prediction IoU tensor
    :return: loss output tensor
    """
    u_hmask = (gt_iou > 0.6).float()        # map中 >0.6的为1       [100,100]
    u_mmask = ((gt_iou <= 0.6) & (gt_iou > 0.2)).float()    # map中 <=0.6&>0.2的为1     [100,100]
    u_lmask = (gt_iou <= 0.2).float() * mask    # map中 <0.2的为1       [100,100]

    u_hmask = u_hmask.view(-1)     # 将上述 >0.6的mask  平展开     [10000的长度]
    u_mmask = u_mmask.view(-1)     # 将上述 <=0.6 & >0.2的mask  平展开      [10000的长度]
    u_lmask = u_lmask.view(-1)     # 将上述 <0.2的mask  平展开      [10000的长度]

    num_h = torch.sum(u_hmask)     # >0.6的数量
    num_m = torch.sum(u_mmask)     # <=0.6 & >0.2的数量
    num_l = torch.sum(u_lmask)     # <0.2的数量

    r_m = 1.0 * num_h / num_m
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())
    u_smmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_mmask # torch.rand(10000)返回长度为100，所有值为0到1的随机数
    u_smmask = (u_smmask > (1.0 - r_m)).float()

    r_l = 2.0 * num_h / num_l
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())

    u_slmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask
                # >0.6的mask  <=0.6 & >0.2的mask的一部分  <0.2的mask的一部分
                # 相加之后作为最终求loss的mask(非1即0)  代表gt>0.6的全部都求loss   <=0.6的随机抽一部分求loss

    gt_iou = gt_iou.view(-1)
    pred_iou = pred_iou.view(-1)

    iou_loss = F.smooth_l1_loss(pred_iou * iou_weights, gt_iou * iou_weights, reduction='none') # gt>0.6的全部都求loss  <=0.6的随机抽一部分求loss
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.max(torch.sum(iou_weights),
                                                             torch.Tensor([1.0]).cuda())
    return iou_loss


def DBG_train(net, dl_iter, optimizer, epoch, writer, training):
    """
    One epoch of runing DBG model
    :param net: DBG network module
    :param dl_iter: data loader
    :param optimizer: optimizer module
    :param epoch: current epoch number
    :param training: bool, training or not
    :return: None
    """
    if training:
        net.train()

        loss_action_val = 0
        loss_iou_val = 0
        loss_start_val = 0
        loss_end_val = 0
        cost_val = 0
        for n_iter, \
            (gt_action, gt_start, gt_end, mask_start, mask_end, feature, iou_label) in tqdm.tqdm(enumerate(dl_iter)):
            gt_action = gt_action.cuda()
            gt_start = gt_start.cuda()
            gt_end = gt_end.cuda()
            mask_start = mask_start.cuda()
            mask_end = mask_end.cuda()
            # GTaction_mask = GTaction_mask.cuda()
            # GTaction_mask = ~GTaction_mask
            # GTaction_mask = GTaction_mask.float()
            feature = feature.cuda()
            iou_label = iou_label.cuda()

            output_dict = net(feature)
            x1 = output_dict['x1']
            x2 = output_dict['x2']
            x3 = output_dict['x3']
            x1_Tr_score = output_dict['x1_Tr_score']
            x2_Tr_score = output_dict['x2_Tr_score']

            boundary_x1 = output_dict['boundary_x1']
            boundary_x2 = output_dict['boundary_x2']
            boundary_x3 = output_dict['boundary_x3']
            x3_Tr_score = output_dict['x3_Tr_score']
            x4_Tr_score = output_dict['x4_Tr_score']
            
            mask_x1 = output_dict['mask_x1']
            mask_x2 = output_dict['mask_x2']
            mask_x3 = output_dict['mask_x3']
            mask_x1_Tr_score = output_dict['mask_x1_Tr_score']
            mask_x2_Tr_score = output_dict['mask_x2_Tr_score']

            iou = output_dict['iou']
            prop_start = output_dict['prop_start']
            prop_end = output_dict['prop_end']

            # calculate action loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_action = binary_logistic_loss(gt_action, x1) + \
                        binary_logistic_loss(gt_action, x2) + \
                        binary_logistic_loss(gt_action, x3)
            loss_action /= 3.0

            # calculate Transformer loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_action_Tr = binary_logistic_loss(gt_action, x1_Tr_score) + \
                        binary_logistic_loss(gt_action, x2_Tr_score)

            # calculate boundary loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_boundary = binary_logistic_loss(gt_start, boundary_x1[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_start, boundary_x2[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_start, boundary_x3[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_end, boundary_x1[:,1,:].contiguous()) + \
                        binary_logistic_loss(gt_end, boundary_x2[:,1,:].contiguous()) + \
                        binary_logistic_loss(gt_end, boundary_x3[:,1,:].contiguous())
            loss_boundary /= 6.0

            # calculate Transformer loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_boundary_Tr = binary_logistic_loss(gt_start, x3_Tr_score[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_start, x4_Tr_score[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_end, x3_Tr_score[:,1,:].contiguous()) + \
                        binary_logistic_loss(gt_end, x4_Tr_score[:,1,:].contiguous())
            loss_boundary_Tr /= 2.0
            
            # calculate action loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_mask_action = binary_logistic_loss(gt_action, mask_x1[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_action, mask_x2[:,0,:].contiguous()) + \
                        binary_logistic_loss(gt_action, mask_x3[:,0,:].contiguous())
            loss_mask_action /= 3.0
            loss_mask_start = binary_logistic_loss(mask_start, mask_x1[:,1,:].contiguous()) + \
                        binary_logistic_loss(mask_start, mask_x2[:,1,:].contiguous()) + \
                        binary_logistic_loss(mask_start, mask_x3[:,1,:].contiguous())
            loss_mask_start /= 3.0
            loss_mask_end = binary_logistic_loss(mask_end, mask_x1[:,2,:].contiguous()) + \
                        binary_logistic_loss(mask_end, mask_x2[:,2,:].contiguous()) + \
                        binary_logistic_loss(mask_end, mask_x3[:,2,:].contiguous())
            loss_mask_end /= 3.0

            # calculate Transformer loss
            # gt_action=bs,1,100           x1,x2,x3=bs,1,100
            loss_mask_Tr = binary_logistic_loss(gt_action, mask_x1_Tr_score) + \
                        binary_logistic_loss(gt_action, mask_x2_Tr_score)

            # calculate IoU loss
            iou_losses = 0.0
            for i in range(batch_size):
                iou_loss = IoU_loss(iou_label[i:i + 1], iou[i:i + 1])
                iou_losses += iou_loss
            loss_iou = iou_losses / batch_size * 10.0

            # calculate starting and ending map loss
            # gt_start=bs,1,100     gt_end=bs,1,100
            gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, tscale)  # gt_start=bs,1,100,100  这里是gt,对于gt的话，map的行代表start，某一行的所有proposal具有相同的start，而这个start的gt概率(注意是gt)就是确定的，即这一行数值应该是相同的
            gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, tscale, 1)     # gt_end=bs,1,100,100   这里是gt,对于gt，map的列代表end，某一列的所有proposal具有相同的end，而这个end点的gt概率就是确定的，所以这一列的数值是相同的
            loss_start = binary_logistic_loss(
                torch.masked_select(gt_start, tmp_mask),
                torch.masked_select(prop_start, tmp_mask) # prop_start=bs,1,T,T
            )
            loss_end = binary_logistic_loss(
                torch.masked_select(gt_end, tmp_mask),
                torch.masked_select(prop_end, tmp_mask)   # prop_end=bs,1,T,T
            )

            # total loss
            cost = 2.0 * loss_action + loss_boundary + loss_iou + loss_start + loss_end + \
                     loss_action_Tr + loss_boundary_Tr + \
                     2.0 * loss_mask_action + loss_mask_start + loss_mask_end + loss_mask_Tr

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            loss_action_val += loss_action.cpu().detach().numpy()
            loss_iou_val += loss_iou.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()

        loss_action_val /= (n_iter + 1)
        loss_iou_val /= (n_iter + 1)
        loss_start_val /= (n_iter + 1)
        loss_end_val /= (n_iter + 1)
        cost_val /= (n_iter + 1)

        writer.add_scalars('log/total', {'train': cost_val}, epoch)
        writer.add_scalars('log/action', {'train': loss_action_val}, epoch)
        writer.add_scalars('log/start', {'train': loss_start_val}, epoch)
        writer.add_scalars('log/end', {'train': loss_end_val}, epoch)
        writer.add_scalars('log/iou', {'train': loss_iou_val}, epoch)
        print(
            "Epoch-%d Train Loss: "
            "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))
    
    else:
        test_one_epoch(net)
        post_processing()
        AR_1, AR_5, AR_10, AR_100, AUC_ = eval()

        writer.add_scalars('log/AR@1', {'test': AR_1}, epoch)
        writer.add_scalars('log/AR@5', {'test': AR_5}, epoch)
        writer.add_scalars('log/AR@10', {'test': AR_10}, epoch)
        writer.add_scalars('log/AR@100', {'test': AR_100}, epoch)
        writer.add_scalars('log/AUC', {'test': AUC_}, epoch)

        # 存储AR1_best
        if AUC_ > net.module.AUC_best:
            print("Saving checkpoint_AUCbest.pth ...")
            net.module.AUC_best = AUC_
            checkpoint_best_path = os.path.join(checkpoint_dir, 'DBG_checkpoint_AUCbest.ckpt')
            # 存储文件
            utils.save_on_master({
                'model': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_best_path)




def set_seed(seed):
    """
    Set randon seed for pytorch
    :param seed:
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('Only train on GPU.')
        exit()
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    set_seed(2020)
    net = DBG(feature_dim)  # feature_dim = 400
    net = nn.DataParallel(net, device_ids=[0]).cuda()

    # set weight decay for different parameters
    Net_bias = []
    for name, p in net.module.named_parameters():
        if 'bias' in name:
            Net_bias.append(p)

    DSBNet_weight = []
    for name, p in net.module.DSBNet.conv1_1.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv1_2.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv2_1.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv2_2.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv1_3.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv2_3.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv3_3.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv4_3.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv3_1.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    for name, p in net.module.DSBNet.conv3_2.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)
    # for name, p in net.module.DSBNet.Tr1_conv_1.named_parameters():
    #     if 'bias' not in name:
    #         DSBNet_weight.append(p)
    # for name, p in net.module.DSBNet.Tr1_conv_2.named_parameters():
    #     if 'bias' not in name:
    #         DSBNet_weight.append(p)
    # for name, p in net.module.DSBNet.Tr2_conv_1.named_parameters():
    #     if 'bias' not in name:
    #         DSBNet_weight.append(p)
    # for name, p in net.module.DSBNet.Tr2_conv_2.named_parameters():
    #     if 'bias' not in name:
    #         DSBNet_weight.append(p)
    



    Transformer_weight = []
    for name, p in net.module.DSBNet.action_ViT1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.action_ViT2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.boundary_ViT1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.boundary_ViT2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr1_conv_1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr1_conv_2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr2_conv_1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr2_conv_2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr3_conv_1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr3_conv_2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr4_conv_1.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)
    for name, p in net.module.DSBNet.Tr4_conv_2.named_parameters():
        if 'bias' not in name:
            Transformer_weight.append(p)

    


    Mask_weight = []
    for name, p in net.module.DSBNet.mask_ViT1.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_ViT2.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_Tr1_conv_1.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_Tr1_conv_2.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_Tr2_conv_1.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_Tr2_conv_2.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv1_1.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv1_2.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv1_3.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv2_1.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv2_2.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv2_3.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)
    for name, p in net.module.DSBNet.mask_conv3.named_parameters():
        if 'bias' not in name:
            Mask_weight.append(p)



    PFG_weight = []
    for name, p in net.module.PropFeatGen.named_parameters():
        if 'bias' not in name:
            PFG_weight.append(p)

    ACR_TBC_weight = []
    for name, p in net.module.ACRNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)
    for name, p in net.module.TBCNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)

    # setup Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': Net_bias, 'weight_decay': 0},
        {'params': DSBNet_weight, 'weight_decay': 2e-3},    # 2e-3
        {'params': Transformer_weight, 'weight_decay': 1e-4},
        {'params': Mask_weight, 'weight_decay': 1e-4},
        {'params': PFG_weight, 'weight_decay': 2e-4},
        {'params': ACR_TBC_weight, 'weight_decay': 2e-5}
    ], lr=1.0)

    # setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda x: learning_rate[x])  # learning_rate = [8个0.001  4个0.0001]
    # setup training and validation data loader
    train_dl = DataLoader(DBGDataSet(mode='training'), batch_size=batch_size,
                          shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(DBGDataSet(mode='validation'), batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    writer = SummaryWriter()   

    # train DBG
    for i in range(epoch_num):
        scheduler.step(i)
        print('current learning rate:', scheduler.get_lr()[0])
        DBG_train(net, train_dl, optimizer, i, writer, training=True)
        DBG_train(net, val_dl, optimizer, i, writer, training=False)
