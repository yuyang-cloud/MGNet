import sys

sys.path.append('.')
import os

import numpy as np
import tqdm

import data_loader
import utils
from model import DBG
from config_loader import dbg_config

import torch
import torch.nn as nn


""" set checkpoint directory """
checkpoint_dir = dbg_config.checkpoint_dir
""" set result directory to save all csv files """
result_dir = dbg_config.result_dir

""" get input feature scale """
tscale = dbg_config.tscale
""" get input feature channel number """
feature_dim = dbg_config.feature_dim

""" get testing batch size """
batch_size = dbg_config.test_batch_size
""" get testing mode: validation or test """
test_mode = dbg_config.test_mode


""" get map mask """
mask = data_loader.gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1) 
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda() # mask=1,1,L,L  右上角为1


"""
This test script is used for evaluating our algorithm 
This script saves all proposals results (csv format)
Then, use post_processing.py to generate the final result
Finally, use eval.py to evaluate the final result
You can got about 68% AUC
"""

""" 
Testing procedure
1.Get Test data
2.Define DBG model
3.Load model weights 
4.Run DBG model
5.Save proposal results (csv format)
"""

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    with torch.no_grad():
        """ setup DBG model and load weights """
        net = DBG(feature_dim)  # feature_dim = 400

        # state_dict = torch.load(os.path.join(checkpoint_dir, 'DBG_checkpoint_best.ckpt'))
        # net.load_state_dict(state_dict)

        print("Loading model from checkpoint ...")
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'DBG_checkpoint_AUCbest.ckpt')) #    
        net.load_state_dict(checkpoint['model'])
        print("Model Loaded from checkpoint successful!")
        print("Checkpoint epoch :", checkpoint['epoch'])

        net = nn.DataParallel(net, device_ids=[0]).cuda()
        net.eval()

        """ get testing dataset """
        train_dict, val_dict, test_dict = data_loader.getDatasetDict(dbg_config.video_info_file)
        if test_mode == 'validation':
            video_dict = val_dict
        else:
            video_dict = test_dict
        print(len(video_dict))

        batch_video_list = data_loader.getBatchListTest(video_dict, batch_size)

        batch_result_xmin = []
        batch_result_xmax = []
        batch_result_iou = []
        batch_result_pstart = []
        batch_result_pend = []

        """ runing DBG model """
        print('Runing DBG model ...')
        for idx in tqdm.tqdm(range(len(batch_video_list))):  # len(batch_video_list)是 batch的总个数 即一个epoch的迭代次数
            batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = \
                data_loader.getProposalDataTest(batch_video_list[idx], dbg_config)  # batch_anchor_xmin=bs,100   batch_anchor_feature=bs,100,400
            in_feature = torch.from_numpy(batch_anchor_feature).float().cuda().permute(0, 2, 1)   # in_feature=bs,400,100
            output_dict = net(in_feature)
            out_iou = output_dict['iou']            # iou=bs,1,T,T
            out_start = output_dict['prop_start']   # out_start=bs,1,T,T
            out_end = output_dict['prop_end']       # out_end=bs,1,T,T

            # fusion starting and ending map score
            out_start = out_start * mask
            out_end = out_end * mask
            out_start = torch.sum(out_start, 3) / torch.sum(mask, 3) # 按行求和/mask该行是1的个数  也就是正常的话，右上角输出有值，左下角为0，对改行中属于右上角的求均值，作为这一行start的预测输出  bs,1,T
            out_end = torch.sum(out_end, 2) / torch.sum(mask, 2)  # 按列求和/mask该列是1的个数  即在该列中，右上角的输出有值，左下角为0，对该列中属于右上角的部分求平均，作为该列end的预测输出
            # out_start=bs,1,T  out_end=bs,1,T  都是概率序列

            batch_result_xmin.append(batch_anchor_xmin) # batch_anchor_xmin=bs,100 
            batch_result_xmax.append(batch_anchor_xmax) # batch_result_xmax=bs,100 
            batch_result_iou.append(out_iou[:, 0].cpu().detach().numpy())   # batch_result_iou=bs,T,T   概率map
            batch_result_pstart.append(out_start[:, 0].cpu().detach().numpy())  # batch_result_pstart=bs,T  概率序列
            batch_result_pend.append(out_end[:, 0].cpu().detach().numpy())  # batch_result_pend=bs,T    概率序列

        utils.save_proposals_result(batch_video_list,   # batch_video_list=N,bs
                                    batch_result_xmin,  # batch_result_xmin=N,bs,100     N=19994//bs+1
                                    batch_result_xmax,  # batch_result_xmax=N,bs,100     N=19994//bs+1
                                    batch_result_iou,   # batch_result_iou=N,bs,T,T      N=19994//bs+1
                                    batch_result_pstart,# batch_result_pstart=N,,bs,100     N=19994//bs+1
                                    batch_result_pend,  # batch_result_pend=N,,bs,100     N=19994//bs+1
                                    tscale, result_dir)
        