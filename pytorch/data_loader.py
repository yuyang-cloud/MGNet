import torch
from torch.utils.data import Dataset

from config_loader import dbg_config
from utils import *

""" Load config"""
""" get input feature temporal scale """
tscale = dbg_config.tscale  # 100
tgap = 1.0 / tscale         # 0.01
""" get video information json file path """
video_info_file = dbg_config.video_info_file # data/video_info_19993.json
""" set filter videos or not """
video_filter = dbg_config.video_filter  # True
""" get feature directory """
data_dir = dbg_config.feat_dir
""" set data augmentation or not """
data_aug = dbg_config.data_aug  # True


class DBGDataSet(Dataset):
    """
    DBG dataset to load ActivityNet-1.3 data
    """
    def __init__(self, mode='training'):
        train_dict, val_dict, test_dict = getDatasetDict(video_info_file, video_filter)
        training = True
        if mode == 'training':
            video_dict = train_dict
        else:
            training = False
            video_dict = val_dict
        self.mode = mode
        self.video_dict = video_dict
        video_num = len(list(video_dict.keys()))
        video_list = np.arange(video_num)

        # load raw data
        if training:
            data_dict, train_video_mean_len = getFullData(video_dict, dbg_config,
                                                          last_channel=False,
                                                          training=True)        
            # data_dict = {                                                       # train_video_mean_len=19993  每个video所有gt的时长mean
            #     "gt_action": batch_label_action,  # 19993, 100
            #     "gt_start": batch_label_start,    # 19993, 100
            #     "gt_end": batch_label_end,        # 19993, 100
            #     "feature": batch_anchor_feature,  # 19993, 400, 100
            #     "iou_label": batch_anchor_iou     # 19993, 100, 100  IoU
            # }

        else:
            data_dict = getFullData(video_dict, dbg_config,
                                    last_channel=False, training=False)

        # transform data to torch tensor
        for key in list(data_dict.keys()):
            # if key is not "GTaction_mask":
                data_dict[key] = torch.Tensor(data_dict[key]).float()
            # else:
            #     data_dict[key] = torch.Tensor(data_dict[key])>0.
        self.data_dict = data_dict

        if data_aug and training:
            # add train video with short proposals
            add_list = np.where(np.array(train_video_mean_len) < 0.2)
            add_list = np.reshape(add_list, [-1])
            video_list = np.concatenate([video_list, add_list[:]], 0) # 这三行操作 把gt<0.2 的video的index  又都附加到了video_list后边

        self.video_list = video_list  # 19993 + gt<0.2的video数量
        np.random.shuffle(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.video_list[idx]
        data_dict = self.data_dict
        gt_action = data_dict['gt_action'][idx].unsqueeze(0)  # 注意加了一个维度
        gt_start = data_dict['gt_start'][idx].unsqueeze(0)
        gt_end = data_dict['gt_end'][idx].unsqueeze(0)
        mask_start = data_dict['mask_start'][idx].unsqueeze(0)
        mask_end = data_dict['mask_end'][idx].unsqueeze(0)
        feature = data_dict['feature'][idx]
        iou_label = data_dict['iou_label'][idx].unsqueeze(0)
        return gt_action, gt_start, gt_end, mask_start, mask_end, feature, iou_label

        #     "gt_action":      1, 100
        #     "gt_start":       1, 100
        #     "gt_end":         1, 100
        #     "feature":        400, 100
        #     "iou_label":      1, 100, 100