import sys
sys.path.append('.')
import torch
import torch.nn as nn
import numpy as np
from model import DBG
import os
import json
import pandas as pd
import multiprocessing as mp
from data_loader import DBG_THUMOS_DataSet
import h5py
import pdb





checkpoint_dir = 'output/model'
feature_dim = 2048
org_feat_dir = '/public/datasets/THUMOS/'

feat_resolution = 5
skip_videoframes = 5
window_len = 128
window_step = window_len // 2

top_number = 2300

mask = np.zeros([window_len, window_len], np.float32)
for i in range(window_len):
    for j in range(i, window_len):
        if j - i < 64:
            mask[i, j] = 1
tf_mask = np.expand_dims(np.expand_dims(mask, 0), 1)
tf_mask = torch.from_numpy(tf_mask).float().requires_grad_(False).cuda()

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def softNMS(df):
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) <= top_number:
        max_index = tscore.index(max(tscore))
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)

        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(np.float)

        idxs = np.where(iou > 0.65)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / 0.75)

        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf

def NMS(df, thresh):
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    t1 = np.array(tstart)
    t2 = np.array(tend)
    scores = np.array(tscore)
    durations = t2 - t1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    rscore = [tscore[i] for i in keep]
    rstart = [tstart[i] for i in keep]
    rend = [tend[i] for i in keep]
    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf

def result_post_process(res, top_number, extracted_frame_num, frame_num, video_name):
    df = pd.DataFrame(res, columns=['xmin', 'xmax', 'score'])
    df = df.sort_values(by='score', ascending=False)
    # df = softNMS(df)
    df = NMS(df, 0.65)
    data = []
    for j in range(min(top_number, len(df))):
        f_end = int(df.xmax.values[j] * extracted_frame_num)    # 恢复到帧idx
        f_init = int(df.xmin.values[j] * extracted_frame_num)   # 恢复到帧idx
        score = df.score.values[j]
        data.append([f_end, f_init, score, frame_num, video_name])
    data = np.stack(data)
    new_df = pd.DataFrame(data, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])
    result_list.append(new_df)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    mp.set_start_method('spawn')
    # mp.set_start_method('forkserver', force=True)

    global result_list
    result_list = mp.Manager().list()

    """ setup DBG model and load weights """
    net = DBG(feature_dim)
    state_dict = torch.load(os.path.join(checkpoint_dir, 'DBG_checkpoint_best.ckpt'))
    net.load_state_dict(state_dict['model'])
    print("epoch: %d",state_dict['epoch'])
    net = nn.DataParallel(net, device_ids=[0]).cuda()
    net.eval()



    # test_loader = torch.utils.data.DataLoader(DBG_THUMOS_DataSet(subset="validation", mode='inference'),
    #                                           batch_size=1, shuffle=False,
    #                                           num_workers=8, pin_memory=True, drop_last=False)



    annos = pd.read_csv('./data/thumos_annotations/thumos14_test_groundtruth.csv')
    video_list = sorted(list(set(annos['video-name'].values[:])))

    dfs = []
    processes = []
    for video_name in video_list[:]:
        print(video_name)

        flow_test = h5py.File(org_feat_dir+'/flow_test.h5', 'r')
        rgb_test = h5py.File(org_feat_dir+'/rgb_test.h5', 'r')
        feature_h5s = [
                    flow_test[video_name][::skip_videoframes,...],
                    rgb_test[video_name][::skip_videoframes,...]
                ]
        feature_len = min([h5.shape[0] for h5 in feature_h5s])  # num_snippet = T 每个视频的T不同
        video_feature = np.concatenate([h5[:feature_len, :]
                                      for h5 in feature_h5s],   # df_data=T,4096 每个视频的T不同
                                     axis=1)
        

        video_info = annos.loc[annos['video-name'] == video_name]
        duration = video_info['video-duration'].values[0]
        frame_num = video_info['video-frames'].values[0] # 视频真实帧数

        extracted_frame_num = feature_len * feat_resolution # 抽取帧数frame = T*5  因为intervel=5,视频一行代表5帧
        corrected_second = float(extracted_frame_num) / frame_num * duration
        if feature_len < window_len:
            # feature padding
            pad_len = window_len - feature_len
            video_feature = np.pad(video_feature, [[0, pad_len], [0, 0]], mode='constant')
            corrected_second = corrected_second / feature_len * window_len
            extracted_frame_num = extracted_frame_num // feature_len * window_len
            feature_len = window_len

        tscale = feature_len    # T 不同video的T不同
        tgap = 1.0 / tscale     # 1/T

        xmins = [tgap*i + tgap*0.5 for i in range(tscale)]
        xmaxs = [tgap*i + tgap*0.5 for i in range(1, tscale+1)]

        # generate windows
        windows = []
        for i in range(0, feature_len - window_len + 1, window_step):
            windows.append([i, i + window_len])
        if feature_len % window_len:
            windows.append([feature_len - window_len, feature_len]) # windows是每个window的[start_frmae,end_frame]   [ [0,128], [64,192], [128,256] ...]


        res = []
        for s, e in windows:
            input_feat = video_feature[s:e] # 取当前window内的feature: 128,400
            input_feat = np.expand_dims(input_feat, 0)
            in_feature = torch.from_numpy(input_feat).float().cuda()
            in_feature = in_feature.permute(0,2,1)
            
            with torch.no_grad():
                output_dict = net(in_feature)
            out_iou = output_dict['iou']
            out_start = output_dict['prop_start']
            out_end = output_dict['prop_end']
            
            # fusion starting and ending map score
            out_start = out_start * tf_mask
            out_end = out_end * tf_mask
            out_iou = out_iou * tf_mask
            out_start = torch.sum(out_start, 3) / torch.sum(tf_mask, 3)
            out_end = torch.sum(out_end, 2) / torch.sum(tf_mask, 2)
            pstart = out_start[0].squeeze().cpu().detach().numpy()
            pend = out_end[0].squeeze().cpu().detach().numpy()
            iou = out_iou[0].squeeze().cpu().detach().numpy()

            

            max_start = max(pstart) * 0.5
            max_end = max(pend) * 0.5
            start_set = [0, window_len-1]   # [0,127]
            end_set = [0, window_len-1]
            for i in range(1, window_len-1):    # (1:127)
                start = pstart[i]
                if start > max_start:
                    start_set.append(i) # 如果当前intervel的start大于max_start，则认为当前frame属于start,把其index加入
                elif pstart[i-1] < start and pstart[i+1] < start:   # 如果当前frame的start是个峰值
                    start_set.append(i)
            for i in range(1, window_len-1):
                end = pend[i]
                if end > max_end:
                    end_set.append(i)
                elif pend[i-1] < end and pend[i+1] < end: # 如果当前frmae的end大于max_end，或者是个峰值，则认为当前frmae属于end,把其index加入
                    end_set.append(i)

            # 到这里 start_set是所有可能为start的frame_idx  (当前window内)
            #       end_set是所有可能为end的frame_idx      (当前window内)


            for i in start_set:
                start = pstart[i]   # i帧对应的start概率
                for j in end_set:
                    if j - i <= 3 or j - i >= 64:
                        continue
                    end = pend[j]   # j帧对应的end概率
                    score = start * end * iou[i, j] 
                    res.append([xmins[s + i], xmaxs[s + j], score]) # 在这里 xmins归一化的东西  s+i把s加上  所以这里存储的xmins[s+i]就是相对于video而言

        res = np.stack(res)
        result_post_process(res, top_number, extracted_frame_num, frame_num, video_name)

        # # NMS
        # p = mp.Process(target=result_post_process, args=(res, top_number, extracted_frame_num, frame_num, video_name)) # top_number=1100
        # processes.append(p)
        # p.start()

    # for p in processes:
    #     p.join()

    dfs = list(result_list)
    final_df = pd.concat(dfs)
    final_df.to_csv('result_v1_nms.csv', sep=' ', index=False)




