import numpy as np
from numpy.lib.function_base import place
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from custom_op.prop_tcfg_op import PropTcfg
from action_ViT import Action_VisionTransformer
from mask_VIT import Mask_VisionTransformer
from boundary_ViT import Boundary_VisionTransformer
import pdb

def conv1d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv1D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv1D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )


def conv2d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv2D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv2D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )


def new_conv2d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv2D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv2D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
        )




# class gen_mask(nn.Module):
#     def __init__(self, embed_dim=400, output_dim=1):
#         super(gen_mask, self).__init__()
#         self.embed_dim = embed_dim
#         self.output_dim = output_dim

#         self.TEM = nn.Sequential(
#             conv1d(self.embed_dim, 256, 3),
#             conv1d(256, 128, 3),
#             conv1d(128, self.output_dim, kernel_size=1, is_relu=False)
#         )

#     def forward(self, x):
#         # x=bs,C,T
#         x = self.TEM(x)

#         if self.output_dim == 1:
#             action_score = torch.sigmoid(x)    # bs,T
#             action_score = action_score.squeeze(1)
#             max_action = torch.max(action_score,1)[0].unsqueeze(1) # bs,1
#             action_mask = action_score>(0.5*max_action)    # bs,T
#             action_mask = action_mask.cuda()
#             return action_score, action_mask    # bs,T
#         if self.output_dim ==2:
#             boundary_score = torch.sigmoid(x)  # bs,2,T
#             start_score = boundary_score[:,0,:] # bs,T
#             end_score = boundary_score[:,1,:]   # bs,T
#             max_start = torch.max(start_score,1)[0].unsqueeze(1)    # bs,1
#             max_end = torch.max(end_score,1)[0].unsqueeze(1)        # bs,1
#             start_mask = start_score>(0.5*max_start)       # bs,T
#             start_mask = start_mask.cuda()
#             end_mask = end_score>(0.5*max_end)             # bs,T
#             end_mask = end_mask.cuda()
#             boundary_mask = start_mask + end_mask           # bs,T
#             return start_score, end_score, boundary_mask    # bs,T


class gen_mask(nn.Module):
    def __init__(self, embed_dim=400, output_dim=1):
        super(gen_mask, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.hidden_dim_1d = 256

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            # nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            # nn.ReLU(inplace=True)
        )
        # Temporal Evaluation Module
        self.x_1d_a = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.output_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x=bs,C,T
        base_feature = self.x_1d_b(x)
        action_score = self.x_1d_a(base_feature).squeeze(1)
        max_action = torch.max(action_score,1)[0].unsqueeze(1) # bs,1
        action_mask = action_score>(0.5*max_action)    # bs,T
        action_mask = action_mask.cuda()
        return action_score, action_mask    # bs,T






## 串联
class DSBaseNet(nn.Module):
    """
    Setup dual stream base network (DSB)
    """
    def __init__(self, feature_dim):
        super(DSBaseNet, self).__init__()
        feature_dim = feature_dim // 2  # feature_dim = 400//2 = 200
        self.feature_dim = feature_dim  # feature_dim = 200

        self.mask_ViT1 = Mask_VisionTransformer(in_chans=200, embed_dim=512, depth=1, num_heads=8, mlp_ratio=4., pos_type='sin',
                                    qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)

        self.mask_ViT2 = Mask_VisionTransformer(in_chans=200, embed_dim=512, depth=1, num_heads=8, mlp_ratio=4., pos_type='sin',
                                    qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)

        self.mask_Tr1_conv_1 = conv1d(512, 128, 1)
        self.mask_Tr1_conv_2 = conv1d(128, 1, 1, is_relu=False)

        self.mask_Tr2_conv_1 = conv1d(512, 128, 1)
        self.mask_Tr2_conv_2 = conv1d(128, 1, 1, is_relu=False)

        self.mask_conv1_1 = conv1d(512, 256, 3)
        self.mask_conv1_2 = conv1d(256, 128, 3)
        self.mask_conv1_3 = conv1d(128, 3, 1, is_relu=False)

        self.mask_conv2_1 = conv1d(512, 256, 3)
        self.mask_conv2_2 = conv1d(256, 128, 3)
        self.mask_conv2_3 = conv1d(128, 3, 1, is_relu=False)

        self.mask_conv3 = conv1d(128, 3, 1, is_relu=False)


        # depth=1   heads=4   mlp=2   wd=2e-3   76.84   68.46
        # depth=1   heads=4   mlp=4   wd=2e-3   77.02   68.71
        # depth=1   heads=4   mlp=4   wd=1e-4
        self.action_ViT1 = Action_VisionTransformer(in_chans=200, embed_dim=512, depth=3, num_heads=4, mlp_ratio=4., pos_type='learn',
                                    qkv_bias=False, drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)
        
        self.action_ViT2 = Action_VisionTransformer(in_chans=200, embed_dim=512, depth=3, num_heads=4, mlp_ratio=4., pos_type='learn',
                                    qkv_bias=False, drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)

        self.Tr1_conv_1 = conv1d(512, 128, 1)
        self.Tr1_conv_2 = conv1d(128, 1, 1, is_relu=False)
        
        self.Tr2_conv_1 = conv1d(512, 128, 1)
        self.Tr2_conv_2 = conv1d(128, 1, 1, is_relu=False)



        self.boundary_ViT1 = Boundary_VisionTransformer(in_chans=200, embed_dim=512, depth=3, num_heads=4, mlp_ratio=4., pos_type='learn',
                                    qkv_bias=False, drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)
        
        self.boundary_ViT2 = Boundary_VisionTransformer(in_chans=200, embed_dim=512, depth=3, num_heads=4, mlp_ratio=4., pos_type='learn',
                                    qkv_bias=False, drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1)

        self.Tr3_conv_1 = conv1d(512, 128, 1)
        self.Tr3_conv_2 = conv1d(128, 2, 1, is_relu=False)
        
        self.Tr4_conv_1 = conv1d(512, 128, 1)
        self.Tr4_conv_2 = conv1d(128, 2, 1, is_relu=False)

        self.conv1_1 = conv1d(200, 256, 3)
        self.conv1_2 = conv1d(256, 128, 3)
        self.conv2_1 = conv1d(200, 256, 3)
        self.conv2_2 = conv1d(256, 128, 3)

        
        self.conv1_3 = conv1d(128, 1, 1, is_relu=False)
        self.conv2_3 = conv1d(128, 1, 1, is_relu=False)
        self.conv3_3 = conv1d(128, 2, 1, is_relu=False)
        self.conv4_3 = conv1d(128, 2, 1, is_relu=False)

        self.conv3_1 = conv1d(128, 1, 1, is_relu=False)
        self.conv3_2 = conv1d(128, 2, 1, is_relu=False)

    def forward(self, x):
        # x=bs,C,T
        x_RGB, x_flow = torch.split(x, self.feature_dim, 1)    # bs,200,T

        # Mask 2
        mask_x1_Tr = self.mask_ViT1(x_RGB, mask=None)  # bs,T,512
        mask_x1_Tr = mask_x1_Tr.permute(0,2,1)  # bs,512,T
        mask_x1_Tr_score = self.mask_Tr1_conv_1(mask_x1_Tr)  # bs,128,T
        mask_x1_Tr_score = torch.sigmoid(self.mask_Tr1_conv_2(mask_x1_Tr_score))
        mask_x1_feat = self.mask_conv1_1(mask_x1_Tr)
        mask_x1_feat = self.mask_conv1_2(mask_x1_feat)   # bs,128,T
        mask_x1 = torch.sigmoid(self.mask_conv1_3(mask_x1_feat))    # bs,3,T

        mask_x2_Tr = self.mask_ViT2(x_flow, mask=None)  # bs,T,512
        mask_x2_Tr = mask_x2_Tr.permute(0,2,1)    # bs,512,T
        mask_x2_Tr_score = self.mask_Tr2_conv_1(mask_x2_Tr)  # bs,128,T
        mask_x2_Tr_score = torch.sigmoid(self.mask_Tr2_conv_2(mask_x2_Tr_score))
        mask_x2_feat = self.mask_conv2_1(mask_x2_Tr)
        mask_x2_feat = self.mask_conv2_2(mask_x2_feat) # bs,128,T
        mask_x2 = torch.sigmoid(self.mask_conv2_3(mask_x2_feat))    # bs,3,T

        mask_xc = mask_x1_feat + mask_x2_feat      # bs,128,T
        mask_x3 = torch.sigmoid(self.mask_conv3(mask_xc))   # bs,3,T

        action_score = (mask_x1[:,0,:] + mask_x2[:,0,:] + mask_x3[:,0,:]) / 3.0  # bs,1,T
        action_score = action_score.squeeze()
        # action_mask = action_score>0.5    # bs,T
        action_mask = action_score
        start_score = (mask_x1[:,1,:] + mask_x2[:,1,:] + mask_x3[:,1,:]) / 3.0  # bs,1,T
        start_score = start_score.squeeze()
        # start_mask = start_score>0.5    # bs,T
        start_mask = start_score
        end_score = (mask_x1[:,2,:] + mask_x2[:,2,:] + mask_x3[:,2,:]) / 3.0  # bs,1,T
        end_score = end_score.squeeze()
        # end_mask = end_score>0.5    # bs,T
        end_mask = end_score

        # Action
        x1_Tr = self.action_ViT1(x_RGB, action_mask)  # bs,T,512
        x1_Tr = x1_Tr.permute(0,2,1)  # bs,512,T
        x1_Tr = self.Tr1_conv_1(x1_Tr)  # bs,128,T
        x1_Tr_score = torch.sigmoid(self.Tr1_conv_2(x1_Tr))
        x1_Conv = self.conv1_1(x_RGB)
        x1_Conv = self.conv1_2(x1_Conv)   # bs,128,T
        x1_feat = x1_Tr + x1_Conv   # bs,128,T
        x1 = torch.sigmoid(self.conv1_3(x1_feat))

        x2_Tr = self.action_ViT2(x_flow, action_mask)  # bs,T,512
        x2_Tr = x2_Tr.permute(0,2,1)    # bs,512,T
        x2_Tr = self.Tr2_conv_1(x2_Tr)  # bs,128,T
        x2_Tr_score = torch.sigmoid(self.Tr2_conv_2(x2_Tr))
        x2_Conv = self.conv2_1(x_flow)
        x2_Conv = self.conv2_2(x2_Conv) # bs,128,T
        x2_feat = x2_Tr + x2_Conv   # bs,128,T
        x2 = torch.sigmoid(self.conv2_3(x2_feat))

        xc = x1_feat + x2_feat      # bs,128,T
        x3 = torch.sigmoid(self.conv3_1(xc))
        score = (x1 + x2 + x3) / 3.0

        # Boundary
        x3_Tr = self.boundary_ViT1(x_RGB, start_mask, end_mask)  # bs,T,512
        x3_Tr = x3_Tr.permute(0,2,1)  # bs,512,T
        x3_Tr = self.Tr3_conv_1(x3_Tr)  # bs,128,T
        x3_Tr_score = torch.sigmoid(self.Tr3_conv_2(x3_Tr))
        x3_feat = x3_Tr + x1_Conv   # bs,128,T
        boundary_x1 = torch.sigmoid(self.conv3_3(x3_feat))

        x4_Tr = self.boundary_ViT2(x_flow, start_mask, end_mask)  # bs,T,512
        x4_Tr = x4_Tr.permute(0,2,1)    # bs,512,T
        x4_Tr = self.Tr4_conv_1(x4_Tr)  # bs,128,T
        x4_Tr_score = torch.sigmoid(self.Tr4_conv_2(x4_Tr))
        x4_feat = x4_Tr + x2_Conv   # bs,128,T
        boundary_x2 = torch.sigmoid(self.conv4_3(x4_feat))

        xc_feat = x3_feat + x4_feat      # bs,128,T
        boundary_x3 = torch.sigmoid(self.conv3_2(xc_feat))

        output_dict = {
            'score': score,
            'xc_feat': xc_feat,

            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x1_Tr_score': x1_Tr_score,
            'x2_Tr_score': x2_Tr_score,

            'boundary_x1': boundary_x1,
            'boundary_x2': boundary_x2,
            'boundary_x3': boundary_x3,
            'x3_Tr_score': x3_Tr_score,
            'x4_Tr_score': x4_Tr_score,

            'mask_x1': mask_x1,
            'mask_x2': mask_x2,
            'mask_x3': mask_x3,
            'mask_x1_Tr_score': mask_x1_Tr_score,
            'mask_x2_Tr_score': mask_x2_Tr_score,
        }
        return output_dict











class ProposalFeatureGeneration(nn.Module):
    """
    Setup proposal feature generation module
    """
    def __init__(self, in_channels=128):
        super(ProposalFeatureGeneration, self).__init__()
        self.prop_tcfg = PropTcfg()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 512, kernel_size=(32, 1, 1)),
            # nn.BatchNorm3d(512),            # BN+Relu : 31.35  50.54  57.94  76.73  68.54
        )
        # self.conv3d = nn.Sequential(
        #     nn.Conv3d(in_channels, 512, kernel_size=(32, 1, 1), stride=(32, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, action_score, xc_feat):   # action_score=bs,1,T   xc_feat=bs,128,T
        action_feat = self.prop_tcfg(action_score)  # B x 1 x 32 x T x T
        action_feat = torch.squeeze(action_feat, 1)  # B x 32 x T x T
        net_feat = self.prop_tcfg(xc_feat)  # B x 128 x 32 x T x T
        net_feat = self.conv3d(net_feat)  # B x 512 x 1 x T x T
        net_feat = torch.squeeze(net_feat, 2)  # B x 512 x T x T

        return action_feat, net_feat  # action_feat=bs,32,T,T   net_feat=bs,512,T,T


class ACRNet(nn.Module):
    """
    Setup action classification regression network (ACR)
    """
    def __init__(self, in_channels=32):
        super(ACRNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, kernel_size=1),
            nn.Dropout(p=0.3),
            conv2d(256, 1, 1, is_relu=False)
        )

    def forward(self, action_feat):
        iou = self.conv2d(action_feat)
        iou = torch.sigmoid(iou)
        return iou


class TBCNet(nn.Module):
    """
    Setup temporal boundary classification network (TBC)
    """
    def __init__(self, in_channels=512):
        super(TBCNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, kernel_size=1),
            nn.Dropout(p=0.3),
            conv2d(256, 2, 1, is_relu=False)
        )

        # ## conv3d+BN+Relu 512-128-(ker3)128-(ker3)128-2    31.53  49.75  57.46  76.37  68.20
        # ## conv3d         512-128-(ker3)128-(ker3)128-2    31.31  49.51  57.34  76.55  68.27
        # ## conv3d         512-256-(ker3)256-(ker3)256-2    31.69  49.69  57.09  76.37  68.10
        # ## conv3d+BN+Relu 512-128-(ker3)128-2              
        # self.conv2d = nn.Sequential(
        #     conv2d(in_channels, 256, 1),
        #     nn.Dropout(p=0.3),
        #     conv2d(256, 256, kernel_size=3),
        #     nn.Dropout(p=0.3),
        #     # conv2d(256, 256, kernel_size=3),
        #     # nn.Dropout(p=0.3),
        #     conv2d(256, 2, 1, is_relu=False)
        # )

    def forward(self, net_feat):
        x = self.conv2d(net_feat)
        x = torch.sigmoid(x)

        prop_start = x[:, :1].contiguous()
        prop_end = x[:, 1:].contiguous()
        return prop_start, prop_end


class DBG(nn.Module):
    """
    Setup dense boundary generator framework (DBG)
    """
    def __init__(self, feature_dim):
        super(DBG, self).__init__()

        self.DSBNet = DSBaseNet(feature_dim)  # feature_dim = 400
        self.PropFeatGen = ProposalFeatureGeneration()
        self.ACRNet = ACRNet()
        self.TBCNet = TBCNet()

        self.AUC_best = 0
        self.reset_params() # reset all params by glorot uniform

    @staticmethod
    def glorot_uniform_(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = 1.0
        scale /= max(1., (fan_in + fan_out) / 2.)
        limit = np.sqrt(3.0 * scale)
        return nn.init._no_grad_uniform_(tensor, -limit, limit)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            DBG.glorot_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        DSB_output = self.DSBNet(x)  # score=bs,1,T   xc_feat=bs,256,T
        action_feat, net_feat = self.PropFeatGen(DSB_output['score'], DSB_output['xc_feat']) # action_feat=bs,32,T,T   net_feat=bs,512,T,T
        iou = self.ACRNet(action_feat)   # iou=bs,1,T,T
        prop_start, prop_end = self.TBCNet(net_feat)  # prop_start=bs,1,T,T  prop_end=bs,1,T,T

        output_dict = {
            'x1': DSB_output['x1'],   # x1=bs,1,T    sigmoid之后
            'x2': DSB_output['x2'],   # x2=bs,1,T
            'x3': DSB_output['x3'],   # x3=bs,1,T
            'x1_Tr_score': DSB_output['x1_Tr_score'],
            'x2_Tr_score': DSB_output['x2_Tr_score'],

            'boundary_x1': DSB_output['boundary_x1'],
            'boundary_x2': DSB_output['boundary_x2'],
            'boundary_x3': DSB_output['boundary_x3'],
            'x3_Tr_score': DSB_output['x3_Tr_score'],
            'x4_Tr_score': DSB_output['x4_Tr_score'],

            'mask_x1': DSB_output['mask_x1'],
            'mask_x2': DSB_output['mask_x2'],
            'mask_x3': DSB_output['mask_x3'],
            'mask_x1_Tr_score': DSB_output['mask_x1_Tr_score'],
            'mask_x2_Tr_score': DSB_output['mask_x2_Tr_score'],

            'iou': iou,               # iou=bs,1,T,T
            'prop_start': prop_start, # prop_start=bs,1,T,T
            'prop_end': prop_end      # prop_end=bs,1,T,T
        }
        return output_dict
