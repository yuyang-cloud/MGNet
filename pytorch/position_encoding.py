# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Positional encodings for the transformer.
"""
import math
from re import S
import torch
import numpy as np
from torch import nn
import pdb

class PositionalEncoding(nn.Module):
	
	def __init__(self, d_model=512, seq_len=100):
		super(PositionalEncoding, self).__init__()
		
		# Not a parameter
		self.register_buffer('pos_table', self._get_sinusoid_encoding_table(seq_len, d_model))

	
	def _get_sinusoid_encoding_table(self, seq_len, d_model):
		''' Sinusoid position encoding table '''
		
		# TODO: make it with torch instead of numpy
		
		def get_position_angle_vec(position):
			return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]
		
		sinusoid_table = np.array(
			[get_position_angle_vec(pos_i) for pos_i in range(seq_len)]) 
		
		sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
		sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
		
		sinusoid_table = torch.from_numpy(sinusoid_table).float()
		sinusoid_table = sinusoid_table.unsqueeze(0)	# 1,T,C
		
		return sinusoid_table  # bs,T,C
	
	def forward(self, x):  # src=[seq_len,bs,d_model]
		pos_table = self.pos_table.repeat(x.size(0), 1, 1)	# bs,T,C
		return x + pos_table[:x.size(0), :x.size(1), :x.size(2)].clone().detach()
