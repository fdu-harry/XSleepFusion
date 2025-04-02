# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent

batch_size = 512*8
# batch_size = 32*20
d_model = 128
num_layers = 3
num_heads = 4
class_num = 2
d_inner = 512
dropout = 0.0
warm_steps = 4000
fea_num = 7
epoch = 200
PAD = 0
KS = 11
Fea_PLUS = 2
SIG_LEN = 256
ecg_lead=3
feature_attn_len = 64

import warnings
warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib.font_manager')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm1d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
    

    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = SDPAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

#         mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class SDPAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.BN = nn.BatchNorm1d(feature_attn_len)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.BN(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
    
#block.py   -> main.py 7 click
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu((self.w_1(output))))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
#         enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
#         enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

#model.py   -> main.py 6 click
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature+1
#         self.src_word_emb = nn.Conv1d(ecg_lead, d_model, kernel_size=11, stride=1, padding=int((11 - 1) / 2))
#         self.pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(int(d_model/2), d_model, kernel_size=15, stride=1, padding=int((15 - 1) / 2))
#         self.pool2 = nn.MaxPool1d(2)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
#         non_pad_mask = get_non_pad_mask(src_seq)
        non_pad_mask = None
#         slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        slf_attn_mask = None
        # -- Forward
#         enc_output = src_seq.unsqueeze(1)
#         enc_output = self.src_word_emb(src_seq)
        enc_output = src_seq

#         enc_output = self.pool(enc_output)
        enc_output = enc_output.transpose(1, 2)
        b1, b2, b3 = self.position_enc(src_pos).shape
#         enc_output.add_(nn.Parameter(torch.randn(1, b2, b3).to(torch.device("cuda:1"))))#可训练位置编码
    
        enc_output.add_(self.position_enc(src_pos))#不可训练位置编码
        
        attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            attn_list.append(enc_slf_attn)
        return enc_output, attn_list

class MultiBranchSEModule(nn.Module):
    def __init__(self, ecg_lead, d_model):
        super(MultiBranchSEModule, self).__init__()
        self.src_word_emb = nn.Conv1d(ecg_lead, 32, kernel_size=15, stride=1, padding=int((15 - 1) / 2))
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=9, stride=1, padding=int((9 - 1) / 2))
        self.pool2 = nn.MaxPool1d(2)
        
        self.src_word_emb1 = nn.Conv1d(ecg_lead, 32, kernel_size=13, stride=1, padding=int((13 - 1) / 2))
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, d_model, kernel_size=7, stride=1, padding=int((7 - 1) / 2))
        self.pool4 = nn.MaxPool1d(2)
        
        self.src_word_emb2 = nn.Conv1d(ecg_lead, 32, kernel_size=11, stride=1, padding=int((11 - 1) / 2))
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=int((5 - 1) / 2))
        self.pool6 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(3 * d_model, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, d_model // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 16, 3 * d_model, bias=False),
            nn.Sigmoid())
        
        self.residual_conv1 = nn.Conv1d(ecg_lead, 32, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.residual_pool1 = nn.MaxPool1d(2)
        self.residual_conv2 = nn.Conv1d(32, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.residual_pool2 = nn.MaxPool1d(2)
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, src_seq):
        res = self.residual_conv1(src_seq)
        res = self.residual_pool1(res)
        res = self.residual_conv2(res)
        res = self.residual_pool2(res)
        
        src_seq1 = self.src_word_emb(src_seq)
        src_seq1 = self.pool1(src_seq1)
#         src_seq1 = self.dropout(src_seq1)
        src_seq1 = self.conv2(src_seq1)
        src_seq1 = self.pool2(src_seq1)
        
        src_seq2 = self.src_word_emb1(src_seq)
        src_seq2 = self.pool3(src_seq2)
#         src_seq2 = self.dropout(src_seq2)
        src_seq2 = self.conv4(src_seq2)
        src_seq2 = self.pool4(src_seq2)
        
        src_seq3 = self.src_word_emb2(src_seq)
        src_seq3 = self.pool5(src_seq3)
#         src_seq3 = self.dropout(src_seq3)
        src_seq3 = self.conv6(src_seq3)
        src_seq3 = self.pool6(src_seq3)

        
        seq_all = torch.cat([src_seq1, src_seq2, src_seq3], dim=1)
        src_seq = seq_all
#         src_seq = self.dropout(src_seq)
        
#         print(src_seq.shape)
        b1, c1, _ = src_seq.size()
        se = self.globalpool(src_seq).view(b1, c1)
        se = self.fc(se).view(b1, c1, 1)
        src_seq = src_seq * se.expand_as(src_seq)
        src_seq = self.conv7(src_seq)
        
        
        src_seq = torch.add(src_seq, res)
        
        return src_seq
    
class MultiBranchSEModule2(nn.Module):
    def __init__(self, ecg_lead, d_model):
        super(MultiBranchSEModule2, self).__init__()
        self.src_word_emb = nn.Conv1d(ecg_lead, 32, kernel_size=15, stride=1, padding=int((15 - 1) / 2))
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=9, stride=1, padding=int((9 - 1) / 2))
        self.pool2 = nn.MaxPool1d(2)
        
        self.src_word_emb1 = nn.Conv1d(ecg_lead, 32, kernel_size=13, stride=1, padding=int((13 - 1) / 2))
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, d_model, kernel_size=7, stride=1, padding=int((7 - 1) / 2))
        self.pool4 = nn.MaxPool1d(2)
        
        self.src_word_emb2 = nn.Conv1d(ecg_lead, 32, kernel_size=11, stride=1, padding=int((11 - 1) / 2))
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=int((5 - 1) / 2))
        self.pool6 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(3 * d_model, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, d_model // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 16, 3 * d_model, bias=False),
            nn.Sigmoid())
        
        self.residual_conv1 = nn.Conv1d(ecg_lead, 32, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.residual_pool1 = nn.MaxPool1d(2)
        self.residual_conv2 = nn.Conv1d(32, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.residual_pool2 = nn.MaxPool1d(2)
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, src_seq):
        res = self.residual_conv1(src_seq)
        res = self.residual_pool1(res)
        res = self.residual_conv2(res)
        res = self.residual_pool2(res)
        
        src_seq1 = self.src_word_emb(src_seq)
        src_seq1 = self.pool1(src_seq1)
#         src_seq1 = self.dropout(src_seq1)
        src_seq1 = self.conv2(src_seq1)
        src_seq1 = self.pool2(src_seq1)
        
        src_seq2 = self.src_word_emb1(src_seq)
        src_seq2 = self.pool3(src_seq2)
#         src_seq2 = self.dropout(src_seq2)
        src_seq2 = self.conv4(src_seq2)
        src_seq2 = self.pool4(src_seq2)
        
        src_seq3 = self.src_word_emb2(src_seq)
        src_seq3 = self.pool5(src_seq3)
#         src_seq3 = self.dropout(src_seq3)
        src_seq3 = self.conv6(src_seq3)
        src_seq3 = self.pool6(src_seq3)
        seq_all = torch.cat([src_seq1, src_seq2, src_seq3], dim=1)
        src_seq = seq_all
#         src_seq = self.dropout(src_seq)
        
#         print(src_seq.shape)
        b1, c1, _ = src_seq.size()
        se = self.globalpool(src_seq).view(b1, c1)
        se = self.fc(se).view(b1, c1, 1)
        src_seq = src_seq * se.expand_as(src_seq)
        src_seq = self.conv7(src_seq)
        
        
        src_seq = torch.add(src_seq, res)
        
        return src_seq
    
    
class CNN_Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            class_num=5):

        super().__init__()
        self.src_word_emb = nn.Conv1d(ecg_lead, 32, kernel_size=15, stride=1, padding=int((15 - 1) / 2))
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=9, stride=1, padding=int((9 - 1) / 2))
        self.pool2 = nn.MaxPool1d(2)
        
        self.src_word_emb1 = nn.Conv1d(ecg_lead, 32, kernel_size=13, stride=1, padding=int((13 - 1) / 2))
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, d_model, kernel_size=7, stride=1, padding=int((7 - 1) / 2))
        self.pool4 = nn.MaxPool1d(2)
        
        self.src_word_emb2 = nn.Conv1d(ecg_lead, 32, kernel_size=11, stride=1, padding=int((11 - 1) / 2))
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=int((5 - 1) / 2))
        self.pool6 = nn.MaxPool1d(2)
        self.conv7 = nn.Conv1d(3*d_model, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(3*d_model, d_model//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//16, 3*d_model, bias=False),
            nn.Sigmoid())
        
        self.resconv1=nn.Conv1d(ecg_lead, 32, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.respool1=nn.MaxPool1d(2)
        self.resconv2=nn.Conv1d(32, d_model, kernel_size=1, stride=1, padding=int((1 - 1) / 2))
        self.respool2=nn.MaxPool1d(2)
        
        self.encoder = Encoder(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(feature_attn_len, 1, kernel_size=1)
#         self.linear1_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model*2, class_num)#64=>2
        self.linear1_cov0 = nn.Conv1d(d_model, ecg_lead, kernel_size=1)
        self.linear1_cov1 = nn.Conv1d(16, 1, kernel_size=1)
#         self.linear1_linear = nn.Linear(feature_attn_len*2, class_num)
#         # try different linear style(未連接手工特徵)
#         self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
#         self.linear2_linear = nn.Linear(d_feature, class_num)

        self.CNN_branch = MultiBranchSEModule(ecg_lead,d_model)
        self.CNN_branch2 = MultiBranchSEModule2(ecg_lead,d_model)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, src_seq):
#         print(src_seq.shape)
#         x1 = src_seq
        
#         std_low = 0.005
#         std_high = 0.01
#         for k in range(x1.shape[0]):
#             std_dev = random.uniform(std_low,std_high)
#             noise = torch.normal(mean=0, std=std_dev, size=(1, 256),device=x1.device)
#             x1[k] = torch.add(x1[k], noise)
        x1 = src_seq.clone()  # 创建 src_seq 的副本
        x1_noise = x1.clone()  # 创建 x1 的副本，用于保存添加噪声后的结果

        std_low = 0.005
        std_high = 0.01
        for k in range(x1_noise.shape[0]):
            std_dev = random.uniform(std_low, std_high)
            noise = torch.normal(mean=0, std=std_dev, size=(1, SIG_LEN), device=x1_noise.device)
            x1_noise[k] = torch.add(x1_noise[k], noise)
    
        res = self.resconv1(src_seq)
        res = self.respool1(res)
        res = self.resconv2(res)
        res = self.respool2(res)
        
        src_seq1 = self.src_word_emb(src_seq)#嵌入
        src_seq1 = self.pool1(src_seq1)
#         src_seq1 = self.dropout(src_seq1)
        src_seq1 = self.conv2(src_seq1)
        src_seq1 = self.pool2(src_seq1)
        
        src_seq2 = self.src_word_emb1(src_seq)#嵌入
        src_seq2 = self.pool3(src_seq2)
#         src_seq2 = self.dropout(src_seq2)
        src_seq2 = self.conv4(src_seq2)
        src_seq2 = self.pool4(src_seq2)
        
        src_seq3 = self.src_word_emb2(src_seq)#嵌入
        src_seq3 = self.pool5(src_seq3)
#         src_seq3 = self.dropout(src_seq3)
        src_seq3 = self.conv6(src_seq3)
        src_seq3 = self.pool6(src_seq3)
        seq_all = torch.cat([src_seq1,src_seq2,src_seq3],dim=1)
        src_seq = seq_all
#         src_seq = self.dropout(src_seq)
            

        b1,c1,_ = src_seq.size()
        se = self.globalpool(src_seq).view(b1,c1)
        se = self.fc(se).view(b1,c1,1)
        src_seq = src_seq*se.expand_as(src_seq)
        src_seq = self.conv7(src_seq)
        
        src_seq = torch.add(src_seq,res)
        
#         print(src_seq.size())#输出特征维度
        b, _, l = src_seq.size()
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, attn_map = self.encoder(src_seq, src_pos)
        dec_output = enc_output
#         dec_output = dec_output.permute(0, 2, 1)
#         print(dec_output.shape)
        res = self.linear1_cov(dec_output)
#         print(res.shape)
        res = res.contiguous().view(res.size()[0], -1)
#         print(res.shape)
#         res = self.linear1_linear(res)
        
#         print(res.shape)
        x = self.CNN_branch(x1_noise)
#         print(x.shape)
        x = self.linear1_cov0(x)
#         print(x.shape)
        x = self.CNN_branch2(x)
#         print(x.shape)
        x = x.permute(0,2,1)
        x = self.linear1_cov1(x)
        x = x.contiguous().view(x.size()[0], -1)
        branch_cnn_z = x
        branch_ct_z = res
#         print(x.shape)
        out = torch.cat((x,res),dim=1)
#         print(out.shape)
        out = self.linear1_linear(out)
        out = self.sigmoid(out)
#         print(out.shape)
        # return out, branch_cnn_z, branch_ct_z
        return out
    
# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1=64, size2=64):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


# Schedulers for beta
class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()


class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value


class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value

    
class MIBLossCalculator():
    def __init__(self, model, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000):
        self.model = model
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)
        self.mi_estimator = MIEstimator().to(device)  # Instantiate MIEstimator
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_mutual_information(self, z1, z2):
        # Pass z1 and z2 through the mi_estimator
        z1 = z1.to(device)
        z2 = z2.to(device)
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)

        # Return the gradient and estimation
        return mi_gradient, mi_estimation
    
    def compute_loss(self, v1, v2):
        # Encode a batch of data
        _,p_z1_given_v1,_ = self.model(v1)
        _,_,p_z2_given_v2 = self.model(v2)
#         # Sample from the posteriors with reparametrization
#         z1 = p_z1_given_v1.rsample()
#         z2 = p_z2_given_v2.rsample()
#         # Sample from the posteriors with reparameterization
#         z1 = self.reparameterize(p_z1_given_v1.mean, p_z1_given_v1.log_var)
#         z2 = self.reparameterize(p_z2_given_v2.mean, p_z2_given_v2.log_var)
        # Sample from the posteriors with reparameterization
        mu1, sigma1 = p_z1_given_v1[:, :64], p_z1_given_v1[:, 64:]
        mu2, sigma2 = p_z2_given_v2[:, :64], p_z2_given_v2[:, 64:]
        sigma1 = softplus(sigma1) + 1e-7  
        sigma2 = softplus(sigma2) + 1e-7  
        p_z1_given_v1 = Independent(Normal(loc=mu1, scale=sigma1), 1) 
        p_z2_given_v2 = Independent(Normal(loc=mu2, scale=sigma2), 1) 
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()
#         print(z1.size())
#         print(z2.size())
        # Mutual information estimation
        mi_gradient, mi_estimation = self.compute_mutual_information(z1, z2)
        
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()

        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
        kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1).mean() / 2.

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(50)

        # Logging the components
        # You need to implement _add_loss_item method
        # self._add_loss_item('loss/I_z1_z2', mi_estimation.item())
        # self._add_loss_item('loss/SKL_z1_z2', skl.item())
        # self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = - mi_gradient + beta * skl

        return loss
    
    
# model = CNN_Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner, n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)