
import pickle
from scipy.signal import resample
import time
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from torchvision import datasets, transforms
from matplotlib.ticker import MultipleLocator
import scipy.signal as signal
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# d_model = 64
# num_layers = 3
# num_heads = 4
# class_num = 1
# d_inner = 512
# dropout = 0.0
# warm_steps = 4000
# fea_num = 7
# epoch = 50
PAD = 0
# KS = 11
# Fea_PLUS = 2
# SIG_LEN = 1250
ecg_lead=1
feature_attn_len=64
# k=0


# In[5]:


#sublayer.py   -> main.py 8 click
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


# In[6]:


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


# In[7]:


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
#         enc_output.add_(nn.Parameter(torch.randn(1, b2, b3).to(torch.device("cuda:0"))))#可训练位置编码
    
        enc_output.add_(self.position_enc(src_pos))#不可训练位置编码
        
        attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            attn_list.append(enc_slf_attn)
        return enc_output, attn_list


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            class_num=1):

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
        self.drop=nn.Dropout(0.2)
        self.linear1_linear = nn.Linear(d_model, 1)#64=>2
#         # try different linear style(未連接手工特徵)
#         self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
#         self.linear2_linear = nn.Linear(d_feature, class_num)

    def forward(self, src_seq):
        res = self.resconv1(src_seq)
        res = self.respool1(res)
        res = self.resconv2(res)
        res = self.respool2(res)

        
        src_seq1 = self.src_word_emb(src_seq)#嵌入
        src_seq1 = self.pool1(src_seq1)
        src_seq1 = self.conv2(src_seq1)
        src_seq1 = self.pool2(src_seq1)
        src_seq2 = self.src_word_emb1(src_seq)#嵌入
        src_seq2 = self.pool3(src_seq2)
        src_seq2 = self.conv4(src_seq2)
        src_seq2 = self.pool4(src_seq2)
        src_seq3 = self.src_word_emb2(src_seq)#嵌入
        src_seq3 = self.pool5(src_seq3)
        src_seq3 = self.conv6(src_seq3)
        src_seq3 = self.pool6(src_seq3)
        seq_all = torch.cat([src_seq1,src_seq2,src_seq3],dim=1)
        src_seq = seq_all
        
        b1,c1,_ = src_seq.size()
        se = self.globalpool(src_seq).view(b1,c1)
        se = self.fc(se).view(b1,c1,1)
        src_seq = src_seq*se.expand_as(src_seq)
        src_seq = self.conv7(src_seq)
        src_seq = torch.add(src_seq,res)

        
        # print(src_seq.size())#输出特征维度
        x_transposed = torch.transpose(src_seq, 1, 2)
        # print(x_transposed.size())#输出特征维度
        b, _, l = src_seq.size()
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)
        enc_output, attn_map = self.encoder(src_seq, src_pos)
   
        dec_output = enc_output

        res = self.linear1_cov(x_transposed)

        res = res.contiguous().view(res.size()[0], -1)

        res = self.drop(res)
        res = self.linear1_linear(res)
        return res, attn_map





