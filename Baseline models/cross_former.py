import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_encoder import Encoder
from cross_decoder import Decoder
from attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from cross_embed import DSW_embedding

from math import ceil

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return base + predict_y[:, :self.out_len, :]
    
def create_crossformer_classifier(seq_len=256, num_classes=2):
    """
    创建一个用于分类的Crossformer模型，固定序列长度为256
    Args:
        seq_len: 输入序列长度，固定为256
        num_classes: 分类类别数
    Returns:
        CrossformerClassifier: 用于分类的Crossformer模型
    """
    if seq_len != 256:
        print("Warning: Crossformer模型已被优化为固定使用256的序列长度，输入的seq_len参数将被忽略")
    
    class Config:
        def __init__(self):
            self.seq_len = 256           # 固定输入序列长度为256
            self.pred_len = 64           # 预测长度为序列长度的1/4
            self.seg_len = 16            # 段长度设为16，因为256可以被16整除
            self.win_size = 4             # 窗口大小
            self.data_dim = 3             # 输入特征维度
            self.d_model = 512            # 模型维度
            self.d_ff = 2048             # 前馈网络维度
            self.n_heads = 8              # 注意力头数
            self.e_layers = 3             # 编码器层数
            self.dropout = 0.1            # dropout率
            self.factor = 5               # 注意力因子
            self.baseline = False         # 是否使用baseline
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CrossformerClassifier(nn.Module):
        def __init__(self, configs, num_classes):
            super(CrossformerClassifier, self).__init__()
            
            self.crossformer = Crossformer(
                data_dim=configs.data_dim,
                in_len=configs.seq_len,
                out_len=configs.pred_len,
                seg_len=configs.seg_len,
                win_size=configs.win_size,
                factor=configs.factor,
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                n_heads=configs.n_heads,
                e_layers=configs.e_layers,
                dropout=configs.dropout,
                baseline=configs.baseline,
                device=configs.device
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(configs.data_dim, 64),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(64, num_classes)
            )
            
            self.configs = configs

        def forward(self, x):
            # x shape: (batch_size, 3, 256)
            if x.shape[-1] != 256:
                raise ValueError(f"输入序列长度必须为256，但得到的是{x.shape[-1]}")
                
            x = x.transpose(1, 2)  # (batch_size, 256, 3)
            
            # Crossformer特征提取
            features = self.crossformer(x)
            
            # 取最后一个时间步的特征进行分类
            features = features[:, -1, :]  # (batch_size, 3)
            
            # 分类
            logits = self.classifier(features)
            return logits

    # 创建配置和模型
    configs = Config()
    model = CrossformerClassifier(configs, num_classes)
    
    return model