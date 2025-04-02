import torch
import torch.nn as nn
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import ReformerLayer
from Embed import DataEmbedding


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=configs.bucket_size,
                                  n_hashes=configs.n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]

        
def create_reformer_classifier(seq_len, num_classes):
    """
    创建一个用于分类的Reformer模型
    Args:
        seq_len: 输入序列长度
        num_classes: 分类类别数
    Returns:
        ReformerClassifier: 用于分类的Reformer模型
    """
    class Config:
        def __init__(self):
            self.seq_len = seq_len      # 输入序列长度
            self.pred_len = 1           # 预测长度
            self.output_attention = False
            self.enc_in = 3             # 输入特征维度
            self.d_model = 512          # 模型维度
            self.embed = 'timeF'        # 时间特征编码
            self.freq = 'h'             # 时间频率
            self.dropout = 0.1          # dropout率
            self.n_heads = 8            # 注意力头数
            self.d_ff = 2048           # 前馈网络维度
            self.e_layers = 3           # 编码器层数
            self.bucket_size = 64       # LSH bucket大小
            self.n_hashes = 4           # LSH哈希数
            self.activation = 'gelu'    # 激活函数
            self.c_out = 3              # 输出维度

    class ReformerClassifier(nn.Module):
        def __init__(self, configs, num_classes):
            super(ReformerClassifier, self).__init__()
            self.reformer = Model(configs)
            self.classifier = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes)
            )
            self.configs = configs

        def forward(self, x):
            # x shape: (batch_size, 3, seq_len)
            x = x.transpose(1, 2)  # (batch_size, seq_len, 3)
            
            # 创建必要的时间特征和占位符
            batch_size = x.shape[0]
            x_mark = torch.zeros(batch_size, self.configs.seq_len, 4).to(x.device)
            x_dec = torch.zeros(batch_size, self.configs.pred_len, 3).to(x.device)
            x_mark_dec = torch.zeros(batch_size, self.configs.pred_len, 4).to(x.device)
            
            # Reformer特征提取
            features = self.reformer(x, x_mark, x_dec, x_mark_dec)
            
            # 取最后一个时间步的特征进行分类
            features = features[:, -1, :]  # (batch_size, 3)
            
            # 分类
            logits = self.classifier(features)
            return logits

    # 创建配置和模型
    configs = Config()
    model = ReformerClassifier(configs, num_classes)
    
    return model