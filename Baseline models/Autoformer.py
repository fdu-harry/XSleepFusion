import torch
import torch.nn as nn
from Embed import DataEmbedding_wo_pos
from AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


        
# #在文件末尾添加新的函数
def create_autoformer_classifier(seq_len, num_classes):
    """
    创建一个用于分类的Autoformer模型
    Args:
        seq_len: 输入序列长度
        num_classes: 分类类别数
    Returns:
        AutoformerClassifier: 用于分类的Autoformer模型
    """
    class Config:
        def __init__(self):
            self.seq_len = seq_len      # 输入序列长度
            self.label_len = seq_len//2 # 标签长度
            self.pred_len = 1           # 预测长度
            self.output_attention = False
            self.moving_avg = 25
            self.enc_in = 3             # 输入特征维度
            self.dec_in = 3             # 解码器输入维度
            self.c_out = 3              # 输出维度
            self.d_model = 512
            self.embed = 'timeF'
            self.freq = 'h'
            self.dropout = 0.1
            self.factor = 1
            self.n_heads = 8
            self.d_ff = 2048
            self.e_layers = 2
            self.d_layers = 1
            self.activation = 'gelu'

    class AutoformerClassifier(nn.Module):
        def __init__(self, configs, num_classes):
            super(AutoformerClassifier, self).__init__()
            self.autoformer = Model(configs)
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
            
            # 创建必要的时间特征和解码器输入
            batch_size = x.shape[0]
            x_mark = torch.zeros(batch_size, self.configs.seq_len, 4).to(x.device)
            x_dec = torch.zeros(batch_size, self.configs.label_len + self.configs.pred_len, 3).to(x.device)
            x_mark_dec = torch.zeros(batch_size, self.configs.label_len + self.configs.pred_len, 4).to(x.device)
            
            # Autoformer特征提取
            features = self.autoformer(x, x_mark, x_dec, x_mark_dec)
            
            # 取最后一个时间步的特征进行分类
            features = features[:, -1, :]  # (batch_size, 3)
            
            # 分类
            logits = self.classifier(features)
            return logits

    # 创建配置和模型
    configs = Config()
    model = AutoformerClassifier(configs, num_classes)
    
    return model