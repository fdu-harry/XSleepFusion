import torch
import torch.nn as nn
from Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        
def create_transformer_classifier(seq_len, num_classes):
    """
    创建一个用于分类的Transformer模型
    Args:
        seq_len: 输入序列长度
        num_classes: 分类类别数
    Returns:
        TransformerClassifier: 用于分类的Transformer模型
    """
    class Config:
        def __init__(self):
            self.seq_len = seq_len      # 输入序列长度
            self.label_len = seq_len//2 # 标签长度
            self.pred_len = 1           # 预测长度
            self.output_attention = False
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

    class TransformerClassifier(nn.Module):
        def __init__(self, configs, num_classes):
            super(TransformerClassifier, self).__init__()
            self.transformer = Model(configs)
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
            
            # Transformer特征提取
            features = self.transformer(x, x_mark, x_dec, x_mark_dec)
            
            # 取最后一个时间步的特征进行分类
            features = features[:, -1, :]  # (batch_size, 3)
            
            # 分类
            logits = self.classifier(features)
            return logits

    # 创建配置和模型
    configs = Config()
    model = TransformerClassifier(configs, num_classes)
    
    return model