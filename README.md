# XSleepFusion

## Publication
**Our paper:** *A Dual-stage Information Bottleneck Fusion Framework for Interpretable Multimodal Sleep Analysis*  
**Status: 🔄 Major revision**  
**Journal:** Information Fusion, 2025
![image](https://github.com/fdu-harry/XSleepFusion/blob/main/XSleepFusion.jpg)

# Time Series Classification Models

This repository contains implementations of various deep learning models for time series classification. Each model accepts input shape of (batch_size, 3, 256).

Note: For comparison models, the number of input channels can be adjusted by modifying the intermediate convolutional channel parameters in their respective .py files. The default setting is 3 channels.

## Model Files and Their Corresponding Papers

### Core Models
- `Pyramid_CNN.py`: Pyramid CNN model
  > "EEGWaveNet: Multiscale CNN-based spatiotemporal feature extraction for EEG seizure detection" (2021)

- `ResNet_1D.py`: 1D ResNet model
  > "Deep residual learning for image recognition" (2016)

- `MACNN.py`: Multi-Attention CNN model
  > "Multi-scale attention convolutional neural network for time series classification" (2021)

- `Hybrid_Net.py`: Hybrid CNN-Transformer model
  > "A hybrid transformer model for obstructive sleep apnea detection based on self-attention mechanism using single-lead ECG" (2022)

- `Transformer.py`: Standard Transformer model
  > "Attention is all you need" (2017)

- `ViT_1D.py`: Vision Transformer for 1D signals
  > "An image is worth 16x16 words: Transformers for image recognition at scale" (2020)

- `SwinT_1D.py`: Swin Transformer for 1D signals
  > "Swin transformer: Hierarchical vision transformer using shifted windows" (2021)

- `Reformer.py`: Reformer model
  > "Reformer: The efficient transformer" (2020)

- `Informer.py`: Informer model
  > "Informer: Beyond efficient transformer for long sequence time-series forecasting" (2021)

- `Autoformer.py`: Autoformer model
  > "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting" (2021)

- `cross_former.py`: Crossformer model
  > "Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting" (2022)

### Dependency Files
The following files are dependencies required by the main models:
- `Autoformer_EncDec.py`: Encoder-Decoder components for Autoformer
- `Embed.py`: Embedding layers implementation
- `SelfAttention_Family.py`: Various self-attention mechanisms
- `Transformer_EncDec.py`: Encoder-Decoder components for Transformer
- `attn.py`: Attention mechanism implementations
- `cross_decoder.py`: Decoder for Crossformer
- `cross_embed.py`: Embedding for Crossformer
- `cross_encoder.py`: Encoder for Crossformer
- `masking.py`: Masking utilities
- `Model_loading.py`: Model loading utilities
- `AutoCorrelation.py`: Auto-correlation layer implementations
