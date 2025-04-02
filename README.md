# XSleepFusion

## Publication
**Our paper:** *A Dual-stage Information Bottleneck Fusion Framework for Interpretable Multimodal Sleep Analysis*  
**Status:** <span style="color: red">**Major revision**</span>  
**Journal:** Information Fusion, 2025

# Time Series Classification Models

This repository contains implementations of various deep learning models for time series classification. Each model accepts input shape of (batch_size, 3, 256).

Note: For comparison models, the number of input channels can be adjusted by modifying the intermediate convolutional channel parameters in their respective .py files. The default setting is 3 channels.

## Model Files and Their Corresponding Papers

### Core Models
- `Pyramid_CNN.py`: Pyramid CNN model
  > "Multi-Scale Convolutional Neural Networks for Time Series Classification" (2020)

- `ResNet_1D.py`: 1D ResNet model
  > "Deep Residual Learning for Image Recognition" (2016)

- `MACNN.py`: Multi-Attention CNN model
  > "Multi-Attention Convolutional Neural Network for Time Series Classification" (2021)

- `Hybrid_Net.py`: Hybrid CNN-Transformer model
  > "Hybrid CNN-Transformer Networks for Time Series Classification" (2022)

- `Transformer.py`: Standard Transformer model
  > "Attention Is All You Need" (2017)

- `ViT_1D.py`: Vision Transformer for 1D signals
  > "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)

- `SwinT_1D.py`: Swin Transformer for 1D signals
  > "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)

- `Reformer.py`: Reformer model
  > "Reformer: The Efficient Transformer" (2020)

- `Informer.py`: Informer model
  > "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (2021)

- `Autoformer.py`: Autoformer model
  > "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (2021)

- `cross_former.py`: Crossformer model
  > "Crossformer: A Versatile Vision Transformer Based on Cross-scale Attention" (2022)

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
- 
![image](https://github.com/fdu-harry/XSleepFusion/blob/main/XSleepFusion.jpg)
