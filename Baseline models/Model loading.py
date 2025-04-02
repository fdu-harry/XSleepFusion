# Import various deep learning models for 1D signal processing
from Pyramid_CNN import pyramid_CNN
from ResNet_1D import resnet50
from MACNN import MACNN
from Hybrid_Net import Transformer

from Transformer import create_transformer_classifier  
from ViT_1D import create_vit_1d_model_b
from SwinT_1D import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224

from Reformer import create_reformer_classifier
from Informer import create_informer_classifier 
from Autoformer import create_autoformer_classifier
from cross_former import create_crossformer_classifier

warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib.font_manager')

# Setting global parameters
batch_size = 512  # Number of samples processed in each batch
d_model = 128     # Model dimension 
d_inner = 512     # Inner dimension of feed forward network
num_layers = 3    # Number of transformer layers
num_heads = 4     # Number of attention heads
class_num = 2     # Number of output classes
dropout = 0.0     # Dropout rate
warm_steps = 4000 # Steps for learning rate warmup
num_epochs = 200  # Total training epochs
SIG_LEN = 256    # Input signal length
ecg_lead = 3     # Number of input channels
feature_attn_len = 64  # Feature attention length

# Setting up GPU/CPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initializing and loading models
# All models below expect input shape of (batch_size, 3, 256)
# where:
# - batch_size: number of samples in a batch 
# - 3: number of input channels
# - 256: length of 1D signal sequence

# model = pyramid_CNN(n_chans=ecg_lead, n_classes=class_num)
# model = resnet50(num_classes=2, include_top=True) 
# model = MACNN()
# model = Transformer(device=device, d_feature=SIG_LEN, d_model=64, d_inner=d_inner,
#             n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
# model = CNN_Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner,
#                         n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
# model = create_vit_1d_model_b()
# model = swin_tiny_patch4_window7_224()
# model = create_transformer_classifier(seq_len=256,num_classes=2)
# model = create_reformer_classifier(seq_len=256,num_classes=2)    
# model = create_informer_classifier(seq_len=256,num_classes=2)
# model = create_autoformer_classifier(seq_len=256,num_classes=2)   
model = create_crossformer_classifier(seq_len=256,num_classes=2)   

# Move model to specified device (GPU/CPU)
model = model.to(device)
