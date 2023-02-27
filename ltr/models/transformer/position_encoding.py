import math
import torch
from torch import nn


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine', avoid_aliasing=False, max_spatial_resolution=None):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

        if avoid_aliasing and max_spatial_resolution == None:
            raise ValueError('Please specify the maxima spatial resolution (h, w) of the feature map')
        elif avoid_aliasing:
            self.factor = max_spatial_resolution/depth
        else:
            self.factor = 1.

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([torch.sin(i * self.factor * math.pi * inputs) for i in self.bases] +
                        [torch.cos(i * self.factor * math.pi * inputs) for i in self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    一个更标准的位置嵌入，与Attention is all you need 论文所使用的版本非常相似，一般用于处理图像
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine',
                 avoid_aliazing=False, max_spatial_resolution=None):
        #num_pos_feats=128, sine_type='lin_sine',avoid_aliazing=True, max_spatial_resolution=feature_sz

        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.sine = NerfPositionalEncoding(num_pos_feats//2, sine_type, avoid_aliazing, max_spatial_resolution)

    @torch.no_grad()
    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask  #~按位取反操作
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  #cumsum给定axis上的累计和
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
        pos = torch.stack([x_embed, y_embed], dim=-1)   #沿最后一个维度对输入的张量进行连接，序列中的所有张量应为相同形状
        return self.sine(pos).permute(0, 3, 1, 2)
