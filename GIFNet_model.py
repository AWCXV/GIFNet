import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  
from args import Args as args
import utils
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out
from PIL import Image
import math

EPSILON = 1e-4
MAX = 1e2


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

#SwinTransformer - Window patition
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

#SwinTransformer - Window reverse
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#SwinTransformer - get relative position
def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

#SwinTransformer - Window attention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.shift_size = shift_size

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, qkv_mfif, trainingTag):
        #MFIF training
        if (trainingTag == 2):
            B_, N, _ = qkv.shape        
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_mfif = qkv_mfif.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q_mfif, k_mfif, v_mfif = qkv_mfif[0], qkv_mfif[1], qkv_mfif[2]  # make torchscript happy (cannot use tensor as tuple)

            #text modality -> vision
            with torch.no_grad():            
                q = q * self.scale
            q_mfif = q_mfif * self.scale
            
            with torch.no_grad():            
                if (self.shift_size == 0):
                    attn = (q_mfif @ k.transpose(-2, -1))            
                else:
                    attn = (q @ k.transpose(-2, -1))            
            attn_mfif = (q_mfif @ k_mfif.transpose(-2, -1))


            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            with torch.no_grad():
                attn = attn + relative_position_bias.unsqueeze(0)        
            attn_mfif = attn_mfif + relative_position_bias.unsqueeze(0)

            with torch.no_grad():
                attn = self.softmax(attn)
            attn_mfif = self.softmax(attn_mfif)

            with torch.no_grad():        
                x_ivif = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
            x_mfif = (attn_mfif @ v_mfif).transpose(1, 2).reshape(B_, N, self.dim)
        elif (trainingTag == 1):
            B_, N, _ = qkv.shape        
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_mfif = qkv_mfif.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q_mfif, k_mfif, v_mfif = qkv_mfif[0], qkv_mfif[1], qkv_mfif[2]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            with torch.no_grad():
                q_mfif = q_mfif * self.scale
            
            attn = (q @ k.transpose(-2, -1))
            
            with torch.no_grad():
                if (self.shift_size == 0):
                    attn_mfif = (q @ k_mfif.transpose(-2, -1))
                else:
                    attn_mfif = (q_mfif @ k_mfif.transpose(-2, -1))

            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            attn = attn + relative_position_bias.unsqueeze(0)        
            with torch.no_grad():
                attn_mfif = attn_mfif + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            with torch.no_grad():
                attn_mfif = self.softmax(attn_mfif)

            x_ivif = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
            with torch.no_grad():        
                x_mfif = (attn_mfif @ v_mfif).transpose(1, 2).reshape(B_, N, self.dim)

        return x_ivif, x_mfif

spe_transformer_cur_depth = 0

def save_feature_maps_as_images(feature_maps, alias, numFeatures = 3, output_folder="visualization"):
    global spe_transformer_cur_depth
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Assuming feature_maps is a PyTorch tensor with dimensions BxCxHxW
    batch_size, num_channels, height, width = feature_maps.size()    
    for i in range(batch_size):
        for j in range(numFeatures):
            # Get the j-th channel of the i-th feature map
            channel_data = feature_maps[i, j, :, :].cpu().detach().numpy()

            # Normalize values to be in the range [0, 255]
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min()) * 255

            # Convert to uint8
            channel_data = channel_data.astype('uint8')

            # Create a PIL Image from the numpy array
            image = Image.fromarray(channel_data)

            # Save the image
            image_path = os.path.join(output_folder, f"content_{alias}_spe_transformer_cur_depth_{spe_transformer_cur_depth}_feature_map_{0}_channel_{j}.jpg")
            image.save(image_path)

#SwinTransformer - Attention
class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
            self.conv_mfif = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.V_mfif = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            self.proj_mfif = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, 2*dim, 1)
            self.QK_mfif = nn.Conv2d(dim, 2*dim, 1)
            self.attn = WindowAttention(dim, window_size, num_heads, shift_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x_ivif, x_mfif, trainingTag):
        B, C, H, W = x_ivif.shape
        
        #MFIF task
        if (trainingTag == 2):
        
            #print(x_ivif.shape);
            if self.conv_type == 'DWConv' or self.use_attn:
                with torch.no_grad():            
                    V = self.V(x_ivif)
                V_mfif = self.V_mfif(x_mfif)

            #save_feature_maps_as_images(V_mfif,"v_mfif");
            
            #print("V.shape:");
            #print(V.shape);

            if self.use_attn:
                with torch.no_grad():            
                    QK = self.QK(x_ivif)
                QK_mfif = self.QK_mfif(x_mfif)

                #save_feature_maps_as_images(QK_mfif[:,:self.dim,:,:],"q_mfif");
                #save_feature_maps_as_images(QK_mfif[:,self.dim:,:,:],"k_mfif");

                with torch.no_grad():
                    QKV = torch.cat([QK, V], dim=1)
                QKV_mfif = torch.cat([QK_mfif, V_mfif], dim=1)



                with torch.no_grad():                
                    shifted_QKV = self.check_size(QKV, self.shift_size > 0)
                shifted_QKV_mfif = self.check_size(QKV_mfif, self.shift_size > 0)
                
                
                Ht, Wt = shifted_QKV.shape[2:]

                # partition windows
                with torch.no_grad():                
                    shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
                shifted_QKV_mfif = shifted_QKV_mfif.permute(0, 2, 3, 1)

                qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
                qkv_mfif = window_partition(shifted_QKV_mfif, self.window_size)  # nW*B, window_size**2, C

                #attn_windows = self.attn(qkv)
                #attn_windows_mfif = self.attn(qkv_mfif)
                attn_windows, attn_windows_mfif = self.attn(qkv, qkv_mfif, trainingTag)
                

                # merge windows
                shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
                shifted_out_mfif = window_reverse(attn_windows_mfif, self.window_size, Ht, Wt)  # B H' W' C

                # reverse cyclic shift
                out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
                out_mfif = shifted_out_mfif[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

                attn_out = out.permute(0, 3, 1, 2)
                attn_out_mfif = out_mfif.permute(0, 3, 1, 2)

                if self.conv_type in ['Conv', 'DWConv']:
                    with torch.no_grad():                
                        conv_out = self.conv(V)
                    conv_out_mfif = self.conv_mfif(V_mfif)
                    
                    with torch.no_grad():                    
                        out = self.proj(conv_out + attn_out)
                    out_mfif = self.proj_mfif(conv_out_mfif + attn_out_mfif)
                else:
                    with torch.no_grad():                                    
                        out = self.proj(attn_out)
                    out_mfif = self.proj_mfif(attn_out_mfif)

            else:
                if self.conv_type == 'Conv':
                    out = self.conv(x_ivif)                # no attention and use conv, no projection
                    out_mfif = self.conv_mfif(x_mfif)                # no attention and use conv, no projection
                elif self.conv_type == 'DWConv':
                    out = self.proj(self.conv(V))
                    out_mfif = self.proj_mfif(self.conv_mfif(V_mfif))
        elif (trainingTag == 1):
            #print(x_ivif.shape);
            if self.conv_type == 'DWConv' or self.use_attn:
                V = self.V(x_ivif)
                with torch.no_grad():
                    V_mfif = self.V_mfif(x_mfif)
            #print("V.shape:");
            #print(V.shape);
            if self.use_attn:
                QK = self.QK(x_ivif)
                with torch.no_grad():
                    QK_mfif = self.QK_mfif(x_mfif)

                QKV = torch.cat([QK, V], dim=1)
                
                #print("QKV.shape:");
                #print(QKV.shape);                
                #
                #print(V.shape);                
                with torch.no_grad():
                    QKV_mfif = torch.cat([QK_mfif, V_mfif], dim=1)

                # shift
                shifted_QKV = self.check_size(QKV, self.shift_size > 0)
                with torch.no_grad():
                    shifted_QKV_mfif = self.check_size(QKV_mfif, self.shift_size > 0)
                Ht, Wt = shifted_QKV.shape[2:]

                # partition windows
                shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
                with torch.no_grad():
                    shifted_QKV_mfif = shifted_QKV_mfif.permute(0, 2, 3, 1)

                qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
                qkv_mfif = window_partition(shifted_QKV_mfif, self.window_size)  # nW*B, window_size**2, C

                #attn_windows = self.attn(qkv)
                #attn_windows_mfif = self.attn(qkv_mfif)
                attn_windows, attn_windows_mfif = self.attn(qkv, qkv_mfif, trainingTag)
                

                # merge windows
                shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
                shifted_out_mfif = window_reverse(attn_windows_mfif, self.window_size, Ht, Wt)  # B H' W' C

                # reverse cyclic shift
                out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
                out_mfif = shifted_out_mfif[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

                attn_out = out.permute(0, 3, 1, 2)
                attn_out_mfif = out_mfif.permute(0, 3, 1, 2)

                if self.conv_type in ['Conv', 'DWConv']:
                    conv_out = self.conv(V)
                    with torch.no_grad():
                        conv_out_mfif = self.conv_mfif(V)
                    
                    out = self.proj(conv_out + attn_out)
                    with torch.no_grad():
                        out_mfif = self.proj_mfif(conv_out_mfif + attn_out_mfif)
                else:
                    out = self.proj(attn_out)
                    with torch.no_grad():                    
                        out_mfif = self.proj_mfif(attn_out_mfif)

            else:
                if self.conv_type == 'Conv':
                    out = self.conv(x_ivif)                # no attention and use conv, no projection
                    out_mfif = self.conv_mfif(x_mfif)                # no attention and use conv, no projection
                elif self.conv_type == 'DWConv':
                    out = self.proj(self.conv(V))
                    out_mfif = self.proj_mfif(self.conv_mfif(V_mfif))
        
        return out, out_mfif


class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

        
		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)

#SwinTransformer - TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.norm1_mfif = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.norm2_mfif = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_mfif = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x_ivif, x_mfif, trainingTag):
    
        #train IVIF task;
        if (trainingTag == 1):
            identity = x_ivif
            identity_mfif = x_mfif
            if self.use_attn: x_ivif, rescale, rebias = self.norm1(x_ivif)
            if self.use_attn: x_mfif, rescale_mfif, rebias_mfif = self.norm1_mfif(x_mfif)

            x_ivif, x_mfif = self.attn(x_ivif, x_mfif, trainingTag)

            if self.use_attn: x_ivif = x_ivif * rescale + rebias
            if self.use_attn: x_mfif = x_mfif * rescale_mfif + rebias_mfif

            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif

            identity = x_ivif
            identity_mfif = x_mfif

            if self.use_attn and self.mlp_norm: x_ivif, rescale, rebias = self.norm2(x_ivif)
            
            if self.use_attn and self.mlp_norm: x_mfif, rescale_mfif, rebias_mfif = self.norm2_mfif(x_mfif)

            x_ivif = self.mlp(x_ivif)
            x_mfif = self.mlp_mfif(x_mfif)

            if self.use_attn and self.mlp_norm: x_ivif = x_ivif * rescale + rebias
            if self.use_attn and self.mlp_norm: x_mfif = x_mfif * rescale_mfif + rebias_mfif


            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif
        elif (trainingTag == 2):
            #trainMFIF task;
            identity = x_ivif
            identity_mfif = x_mfif
            if self.use_attn: x_ivif, rescale, rebias = self.norm1(x_ivif)
            if self.use_attn: x_mfif, rescale_mfif, rebias_mfif = self.norm1_mfif(x_mfif)

            x_ivif, x_mfif = self.attn(x_ivif, x_mfif, trainingTag)

            if self.use_attn: x_ivif = x_ivif * rescale + rebias
            if self.use_attn: x_mfif = x_mfif * rescale_mfif + rebias_mfif

            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif

            identity = x_ivif
            identity_mfif = x_mfif

            if self.use_attn and self.mlp_norm: x_ivif, rescale, rebias = self.norm2(x_ivif)
            
            if self.use_attn and self.mlp_norm: x_mfif, rescale_mfif, rebias_mfif = self.norm2_mfif(x_mfif)

            x_ivif = self.mlp(x_ivif)
            x_mfif = self.mlp_mfif(x_mfif)

            if self.use_attn and self.mlp_norm: x_ivif = x_ivif * rescale + rebias
            if self.use_attn and self.mlp_norm: x_mfif = x_mfif * rescale_mfif + rebias_mfif


            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif
        return x_ivif, x_mfif

#Swintransformer - Basic
class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(depth)])            
        self.weights_mfif = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(depth)])            

    def forward(self, x_ivif, x_mfif, trainingTag):
        global spe_transformer_cur_depth
        # IVIF train
        if (trainingTag == 1):
            for i, blk in enumerate(self.blocks):
                #identity_ivif = x_ivif;
                x_ivif, x_mfif  = blk(x_ivif, x_mfif, trainingTag)
                weight_i = self.weights[i];          

                if (i % 2 == 0):
                    x_ivif = x_ivif + weight_i*x_mfif;

            return x_ivif;
        elif (trainingTag == 2):
        # MFIF train
            for i, blk in enumerate(self.blocks):
                spe_transformer_cur_depth = i;
                x_ivif, x_mfif  = blk(x_ivif, x_mfif, trainingTag)
                weight_i = self.weights_mfif[i];          
                if (i % 2 == 0):
                    x_mfif = x_mfif + weight_i*x_ivif;                
            return x_mfif;

#SwinTransformer - Embed
class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

        #1x1 conv
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x

#SwinTransformer - UnEmbed
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


#SwinTransformer - Main
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
    
        self.patch_size = 4
        embed_dims=[32+32+32+2,48]
        depths = [2]
        num_heads = [2]
        attn_ratio = [1] #all layers use attention
        conv_type = ['DWConv']
        mlp_ratios=[2.]
        window_size=8
        in_chans = embed_dims[0]
        norm_layer=[RLN, RLN, RLN, RLN, RLN]
        # backbone
        self.patch_embed_ivif = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)            
            
        self.patch_embed_mfif = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)                        
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])     
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=embed_dims[0], embed_dim=embed_dims[0], kernel_size=3)                                 
                
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, mod_pad_w, mod_pad_h

    def forward(self, x_ivif, x_mfif, trainingTag):

        #save_feature_maps_as_images(x_ivif,"cnnFeatures_ivif",20);

        x_ivif, mod_pad_w_ivif, mod_pad_h_ivif = self.check_image_size(x_ivif);     
        x_mfif, mod_pad_w_mfif, mod_pad_h_mfif = self.check_image_size(x_mfif);     

        x_ivif = self.patch_embed_ivif(x_ivif);        
        x_mfif = self.patch_embed_mfif(x_mfif);        
        
        x = self.layer1(x_ivif, x_mfif, trainingTag);        
        
        #save_feature_maps_as_images(x_mfif,"afterAttentionMF",20);
        #save_feature_maps_as_images(x_ivif,"afterAttentionIVIF",20);
        
        x = self.patch_unembed(x);        
        _,_,h,w = x.size()
        
        x = x[:, :, :h - mod_pad_h_ivif, :w - mod_pad_w_ivif]        
        
        return x

#Shared feature fusion module
class ComplementFeatureFusionModule(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(ComplementFeatureFusionModule, self).__init__()

        self.height = height
        d = (32+32+32+2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d((32+32+32+2)*2, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)

        attn = self.mlp(in_feats) # mlp(B*C*1*1)->B*(C*2)*1*1       

        return attn

#Specific Feature Extraction & decoder
class TransformerSpecificExtractor(nn.Module):
    def __init__(self):
        super(TransformerSpecificExtractor, self).__init__()

        self.SwinTransformerSpecific = TransformerNet()

    def forward(self, x_ivif, x_mfif, trainingTag):
        
        x = self.SwinTransformerSpecific(x_ivif, x_mfif, trainingTag);

        return x

class CNNspecificDecoder(nn.Module):
    def __init__(self, embed_size, num_decoder_layers):
        super(CNNspecificDecoder, self).__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size*2)        
        # Decoder
        layers = []
        channels = [embed_size,embed_size//2,embed_size//4,1];
        lastOut = embed_size*2
        cur_depth = 0;
        for _ in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if (_==num_decoder_layers-1):
                    layers.append(nn.Tanh());
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth];
            cur_depth += 1;
        self.decoder = nn.Sequential(*layers)        
    
    def forward(self,fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused);
        x = self.decoder(x);
        x = x / 2 + 0.5;        
        return x;
        

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, isLast):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if (isLast == True):
            self.ac = nn.Tanh()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.ac(out)
        return out

#Shared Feature Extraction
class SharedFeatureExtractor(nn.Module):
    def __init__(self, s, n, channel, stride):
        super(SharedFeatureExtractor, self).__init__()
        # n = 64  # number of filters
        # s = 3  # filter size
        # Channel = 3
        self.n = n
        self.conv_1 = ConvLayer(channel*2, 32, s, stride, isLast = False)
        self.conv_2 = ConvLayer(32+2, 32, s, stride, isLast = False)
        self.conv_3 = ConvLayer(32+32+2, 32, s, stride, isLast = False)

    def forward(self, x):
        x_1 = self.conv_1(x)  # Z_0
        x_2 = self.conv_2(torch.cat((x,x_1),1))  # Z_0
        x_3 = self.conv_3(torch.cat((x,x_1,x_2),1))  # Z_0        
        return torch.cat((x,x_1,x_2,x_3),1)

#Shared Feature Extraction
class ReconstructionDecoder(nn.Module):
    def __init__(self, embed_size, num_decoder_layers):
        super(ReconstructionDecoder, self).__init__()
        # Decoder
        layers = []
        channels = [embed_size,embed_size//2,embed_size//4,1];
        lastOut = 32+32+32+2
        cur_depth = 0;
        for _ in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if (_==num_decoder_layers-1):
                layers.append(nn.Tanh());
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth];
            cur_depth += 1;
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x);
        return x


class TwoBranchesFusionNet(nn.Module):
    def __init__(self, s, n, channel, stride):
        super(TwoBranchesFusionNet, self).__init__()
        
        self.getSharedFeatures = SharedFeatureExtractor(s, n, channel, stride)        
        num_decoder_layers = 4;
        self.decoder_rec = ReconstructionDecoder(n,num_decoder_layers);                
        
        embed_size = int(args.n)
        heads = 4
        num_transformer_blocks = 2
        num_decoder_layers = 4

        self.extractor_multask = TransformerSpecificExtractor()
        self.cnnDecoder = CNNspecificDecoder(embed_size, num_decoder_layers);                        
        
    def forward_encoder(self, x, y):
        x = torch.cat((x,y),1)
        fea_x = self.getSharedFeatures(x)
        
        fea_com = fea_x;
        
        return fea_com;    
    
    #trainingTag = 1, IVIF task; trainingTag = 2, MFIF task;
    def forward_MultiTask_branch(self, fea_com_ivif, fea_com_mfif, trainingTag = 2):
        x = self.extractor_multask(fea_com_ivif, fea_com_mfif, trainingTag);
        return x;
    
    def forward_mixed_decoder(self, fea_com, fea_fused):
        x = self.cnnDecoder([fea_com,fea_fused]);
        return x;
        
    def forward_rec_decoder(self, fea_com):
        return self.decoder_rec(fea_com);                
        
    def forward(self, x, y):
        fea_com = self.forward_encoder(x, y);
        output = self.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com, trainingTag = 2)   
        return output