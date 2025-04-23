import numpy as np
import skimage
import torchvision
from matplotlib import pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
import einops
import numpy
DEVICE = 'cuda:1'

class Attention_FRN(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=dim*patch_size*patch_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )
        self.Pad = nn.ZeroPad2d(2)
        # self.k_2_patch = torch.zeros(8, patch_size, patch_size, 256, 9)

    def forward(self, x, y, HSI_Patch, MSI_Patch2):
        HSI_Patch = HSI_Patch.reshape(HSI_Patch.shape[0], HSI_Patch.shape[1], -1).permute(0,2,1)
        MSI_Patch2 = MSI_Patch2.reshape(MSI_Patch2.shape[0], MSI_Patch2.shape[1], -1).permute(0,2,1)
        B, L, C = HSI_Patch.shape
        qkv_1 = self.qkv_1(HSI_Patch)
        qkv_2 = self.qkv_2(MSI_Patch2)

        qkv_1 = einops.rearrange(qkv_1, 'B L (K D) -> K B L D', K=3)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # B H L D
        qkv_2 = einops.rearrange(qkv_2, 'B L (K D) -> K B L D', K=3)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]
        k_2_2D = k_2.permute(0, 2, 1).reshape(B, C, L, 9).permute(0,2,1,3)# B H L D

        attn_1 = torch.sum(q_1.unsqueeze(3).repeat(1, 1, 1, 9) * k_2_2D, dim=-2)*self.scale
        attn_1 = (attn_1).softmax(dim=-1)

        fine_deformation = torch.max(attn_1, dim=-1)[1]
        fine_deformation_2D = torch.zeros(B, L, 2).to(DEVICE)
        fine_deformation_2D[:, :, 0] = fine_deformation // 3-1
        fine_deformation_2D[:, :, 1] = fine_deformation % 3-1

        # attn_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        # attn_1 = (attn_1).softmax(dim=-1)
        # attn_1 = self.attn_drop(attn_1)
        # x = (attn_1 @ v_1).transpose(1, 2).reshape(B, L * C)
        # theta = self.localization_linear(x)

        return fine_deformation_2D.reshape(B, x.shape[2], x.shape[2], 2)


def Grid_function(image):

    b, c, image_size, _ = image.size()
    x_coords = torch.arange(0, image_size+4, 1)
    y_coords = torch.arange(0, image_size+4, 1)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
    coordinates = torch.stack((grid_y, grid_x), dim=2)
    coordinates_middle = coordinates[2:-2, 2:-2, :]
    coordinates_sround = torch.zeros(image_size, image_size, 9, 2)

    # 左上
    coordinates_sround[:, :, 0, 0] = coordinates_middle[:, :, 0] - 1
    coordinates_sround[:, :, 0, 1] = coordinates_middle[:, :, 1] - 1
    # 中上
    coordinates_sround[:, :, 1, 0] = coordinates_middle[:, :, 0]
    coordinates_sround[:, :, 1, 1] = coordinates_middle[:, :, 1] - 1
    # 右上
    coordinates_sround[:, :, 2, 0] = coordinates_middle[:, :, 0] + 1
    coordinates_sround[:, :, 2, 1] = coordinates_middle[:, :, 1] - 1
    # 左
    coordinates_sround[:, :, 3, 0] = coordinates_middle[:, :, 0] - 1
    coordinates_sround[:, :, 3, 1] = coordinates_middle[:, :, 1]
    # 中
    coordinates_sround[:, :, 4, 0] = coordinates_middle[:, :, 0]
    coordinates_sround[:, :, 4, 1] = coordinates_middle[:, :, 1]
    # 右
    coordinates_sround[:, :, 5, 0] = coordinates_middle[:, :, 0] + 1
    coordinates_sround[:, :, 5, 1] = coordinates_middle[:, :, 1]
    # 左下
    coordinates_sround[:, :, 6, 0] = coordinates_middle[:, :, 0] - 1
    coordinates_sround[:, :, 6, 1] = coordinates_middle[:, :, 1] + 1
    # 中下
    coordinates_sround[:, :, 7, 0] = coordinates_middle[:, :, 0]
    coordinates_sround[:, :, 7, 1] = coordinates_middle[:, :, 1] + 1
    # 中下
    coordinates_sround[:, :, 8, 0] = coordinates_middle[:, :, 0] + 1
    coordinates_sround[:, :, 8, 1] = coordinates_middle[:, :, 1] + 1

    batch_coordinates = coordinates.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1, 1)
    batch_coordinates_sround = coordinates_sround .unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1, 1, 1)


class FineRegistrationNetwork(nn.Module):
    def __init__(self, patch_size=128, in_ch_msi=3, in_ch_hsi=31):
        super(FineRegistrationNetwork, self).__init__()
        self.Conv_HSI_Embedding = nn.Conv2d(in_ch_hsi*9, 256, 1, 1, 0)
        self.Conv_MSI_Embedding = nn.Conv2d(in_ch_msi*9, 256, 1, 1, 0)
        self.FRN = Attention_FRN(256, patch_size=patch_size)
        self.Conv_half = nn.Conv2d(in_ch_hsi*2, in_ch_hsi, 1, 1, 0)

    def forward(self, x, y, HSI_Patch, MSI_Patch2):

        HSI_Patch = self.Conv_HSI_Embedding(HSI_Patch)
        MSI_Patch2 = self.Conv_MSI_Embedding(MSI_Patch2)
        x = self.Conv_half(x)
        Grid = self.FRN(x, y, HSI_Patch, MSI_Patch2)
        x_org = F.grid_sample(x, Grid)
        # print('OK')
        # Grid = Grid_function(x)
        return x_org


def PCA_Batch_Feat(X, k=1, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return:
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    Y = torch.bmm(X, V[:, :, :k])
    Y = Y.reshape(B, H, W, k)
    Y = Y.permute(0, 3, 1, 2)  # BxkxHxW
    Y = Y.repeat(1, 256, 1, 1)

    return Y


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        # 其实这个就是 Mean Filter
        self.box_filter = nn.Conv2d(256, 256, kernel_size=3, padding=radius, dilation=radius, bias=False)
        self.conv_a = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                    norm(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    norm(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_Guide):
        y_lr = PCA_Batch_Feat(y_Guide)
        b, c, h_lrx, w_lrx = x_lr.size()

        N = self.box_filter(x_lr.data.new().resize_((b, c, h_lrx, w_lrx)).fill_(1.0))
        # 下面几个计算公式与引导滤波一致
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        # A 这里引入了卷积求解 ak
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        # b
        b = mean_y - A * mean_x

        # 最终用双线性插值，放大特征图，获得最终的大尺寸的输出 O_H

        return A * x_lr +b


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResBlock, self).__init__()
        self.same_shape = same_shape
        # if not same_shape:
        #     strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)

        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class StructureTensor(nn.Module):
    def __init__(self, patch_size=128):
        super(StructureTensor, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(4, 4, 3, 1, 1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).repeat(4,4,1,1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).repeat(4,4,1,1)

        self.sobel_x_cov = torch.nn.Conv2d(4, 4, (3, 3), stride=1, padding=1, bias=False)
        self.sobel_x_cov.weight.data = sobel_x
        self.sobel_y_cov = torch.nn.Conv2d(4, 4, (3, 3), stride=1, padding=1, bias=False)
        self.sobel_y_cov.weight.data = sobel_y

        self.sobel_x_cov.weight.requires_grad = False
        self.sobel_y_cov.weight.requires_grad = False
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, y):
        y_org = y
        # 空间增强分支
        y = self.conv(y)
        y = torch.nn.functional.pad(y, (2, 2, 2, 2), mode='replicate')
        x_axis = torch.sum(self.sobel_x_cov(y), dim=1).unsqueeze(1)
        y_axis = torch.sum(self.sobel_y_cov(y), dim=1).unsqueeze(1)

        Gradient = torch.cat([x_axis, y_axis], dim=1)
        Gradient_1 = torch.cat([x_axis, y_axis], dim=1).unsqueeze(1).reshape(Gradient.shape[0], Gradient.shape[1], 1, -1).permute(0,3,1,2)
        Gradient_2 = torch.cat([x_axis, y_axis], dim=1).unsqueeze(2).reshape(Gradient.shape[0], 1, Gradient.shape[1], -1).permute(0,3,1,2)
        Structure_tensor_mine = (Gradient_1 @ Gradient_2)+0.1
        attention_map = Structure_tensor_mine[:, :, 0, 0] * Structure_tensor_mine[:, :, 1, 1]-Structure_tensor_mine[:,: , 0, 1] * Structure_tensor_mine[:, :, 1, 0]
        attention_map = attention_map.reshape(attention_map.shape[0], 1, self.patch_size+4, self.patch_size+4)
        attention_map = attention_map[:, :, 2:-2, 2:-2]
        attention_map = self.batchnorm(attention_map)

        # 可视化注意力特征
        # for i in range(8):
        #     img = attention_map[i].squeeze().unsqueeze(2).cpu().detach().numpy()
        #     plt.figure(figsize=(15, 10))
        #     plt.subplot(1, 2, 1)
        #     plt.axis("off")
        #     plt.title("Low-ResolutionInputs")
        #     plt.imshow(img)
        #
        #     plt.subplot(1, 2, 2)
        #     plt.axis("off")
        #     plt.title("Low-ResolutionInputs")
        #     plt.imshow(y_org[i].squeeze().permute(1,2,0).cpu().detach().numpy())
        #     plt.show()

        return attention_map


class FuseModel(nn.Module):
    def __init__(self, patch_size=128, in_ch_msi=3, in_ch_hsi=31):
        super(FuseModel, self).__init__()
        self.patch_size = patch_size
        self.Conv = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.StructureTensor = StructureTensor(patch_size=patch_size)
        self.ResBlock_MSI = ResBlock(in_ch_msi, 64, same_shape=False)
        self.ResBlock_MSI_2 = ResBlock(64*2, 64, same_shape=False)
        self.ResBlock_HSI_1 = ResBlock(256, 256, same_shape=False)
        self.ResBlock_HSI_reshape = ResBlock(256*2, 256, same_shape=False)
        self.ResBlock_HSI = nn.ModuleList([
            ResBlock(256+64+256, 256, same_shape=False),
            nn.ReLU(),
            ResBlock(256, 256),
            nn.ReLU(),
            # ResBlock(256, 256),
            # nn.ReLU(),
            ResBlock(256, 128, same_shape=False),
            nn.ReLU(),
            # ResBlock(128, 128, same_shape=False),
            # nn.ReLU(),
            nn.Conv2d(128, in_ch_hsi, 3, 1, 1)
        ])

        self.ResBlock_final = nn.ModuleList([
            ResBlock(in_ch_hsi * 2, in_ch_hsi * 2, same_shape=True),
            ResBlock(in_ch_hsi * 2, in_ch_hsi, same_shape=False),
            ResBlock(in_ch_hsi, in_ch_hsi, same_shape=True),
            ])

        self.Channel_Att = nn.Conv1d(in_channels=patch_size*patch_size, out_channels=256, kernel_size=3, padding=1)
        self.ConvGuidedFilter = ConvGuidedFilter()
        self.Conv2D = nn.Conv2d(in_ch_hsi, in_ch_hsi, 3, 1, 1)

    def forward(self, x, y):
        x_org = x
        # 结构张量分支
        spa_attention_map = self.StructureTensor(y)
        # attention分支
        y = self.ResBlock_MSI(y)
        y_att = y*spa_attention_map
        # y_hat = self.ResBlock_MSI_2(torch.cat([y, y_att], dim=1))
        y_hat = self.ResBlock_MSI_2(torch.cat([y, y_att], dim=1))
        # 高光谱分支
        x = self.Conv(x)
        x = self.ResBlock_HSI_1(x)
        x_reshape = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        spe_attation = self.Channel_Att(x_reshape)
        x_hat = torch.matmul(x_reshape, spe_attation).permute(0, 2, 1).reshape(x.shape[0], x.shape[1], self.patch_size, self.patch_size)
        x_hat = self.ResBlock_HSI_reshape(torch.cat([x, x_hat], dim=1))

        # 引导滤波
        x_gudie = self.ConvGuidedFilter(x_hat, y_hat)
        x_gudie = x_hat+x_gudie
        x = torch.cat([x_hat, x_gudie, y_hat], dim=1)
        for block in self.ResBlock_HSI:
            x = block(x)

        x = self.Conv2D(x)
        # x = torch.cat([x, x_org], dim=1)
        # for block in self.ResBlock_final:
        #     x = block(x)
        # out = x
        return x


# 对形变进行约束 表示最大形变限度
def norm_(input):
    b, h, w = input.shape
    for i in range(b):
        if input[i,0,0]>1 or input[i,0,0]<0:
            input[i,0,0]=1
        if input[i,0,1]>0.1 or input[i,0,1]<0:
            input[i,0,1]=0
        if input[i,1,0]<-0.1 or input[i,1,0]>0:
            input[i,1,0]=-0
        if input[i,1,1]>1 or input[i,1,1]<0:
            input[i,1,1]=1
        if input[i,0,2]>3 or input[i,0,2]<0:
            input[i,0,2]=0.05
        if input[i,1,2]>3 or input[i,1,2]<0:
            input[i,1,2]=0.05
    return input


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=dim*patch_size*patch_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )

    def forward(self, x, y):

        B, L, C = x.shape
        qkv_1 = self.qkv_1(x)
        qkv_2 = self.qkv_2(y)

        qkv_1 = einops.rearrange(qkv_1, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # B H L D
        qkv_2 = einops.rearrange(qkv_2, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]  # B H L D
        attn_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        attn_1 = (attn_1).softmax(dim=-1)
        attn_1 = self.attn_drop(attn_1)
        x = (attn_1 @ v_1).transpose(1, 2).reshape(B, L * C)

        theta = self.localization_linear(x)

        return theta


class CoarseRegistrationNetwork(nn.Module):

    def __init__(self, patch_size=32, dim=256, num_heads=4, qkv_bias=False, qk_scale=None, in_ch_msi=3, in_ch_hsi = 31):
        super(CoarseRegistrationNetwork, self).__init__()
        patch_size = patch_size//4
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size ** 2, dim))
        self.Embedding_HSI = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.Embedding_MSI = nn.Conv2d(in_ch_msi, 256, 3, 1, 1)
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=patch_size)

    def forward(self, x, x_org, y):
        x_input = x
        x_org = self.Embedding_HSI(x_org).flatten(2).transpose(1, 2)
        y = self.Embedding_MSI(y).flatten(2).transpose(1, 2)
        x_org = x_org + self.pos_embed
        y = y + self.pos_embed
        theta = self.attn(self.norm(x_org), self.norm(y))


        theta = theta.view(-1, 2, 3)

        # theta = norm_(theta)
        # theta[:,:,2]=0
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class CCFnet(nn.Module):

    def __init__(self, patch_size, in_ch_msi=3, in_ch_hsi = 31):
        super(CCFnet, self).__init__()
        self.conv = nn.Conv2d(in_ch_hsi, 64, 3, 1, 1)
        self.conv_final = nn.Conv2d(256, in_ch_hsi, 3, 1, 1)
        self.Conv_256 = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.upSample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.downSample = nn.Upsample(scale_factor=0.25, mode='bicubic')
        # 粗配准模块初始化
        self.CRN = CoarseRegistrationNetwork(patch_size=patch_size, dim=256, num_heads=4, in_ch_msi=in_ch_msi, in_ch_hsi = in_ch_hsi)
        # 融合模块初始化
        self.FuseModel = FuseModel(patch_size=patch_size, in_ch_msi=in_ch_msi, in_ch_hsi=in_ch_hsi)
        # 精细变形场
        self.FRN = FineRegistrationNetwork(patch_size=patch_size, in_ch_msi=in_ch_msi, in_ch_hsi=in_ch_hsi)

    def forward(self, HSI, MSI, HSI_Patch, MSI_Patch2):

        x = self.FRN(HSI, MSI, HSI_Patch, MSI_Patch2)
        x = self.FuseModel(x, MSI)

        return x


