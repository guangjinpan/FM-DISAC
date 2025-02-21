import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from random import randrange


def to_2tuple(x):
    """和原先保持一致，将单个数字变为2元tuple"""
    if isinstance(x, tuple):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
        与原代码中类似：将输入 (B, in_chans, H, W) -> (B, num_patches, embed_dim)
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, in_chans, H, W)
        输出: (B, num_patches, embed_dim)
        """
        # proj 后形状为 (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)  
        # flatten(2) 后形状: (B, embed_dim, num_patches)，然后转置为 (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """正弦位置编码，返回 shape: (1, n_position, d_hid)
       也可根据需要换成可学习的 nn.Parameter(torch.zeros(1, n_position, d_hid))
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)      # (1, n_position, d_hid)


class SimpleTransformer(nn.Module):
    """
    一个基于 PyTorch nn.TransformerEncoder 的简单 Transformer，用来替换原先基于 timm 的AST模型。
    主要包含：
      1. PatchEmbed
      2. 可选的 [CLS] token（或不用）
      3. 位置编码（正弦/可学习）
      4. 多层 TransformerEncoder
      5. 任务相关的头部，如 mask_antenna_pre、mpg、singleBSLoc_wo_pre
    """
    def __init__(
        self,
        label_dim=567,
        # patch/输入相关
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=64, input_tdim=32, input_fmap=2,

        # Transformer相关
        embed_dim=512,
        depth=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,

        # 是否在做pretrain
        pretrain_stage=True,

        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.pretrain_stage = pretrain_stage

        # ------------------------
        # 1) Patch Embedding
        # 注：这里的 patch_size = (fshape, tshape)，stride = (fstride, tstride) 逻辑可自行根据需要修改
        #    也可直接不实现重叠 patch，仅仅做一个简单的 fstride=fshape, tstride=tshape
        #    下面仍然对照你的原始逻辑写，但你可以酌情精简
        # ------------------------
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.input_fmap = input_fmap

        # 实际 patch 大小和 stride(此处仅简化写法，如要完全模拟原逻辑，需要自己处理 overlap 等)
        patch_size = (fshape, tshape)
        in_chans = input_fmap
        self.patch_embed = PatchEmbed(img_size=(input_fdim, input_tdim),
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches

        # 这里也可以视情况加一个 [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_num = 1

        # ------------------------
        # 2) 位置编码
        # ------------------------
        # 这里演示用正弦位置编码。也可换成 learnable。
        pos_embed = get_sinusoid_encoding(self.num_patches + self.cls_token_num, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)  # 如果想可学习就设 True

        # ------------------------
        # 3) 搭建 TransformerEncoder
        # ------------------------
        # PyTorch 自带的：nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, ...)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # 让输入是 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # ------------------------
        # 4) 为了模拟你原代码中的一些“预训练目标”和“定位预测”头等：
        #    - mpg (mask patch generative)
        #    - mask_antenna
        #    - singleBSLoc_wo_pre
        #  在这里定义若干仿照的线性层。
        # ------------------------
        self.gpredlayer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, fshape * tshape * input_fmap)
        )
        self.positioninglayer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )

        # mask embedding
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, embed_dim]))
        nn.init.xavier_normal_(self.mask_embed)

        # mask_antenna 的可学习 token
        self.mask_antenna = nn.Parameter(torch.zeros([1, self.input_fmap, self.input_fdim, self.input_tdim]))
        self.mask_antenna = torch.nn.init.xavier_normal_(self.mask_antenna)

        # unfold 用于把输入在 time/freq 维度上分块，以便和 patch 一一对应
        self.unfold = nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # 记录一下
        print(f"PatchEmbed num_patches: {self.num_patches}, embed_dim: {embed_dim}")

    # ----------------------------------------------------------------
    # 下面几个函数是对应你原先的逻辑，比如随机mask、或随机mask天线等。
    # ----------------------------------------------------------------
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []
        cur_clus = randrange(cluster) + 3
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cur_mask = []
            for i in range(cur_clus):
                for j in range(cur_clus):
                    mask_cand = start_id + int(np.sqrt(sequence_len)) * i + j
                    if 0 <= mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        return torch.tensor(random.sample(range(0, sequence_len), mask_size))

    # 预训练：masked patch generative (mpg)
    def mpg(self, input, mask_patch=100, cluster=True):
        """
        参考原逻辑：
          1) 先拿 x = patch_embed(input)，并 unfold 做 target
          2) 随机mask一部分 patch
          3) 过Transformer
          4) 用 self.gpredlayer 做重构，计算 MSE
        """
        B = input.shape[0]
        # x: (B, num_patches, embed_dim)
        x = self.patch_embed(input)
        # 展开后的原始 patch 用于当做 target, shape: (B, #patches, patch_size * in_chans)
        input_unfold = self.unfold(input).transpose(1, 2)

        # 随机 mask
        mask_index = torch.empty((B, mask_patch), device=x.device, dtype=torch.long)
        mask_dense = torch.ones((B, x.shape[1], x.shape[2]), device=x.device)
        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(sequence_len=self.num_patches, mask_size=mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(sequence_len=self.num_patches, mask_size=mask_patch)
            mask_dense[i, mask_index[i]] = 0

        # 用可学习 mask_embed 替换
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        # 加上 cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        # 加位置编码
        x = x + self.pos_embed[:, : x.shape[1], :]
        # 过 Transformer
        x = self.transformer_encoder(x)
        x = self.norm(x)

        # 计算重构loss
        pred = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device)
        target = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device)
        for i in range(B):
            # x[i, mask_index[i] + 1, : ]  +1是因为 cls_token
            x_masked = x[i, mask_index[i] + self.cls_token_num, :]
            pred_i = self.gpredlayer(x_masked)  # (mask_patch, fshape*tshape*in_chans)
            # 变形
            pred[i] = pred_i.reshape(mask_patch, self.input_fmap, self.fshape*self.tshape).permute(1, 0, 2)
            target_i = input_unfold[i, mask_index[i], :]
            target[i] = target_i.reshape(mask_patch, self.input_fmap, self.fshape*self.tshape).permute(1, 0, 2)

        mse = torch.mean((pred - target) ** 2)
        return mse

    # 预训练：mask_antenna
    def mask_antenna_pre(self, input, mask_antenna_number=8):
        """
        逻辑：
          1) 在 time 维度上随机mask几个“列”，用 self.mask_antenna 替换
          2) 过 patch_embed, Transformer
          3) 重构原始 unfold 后的全部 patch
        """
        B = input.shape[0]
        # 在 input 的最后一维 (宽度) 上随机mask
        mask_index = torch.empty((B, mask_antenna_number), device=self.device, dtype=torch.long)
        mask_dense = torch.ones_like(input)
        for i in range(B):
            # 随机挑选一些时间帧
            mask_index[i] = self.gen_maskid_frame(sequence_len=self.input_tdim, mask_size=mask_antenna_number)
            # 置0
            mask_dense[i, :, :, mask_index[i]] = 0

        # mask_antenna 替换
        mask_tokens = self.mask_antenna.expand(B, -1, -1, -1)  # (B, in_chans, fdim, tdim)

        print(mask_tokens.shape,mask_dense.shape,input.shape)
        x = input * mask_dense + (1 - mask_dense) * mask_tokens

        # unfold 做 target
        target_unfold = self.unfold(input).transpose(1, 2)  # (B, #patches, patch_size*in_chans)
        # patch_embed
        x = self.patch_embed(x)

        # 加上 cls_token + pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        # Transformer
        x = self.transformer_encoder(x)
        x = self.norm(x)

        # 重构
        # x[:, 1:, :] 对应所有patch(不含cls)
        pred_i = self.gpredlayer(x[:, self.cls_token_num:, :])  # (B, num_patches, patch_size * in_chans)
        pred = pred_i.view(B, self.num_patches, self.input_fmap, self.fshape * self.tshape).permute(0,2,1,3)
        target = target_unfold.view(B, self.num_patches, self.input_fmap, self.fshape * self.tshape).permute(0,2,1,3)

        mse = torch.mean((pred - target) ** 2)
        return mse

    # 单基站定位——没有预训练的分支
    def singleBSLoc_wo_pre(self, input, y_position):
        """
        1) 将输入embed
        2) 加 cls token & pos_embed
        3) 过Transformer
        4) 均值后 -> 线性层 -> (x, y)
        5) MSE
        """
        B = input.shape[0]
        x = self.patch_embed(input)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]

        x = self.transformer_encoder(x)
        x = self.norm(x)  # (B, num_patches+1, embed_dim)

        # 这里简单地将所有token求平均再做两维回归
        # 也可以只拿 x[:, 0, :] 做cls token回归
        x_mean = torch.mean(x, dim=1)  # (B, embed_dim)
        pred = self.positioninglayer(x_mean)  # (B, 2)

        mse = torch.mean((pred - y_position) ** 2)
        return mse

    def inference_SingleBSLoc(self, input, y_position):
        """
        和 singleBSLoc_wo_pre 的推理类似，这里可以返回预测值 + loss
        """
        B = input.shape[0]
        x = self.patch_embed(input)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]

        x = self.transformer_encoder(x)
        x = self.norm(x)

        x_mean = torch.mean(x, dim=1)
        pred = self.positioninglayer(x_mean)
        mse = torch.mean((pred - y_position) ** 2)
        return pred, mse

    def forward(self, x, y_position, task,
                cluster=True, mask_patch=400, mask_antenna_number=8):
        """
        x: (B, input_fmap, input_tdim, input_fdim)，
           如果跟你原先一样是 (B, time, freq)，请先自行在外部转置/reshape。
        task: 指定要执行的任务
        """
        # 先转一下维度，模仿原始
        # [B, input_fmap, input_tdim, input_fdim] => [B, in_chans, H, W]
        # 由于原代码是 x.transpose(2,3)，看你需要，这里不一定需要。只要和 patch 的设定对上即可。
        # 下面示例假设 x 传进来时已是 [B, input_fmap, input_fdim, input_tdim] 并且 patch_embed 的输入就是 (B, C, H, W)
        # 所以可以：
        x = x.transpose(2, 3)  # (B, input_fmap, input_tdim, input_fdim) => (B, in_chans, H, W)
        
        if task == "woFT_SingleBSLoc":
            return self.singleBSLoc_wo_pre(x, y_position)
        elif task == "FT_SingleBSLoc":
            return self.singleBSLoc_wo_pre(x, y_position)
        elif task == "pretrain_mpg":
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == "pretrain_antenna":
            return self.mask_antenna_pre(x, mask_antenna_number=mask_antenna_number)
        elif task == "inference_SingleBSLoc":
            return self.inference_SingleBSLoc(x, y_position)
        else:
            raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    # 简单测试
    # 假设输入大小：B=10, input_fmap=2, input_fdim=64, input_tdim=32
    B = 10
    input_fmap = 2
    input_fdim = 64
    input_tdim = 32

    # 输入形状 [B, 2, 64, 32]
    test_input = torch.randn([B, input_fmap, input_tdim, input_fdim])

    # 初始化网络
    model = SimpleTransformer(
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=input_fdim, input_tdim=input_tdim, input_fmap=input_fmap,
        embed_dim=256,  # 可调
        depth=4,        # 可调
        num_heads=4,    # 可调
        device="cpu",
    )

    # 预训练任务：mask_antenna
    loss_pre_antenna = model(test_input, y_position=torch.zeros((B, 2)), 
                             task="pretrain_antenna", mask_antenna_number=8)
    print("loss_pre_antenna:", loss_pre_antenna.item())

    # 预训练任务：mpg
    loss_pre_mpg = model(test_input, y_position=torch.zeros((B, 2)),
                         task="pretrain_mpg", mask_patch=20, cluster=False)
    print("loss_pre_mpg:", loss_pre_mpg.item())

    # 单基站定位
    y_true = torch.zeros((B, 2))
    loss_loc = model(test_input, y_position=y_true, task="woFT_SingleBSLoc")
    print("loss_loc:", loss_loc.item())

    # 推理
    pred_loc, mse_inf = model(test_input, y_position=y_true, task="inference_SingleBSLoc")
    print("inference loc pred:", pred_loc)
    print("inference loc mse:", mse_inf.item())
