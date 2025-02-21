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
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """正弦位置编码，返回 shape: (1, n_position, d_hid)"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# ====================== 新增：一个简单的 Decoder 类 ====================== #
class SimpleDecoder(nn.Module):
    """
    演示用的 Decoder：从 latent_dim -> embed_dim，然后通过 TransformerDecoder
    最终投影到重构维度 (fshape * tshape * in_chans)，和之前 gpredlayer 类似。
    如果想对 mpg、mask_antenna 用不同 Decoder，可以再定义多份或多实例。
    """
    def __init__(self, latent_dim=512, embed_dim=512,
                 depth=2, num_heads=8, dim_feedforward=2048, dropout=0.1,
                 final_out_dim=256  # 用于重构的最终维度
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # 先把 latent_dim -> embed_dim
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim, bias=True)

        # 这里使用一个简易的 TransformerDecoder 结构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # 最终投影到重构维度
        self.proj_out = nn.Linear(embed_dim, final_out_dim)

        # 如果需要 mask token，也可以定义
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.xavier_normal_(self.mask_token)

    def forward(self, memory, tgt=None, mask_idx=None):
        """
        memory: Encoder 输出, shape: (B, N, latent_dim)
        tgt:    作为 Decoder 输入，如果做自回归可用；也可直接用 memory 作为 tgt
        mask_idx: 用于替换被mask的 token为 mask_token (在latent空间 or embed空间)
        这里演示：直接把 memory 先映射成 embed 以后，当做 decoder 的 tgt + memory
                 (类似 non-autoregressive decode)
        """
        B, N, _ = memory.shape
        # 先映射 latent->embed 作为 memory_embed
        memory_embed = self.latent_to_embed(memory)  # (B, N, embed_dim)

        if tgt is None:
            tgt_embed = memory_embed.clone()
        else:
            tgt_embed = tgt  # 自定义 tgt 时，可以外部处理

        # 如果需要在 tgt_embed 里对被 mask 的位置替换 mask_token，可以在这里操作:
        if mask_idx is not None:
            mask_token_embed = self.latent_to_embed(self.mask_token)  # (1,1,embed_dim)
            for i in range(B):
                idx_i = mask_idx[i]
                tgt_embed[i, idx_i, :] = mask_token_embed[0,0,:]

        # 做 TransformerDecoder
        out = self.transformer_decoder(tgt_embed, memory_embed)  # (B, N, embed_dim)
        out = self.norm(out)
        # 映射到重构维度
        out = self.proj_out(out)  # (B, N, final_out_dim)
        return out


class wireless_loc_fm(nn.Module):
    """
    原有的不依赖 timm 的简易 Transformer，
    现在做增量修改，加入了 "Encoder->Decoder" 机制。
    """
    def __init__(
        self,
        label_dim=567,
        # patch/输入相关
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=64, input_tdim=32, input_fmap=2,

        # Transformer(Encoder)相关
        embed_dim=512,
        depth=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,

        # ============ 新增: latent_dim，用来做 encoder 输出的特征压缩
        latent_dim=256,
        # ============

        # 是否在做pretrain
        pretrain_stage=True,

        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.pretrain_stage = pretrain_stage

        # ----------------------------------------------------
        # 1) Patch Embedding (Encoder 输入)
        # ----------------------------------------------------
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.input_fmap = input_fmap

        patch_size = (fshape, tshape)
        in_chans = input_fmap

        # patch_embed
        self.patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # 一个可学习的 cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_num = 1

        # 位置编码
        pos_embed = get_sinusoid_encoding(self.num_patches + self.cls_token_num, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        # ----------------------------------------------------
        # 2) Encoder部分：TransformerEncoder + LayerNorm
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # ============ 新增：Encoder输出 -> latent_dim 的线性映射  ============
        self.proj_to_latent = nn.Linear(embed_dim, latent_dim)

        # ----------------------------------------------------
        # 3) 为了保留你原来的 pretrain/finetune 功能，一些头层保持不变
        #    下面是旧版的 generative 预测层 / 定位层等
        # ----------------------------------------------------
        # 生成式预测层 (gpredlayer)
        self.gpredlayer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, fshape * tshape * input_fmap)
        )
        # 定位层
        self.positioninglayer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )

        # mask embed
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, embed_dim]))
        nn.init.xavier_normal_(self.mask_embed)

        # mask 贴片
        self.mask_antenna = nn.Parameter(torch.zeros([1, input_fmap, input_fdim, input_tdim]))
        nn.init.xavier_normal_(self.mask_antenna)

        self.unfold = nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # ----------------------------------------------------
        # 4) 新增：Decoder 部分
        #    这里我们演示给 mpg 和 antenna 各写一个简单 Decoder。
        # ----------------------------------------------------
        # mpg decoder
        self.decoder_mpg = SimpleDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=2,  # 你可调整
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        )
        # antenna decoder
        self.decoder_antenna = SimpleDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        )
        # ----------------------------------------------------

        print(f"PatchEmbed num_patches: {self.num_patches}, embed_dim: {embed_dim}")


    # ================== 原有的随机mask函数，保持不变 ================== #
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


    # ================== 原来的 mpg，不使用Decoder ================== #
    def mpg(self, input, mask_patch=100, cluster=True):
        B = input.shape[0]
        x = self.patch_embed(input)
        input_unfold = self.unfold(input).transpose(1, 2)

        mask_index = torch.empty((B, mask_patch), device=x.device, dtype=torch.long)
        mask_dense = torch.ones((B, x.shape[1], x.shape[2]), device=x.device)
        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(sequence_len=self.num_patches, mask_size=mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(sequence_len=self.num_patches, mask_size=mask_patch)
            mask_dense[i, mask_index[i]] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.norm(x)

        # 原生的 gpredlayer
        pred = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device)
        target = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device)
        for i in range(B):
            x_masked = x[i, mask_index[i] + self.cls_token_num, :]
            pred_i = self.gpredlayer(x_masked)
            pred[i] = pred_i.reshape(mask_patch, self.input_fmap, self.fshape*self.tshape).permute(1, 0, 2)
            target_i = input_unfold[i, mask_index[i], :]
            target[i] = target_i.reshape(mask_patch, self.input_fmap, self.fshape*self.tshape).permute(1, 0, 2)

        mse = torch.mean((pred - target) ** 2)
        return mse

    # ============== 新增：mpg 的 Encoder->Decoder 实现 ============== #
    def mpg_encoder_decoder(self, input, mask_patch=100, cluster=True):
        """
        与上面 mpg 类似，但改为:
          1) Encoder(得到 latent)
          2) Decoder(重构)
          3) 计算 MSE
        """
        B = input.shape[0]
        # ---- (A) Encoder ----
        # 先 normal encode
        x = self.patch_embed(input)  # (B, num_patches, embed_dim)

        # mask
        mask_index = torch.empty((B, mask_patch), device=x.device, dtype=torch.long)
        mask_dense = torch.ones((B, x.shape[1], x.shape[2]), device=x.device)
        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i]] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]

        x = self.transformer_encoder(x)  # (B, num_patches+1, embed_dim)
        x = self.norm(x)
        # 压缩到 latent_dim
        latent = self.proj_to_latent(x)  # (B, N+1, latent_dim)
        print(latent.shape)
        # ---- (B) Decoder ----
        # 这里 unfold 作为重构目标
        input_unfold = self.unfold(input).transpose(1, 2)  # (B, #patches, patch_size*in_chans)
        # 用 decoder_mpg 解码: 
        #   - memory=latent
        #   - mask_idx=mask_index + cls_token_num (因为有cls)
        #   - 目标是与 input_unfold 对齐
        #   - 其中N = num_patches+1
        full_mask_idx = mask_index + self.cls_token_num
        rec = self.decoder_mpg(memory=latent, mask_idx=full_mask_idx)
        # rec shape: (B, N, final_out_dim) => (B, N, fshape*tshape*in_chans)

        # 只对被mask的patch 计算loss (也可对全部 patch 计算)
        # 这里演示只对mask处计算
        # unfold里的第 i 个patch 对应 decoder输出的第 i+cls_token_num 行
        # 先把 rec( B, N, patch_dim ) => 取 N-1 处 (不含cls) => (B, num_patches, patch_dim)
        rec_wo_cls = rec[:, self.cls_token_num:, :]  # (B, num_patches, patch_dim)
        # 目标也对应 (B, num_patches, patch_dim)
        target_wo_cls = input_unfold
        # 取 mask_index 位置
        pred_masked = []
        target_masked = []
        for i in range(B):
            idx_i = mask_index[i]
            pred_masked.append(rec_wo_cls[i, idx_i, :])
            target_masked.append(target_wo_cls[i, idx_i, :])
        pred_masked = torch.cat(pred_masked, dim=0)
        target_masked = torch.cat(target_masked, dim=0)

        mse = torch.mean((pred_masked - target_masked)**2)
        return mse


    # ============== 原来的 mask_antenna_pre，不使用Decoder ============== #
    def mask_antenna_pre(self, input, mask_antenna_number=8):
        B = input.shape[0]
        mask_index = torch.empty((B, mask_antenna_number), device=self.device, dtype=torch.long)
        mask_dense = torch.ones([B, input.shape[1], input.shape[2], input.shape[3]], device=self.device)
        for i in range(B):
            mask_index[i] = self.gen_maskid_frame(sequence_len=self.input_tdim, mask_size=mask_antenna_number)
            mask_dense[i, :, :, mask_index[i]] = 0
        mask_tokens = self.mask_antenna.expand(B, input.shape[1], input.shape[2], -1)
        x = input * mask_dense + (1 - mask_dense) * mask_tokens

        input_unfold = self.unfold(input).transpose(1, 2) 
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.norm(x)
        pred = torch.empty((B, self.input_fmap, self.num_patches, self.fshape * self.tshape), device=x.device)
        target = torch.empty((B, self.input_fmap, self.num_patches, self.fshape * self.tshape), device=x.device)
        for i in range(B):
            pred_i = self.gpredlayer(x[i, self.cls_token_num:, :])
            pred[i] = pred_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)
            target_i = input_unfold[i, :, :]
            target[i] = target_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)

        mse = torch.mean((pred - target) ** 2)
        return mse

    # ============== 新增：mask_antenna_pre 的 Encoder->Decoder 实现 ============== #
    def mask_antenna_pre_encoder_decoder(self, input, mask_antenna_number=8):
        B = input.shape[0]
        # mask天线
        mask_index = torch.empty((B, mask_antenna_number), device=self.device, dtype=torch.long)
        mask_dense = torch.ones_like(input)
        for i in range(B):
            mask_index[i] = self.gen_maskid_frame(self.input_tdim, mask_antenna_number)
            mask_dense[i, :, :, mask_index[i]] = 0

        mask_tokens = self.mask_antenna.expand(B, input.shape[1], input.shape[2], -1)
        x_masked = input * mask_dense + (1 - mask_dense) * mask_tokens

        # ---- (A) Encoder ----
        x = self.patch_embed(x_masked)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.norm(x)
        latent = self.proj_to_latent(x)  # (B, N+cls, latent_dim)

        # ---- (B) Decoder ----
        input_unfold = self.unfold(input).transpose(1, 2)
        rec = self.decoder_antenna(latent)  # (B, N, patch_dim)
        # 不同于 mpg，这里演示直接对全部 patch 计算 MSE
        rec_wo_cls = rec[:, self.cls_token_num:, :]  # (B, num_patches, patch_dim)
        # target
        target_wo_cls = input_unfold  # (B, num_patches, patch_dim)
        mse = torch.mean((rec_wo_cls - target_wo_cls)**2)
        return mse


    # ================== 单基站定位：只用 Encoder 输出 ================== #
    def singleBSLoc_wo_pre(self, input, y_position):
        B = input.shape[0]
        x = self.patch_embed(input)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.norm(x)
        print(x.shape)
        # 原先是直接在 embed_dim 上定位，这里保持不变，
        # 如果想用 latent_dim，可以把 x -> proj_to_latent -> MLP
        x_mean = torch.mean(x, dim=1)
        pred = self.positioninglayer(x_mean)
        mse = torch.mean((pred - y_position) ** 2)
        return mse

    def inference_SingleBSLoc(self, input, y_position):
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
        x: (B, input_fmap, input_fdim, input_tdim)
        task: 
            - "woFT_SingleBSLoc", "FT_SingleBSLoc", "inference_SingleBSLoc"
            - "pretrain_mpg", "pretrain_antenna" 
            - 新增: "pretrain_mpg_ed", "pretrain_antenna_ed"
        """
        # 原先的转置
        x = x.transpose(2, 3)  # [B, in_chans, H, W]

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

        # ============== 新增的 Encoder->Decoder 的预训练分支 ==============
        elif task == "pretrain_mpg_ed":
            return self.mpg_encoder_decoder(x, mask_patch=mask_patch, cluster=cluster)
        elif task == "pretrain_antenna_ed":
            return self.mask_antenna_pre_encoder_decoder(x, mask_antenna_number=mask_antenna_number)
        # ===============================================================

        else:
            raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    # 简单测试
    B = 10
    input_fmap = 2
    input_fdim = 64
    input_tdim = 32

    # 输入形状 [B, 2, 64, 32]
    test_input = torch.randn([B, input_fmap, input_tdim, input_fdim])

    model = wireless_loc_fm(
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=input_fdim, input_tdim=input_tdim, input_fmap=input_fmap,
        embed_dim=256,  # 可调
        depth=4,        
        num_heads=4,    
        device="cpu",
        latent_dim=128  # 新增的latent维度
    )

    # 1) 原始 mpg
    loss_pre_mpg = model(test_input, y_position=torch.zeros((B, 2)),
                         task="pretrain_mpg", mask_patch=10)
    print("[old mpg] loss:", loss_pre_mpg.item())

    # 2) 新增 encoder-decoder mpg
    loss_pre_mpg_ed = model(test_input, y_position=torch.zeros((B, 2)),
                            task="pretrain_mpg_ed", mask_patch=10)
    print("[encoder-decoder mpg] loss:", loss_pre_mpg_ed.item())

    # 3) 原始 mask_antenna
    loss_ant = model(test_input, torch.zeros((B,2)), task="pretrain_antenna", mask_antenna_number=5)
    print("[old antenna] loss:", loss_ant.item())

    # 4) 新增 encoder-decoder antenna
    loss_ant_ed = model(test_input, torch.zeros((B,2)), task="pretrain_antenna_ed", mask_antenna_number=5)
    print("[encoder-decoder antenna] loss:", loss_ant_ed.item())

    # 5) 定位
    y_true = torch.zeros((B, 2))
    loc_loss = model(test_input, y_position=y_true, task="woFT_SingleBSLoc")
    print("[loc] loss:", loc_loss.item())

    pred_loc, mse_inf = model(test_input, y_position=y_true, task="inference_SingleBSLoc")
    print("[loc inference] pred:", pred_loc, ", mse:", mse_inf.item())
