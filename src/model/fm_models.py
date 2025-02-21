import torch.nn as nn
import torch
import sys

sys.path.append("../../../pytorch-image-models")
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=4, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
    
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ASTModel(nn.Module):
    def __init__(self, label_dim=567,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, input_fmap=1, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None, device = "cpu"):

        super(ASTModel, self).__init__()
        # assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.device = device
        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_tiny_patch16_224', pretrained=False).to(device)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 1
            elif model_size == 'small':
                self.v = timm.create_model('vit_small_patch16_36x1_224', pretrained=False).to(device)
                self.heads, self.v.depth = 4, 4
                self.cls_token_num = 1
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False).to(device)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False).to(device)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim, self.input_fmap = input_fdim, input_tdim, input_fmap
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, self.fshape * self.tshape * self.input_fmap))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, self.fshape * self.tshape * self.input_fmap))
            self.positioninglayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 2))
            
            self.positioninglayer2 = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 2))
            self.positioninglayer3 = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 2))


            self.v.cls_token = nn.Parameter(torch.ones(1, 1, self.original_embedding_dim) * 0.1, requires_grad=False)


            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            self.mask_antenna = nn.Parameter(torch.zeros([1, self.input_fmap, self.input_fdim, self.input_tdim]))
            self.mask_antenna = torch.nn.init.xavier_normal_(self.mask_antenna)


            # get the intermediate shape
            print(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(self.input_fmap, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj
            self.v.patch_embed.img_size = (input_fdim, input_tdim)  # 注意：input_fdim对应高度，input_tdim对应宽度
            self.v.patch_embed.patch_size = (fshape, tshape)
            print(self.v.cls_token.shape)

            # use trainable positional embedding
            # new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            new_pos_embed = get_sinusoid_encoding(self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim)
            self.v.pos_embed = torch.nn.Parameter(new_pos_embed, requires_grad=False)

            trunc_normal_(self.v.pos_embed, std=.02)
            self.last_loss=0

    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)


    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        # print(input.shape, self.v.patch_embed)
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens
        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, self.input_fmap, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256
        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred_i = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            pred[i] = pred_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)
            target_i = input[i, mask_index[i], :]
            target[i] = target_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse


    def singleBSLoc_wo_pre(self, input, y_position):
        B = input.shape[0]
        # print(input.shape, self.v.patch_embed)
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = self.positioninglayer2(x)
        pred1 = torch.mean(pred,1)
        target = y_position
        # calculate the MSE loss

        
        mse = torch.mean((pred1 - target) ** 2)
        

        # if torch.isnan(mse):
        #     print("Loss is NaN. Stopping training.")
        #     print(self.last_loss, self.last_pos, pred1,target,pred)
        #     print("x",x)
        #     print("self.last_input ", self.last_input)
        #     print("input",input)
        #     print("self.v.pos_embed",self.v.pos_embed, self.last_cls_tokens)
        #     print("cls_tokens",cls_tokens,self.last_pos_embed)
        
        # self.last_loss = mse.item()
        # self.last_input = input.clone()
        # self.last_cls_tokens = cls_tokens.clone()
        # self.last_pos_embed= self.v.pos_embed.clone()
        # self.last_pos = y_position.clone()
        # torch.autograd.set_detect_anomaly(True)
        return mse

    def mask_antenna_pre(self, input, mask_antenna_number):
        B = input.shape[0]
        

        mask_index = torch.empty((B, mask_antenna_number), device=self.device, requires_grad=False).long()
        mask_dense = torch.ones([input.shape[0], input.shape[1], input.shape[2], input.shape[3]], device=self.device)
        for i in range(B):
            # use this if you are masking frame, i.e., 128*2 patches
            mask_index[i] = self.gen_maskid_frame(sequence_len = self.input_tdim, mask_size = mask_antenna_number)
            mask_dense[i, :, :, mask_index[i]] = 0
        mask_tokens = self.mask_antenna.expand(B, input.shape[1], input.shape[2], -1)
        x = input * mask_dense + (1-mask_dense) * mask_tokens

        
        input = self.unfold(input).transpose(1, 2) 
        x = self.v.patch_embed(x)

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, self.input_fmap, self.num_patches, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, self.input_fmap, self.num_patches, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256
        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred_i = self.gpredlayer(x[i, self.cls_token_num:, :])
            pred[i] = pred_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)
            target_i = input[i, :, :]
            target[i] = target_i.reshape((pred_i.shape[0], self.input_fmap, self.fshape * self.tshape)).permute(1, 0, 2)

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def inference_SingleBSLoc(self, input, y_position):
        B = input.shape[0]
        # print(input.shape, self.v.patch_embed)
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = self.positioninglayer2(x)
        pred = torch.mean(pred,1)
        mse = torch.mean((pred - y_position) ** 2)

        return pred, mse

    def forward(self, x, y_position, task, cluster=True, mask_patch=400, mask_antenna_number = 8):

        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        # x = x.unsqueeze(1)

        x = x.transpose(2, 3) #[B, input_fmap, input_fdim, input_tdim]
        # print(11,x.shape)
        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == "woFT_SingleBSLoc":
            return self.singleBSLoc_wo_pre(x, y_position)
        elif task == "FT_SingleBSLoc":
            return self.singleBSLoc_wo_pre(x, y_position)
        elif task == "pretrain_mpg":
            return self.mpg(x, mask_patch = mask_patch, cluster=cluster)
        elif task == "pretrain_antenna":
            return self.mask_antenna_pre(x, mask_antenna_number = mask_antenna_number)
        elif task == "inference_SingleBSLoc":
            return self.inference_SingleBSLoc(x, y_position)
        # if task == 'ft_avgtok':
        #     return self.finetuningavgtok(x)
        # # alternatively, use the [cls] token output as clip-level representation.
        # elif task == 'ft_cls':
        #     return self.finetuningcls(x)
        # # pretraining, masked patch classification (discriminative objective)
        # elif task == 'pretrain_mpc':
        #     return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # # pretraining, masked patch reconstruction (generative objective)
        # elif task == 'pretrain_mpg':
        #     return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        # elif task == 'visualize_mask':
        #     return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        # else:
        #     raise Exception('Task unrecognized.')

        return 0


if __name__ == '__main__':
    # this is an example of how to use the SSAST model

    # pretraining stage
    # suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
    input_tdim = 32
    input_fdim = 64
    input_fmap = 2

    # input_tdim = 1024
    # input_fdim = 128
    # input_fmap = 2
    print(timm.list_models("*vit*tiny*"))  # 模糊匹配特定模型

    # create a 16*16 patch based AST model for pretraining.
    # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
    ast_mdl = ASTModel(
                 fshape=4, tshape=4, fstride=4, tstride=4,
                 input_fdim=input_fdim, input_tdim=input_tdim, input_fmap = input_fmap, model_size='tiny',
                 pretrain_stage=True)
    print(ASTModel)
    # # alternatively, create a frame based AST model
    # ast_mdl = ASTModel(
    #              fshape=128, tshape=2, fstride=128, tstride=2,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain=True)

    # do pretraining, see src/traintest_mask.py for our full pretraining code
    # input in shape [batch_size, input_tdim, input_fdim]
    test_input = torch.zeros([10, input_fmap, input_tdim, input_fdim])
    # mask 100 patches for both discriminative and generative loss
    # acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    mse_loss = ast_mdl(test_input, y_position=torch.zeros((10,2)), task='pretrain_antenna', mask_patch=13, mask_antenna_number=8)
    print(mse_loss)
    # loss = nce_loss + 10 * mse_loss