"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import hashlib
from argparse import ArgumentParser

import torch

from torch.nn import functional as F
from fastmri.mri_ixi_module_t2net import MriModule

from models.T2net import T2Net
import numpy as np


class UnetModule(MriModule):
    """
    Unet training module.
    """

    def __init__(
        self,
        n_resgroups = 5,    #10
        n_resblocks = 10,    #20
        n_feats = 64,       #64
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
                model.
            chans (int): Number of output channels of the first convolution
                layer.
            num_pool_layers (int): Number of down-sampling and up-sampling
                layers.
            drop_prob (float): Dropout probability.
            mask_type (str): Type of mask from ("random", "equispaced").
            center_fractions (list): Fraction of all samples to take from
                center (i.e., list of floats).
            accelerations (list): List of accelerations to apply (i.e., list
                of ints).
            lr (float): Learning rate.
            lr_step_size (int): Learning rate step size.
            lr_gamma (float): Learning rate gamma decay.
            weight_decay (float): Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)

        # self.n_channels_in = n_channels_in
        # self.n_channels_out = n_channels_out
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        upscale_factor = 2
        input_channels =1
        target_channels =1
        self.unet = T2Net(upscale_factor, input_channels, target_channels, n_resblocks=n_resblocks, n_feats=n_feats, res_scale=.1, bn=None)

    def forward(self, slice_LRT2):
        return self.unet(slice_LRT2)

    def training_step(self, batch, batch_idx):
        #T1
        slice_LRT1 = batch['slice_LR_T2'].cuda().float()#[4, 1, 278, 278]
        target_Kspace_T1 = batch['target_Kspace_T2'].cuda().float()# [4, 2, 556, 556]
        target_HR_T1 = batch['target_img_T2'].cuda().float()#[4, 1, 556, 556]
        LRT1_Ori=batch['slice_LR_T2_ori'].cuda().float()

        fname = batch['fname']
        slice_num = batch['slice_num']
        meanT1 = batch['meanT2'].cuda().float()
        stdT1 = batch['stdT2'].cuda().float()

        output_A,output_B= self.forward(slice_LRT1)

        beta_A=0.8
        beta_B=0.2

        loss_T1 = F.l1_loss(output_A, target_HR_T1)*beta_A+F.l1_loss(output_B,LRT1_Ori)*beta_B


        loss = loss_T1
        logs = {"loss": loss.detach()}

        return dict(loss=loss, log=logs)

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)#last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0]**2 + img_cmplx[:, :, :, 1]**2)
        img = img[:, None, :, :]
        return img

    def contrastStretching(self,img, saturated_pixel=0.004):
        """ constrast stretching according to imageJ
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
        values = np.sort(img, axis=None)
        nr_pixels = np.size(values)  # 像素数目
        lim = int(np.round(saturated_pixel*nr_pixels))
        v_min = values[lim]
        v_max = values[-lim-1]
        img = (img - v_min)*(255.0)/(v_max - v_min)
        img = np.minimum(255.0, np.maximum(0.0, img))  # 限制到0-255区间
        return img

    def validation_step(self, batch, batch_idx):
        slice_LRT1 = batch['slice_LR_T2'].cuda().float()  # [4, 1, 278, 278]
        target_Kspace_T1 = batch['target_Kspace_T2'].cuda().float()  # [4, 2, 556, 556]
        target_HR_T1 = batch['target_img_T2'].cuda().float()  # [4, 1, 556, 556]
        LRT1_Ori = batch['slice_LR_T2_ori'].cuda().float()

        fname = batch['fname']
        slice_num = batch['slice_num']
        meanT1 = batch['meanT2'].cuda().float()
        stdT1 = batch['stdT2'].cuda().float()

        output_A, output_B = self.forward(slice_LRT1)

        beta_A = 0.8
        beta_B = 0.2

        loss_T1 = F.l1_loss(output_A, target_HR_T1)*beta_A+F.l1_loss(output_B,LRT1_Ori)*beta_B

        loss = loss_T1


        # hash strings to int so pytorch can concat them
        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output_A.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (
                int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        return {
            "fname": fnumber,
            "slice": slice_num,
            # "output": output * std + mean,
            # "target": target * std + mean,
            "output_T2": output_A* stdT1.reshape([-1,1,1,1]) + meanT1.reshape([-1,1,1,1]),
            "target_im_T2": target_HR_T1* stdT1.reshape([-1,1,1,1]) + meanT1.reshape([-1,1,1,1]),
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        slice_LRT1 = batch['slice_LR_T2'].cuda().float()  # [4, 1, 278, 278]
        target_Kspace_T1 = batch['target_Kspace_T2'].cuda().float()  # [4, 2, 556, 556]
        target_HR_T1 = batch['target_img_T2'].cuda().float()  # [4, 1, 556, 556]
        LRT1_Ori = batch['slice_LR_T2_ori'].cuda().float()

        fname = batch['fname']
        slice_num = batch['slice_num']
        meanT1 = batch['meanT2'].cuda().float()
        stdT1 = batch['stdT2'].cuda().float()

        output_A, output_B = self.forward(slice_LRT1)

        beta_A = 0.8
        beta_B = 0.2

        loss_T1 = F.l1_loss(output_A, target_HR_T1) * beta_A + F.l1_loss(output_B, LRT1_Ori) * beta_B

        loss = loss_T1

        # hash strings to int so pytorch can concat them
        fnumber = torch.zeros(len(fname), dtype=torch.long, device=output_A.device)
        for i, fn in enumerate(fname):
            fnumber[i] = (
                    int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        return {
            "fname": fname,
            "slice": slice_num,
            # "output": output * std + mean,
            # "target": target * std + mean,
            "output_T2": output_B* stdT1.reshape([-1,1,1,1]) + meanT1.reshape([-1,1,1,1]),
            "target_im_T2": LRT1_Ori * stdT1.reshape([-1, 1, 1, 1]) + meanT1.reshape([-1, 1, 1, 1]),
            "test_loss": loss,
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument("--in_chans", default=1, type=int)
        parser.add_argument("--out_chans", default=1, type=int)
        parser.add_argument("--chans", default=1, type=int)
        parser.add_argument("--num_pool_layers", default=4, type=int)
        parser.add_argument("--drop_prob", default=0.0, type=float)

        # data params
        
        # training params (opt)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--lr_step_size", default=40, type=int)
        parser.add_argument("--lr_gamma", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)

        parser.add_argument('--ixi-args', type=dict)

        return parser


