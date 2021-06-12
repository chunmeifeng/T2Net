from os.path import splitext
from os import listdir, path

from torch.utils.data import Dataset
import logging

import h5py
import pickle
from scipy.io import loadmat, savemat
import scipy.io as sio
import torch
import cv2
import numpy as np
import os
from fastmri import transforms

class IXIdataset(Dataset):
    def __init__(self, data_dir, args, validtion_flag=False, load_T2=True):
        self.args = args
        self.data_dir = data_dir
        self.validtion_flag = validtion_flag
        self.load_T2 = load_T2

        print('load T2: ', self.load_T2)

        self.num_input_slices = args.num_input_slices
        self.img_size = args.img_size

        # make an image id's list
        self.file_names = [splitext(file)[0] for file in listdir(data_dir)
                           if not file.startswith('.')]


        self.ids = list()

        reconstruction_root='/home/jc3/YYL/JS_fastMRI/SR_fastMRI-master/experimental/ixi/reconstruction_data_ixi/r2/mat1/'
        self.LR_slice_files = []

        file_len=int(len(self.file_names)*0.75)
        for i in range(file_len):
            try:
                full_file_path = path.join(self.data_dir, self.file_names[i] + '.hdf5')

                with h5py.File(full_file_path, 'r') as f:
                    numOfSlice = f['data'].shape[2]

                if numOfSlice < self.args.slice_range[1]:
                    continue

                for slice in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]:
                    self.ids.append((self.file_names[i], slice))

                for slice in range(0,21):
                    file_name = '%s-%03d.mat' % (self.file_names[i], slice)
                    self.LR_slice_files.append(os.path.join(reconstruction_root,file_name))

            except:
                continue

        if self.validtion_flag:
            logging.info(f'Creating validation dataset with {len(self.ids)} examples')
        else:
            logging.info(f'Creating training dataset with {len(self.ids)} examples')


        masks_dictionary = loadmat(
            '/home/jc3/mycode/T2_IXI_unet/IXI_fastMRI/fastMRI-master/experimental/ixi/mask_mei/1D-Random-3X_256.mat')
        self.masks = masks_dictionary['mask']

        print('masks:', self.masks.shape)
        self.maskedNot = 1 - (self.masks)
        maskedNot = self.maskedNot

        # random noise:
        self.minmax_noise_val = args.minmax_noise_val

        print('dataset :{}'.format(len(self.ids)))
        print('LR: {}'.format(len(self.LR_slice_files)))


    def __len__(self):
        return len(self.ids)

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size) / 2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def niipath2matpath(self, T1,slice_id):
        filedir,filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir,'mat1')
        basename, ext = path.splitext(filename)
        base_name = basename[:-1]
        file_name = '%s-%03d.mat'%(base_name,slice_id)
        T1_file_path = path.join(mat_dir,file_name)
        return T1_file_path

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)  # last dimension=2
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0] ** 2 + img_cmplx[:, :, :, 1] ** 2)
        img = img[:, None, :, :]
        return img

    # @classmethod
    def slice_preprocess(self, kspace_cplx):  # 256,256
        # crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)  # 256,256
        # split to real and imaginary channels
        kspace = np.zeros((self.img_size, self.img_size, 2))  # 256,256,2
        kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
        # target image:
        image = self.ifft2(kspace_cplx)  # 256,256===1,256,256
        # HWC to CHW
        kspace = kspace.transpose((2, 0, 1))  # 2,256,256
        # print('kspace:',kspace.shape)
        HR = self.getHR(image)
        LR, LR_ori = self.getLR(image)


        return LR,LR_ori,kspace, HR

    def getHR(self, hr_data):

        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (256, 256))
        imgifft = np.fft.ifft2(imgfft)
        img_out = abs(imgifft)

        return img_out

    def getLR(self, hr_data):
        # imgfft = np.fft.fft2(hr_data)
        #
        imgfft = self.fft2(hr_data)

        imgfft = self.center_crop(imgfft, (128, 128))

        masks_dictionary = loadmat(
            "/home/jc3/YYL/JS_fastMRI/SR_fastMRI-master/experimental/ixi/mask_mei/1D-Cartesian_6X_128128.mat")
        mask = masks_dictionary['mask']

        t = imgfft

        imgfft = imgfft * mask

        imgifft = np.fft.ifft2(imgfft)
        img_out = abs(imgifft)

        t = np.fft.ifft2(t)
        LR_ori = abs(t)

        return img_out, LR_ori

    def __getitem__(self, i):

        fname, slice_num = self.ids[i]
        target_LR_T2=np.zeros((1, 128, 128))
        target_LR_T2_ori = np.zeros((1, 128, 128))
        target_Kspace_T2 = np.zeros((2, self.img_size, self.img_size))
        target_img_T2 = np.zeros((1, self.img_size, self.img_size))


        if self.num_input_slices == 1:
            slice_range = [slice_num]
        elif self.num_input_slices == 3:
            slice_range = [slice_num - 1, slice_num, slice_num + 1]


        if self.load_T2:
            for ii, slice_id in enumerate(slice_range):
                full_file_path = path.join(self.data_dir, fname + '-{:03d}.mat'.format(slice_id))
                full_file_path = full_file_path.replace('/h5/', '/mat/')

                img_T2 = sio.loadmat(full_file_path)['img']

                img_T2, meanT2, stdT2= transforms.normalize_instance(img_T2, eps=1e-11)

                img_T2_height, img_T2_width = img_T2.shape
                img_T2_matRotate = cv2.getRotationMatrix2D((img_T2_height * 0.5, img_T2_width * 0.5), 90, 1)
                img_T2 = cv2.warpAffine(img_T2, img_T2_matRotate, (img_T2_height, img_T2_width))


                kspace_T2 = self.fft2(img_T2)  # ksapce: (256,256)
                slice_LR_T2,slice_LR_T2_ori, slice_full_Kspace_T2, slice_full_img_T2 = self.slice_preprocess(
                    kspace_T2)

                if slice_id == slice_num:
                    target_LR_T2=slice_LR_T2
                    target_LR_T2_ori=slice_LR_T2_ori
                    target_Kspace_T2 = slice_full_Kspace_T2
                    target_img_T2 = slice_full_img_T2

                    break



        ret = {
            'slice_LR_T2': torch.from_numpy(target_LR_T2),
            'slice_LR_T2_ori': torch.from_numpy(target_LR_T2_ori),
            'target_Kspace_T2': torch.from_numpy(target_Kspace_T2),
            'target_img_T2': torch.from_numpy(target_img_T2),
            'fname': fname,
            'slice_num': slice_num,
            'meanT2':meanT2,
            'stdT2':stdT2
        }

        return ret

    def center_crop(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        # print(data.shape)
        # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
        assert 0 < shape[1] <= data.shape[-1]  # 640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]

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


