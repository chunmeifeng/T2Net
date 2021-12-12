# T2Net

# [Task Transformer Network for Joint MRI Reconstruction and Super-Resolution (MICCAI 2021)](https://arxiv.org/pdf/2106.06742.pdf)

[[Paper](https://link.springer.com/chapter/10.1007%2F978-3-030-87231-1_30)][[Code](https://github.com/chunmeifeng/T2Net)]

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==0.8.1
* h5py==2.10.0
* PyYAML==5.4

## Data Prepare

1. Download and decompress data from the link https://pan.baidu.com/s/1OdIoBwJy3GZB979JPBJS6w  Password: qrlt 

2. Transform .h5 format to .mat format
"python convertH5tomat.py --data_dir XXX/T2Net/h5"

3. You can get the dir of as following:

* h5
    - train
    - val
    - test
 * mat
    - train
    - val
    - test
    
4. Set data_dir = 'XXX/T2Net/h5' at the line 4 of ixi_config.yaml

[[Training code --> T2Net](https://github.com/chunmeifeng/T2Net)]

`git clone https://github.com/chunmeifeng/T2Net.git`

## Train

**single gpu train**
```bash
python ixi_train_t2net.py
```

**multi gpu train**
you can change the 65th line in ixi_tain_t2net.py , set num_gpus = gpu number, then run
```bash
python ixi_train_t2net.py
```

###  :fire: NEWS :fire:
* We have upload the mask file. 
* Before our project, you need to  transform the .nii file to .mat file at first.  
* We have provided the code of converting the .nii file to .mat file.

## Citation

```
@inproceedings{feng2021T2Net,
  title={Task Transformer Network for Joint MRI Reconstruction and Super-Resolution},
  author={Feng, Chun-Mei and Yan, Yunlu and Fu, Huazhu and Chen, Li and Xu, Yong},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2021}
}
```

