# T2Net
# Task Transformer Network for Joint MRI Reconstruction and Super-Resolution (MICCAI 2021)

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==0.8.1
* h5py==2.10.0
* PyYAML==5.4

single gpu train:
"python ixi_train_t2net.py"

multi gpu train :
you can change the 65th line in ixi_tain_t2net.py , set num_gpus = gpu number, then run
"python ixi_train_t2net.py"


**single gpu train**
```bash
python ixi_train_t2net.py
```

**multi gpu train**
you can change the 65th line in ixi_tain_t2net.py , set num_gpus = gpu number, then run
```bash
python ixi_train_t2net.py
```


## We have upload the mask file. In our project, you need to convert the nii file to .mat file first.  
 


Citation

```
@inproceedings{feng2021T2Net,
  title={Task Transformer Network for Joint MRI Reconstruction and Super-Resolution},
  author={Feng, Chun-Mei and Yan, Yunlu and Fu, Huazhu and Chen, Li and Xu, Yong},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2021}
}
```

