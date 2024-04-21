# pytorch-lightning-pointnet
This repo is implementation for [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) in [pytorch-lighting](https://github.com/Lightning-AI/pytorch-lightning) framework.

## Prerequisites
- Python 3.10.6
```
conda create -n pl_pointnet python=3.10.6
```
- Pytorch 2.0.1+cu118 Torchvision 0.15.2+cu118 Torchaudio 2.0.2+cu118
  
  Install pytorch by conda as:
  
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
```
  
  or pip as:
  
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
- Install requirements
```
pip install -r requirements.txt
```

## Data Preparation(ModelNet10/Modelnet40)
### Download
The modelnet10/40 dataset can be downloaded from [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch?tab=readme-ov-file).

## Train
```
python train.py
```

## Hyper-parameter Tutorial
- The common setting (including ddp, early stopping or optimizer and any other tricks) can be set from [here](https://github.com/sutiankang/pytorch-lightning-pointnet/blob/main/configs/default.yaml).
- If you want to speed up dataset loading using offline data, you can set the parameter ```use_cache``` to ```True``` from [here](https://github.com/sutiankang/pytorch-lightning-pointnet/blob/main/configs/datasets/modelnet40_normal_resampled.yaml). 
- If you want to get better performance by adding normals, you can set the paramater ```use_normals``` to ```True``` from[here](https://github.com/sutiankang/pytorch-lightning-pointnet/blob/main/configs/datasets/modelnet40_normal_resampled.yaml).

## Results
| Model | Accuracy |
|--|--|
| PointNet (Official) |  89.2 |
| PointNet2 (Official) | 91.9 |
| Pointnet_Pointnet2_pytorch (Pytorch with normal) |  91.4 |
| pytorch-lightning-pointnet (pytorch-lightning with normal) |  91.7 |

## Weights
You can download our weight after training from [here](https://drive.google.com/drive/folders/14iv_pvSM9Og0rVIoIdCT-_nbuNAXot5f?usp=sharing).

## Reference
[Pytorch_Pointnet_Pointnet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{pytorch-lightning-pointnet,
      Author = {Tiankang Su},
      Title = {pytorch-lightning-pointnet},
      Journal = {https://github.com/sutiankang/pytorch-lightning-pointnet},
      Year = {2024}
}
```
