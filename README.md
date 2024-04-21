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
- The 

## Results



## Reference By
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
