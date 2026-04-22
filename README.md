# MVCTNet
A boundary-aware heterogeneous graph framework forfine-grained segmentation of tree crown-trunkcomponents using UAV LiDAR and multi-view imagery

---
## Environment Setup

### Requirements
This repo provides the MVCTNet source codes, which had been tested with Python 3.8, PyTorch 1.10.0, CUDA 11.3 on Ubuntu 20.04, GPU RTX3090. 

### Install Python dependencies

```bash
conda create -n mvctnet -y python=3.7 numpy=1.20 numba
conda activate mvctnet

conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c pyviz hvplot
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install typing-extensions --upgrade
pip install -r requirements.txt
```

### Install the pointnet++ cuda operation library by running the following command:

```
cd pointops
python3 setup.py install
cd ..
```
---

## Data preprocessing
3D Point Cloud → 2D multi-view images 

```
Generate 2D multi_view images/                      
    1 Generate_multi-view_images.py  
    2 Seg_multi-view_images.py
```

---

## Data Structure

```
data/
 RubberTree/                            # Point cloud dataset
    synsetoffset2category.txt
    train_test_split/
       shuffled_train_file_list.json
       shuffled_val_file_list.json
       shuffled_test_file_list.json
    <category_id>/
        <sample_id>.txt                 # [N, 7]: x y z nx ny nz label
 multi_view_predict_outputs_images/     # Multi-view image dataset
     <sample_name>_multi-view_images/
         <sample_name>_front.jpg
         <sample_name>_back.jpg
         <sample_name>_left.jpg
         <sample_name>_right.jpg
```

---

## Training 

```bash
python train_partseg.py --enable_multimodal --enable_BAHG --enable_ALFE --enable_gucl
```

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[RISurConv](https://github.com/cszyzhang/RISurConv),
[RTreeNet](https://github.com/Chocolate-37/RTreeNet)


