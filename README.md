
## [SF2SE3: Clustering Scene Flow into SE(3)-Motions via Proposal and Selection [GCPR'22]](https://arxiv.org/pdf/2209.08532.pdf)

```
@inproceedings{sommer2022sf2se3,
  title={Sf2se3: Clustering scene flow into se (3)-motions via proposal and selection},
  author={Sommer, Leonhard and Schr{\"o}ppel, Philipp and Brox, Thomas},
  booktitle={DAGM German Conference on Pattern Recognition},
  pages={215--229},
  year={2022},
  organization={Springer}
}
```


## Algorithm
To understand the algorithm start [here](tensor_operations/retrieval/sflow2se3/proposal_selection.py).

## Setup 

### (a) [Docker](scripts/setup/docker.md)    
`sudo docker build -t sflow2se3:v1 -f ./scripts/setup/Dockerfile . `

### (b) Local  
`bash scripts/setup/setup_sflow2se3.sh`  
`source venv_py38_sflow2se3/bin/activate`  

### Datasets
##### (a) Custom Stereo : supplement [datasets/Custom_Stereo](datasets/Custom_Stereo)  
##### (b) Custom RGBD : supplement [datasets/Custom_RGBD](datasets/Custom_RGBD)  
##### (c) FlyingThings3D : [download](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (Subset DispNet/FlowNet2.0) and [adapt & run](scripts/setup/datasets/flyingthings3d_dispnet_2_seq_data.py) 
##### (d) KITTI-2015 : [download](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
##### (e) TUM RGBD : [download](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)

## Inference

### (a) Docker
`sudo docker run --gpus all -v <ext_vol>:<ext_vol> sflow2se3:v1 <configs>`

### (b) Local
`python3.8 eval.py <configs>`  


### Configurations

For setup adjustments look at config-setup e.g. datasets directory and output directory.  
For data adjustments look at config-data e.g. camera settings.

##### (a) Custom Stereo
`--config-setup configs/setup/custom.yaml --config-data configs/data/custom_stereo.yaml --config-sflow2se3 configs/sflow2se3/classic.yaml --config-sflow2se3-data-dependent configs/sflow2se3/custom_stereo/classic.yaml`  

**Note:** Data is expected to have same data types as FlyingThings3D. Otherwise, adjustements to the data loading is necessary.

##### (b) Custom RGBD
`--config-setup configs/setup/custom.yaml --config-data configs/data/custom_rgbd.yaml --config-sflow2se3 configs/sflow2se3/classic.yaml --config-sflow2se3-data-dependent configs/sflow2se3/custom_rgbd/classic.yaml`  

**Note:** Data is expected to have same data types as TUM RGBD. Otherwise, adjustements to the data loading is necessary.

##### (c) FlyingThings3D
`--config-setup configs/setup/custom.yaml --config-data configs/data/flyingthings3d_dispnet.yaml --config-sflow2se3 configs/sflow2se3/classic.yaml --config-sflow2se3-data-dependent configs/sflow2se3/flyingthings3d_dispnet/classic.yaml`  

##### (d) KITTI-2015
`--config-setup configs/setup/custom.yaml --config-data configs/data/kitti.yaml --config-sflow2se3 configs/sflow2se3/classic.yaml --config-sflow2se3-data-dependent configs/sflow2se3/kitti/classic.yaml`  

##### (e) TUM RGBD
`--config-setup configs/setup/custom.yaml --config-data configs/data/tum_rgbd_fr3.yaml --config-sflow2se3 configs/sflow2se3/classic.yaml --config-sflow2se3-data-dependent configs/sflow2se3/tum_rgbd_fr3/classic.yaml`  

![alt text](results/visuals/sf2se3_rigidmask_ft3d_A_0014_4.gif "Logo Title Text 1")
![alt text](results/visuals/kitti.gif "Logo Title Text 1")


## Credits

Optical Flow Neural Network : [RAFT](https://github.com/princeton-vl/RAFT)  
Disparity Neural Network : [LEAStereo](https://github.com/XuelianCheng/LEAStereo)  
