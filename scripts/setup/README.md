### Dependencies Structure
```
raft3d  
└───lietorch
│   └───cuda (environment variable)
│   └───python3.8-dev
└───scikit-sparse
    └───libsuitesparse-dev
    └───libcholmod3
    
rigidmask  
└───ngransac
│   └───python3.8-dev
│   └───opencv_core(ub1804: 3.4.2, ub2004: 4.2/4.5)
│   └───opencv_calib3d (ub1804: 3.4.2, ub2004: 4.2/4.5)
│   └───opencv 3.4.2 (rebuilt with eigen)
│       └───cmake
│       └───libeigen3-dev
└───dcn (v2)
        └───torch
        └───torchvision
        └───timm
        └───cuda 10.1 (during exec for unknown reason)
        └───python3.8-dev
```

*NOTE 1: lietorch, ngransac, dcn might be installed faster using ninja*  
*NOTE 2: *