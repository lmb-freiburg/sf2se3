### Directory Structure
```
datasets  
│
└───KITTI_flow
│   └───testing
│   │   └───image_2
│   │   └───image_3
│   │   └───data_scene_flow_calib
│   │       └───calib_cam_to_cam
│   │       └───calib_imu_to_velo
│   │       └───calib_velo_to_cam
│   └───training
│       └───image_2
│       └───image_3
│       └───data_scene_flow_calib
│       │   └───calib_cam_to_cam
│       │   └───calib_imu_to_velo
│       │   └───calib_velo_to_cam
│       └───disp_occ_0
│       └───...
│       └───flow_occ
└───Brox_SceneFlow
│   └───full
│       └───camera_data
│       │   └───TEST
│       │   │   └───A
│       │   │   │   └───0000
│       │   │   └───...
│       │   │   └───B
│       │   │   └───C
│       │   └───TRAIN
│       └───disparities
│       └───optical_flow
│       └───frames_cleanpass
│       └───frames_finalpass
└───MPI-Sintel
│   └───MPI-Sintel-depth
│   │   └───training
│   │       └───depth
│   │       └───depth_viz
│   │       └───camdata_left
│   └───MPI-Sintel-disp
│   │   └───training
│   │       └───disparities
│   │       └───clean_left
│   │       └───...
│   └───MPI-Sintel-oflow
│       └───training
│           └───flow
│           └───clean
│           └───...
└───Bonn_RGBD
│       └───rgbd_bonn_balloon
│       │   └───rgb
│       │   └───depth
│       │   │   rgb.txt
│       │   │   depth.txt
│       │   │   groundtruth.txt
│       └───rgbd_bonn_crowd
│           │   ...
└───TUM_RGBD
│       └───fr1
│       │   └───rgbd_dataset_freiburg1_xyz
│       │       └───rgb
│       │       └───depth
│       │       │   rgb.txt
│       │       │   depth.txt
│       │       │   groundtruth.txt
│       └───fr2
│       │   └───rgbd_dataset_freiburg2_pioneer_360
│       │   │   │   ...
│       │   └───rgbd_dataset_freiburg2_desk_with_person
│       │       │   ...
