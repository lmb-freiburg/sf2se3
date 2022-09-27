from datasets.flyingthings3d import FlyingThings3D_Dataset
import os
import numpy as np

def dataset_from_args(args):
    args.setup_dataset_dir = os.path.join(
        args.setup_datasets_dir, args.data_dataset_subdir
    )

    data_load_dirs = {'rgb_l' : args.data_dataset_rgb_left_subdir,
                     'rgb_r' : args.data_dataset_rgb_right_subdir,
                     }

    data_seq_load_files = {}

    data_load_offsets = {
        'rgb_l' : [0, 1],
        'rgb_r': [0, 1],
    }

    data_load_types = {'rgb_l' : 'rgb',
                       'rgb_r' : 'rgb',
                       }

    projection_matrix = np.array([[args.data_cam_fx, 0.0, args.data_cam_cx], [0.0, args.data_cam_fy, args.data_cam_cy]], dtype=np.float32)
    reprojection_matrix = np.array(
        [[1 / args.data_cam_fx, 0.0, -args.data_cam_cx / args.data_cam_fx], [0.0, 1 / args.data_cam_fy, -args.data_cam_cy / args.data_cam_fy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    data_fix = {'cam_k1' : 0.0,
                'cam_k2' : 0.0,
                'cam_k3' : 0.0,
                'cam_p1' : 0.0,
                'cam_p2' : 0.0,
                'baseline' : args.data_baseline,
                'projection_matrix' :  projection_matrix,
                'reprojection_matrix' : reprojection_matrix}

    # defaults: camera_transformation -> identiyty
    # perhaps missing:
    #   baseline,
    #   disp2,
    #   camera_transformation
    #   'label' : os.path.join(args.setup_dataset_dir, args.data_dataset_label_subdir)

    dataset = FlyingThings3D_Dataset(
        width=args.data_res_width,
        height=args.data_res_height,
        dev=args.setup_dataloader_device,
        directory_structure=args.data_dataset_directory_structure,
        seqs_dirs_filter_tags=args.data_seqs_dirs_filter_tags,
        directory_structure_seq_depth=args.data_dataset_directory_structure_seq_depth,
        data_seq_load_files=data_seq_load_files,
        data_load_root=args.setup_dataset_dir,
        data_load_dirs=data_load_dirs,
        data_load_offsets=data_load_offsets,
        data_load_types=data_load_types,
        data_fix=data_fix,
        index_shift=args.data_dataset_index_shift,
        meta_use=args.data_meta_use,
        meta_recalc=args.data_meta_recalc
    )

    #max_num_imgs=args.data_dataset_max_num_imgs,
    #index_shift=args.data_dataset_index_shift,
    return dataset
