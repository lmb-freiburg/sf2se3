import os
import torch
import numpy as np
import m4_io.sintel as io_sintel
import m4_io.tum as io_tum
from datasets.seq_dataset import SeqDataset
import PIL
class BonnRGBD_Dataset(SeqDataset):

    def __init__(self, *args, **kwargs):
        super(BonnRGBD_Dataset, self).__init__(*args, **kwargs)

    def read_data(self, fpath, ftype, dkey, offset=0):
        #print('read ', fpath)
        dkey_with_offset = dkey + '_' + str(offset)
        #print('key ', dkey_with_offset)
        data = {}
        if ftype == 'rgb':
            data[dkey_with_offset] = np.array(PIL.Image.open(fpath)).astype(self.dtype_np) / 255.
            data[dkey_with_offset] = self.transform_to_tensor(data[dkey_with_offset])
        elif ftype == 'depth':
            data[dkey_with_offset] = np.array(PIL.Image.open(fpath)).astype(self.dtype_np) / 5000.
            data[dkey_with_offset] = self.transform_to_tensor(data[dkey_with_offset])
            data['depth_valid_' + str(offset)] = ~(data[dkey_with_offset] == 0.)
        elif ftype == 'ego_pose':

             dict_ego_se3 = io_tum.read_trajectory(fpath)
             tstamps = np.array(list(dict_ego_se3.keys()))#.astype(self.dtype_np)
             ego_pose = np.array(list(dict_ego_se3.values())).astype(self.dtype_np)
             ego_pose = torch.from_numpy(ego_pose)
             #print(ego_pose[0])
             #print(ego_pose[6])
             if 'bonn' in fpath:
                 T_ros = np.matrix('-1 0 0 0;\
                                     0 0 1 0;\
                                     0 1 0 0;\
                                     0 0 0 1').astype(self.dtype_np)
                 T_ros = torch.from_numpy(T_ros)[None, ]
                 T_m = np.matrix('1.0157    0.1828   -0.2389    0.0113;\
                                  0.0009   -0.8431   -0.6413   -0.0098;\
                                 -0.3009    0.6147   -0.8085    0.0111;\
                                       0         0         0    1.0000').astype(self.dtype_np)
                 T_m = torch.from_numpy(T_m)[None,]

                 ego_pose = T_ros @ ego_pose @  T_ros @ T_m

                 print('info: additional coordinate frame transformation for bonn dataset')

             #ego_pose_inv = torch.linalg.inv(ego_pose)
             #data['ego_se3'] = torch.matmul(ego_pose_inv[:-1], ego_pose[1:])
             #data['ego_se3_tstamp'] = torch.from_numpy(tstamps[:-1])

             data['ego_pose'] = ego_pose
             #print('pose - 0', data['ego_pose'][0])
             #print('se3 - 0', torch.matmul(torch.linalg.inv(data['ego_pose'][0]), data['ego_pose'][6]))
             #print('se3 - 6', torch.matmul(torch.linalg.inv(data['ego_pose'][6]), data['ego_pose'][20]))
             data['ego_pose_tstamp'] = torch.from_numpy(tstamps)
        else:
            print('error: unknown ftype', ftype)
            return -1.0
        ## TODO: resize data possibly

        for key in data.keys():
            data[key] = data[key].to(self.device)

        return data

def dataset_from_args(args):
    args.setup_dataset_dir = os.path.join(
        args.setup_datasets_dir, args.data_dataset_subdir
    )

    data_load_dirs = {
        'rgb' : args.data_dataset_rgb_subdir,
        'depth' :args.data_dataset_depth_subdir,
                     }

    data_seq_load_files = {
        'ego_pose' : 'groundtruth.txt'
    }

    data_load_offsets = {
        'rgb' : [0, 1],
        'depth': [0, 1],
        'ego_pose': [0, 1],
    }

    data_load_types = {
                        'rgb' : 'rgb',
                        'depth' : 'depth',
                        'ego_pose': 'ego_pose'
                       }

    projection_matrix = np.array([[args.data_cam_fx, 0.0, args.data_cam_cx], [0.0, args.data_cam_fy, args.data_cam_cy]], dtype=np.float32)
    reprojection_matrix = np.array(
        [[1 / args.data_cam_fx, 0.0, -args.data_cam_cx / args.data_cam_fx], [0.0, 1 / args.data_cam_fy, -args.data_cam_cy / args.data_cam_fy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # data - cam - d0: 0.039903
    # data - cam - d1: -0.099343
    # data - cam - d2: -0.000730
    # data - cam - d3: -0.000144
    # data - cam - d4: 0.000000

    data_fix = {'cam_k1' : args.data_cam_d0,
                'cam_k2' : args.data_cam_d1,
                'cam_k3' : args.data_cam_d4,
                'cam_p1' : args.data_cam_d2,
                'cam_p2' : args.data_cam_d3,
                'baseline' : args.data_baseline,
                'projection_matrix' :  projection_matrix,
                'reprojection_matrix' : reprojection_matrix}

    # defaults: camera_transformation -> identiyty
    # perhaps missing:
    #   baseline,
    #   disp2,
    #   camera_transformation
    #   'label' : os.path.join(args.setup_dataset_dir, args.data_dataset_label_subdir)

    timestamp_inter_time_min_diff = 1.0 / args.data_fps
    dataset = BonnRGBD_Dataset(
        width=args.data_res_width,
        height=args.data_res_height,
        dev=args.setup_dataloader_device,
        directory_structure=args.data_dataset_directory_structure,
        seqs_dirs_filter_tags=args.data_seqs_dirs_filter_tags,
        timestamp_inter_time_min_diff=timestamp_inter_time_min_diff,
        timestamp_inter_data_max_diff=args.data_timestamp_inter_data_max_diff,
        data_seq_load_files=data_seq_load_files,
        data_load_root=args.setup_dataset_dir,
        data_load_dirs=data_load_dirs,
        data_load_offsets=data_load_offsets,
        data_load_types=data_load_types,
        data_fix=data_fix,
        index_shift=args.data_dataset_index_shift,
        meta_use = args.data_meta_use,
        meta_recalc = args.data_meta_recalc
    )

    #max_num_imgs=args.data_dataset_max_num_imgs,
    #index_shift=args.data_dataset_index_shift,
    return dataset
