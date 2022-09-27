import os.path
import time
import numpy as np
from datetime import datetime

#from datasets.brox import BroxDataset
#import datasets.brox
#import datasets.sintel
#import datasets.kitti
#import datasets.bonn_rgbd
#import datasets.flyingthings3d
import zmq

import datasets.args2dataloader

import torch
import tensor_operations.visual._2d as o4visual
import tensor_operations.mask.rearrange as o4mask_rearr
import tensor_operations.geometric.pinhole as o4pinhole
import tensor_operations.geometric.se3.transform as o4geo_se3_transf
import tensor_operations.probabilistic.models.gaussian as o4prob_gauss
import tensor_operations.vision.warp as o4warp
import tensor_operations.eval as o4eval
import tensor_operations.vision.similarity as o4vis_sim

import tensor_operations.retrieval.oflow.oflow_raft as oflow_raft
import tensor_operations.retrieval.disp.disp_leastereo as disp_leaststereo

import tensor_operations.log.elemental as o4log

import tensor_operations.geometric.se3.fit.corresp_3d_3d as o4geo_se3_fit_3d_3d
import tensor_operations.geometric.se3.fit.corresp_3d_2d as o4geo_se3_fit_3d_2d

import tensor_operations.geometric.se3.elemental as o4geo_se3

import tensor_operations.clustering.hierarchical as o4cluster_hierarch
import tensor_operations.clustering.elemental as o4cluster

import tensor_operations.geometric.pinhole as o4geo_pin
import tensor_operations.masks.elemental as o4masks
from tensor_operations.preprocess import depth as o4pp_depth

from options.parser import get_args

from tensor_operations.visual.classes import VisualizationSettings


def get_memory_usage(model):
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes

    return mem

import signal

metrics = {}
logger = None
def handle_exit(sig, frame):
    metrics_avg = logger.metrics_2_avg(metrics)
    table_total = metrics_2_table(metrics_avg)
    logger.print_table(table_total)
    raise (SystemExit)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

def main():
    args = get_args()

    # args.setup_dataset_dir = os.path.join(
    #    args.setup_datasets_dir, args.setup_dataset_subdir
    # )

    logger = o4log.Logger(args)

    dataloader = datasets.args2dataloader.args2dataloader(args)

    net_oflow = oflow_raft.RAFT_Wrapper(args)

    if args.sflow2se3_approach == 'raft3d':
        import tensor_operations.retrieval.sflow2rigid.rigid_raft3d as se3_raft3d
        net_raft3d = se3_raft3d.RAFT3D_Wrapper(args)
    elif args.sflow2se3_approach == 'rigidmask':
        import tensor_operations.retrieval.sflow2rigid.rigid_rigidmask as se3_rigidmask
        net_rigidmask = se3_rigidmask.RigidMaskWrapper(args)

    if args.sflow2se3_disp_net_archictecture == "raft":
        net_disp = net_oflow
    elif args.sflow2se3_disp_net_archictecture == "leaststereo":
        net_disp = disp_leaststereo.LEASTStereoWrapper(args)
    else:
        print(
            "error: unknown disp net architecture ",
            args.sflow2se3_disp_net_archictecture,
        )
        return 0

    data_gt = {}
    data_pred_sflow = {}
    data_pred_se3 = {}
    for (i, data_in) in enumerate(dataloader):

        keys_del = []
        for key, val in data_gt.items():
            if not key.startswith('seq_'):
                keys_del.append(key)
        for key in keys_del:
            del data_gt[key]

        keys_del = []
        for key, val in data_pred_sflow.items():
            if not key.startswith('seq_'):
                keys_del.append(key)
        for key in keys_del:
            del data_pred_sflow[key]

        keys_del = []
        for key, val in data_pred_se3.items():
            if not key.startswith('seq_'):
                keys_del.append(key)
        for key in keys_del:
            del data_pred_se3[key]

        torch.cuda.empty_cache()

        metrics_batch = {}
        print("\n\n NEW BATCH ID ", i, " \n\n")

        if "sintel" in args.data_dataset_tags:

            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)
            data_in['projection_matrix'] = data_in['projection_matrix_0']
            data_in['reprojection_matrix'] = data_in['reprojection_matrix_0']
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_l_0'], data_in['rgb_l_1']), dim=1)
            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)

            data_in['oflow'] = data_in['oflow_0']
            data_in['oflow_occ'] = data_in['oflow_occ_0']
            data_in['oflow_valid'] = (~data_in['oflow_invalid_0'])

            data_in['disp_occ_0'] = ~((~data_in['disp_occ_0']) * (~data_in['disp_oof_0']))
            data_in['disp_occ_1'] = ~((~data_in['disp_occ_1']) * (~data_in['disp_oof_1']))
            data_in['disp_valid_0'] = torch.ones_like(data_in['disp_occ_0'])
            data_in['disp_valid_1'] = torch.ones_like(data_in['disp_occ_1'])
            #data_in['objs_labels'] = torch.zeros_like(data_in['disp_valid_0'])

            data_in['objs_labels'] = data_in['objs_labels_0']

            data_in['ego_se3'] = torch.matmul(torch.linalg.inv(data_in['ego_pose_0']), data_in['ego_pose_1'])

            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.8612, 0.0665, 0.5039, -1.7383],
                 [0.1356, 0.9254, -0.3538, 1.4369],
                 [-0.4898, 0.3730, 0.7880, 9.0598],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.05

        elif "kitti" in args.data_dataset_tags:
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_l_0'], data_in['rgb_l_1']), dim=1)
            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)

            if "train" in args.data_dataset_tags:
                data_in['oflow_occ'] = torch.zeros_like(data_in['oflow_valid'])
                data_in['disp_occ_0'] = torch.zeros_like(data_in['disp_valid_0'])
                data_in['disp_occ_f0_1'] = torch.zeros_like(data_in['disp_valid_f0_1'])
                data_in['ego_pose_1'] = data_in['ego_se3']
                device = data_in['ego_pose_1'].device
                dtype = data_in['ego_pose_1'].dtype
                B= data_in['ego_pose_1'].shape[0]
                data_in['ego_pose_0'] = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)

            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[ 9.3827e-01,  6.5216e-03,  3.4585e-01, -4.1100e+00],
                 [ 8.6179e-02,  9.6389e-01, -2.5197e-01,  2.4777e+00],
                 [-3.3501e-01,  2.6622e-01,  9.0382e-01,  9.1466e+00],
                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])

            data_in["eval_pt3d_radius"] = 0.3

        elif "custom_rgbd" in args.data_dataset_tags:
            data_in['rgb_0_distort'] = data_in['rgb_0'].clone()
            data_in = o4pinhole.undistort_data(data_in, keys=['rgb_0', 'rgb_1', 'depth_0', 'depth_1', 'depth_valid_0',
                                                              'depth_valid_1'],
                                               proj_mat_2x3=data_in['projection_matrix'],
                                               reproj_mat_3x3=data_in['reprojection_matrix'],
                                               k1=args.data_cam_d0,
                                               k2=args.data_cam_d1,
                                               p1=args.data_cam_d2,
                                               p2=args.data_cam_d3,
                                               k3=args.data_cam_d4)
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_0'], data_in['rgb_1']), dim=1)
            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.9981, 0.0079, -0.0606, 2.6463],
                 [0.0077, 0.9673, 0.2536, -3.4857],
                 [0.0606, -0.2536, 0.9654, 3.3958],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.16

        elif "custom_stereo" in args.data_dataset_tags:
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_l_0'], data_in['rgb_l_1']), dim=1)
            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)

            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.9981, 0.0079, -0.0606, 2.6463],
                 [0.0077, 0.9673, 0.2536, -3.4857],
                 [0.0606, -0.2536, 0.9654, 3.3958],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.16

        elif "flyingthings3d" in args.data_dataset_tags and "dispnet" in args.data_dataset_tags:
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_l_0'], data_in['rgb_l_1']), dim=1)
            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)

            data_in['oflow'] = data_in['oflow_l_01_0']
            data_in['oflow_occ'] = data_in['oflow_occ_l_01_0']
            data_in['oflow_valid'] = torch.ones_like(data_in['oflow_occ'])

            data_in['disp_0'] = data_in['disp_l_0']
            data_in['disp_occ_0'] = data_in['disp_occ_l_0']
            data_in['disp_valid_0'] = torch.ones_like(data_in['disp_occ_0']) * (~data_in['depth_bound_l_0'])

            data_in['disp_1'] = data_in['disp_l_1']
            data_in['disp_f0_1'] = data_in['disp_l_0'] - data_in['disp_change_l_0']
            #data_in['depth_bound_l_0']
            data_in['disp_valid_f0_1'] = torch.ones_like(data_in['disp_occ_0'])
            data_in['disp_occ_f0_1'] = torch.zeros_like(data_in['disp_occ_0'])

            data_in['disp_occ_1'] = data_in['disp_occ_l_1']
            data_in['disp_valid_1'] = torch.ones_like(data_in['disp_occ_1'])

            data_in['objs_labels'] = data_in['objs_labels_l_0']

            data_in['ego_se3'] = torch.matmul(torch.linalg.inv(data_in['ego_pose_0']), data_in['ego_pose_1'])

            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.9981, 0.0079, -0.0606, 2.6463],
                 [0.0077, 0.9673, 0.2536, -3.4857],
                 [0.0606, -0.2536, 0.9654, 3.3958],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.16

        elif "sceneflow" in args.data_dataset_tags or "flyingthings3d" in args.data_dataset_tags:
            # 'cam_k1', 'cam_k2', 'cam_k3', 'cam_p1', 'cam_p2', 'baseline', 'projection_matrix', 'reprojection_matrix',
            # 'rgb_l_0', 'rgb_l_1', 'rgb_r_0', 'rgb_r_1', 'oflow_l_01_0', 'disp_l_0', 'disp_l_1', 'seq_el_id', 'seq_len'
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_l_0'], data_in['rgb_l_1']), dim=1)
            data_in['rgb_r_01'] = torch.cat((data_in['rgb_r_0'], data_in['rgb_r_1']), dim=1)

            data_in['oflow'] = data_in['oflow_l_01_0']
            data_in['disp_0'] = data_in['disp_l_0']
            data_in['disp_1'] = data_in['disp_l_1']

            data_in['oflow_occ'] = torch.zeros_like(data_in['disp_0']).type(torch.bool)
            data_in['oflow_valid'] = torch.ones_like(data_in['oflow_occ'])

            data_in['disp_occ_0'] = torch.zeros_like(data_in['disp_0']).type(torch.bool)
            data_in['disp_valid_0'] = torch.ones_like(data_in['disp_occ_0'])

            data_in['disp_occ_1'] = torch.zeros_like(data_in['disp_0']).type(torch.bool)
            data_in['disp_valid_1'] = torch.ones_like(data_in['disp_occ_1'])

            data_in['objs_labels'] = data_in['objs_labels_l_0']

            data_in['ego_se3'] = torch.matmul(torch.linalg.inv(data_in['ego_pose_0']), data_in['ego_pose_1'])

            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.9981, 0.0079, -0.0606, 2.6463],
                 [0.0077, 0.9673, 0.2536, -3.4857],
                 [0.0606, -0.2536, 0.9654, 3.3958],
                 [0.0000, 0.0000, 0.0000, 1.0000]]
            )
            data_in["eval_pt3d_radius"] = 0.16

        elif "bonn_rgbd" in args.data_dataset_tags or "tum_rgbd" in args.data_dataset_tags:

            #data_in['objs_labels'] = torch.zeros_like(data_in['depth_valid_0'])
            #data_in['disp_0'] = o4pinhole.depth_2_disp(data_in['depth_0'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            #data_in['disp_valid_0'] = data_in['depth_valid_0']
            #data_in['disp_1'] = o4pinhole.depth_2_disp(data_in['depth_1'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            #data_in['disp_valid_1'] = data_in['depth_valid_1']
            #data_in['disp_occ_0'] = torch.zeros_like(data_in['disp_valid_0'])

            data_in['rgb_0_distort'] = data_in['rgb_0'].clone()

            data_in = o4pinhole.undistort_data(data_in, keys=['rgb_0', 'rgb_1', 'depth_0', 'depth_1', 'depth_valid_0', 'depth_valid_1'],
                                                       proj_mat_2x3=data_in['projection_matrix'],
                                                       reproj_mat_3x3=data_in['reprojection_matrix'],
                                                       k1=args.data_cam_d0,
                                                       k2=args.data_cam_d1,
                                                       p1=args.data_cam_d2,
                                                       p2=args.data_cam_d3,
                                                       k3=args.data_cam_d4)
            
            data_in['rgb_l_01'] = torch.cat((data_in['rgb_0'], data_in['rgb_1']), dim=1)

            data_in['ego_se3'] = torch.matmul(torch.linalg.inv(data_in['ego_pose_0']), data_in['ego_pose_1'])
            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.8104, -0.0952, 0.5782, -1.4318],
                 [0.2848, 0.9263, -0.2468, 0.6640],
                 [-0.5121, 0.3646, 0.7777, 2.8763],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.05

        elif "remote_rgbd" in args.data_dataset_tags:
            args.eval_live = True
            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.8104, -0.0952, 0.5782, -1.4318],
                 [0.2848, 0.9263, -0.2468, 0.6640],
                 [-0.5121, 0.3646, 0.7777, 2.8763],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.05
        else:
            data_in['eval_camera_extrinsics'] = torch.Tensor(
                [[0.8104, -0.0952, 0.5782, -1.4318],
                 [0.2848, 0.9263, -0.2468, 0.6640],
                 [-0.5121, 0.3646, 0.7777, 2.8763],
                 [0.0000, 0.0000, 0.0000, 1.0000]])
            data_in["eval_pt3d_radius"] = 0.05

        data_in["visual_settings"] = VisualizationSettings(intrinsics=data_in['projection_matrix'][0],
                                                           extrinsics=data_in['eval_camera_extrinsics'],
                                                           H=data_in['rgb_l_01'].shape[2],
                                                           W=data_in['rgb_l_01'].shape[3],
                                                           pt3d_radius=data_in["eval_pt3d_radius"])
        data_gt['fps'] = args.data_fps
        data_pred_sflow['projection_matrix'] = data_in['projection_matrix']
        data_pred_sflow['reprojection_matrix'] = data_in['reprojection_matrix']
        data_pred_sflow['baseline'] = data_in['baseline']
        data_pred_sflow['rgb_l_01'] = data_in['rgb_l_01']
        #dz/z = tan(alpha) * (dx/z) = tan(alpha) * (du/W) * (W/fx) # 45 -> 89.999999
        args.sflow2se3_model_euclidean_nn_rel_depth_dev_std = torch.tan(torch.Tensor([89.999999 / 360. * 2 * np.pi])) * args.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std * data_in['rgb_l_01'].shape[3]/data_in['projection_matrix'][0, 0, 0].to('cpu')
        args.sflow2se3_model_euclidean_nn_rel_depth_dev_std = args.sflow2se3_model_euclidean_nn_rel_depth_dev_std.item()
        #data_in['rgb_l_01'] = torch.cat(())


        #args.eval_visualize_dataset = True
        if args.eval_visualize_dataset:
            o4visual.visualize_data(data_in, height=args.eval_visualize_height_max, width=args.eval_visualize_width_max)

        time_start = time.time()

        # potential inputs:
        # rgb_0, rgb_l_0, rgb_r_0, rgb_1, rgb_l_1, rgb_r_1
        # disp_0, disp_1, disp_f0_1, disp_valid_0, disp_valid_1, disp_valid_f0_1,
        # depth_0, depth_1, depth_valid_0, depth_valid_1
        #

        if 'objs_labels' in data_in.keys():
            data_gt['objs_labels'] = data_in['objs_labels']
            data_gt['objs_masks'] = o4mask_rearr.label2unique2onehot(data_gt['objs_labels'])[0]


        data_gt['seq_tag'] = data_in['seq_tag'][0]
        data_gt['seq_el_id'] = data_in['seq_el_id']
        data_gt['seq_len'] = data_in['seq_len']

        print('batch:', i, 'seq', data_gt['seq_tag'], 'seq-el-id', data_gt['seq_el_id'].item() + 1, ':', data_gt['seq_len'].item())

        #####   AGGREGATION GROUND TRUTH    #####
        #####             START             #####
        if 'ego_pose_0' in data_in.keys():
            data_gt['ego_pose_0'] = data_in['ego_pose_0']
            print(data_in['seq_el_id'])
            if 'seq_ego_poses_0' not in data_gt.keys() or data_in['seq_el_id'] == 0:
                data_gt['seq_ego_poses_0'] = []
            data_gt['seq_ego_poses_0'].append(data_in['ego_pose_0'])
        if 'ego_pose_1' in data_in.keys():
            data_gt['ego_pose_1'] = data_in['ego_pose_1']
            if 'seq_ego_poses_1' not in data_gt.keys() or data_in['seq_el_id'] == 0:
                data_gt['seq_ego_poses_1'] = []
            data_gt['seq_ego_poses_1'].append(data_in['ego_pose_1'])
        
        if 'ego_se3' in data_in.keys():
            #print('ego se3',  data_in['ego_se3'])
            data_gt['ego_se3'] = data_in['ego_se3']

        data_in_keys = data_in.keys()

        if 'oflow' in data_in.keys():
            data_gt['oflow'] = data_in['oflow']
            data_gt['oflow_valid'] = data_in['oflow_valid']
            data_gt['oflow_occ'] = data_in['oflow_occ']

        if 'depth_0' in data_in_keys:
            data_gt['depth_0'] = data_in['depth_0']
            data_gt['depth_valid_0'] = data_in['depth_valid_0']

            data_gt['disp_0'] = o4pinhole.depth_2_disp(data_gt['depth_0'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            data_gt['disp_valid_0'] = data_in['depth_valid_0']
            data_gt['disp_occ_0'] = torch.zeros_like(data_gt['disp_valid_0'])

            data_gt['pt3d_0'] = o4pinhole.depth_2_pt3d(data_gt['depth_0'], reproj_mats=data_in['reprojection_matrix'])
            data_gt['depth_inbounds_0'] = (data_gt['depth_0'] >= args.sflow2se3_depth_min) * (data_gt['depth_0'] <= args.sflow2se3_depth_max)
            data_gt['pt3d_valid_0'] = data_gt['depth_valid_0'] * data_gt['depth_inbounds_0']

        elif 'disp_0' in data_in_keys:
            data_gt['disp_0'] = data_in['disp_0']
            data_gt['disp_occ_0'] = data_in['disp_occ_0']
            data_gt['disp_valid_0'] = data_in['disp_valid_0']

            data_gt['depth_0'] = o4pinhole.disp_2_depth(data_gt['disp_0'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            data_gt['depth_valid_0'] = data_in['disp_valid_0']

            data_gt['depth_inbounds_0'] = (data_gt['depth_0'] >= args.sflow2se3_depth_min) * (
                        data_gt['depth_0'] <= args.sflow2se3_depth_max)

            data_gt['pt3d_0'] = o4pinhole.depth_2_pt3d(data_gt['depth_0'], reproj_mats=data_in['reprojection_matrix'])

            data_gt['pt3d_valid_0'] = data_gt['depth_valid_0'] * data_gt['depth_inbounds_0']

        if 'depth_1' in data_in_keys:
            data_gt['depth_1'] = data_in['depth_1']
            data_gt['depth_valid_1'] = data_in['depth_valid_1']
            data_gt['disp_1'] = o4pinhole.depth_2_disp(data_gt['depth_1'],
                                                               fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            data_gt['disp_valid_1'] = data_in['depth_valid_1']
            data_gt['disp_occ_1'] = torch.zeros_like(data_in['depth_valid_1'])

            data_gt['pt3d_1'] = o4pinhole.depth_2_pt3d(data_gt['depth_1'],
                                                               reproj_mats=data_in['reprojection_matrix'])

            data_gt['depth_inbounds_1'] = (data_gt['depth_1'] >= args.sflow2se3_depth_min) * (
                        data_gt['depth_1'] <= args.sflow2se3_depth_max)
            data_gt['pt3d_valid_1'] = data_in['depth_valid_1'] * data_gt['depth_inbounds_1']

        elif 'disp_f0_1' in data_in_keys:
            data_gt['disp_f0_1'] = data_in['disp_f0_1']
            data_gt['disp_valid_f0_1'] = data_in['disp_valid_f0_1']
            data_gt['disp_occ_f0_1'] = data_in['disp_occ_f0_1']

            data_gt['depth_f0_1'] = o4pinhole.disp_2_depth(data_gt['disp_f0_1'],
                                                           fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])

            data_gt['depth_inbounds_f0_1'] = (data_gt['depth_f0_1'] >= args.sflow2se3_depth_min) * (
                        data_gt['depth_f0_1'] <= args.sflow2se3_depth_max)

            data_gt['depth_valid_f0_1'] = data_in['disp_valid_f0_1']

            data_gt['pt3d_f0_1'] = o4pinhole.depth_2_pt3d(data_gt['depth_f0_1'], reproj_mats=data_in['reprojection_matrix'], oflow=data_gt['oflow'])
            data_gt['pt3d_valid_f0_1'] = data_gt['depth_valid_f0_1'] * data_gt['depth_inbounds_f0_1']

        elif 'disp_1' in data_in_keys:
            data_gt['disp_1'] = data_in['disp_1']
            data_gt['disp_valid_1'] = data_in['disp_valid_1']
            data_gt['disp_occ_1'] = data_in['disp_occ_1']

            data_gt['depth_1'] = o4pinhole.disp_2_depth(data_gt['disp_1'],
                                                               fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])

            data_gt['depth_inbounds_1'] = (data_gt['depth_1'] >= args.sflow2se3_depth_min) * (
                        data_gt['depth_1'] <= args.sflow2se3_depth_max)

            data_gt['depth_valid_1'] = data_in['disp_valid_1']

            data_gt['pt3d_1'] = o4pinhole.depth_2_pt3d(data_gt['depth_1'],
                                                               reproj_mats=data_in['reprojection_matrix'])
            data_gt['pt3d_valid_1'] = data_gt['depth_valid_1'] * data_gt['depth_inbounds_1']

        #####   AGGREGATION GROUND TRUTH    #####
        #####              END              #####

        #####   AGGREGATION WARPED SCENE FLOW    #####
        #####               START                #####
        metrics_batch["aggreg_warped_sflow_start"] = torch.DoubleTensor([time.time()])
        metrics_batch["aggreg_oflow_start"] = torch.DoubleTensor([time.time()])
        if args.sflow2se3_sflow_use_oflow_if_available and 'oflow' in data_gt.keys():
            data_pred_sflow['oflow'] = data_gt['oflow']
            data_pred_sflow['oflow_valid'] = data_gt['oflow_valid']
            data_pred_sflow['oflow_occ'] = data_gt['oflow_occ']
        else:
            pred_flows_fwd, pred_flows_bwd = net_oflow.forward_backward(data_in['rgb_l_01'])
            data_pred_sflow['oflow'] = pred_flows_fwd
            data_pred_sflow['oflow_bwd'] = pred_flows_bwd

            if args.sflow2se3_oflow_disp_std_auto:
                flow_bwd_bwdwrpd, flow_inside = o4warp.warp(
                    pred_flows_bwd.clone(), pred_flows_fwd.clone(), return_masks_flow_inside=True
                )
                data_pred_sflow['oflow_bwd'] = flow_bwd_bwdwrpd
                #flow_fwd_bwd_norm_avg = (torch.norm(pred_flows_fwd, dim=1, keepdim=True)
                #                        + torch.norm(flow_bwd_bwdwrpd, dim=1, keepdim=True)
                #                ) / 2.0
                #flow_fwd_bwd_norm_avg = (pred_flows_fwd.abs() + flow_bwd_bwdwrpd.abs()) / 2.0
                flow_fwd_bwd_devs = (pred_flows_fwd + flow_bwd_bwdwrpd) #.norm(dim=1, keepdim=True)

                args.sflow2se3_model_se3_likelihood_oflow_abs_std = o4prob_gauss.estimate_std(
                    dev=flow_fwd_bwd_devs, # / 2.0,
                    valid=flow_inside * ((flow_fwd_bwd_devs).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max * np.sqrt(2)), #((flow_fwd_bwd_devs / 2.0).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max),
                    dev_trusted_perc=args.sflow2se3_oflow_disp_trusted_perc,
                    valid_min_perc=args.sflow2se3_oflow_disp_std_valid_min_perc,
                    std_min=args.sflow2se3_oflow_disp_std_abs_min * np.sqrt(2),
                    std_max=args.sflow2se3_oflow_disp_std_abs_max * np.sqrt(2),
                    correct_trunc_factor=args.sflow2se3_oflow_disp_std_correct_truncation)[0] / np.sqrt(2)

                metrics_batch["est_oflow_x_abs_std"] = torch.Tensor([args.sflow2se3_model_se3_likelihood_oflow_abs_std[0]])
                metrics_batch["est_oflow_y_abs_std"] = torch.Tensor([args.sflow2se3_model_se3_likelihood_oflow_abs_std[1]])

                """
                oflow_gt_std = o4prob_gauss.estimate_std(dev=pred_flows_fwd - data_gt['oflow'],
                                                         valid=flow_inside * (~data_gt['oflow_occ']) * (data_gt['oflow_valid']) * ((pred_flows_fwd - data_gt['oflow']).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max),
                                                         dev_trusted_perc=args.sflow2se3_oflow_disp_trusted_perc,
                                                         valid_min_perc=args.sflow2se3_oflow_disp_std_valid_min_perc,
                                                         std_min=args.sflow2se3_oflow_disp_std_abs_min,
                                                         std_max=args.sflow2se3_oflow_disp_std_abs_max,
                                                         correct_trunc_factor=args.sflow2se3_oflow_disp_std_correct_truncation)[0]
                metrics_batch["gt_oflow_x_abs_std"] = torch.Tensor([oflow_gt_std[0]])
                metrics_batch["gt_oflow_y_abs_std"] = torch.Tensor([oflow_gt_std[1]])
                """

                """
                o4prob_gauss.visualize_estimate_std(dev=pred_flows_fwd - data_gt['oflow'],
                                                    norm=pred_flows_fwd.norm(dim=1, keepdim=True), #data_gt['oflow'].norm(dim=1, keepdim=True),
                                                    valid=flow_inside * (~data_gt['oflow_occ']) * (data_gt['oflow_valid']) * ((pred_flows_fwd - data_gt['oflow']).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max),
                                                    std_min=args.sflow2se3_oflow_disp_std_abs_min,
                                                    std_max=args.sflow2se3_oflow_disp_std_abs_max,
                                                    tag="oflow")
                """

                #metrics_batch["oflow_rel_std"] = torch.Tensor([args.sflow2se3_oflow_fwdbwd_dev_rel_max / 2.0])
            else:
                args.sflow2se3_model_se3_likelihood_oflow_abs_std = torch.Tensor(args.args.sflow2se3_model_se3_likelihood_oflow_abs_std).to(device)


            if args.sflow2se3_oflow_occl_source == 'fwdbwd':
                oflow_noc, oflow_fwdbwd_dev = o4pinhole.oflow_2_mask_valid(pred_flows_fwd, pred_flows_bwd,
                                                                           std_pixel_dev=args.sflow2se3_model_se3_likelihood_oflow_abs_std * np.sqrt(2),
                                                                           return_dev=True,
                                                                           inlier_thresh=args.sflow2se3_model_inlier_hard_threshold)

                data_pred_sflow['oflow_occ'] = ~oflow_noc
                data_pred_sflow['oflow_fwdbwd_dev'] = oflow_fwdbwd_dev
                #data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'] = \
                #    o4prob_gauss.calc_inlier_prob(oflow_fwdbwd_dev[:, 0:1], std=args.sflow2se3_model_se3_likelihood_oflow_abs_std[0] * np.sqrt(2)) * \
                #    o4prob_gauss.calc_inlier_prob(oflow_fwdbwd_dev[:, 1:2], std=args.sflow2se3_model_se3_likelihood_oflow_abs_std[1] * np.sqrt(2))
                #o4visual.visualize_img(data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0])
                ##data_pred_sflow['oflow_fwdbwd_dev']
            elif args.sflow2se3_oflow_occl_source == 'warp':
                data_pred_sflow['oflow_occ'] = ~o4vis_sim.oflow_2_mask_non_occl(
                    pred_flows_fwd,
                    data_in['rgb_l_01'],
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                )
            elif args.sflow2se3_oflow_occl_source == 'fwdbwd+warp':
                data_pred_sflow['oflow_occ'] = ~(o4vis_sim.oflow_2_mask_non_occl(
                    pred_flows_fwd,
                    data_in['rgb_l_01'],
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                ) + (o4pinhole.oflow_2_mask_valid(pred_flows_fwd, pred_flows_bwd,
                                                  std_pixel_dev=args.sflow2se3_model_se3_likelihood_oflow_abs_std * np.sqrt(2),
                                                  inlier_thresh=args.sflow2se3_model_inlier_hard_threshold)))
            else:
                print('error: unknown oflow-occl-source ', args.sflow2se3_oflow_occl_source)
            data_pred_sflow['oflow_valid'] = ~data_pred_sflow['oflow_occ']# torch.ones_like(data_pred_sflow['oflow_occ'])

        metrics_batch["aggreg_oflow_duration"] = torch.DoubleTensor([time.time()]) - metrics_batch["aggreg_oflow_start"]

        if args.sflow2se3_approach == 'rigidmask' or args.sflow2se3_approach == 'raft3d':
            metrics_batch["aggreg_oflow_duration"] *= 0.

        ### POINTS 3D T0
        metrics_batch["aggreg_depth_start"] = torch.DoubleTensor([time.time()])
        if args.sflow2se3_sflow_use_depth_if_available and 'depth_0' in data_gt.keys():
            data_pred_sflow['depth_0'] = data_gt['depth_0'].clone()
            data_pred_sflow['depth_valid_0'] = data_gt['depth_valid_0'].clone()
            if args.sflow2se3_depth_complete_invalid:
                data_pred_sflow['depth_0'], data_pred_sflow['depth_valid_0'] = o4pp_depth.complete(data_pred_sflow['depth_0'],
                                                                                                   data_pred_sflow['depth_valid_0'])
                data_pred_sflow['depth_valid_0'] = data_gt['depth_valid_0']

            data_pred_sflow['disp_0'] = o4pinhole.depth_2_disp(data_pred_sflow['depth_0'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            data_pred_sflow['disp_valid_0'] = data_in['depth_valid_0']
            data_pred_sflow['disp_occ_0'] = torch.zeros_like(data_gt['disp_valid_0'])

            data_pred_sflow['pt3d_0'] = o4pinhole.depth_2_pt3d(data_pred_sflow['depth_0'], reproj_mats=data_in['reprojection_matrix'])
            data_pred_sflow['depth_inbounds_0'] = (data_pred_sflow['depth_0'] >= args.sflow2se3_depth_min) * (data_pred_sflow['depth_0'] <= args.sflow2se3_depth_max)
            data_pred_sflow['pt3d_valid_0'] = data_pred_sflow['depth_valid_0'] * data_pred_sflow['depth_inbounds_0']
            args.sflow2se3_model_se3_likelihood_disp_abs_std = torch.Tensor([args.sflow2se3_model_se3_likelihood_disp_abs_std]).to(data_pred_sflow['pt3d_valid_0'].device)

        else:
            imgpairs1_stereo = torch.cat(
                (data_in['rgb_l_01'][:, :3], data_in['rgb_r_01'][:, :3]), dim=1
            )
            pred_disps1_fwd, pred_disps1_bwd = net_disp.forward_backward(
                imgpairs1_stereo, disp=True
            )

            if args.sflow2se3_oflow_disp_std_auto:
                B, _, H, W = pred_disps1_fwd.shape
                dtype = pred_disps1_fwd.dtype
                device = pred_disps1_fwd.device
                disp_fwd_oflow_fwd = torch.zeros(size=(B, 2, H, W), dtype=dtype, device=device)
                disp_fwd_oflow_fwd[:, :1] = -pred_disps1_fwd[:, :1]

                disp_bwd_bwdwrpd, disp_inside = o4warp.warp(
                    pred_disps1_bwd.clone(), disp_fwd_oflow_fwd.clone(), return_masks_flow_inside=True
                )

                #disp_fwd_bwd_norm_avg = (torch.norm(pred_disps1_fwd, dim=1, keepdim=True)
                #                        + torch.norm(disp_bwd_bwdwrpd, dim=1, keepdim=True)
                #                ) / 2.0
                #disp_fwd_bwd_norm_avg = (pred_disps1_fwd.abs() + disp_bwd_bwdwrpd.abs()) / 2.0

                #disp_fwd_bwd_devs = (pred_disps1_fwd - disp_bwd_bwdwrpd).norm(dim=1, keepdim=True)
                disp_fwd_bwd_devs = (pred_disps1_fwd - disp_bwd_bwdwrpd).abs()

                args.sflow2se3_model_se3_likelihood_disp_abs_std = o4prob_gauss.estimate_std(
                    dev=disp_fwd_bwd_devs, # / 2.0,
                    valid=disp_inside * (disp_fwd_bwd_devs < args.sflow2se3_oflow_disp_std_abs_max * np.sqrt(2)), # ((disp_fwd_bwd_devs / 2.0) < args.sflow2se3_oflow_disp_std_abs_max),
                    dev_trusted_perc=args.sflow2se3_oflow_disp_trusted_perc,
                    valid_min_perc=args.sflow2se3_oflow_disp_std_valid_min_perc,
                    std_min=args.sflow2se3_oflow_disp_std_abs_min * np.sqrt(2),
                    std_max=args.sflow2se3_oflow_disp_std_abs_max * np.sqrt(2),
                    correct_trunc_factor=args.sflow2se3_oflow_disp_std_correct_truncation)[0] / np.sqrt(2)

                metrics_batch["est_disp_abs_std"] = torch.Tensor([args.sflow2se3_model_se3_likelihood_disp_abs_std[0]])

                """
                disp_gt_std = o4prob_gauss.estimate_std(
                    dev=pred_disps1_fwd - data_gt['disp_0'],
                    valid=disp_inside * (~data_gt['disp_occ_0']), #* (data_gt['disp_valid_0']) * ((pred_disps1_fwd - data_gt['disp_0']).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max),
                    dev_trusted_perc=args.sflow2se3_oflow_disp_trusted_perc,
                    valid_min_perc=args.sflow2se3_oflow_disp_std_valid_min_perc,
                    std_min=args.sflow2se3_oflow_disp_std_abs_min,
                    std_max=args.sflow2se3_oflow_disp_std_abs_max,
                    correct_trunc_factor=args.sflow2se3_oflow_disp_std_correct_truncation)[0]
                metrics_batch["gt_disp_abs_std"] = torch.Tensor([disp_gt_std[0]])

                
                o4prob_gauss.visualize_estimate_std(dev=pred_disps1_fwd - data_gt['disp_0'],
                                                    norm=data_gt['disp_0'].norm(dim=1, keepdim=True),
                                                    valid=disp_inside * (~data_gt['disp_occ_0']) * (data_gt['disp_valid_0']) * ((pred_disps1_fwd - data_gt['disp_0']).norm(dim=1, keepdim=True) < args.sflow2se3_oflow_disp_std_abs_max),
                                                    std_min=args.sflow2se3_oflow_disp_std_abs_min,
                                                    std_max=args.sflow2se3_oflow_disp_std_abs_max,
                                                    tag="disp")
                """
            else:
                args.sflow2se3_model_se3_likelihood_disp_abs_std = torch.Tensor([args.sflow2se3_model_se3_likelihood_disp_abs_std]).to(pred_disps1_fwd.device)

            if args.sflow2se3_disp_occl_source == 'fwdbwd':
                disp_noc, disp_fwdbwd_dev = o4pinhole.disp_2_mask_valid(
                    pred_disps1_fwd,
                    pred_disps1_bwd,
                    std_pixel_dev=args.sflow2se3_model_se3_likelihood_disp_abs_std * np.sqrt(2),
                    inlier_thresh=args.sflow2se3_model_inlier_hard_threshold,
                    return_dev=True
                )
                data_pred_sflow['disp_occ_0'] = ~disp_noc
                data_pred_sflow['disp_fwdbwd_dev'] = disp_fwdbwd_dev

            elif args.sflow2se3_disp_occl_source == 'warp':
                data_pred_sflow['disp_occ_0'] = ~o4vis_sim.disp_2_mask_non_occl(
                    pred_disps1_fwd,
                    imgpairs1_stereo,
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                )
            elif args.sflow2se3_disp_occl_source == 'fwdbwd+warp':
                data_pred_sflow['disp_occ_0'] = ~(o4vis_sim.disp_2_mask_non_occl(
                    pred_disps1_fwd,
                    imgpairs1_stereo,
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                ) + (o4pinhole.disp_2_mask_valid(
                    pred_disps1_fwd,
                    pred_disps1_bwd,
                    std_pixel_dev=args.sflow2se3_model_se3_likelihood_disp_abs_std * np.sqrt(2),
                    inlier_thresh=args.sflow2se3_model_inlier_hard_threshold
                )))

            else:
                print('error: unknown disp-occl-source ', args.sflow2se3_disp_occl_source)

            data_pred_sflow['disp_0'] = pred_disps1_fwd
            data_pred_sflow['depth_0'] = o4pinhole.disp_2_depth(data_pred_sflow['disp_0'],
                                                                   fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])

            data_pred_sflow['depth_inbounds_0'] = (data_pred_sflow['depth_0'] >= args.sflow2se3_depth_min) * (
                        data_pred_sflow['depth_0'] <= args.sflow2se3_depth_max)

            data_pred_sflow['depth_valid_0'] = torch.ones_like(data_pred_sflow['depth_0']).type(torch.bool) # data_pred_sflow['depth_inbounds_0']
            data_pred_sflow['disp_valid_0'] = torch.ones_like(data_pred_sflow['depth_0']).type(torch.bool) # data_pred_sflow['depth_inbounds_0']

            data_pred_sflow['pt3d_0'] = o4pinhole.depth_2_pt3d(data_pred_sflow['depth_0'], reproj_mats=data_in['reprojection_matrix'])

        ### POINTS 3D T1

        if args.sflow2se3_sflow_use_depth_if_available and 'depth_1' in data_gt.keys():
            if 'oflow' not in data_gt.keys():
                data_gt['oflow'] = data_pred_sflow['oflow']
                data_gt['oflow_valid'] = data_pred_sflow['oflow_valid']
                data_gt['oflow_occ'] = data_pred_sflow['oflow_occ']
            oflow_valid_and_noc = (~data_gt['oflow_occ']) * data_gt['oflow_valid']
            data_gt['depth_f0_1'] = o4warp.warp(data_gt['depth_1'], data_gt['oflow'],
                                                mode='nearest') * oflow_valid_and_noc
            data_gt['depth_valid_f0_1'] = o4warp.warp(data_gt['depth_valid_1'], data_gt['oflow'],
                                                      mode='nearest') * oflow_valid_and_noc
            data_gt['depth_inbounds_f0_1'] = o4warp.warp(data_gt['depth_inbounds_1'], data_gt['oflow'],
                                                         mode='nearest') * oflow_valid_and_noc
            data_gt['disp_f0_1'] = o4warp.warp(data_gt['disp_1'], data_gt['oflow'],
                                               mode='nearest') * oflow_valid_and_noc
            data_gt['disp_valid_f0_1'] = o4warp.warp(data_gt['disp_valid_1'], data_gt['oflow'],
                                                     mode='nearest') * oflow_valid_and_noc
            data_gt['disp_occ_f0_1'] = ~(
                        (~o4warp.warp(data_gt['disp_occ_1'], data_gt['oflow'], mode='nearest')) * oflow_valid_and_noc)
            data_gt['pt3d_f0_1'] = o4warp.warp(data_gt['pt3d_1'], data_gt['oflow'],
                                               mode='nearest') * oflow_valid_and_noc
            data_gt['pt3d_valid_f0_1'] = data_gt['depth_valid_f0_1'] * data_gt['depth_inbounds_f0_1']


            data_pred_sflow['depth_1'] = data_gt['depth_1'].clone()
            data_pred_sflow['depth_valid_1'] = data_gt['depth_valid_1'].clone()
            if args.sflow2se3_depth_complete_invalid:
                data_pred_sflow['depth_1'], data_pred_sflow['depth_valid_1'] = o4pp_depth.complete(data_pred_sflow['depth_1'],
                                                                                                   data_pred_sflow['depth_valid_1'])
                data_pred_sflow['depth_valid_1'] = data_gt['depth_valid_1']

            data_pred_sflow['disp_1'] = o4pinhole.depth_2_disp(data_pred_sflow['depth_1'], fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])
            data_pred_sflow['disp_valid_1'] = data_in['depth_valid_1']
            data_pred_sflow['disp_occ_1'] = torch.zeros_like(data_gt['disp_valid_1'])

            data_pred_sflow['pt3d_1'] = o4pinhole.depth_2_pt3d(data_pred_sflow['depth_1'], reproj_mats=data_in['reprojection_matrix'])
            data_pred_sflow['depth_inbounds_1'] = (data_pred_sflow['depth_1'] >= args.sflow2se3_depth_min) * (data_pred_sflow['depth_1'] <= args.sflow2se3_depth_max)
            data_pred_sflow['pt3d_valid_1'] = data_pred_sflow['depth_valid_1'] * data_pred_sflow['depth_inbounds_1']

            oflow_valid_and_noc = (~data_pred_sflow['oflow_occ']) * data_pred_sflow['oflow_valid']
            data_pred_sflow['depth_f0_1'] = o4warp.warp(data_pred_sflow['depth_1'], data_pred_sflow['oflow'], mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['depth_valid_f0_1'] = o4warp.warp(data_pred_sflow['depth_valid_1'], data_pred_sflow['oflow'],
                                                      mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['depth_inbounds_f0_1'] = o4warp.warp(data_pred_sflow['depth_inbounds_1'], data_pred_sflow['oflow'],
                                                         mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['disp_f0_1'] = o4warp.warp(data_pred_sflow['disp_1'], data_pred_sflow['oflow'],
                                               mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['disp_valid_f0_1'] = o4warp.warp(data_pred_sflow['disp_valid_1'], data_pred_sflow['oflow'],
                                                     mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['disp_occ_f0_1'] = ~(
                        (~o4warp.warp(data_pred_sflow['disp_occ_1'], data_pred_sflow['oflow'], mode='nearest')) * oflow_valid_and_noc)
            data_pred_sflow['pt3d_f0_1'] = o4warp.warp(data_pred_sflow['pt3d_1'], data_pred_sflow['oflow'],
                                               mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['pt3d_valid_f0_1'] = data_pred_sflow['depth_valid_f0_1'] * data_pred_sflow['depth_inbounds_f0_1']

        else:
            imgpairs2_stereo = torch.cat(
                (data_in['rgb_l_01'][:, 3:], data_in['rgb_r_01'][:, 3:]), dim=1
            )
            pred_disps2_fwd, pred_disps2_bwd = net_disp.forward_backward(
                imgpairs2_stereo, disp=True
            )

            if args.sflow2se3_disp_occl_source == 'fwdbwd':
                data_pred_sflow['disp_occ_1'] = ~o4pinhole.disp_2_mask_valid(
                    pred_disps2_fwd,
                    pred_disps2_bwd,
                    std_pixel_dev=args.sflow2se3_model_se3_likelihood_disp_abs_std * np.sqrt(2),
                    inlier_thresh=args.sflow2se3_model_inlier_hard_threshold
                )
            elif args.sflow2se3_disp_occl_source == 'warp':
                data_pred_sflow['disp_occ_1'] = ~o4vis_sim.disp_2_mask_non_occl(
                    pred_disps2_fwd,
                    imgpairs2_stereo,
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                )
            elif args.sflow2se3_disp_occl_source == 'fwdbwd+warp':
                data_pred_sflow['disp_occ_1'] = ~(o4vis_sim.disp_2_mask_non_occl(
                    pred_disps2_fwd,
                    imgpairs2_stereo,
                    max_ddsim=args.sflow2se3_occl_warp_dssim_max,
                )+ o4pinhole.disp_2_mask_valid(
                    pred_disps2_fwd,
                    pred_disps2_bwd,
                    std_pixel_dev=args.sflow2se3_model_se3_likelihood_disp_abs_std * np.sqrt(2),
                    inlier_thresh=args.sflow2se3_model_inlier_hard_threshold
                ))
            else:
                print('error: unknown disp-occl-source ', args.sflow2se3_disp_occl_source)

            data_pred_sflow['disp_1'] = pred_disps2_fwd

            data_pred_sflow['depth_1'] = o4pinhole.disp_2_depth(data_pred_sflow['disp_1'],
                                                                   fx=data_in['projection_matrix'][:, 0, 0], baseline=data_in['baseline'])

            data_pred_sflow['depth_inbounds_1'] = (data_pred_sflow['depth_1'] >= args.sflow2se3_depth_min) * (
                        data_pred_sflow['depth_1'] <= args.sflow2se3_depth_max)
            data_pred_sflow['depth_valid_1'] = torch.ones_like(data_pred_sflow['depth_1']).type(torch.bool) # data_pred_sflow['depth_inbounds_1']
            data_pred_sflow['disp_valid_1'] = torch.ones_like(data_pred_sflow['depth_1']).type(torch.bool) # data_pred_sflow['depth_inbounds_1']

            data_pred_sflow['pt3d_1'] = o4pinhole.depth_2_pt3d(data_pred_sflow['depth_1'],
                                                               reproj_mats=data_in['reprojection_matrix'])

            data_pred_sflow['depth_inbounds_1'] = (data_pred_sflow['depth_1'] >= args.sflow2se3_depth_min) * (
                    data_pred_sflow['depth_1'] <= args.sflow2se3_depth_max)
            data_pred_sflow['pt3d_valid_1'] =  data_pred_sflow['depth_valid_1'] * data_pred_sflow['depth_inbounds_1'] # (~data_pred_sflow['disp_occ_1'])

            oflow_valid_and_noc = (~data_pred_sflow['oflow_occ']) * data_pred_sflow['oflow_valid']
            data_pred_sflow['depth_f0_1'] = o4warp.warp(data_pred_sflow['depth_1'], data_pred_sflow['oflow'], mode='nearest') #* oflow_valid_and_noc
            data_pred_sflow['depth_valid_f0_1'] = o4warp.warp(data_pred_sflow['depth_valid_1'], data_pred_sflow['oflow'], mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['depth_inbounds_f0_1'] = o4warp.warp(data_pred_sflow['depth_inbounds_1'], data_pred_sflow['oflow'], mode='nearest') # * oflow_valid_and_noc
            data_pred_sflow['disp_f0_1'] = o4warp.warp(data_pred_sflow['disp_1'], data_pred_sflow['oflow'], mode='nearest') # * oflow_valid_and_noc
            data_pred_sflow['disp_valid_f0_1'] = o4warp.warp(data_pred_sflow['disp_valid_1'], data_pred_sflow['oflow'], mode='nearest') * oflow_valid_and_noc
            data_pred_sflow['disp_occ_f0_1'] = ~((~o4warp.warp(data_pred_sflow['disp_occ_1'], data_pred_sflow['oflow'], mode='nearest')) * oflow_valid_and_noc)
            data_pred_sflow['pt3d_f0_1'] = o4warp.warp(data_pred_sflow['pt3d_1'], data_pred_sflow['oflow'], mode='nearest') #* oflow_valid_and_noc


        data_pred_sflow['pt3d_valid_0'] = data_pred_sflow['depth_valid_0'] * data_pred_sflow['depth_inbounds_0']
        data_pred_sflow['pt3d_valid_f0_1'] = data_pred_sflow['depth_valid_f0_1'] * data_pred_sflow['depth_inbounds_f0_1'] * (~data_pred_sflow['oflow_occ']) * data_pred_sflow['oflow_valid']

        if args.sflow2se3_pt3d_valid_req_disp_nocc:
            data_pred_sflow['pt3d_valid_0'] *= ~data_pred_sflow['disp_occ_0']
            data_pred_sflow['pt3d_valid_f0_1'] *= ~data_pred_sflow['disp_occ_f0_1']

        data_pred_sflow['pt3d_pair_valid'] = data_pred_sflow['pt3d_valid_0'] * data_pred_sflow['pt3d_valid_f0_1']

        data_pred_sflow['visual_settings'] = data_in['visual_settings']

        metrics_batch["aggreg_depth_duration"] = torch.DoubleTensor([time.time()]) - metrics_batch["aggreg_depth_start"]
        metrics_batch["aggreg_warped_sflow_duration"] = torch.DoubleTensor([time.time()]) - metrics_batch["aggreg_warped_sflow_start"]
        #####   AGGREGATION WARPED SCENE FLOW    #####
        #####                END                 #####

        #####       COMPLETION GROUND TRUTH      #####
        #####                START               #####
        if 'oflow' in data_gt.keys():
            gt_std_abs, gt_std_rel = o4eval.calc_std_from_inlier(data_gt['oflow'], data_pred_sflow['oflow'],
                                                                 mask=(~data_gt['oflow_occ']) * data_gt['oflow_valid'])
            #metrics_batch["gt_oflow_abs_std"] = torch.Tensor([gt_std_abs])

        if 'oflow' not in data_gt.keys():
            data_gt['oflow'] = data_pred_sflow['oflow']
            data_gt['oflow_valid'] = data_pred_sflow['oflow_valid']
            data_gt['oflow_occ'] = data_pred_sflow['oflow_occ']

        # in case depth is not given as ground truth
        if 'pt3d_0' not in data_gt.keys():
            data_gt['depth_0'] = data_pred_sflow['depth_0']
            data_gt['disp_0'] = data_pred_sflow['disp_0']
            data_gt['pt3d_0'] = data_pred_sflow['pt3d_0']
            data_gt['depth_valid_0'] = data_pred_sflow['depth_valid_0']
            data_gt['depth_inbounds_0'] = data_pred_sflow['depth_inbounds_0']
            data_gt['disp_valid_0'] = data_pred_sflow['disp_valid_0']
            data_gt['disp_occ_0'] = data_pred_sflow['disp_occ_0']

        if 'pt3d_f0_1' not in data_gt.keys():
            if 'pt3d_1' in data_gt.keys():
                data_gt['pt3d_valid_f0_1'] = o4warp.warp(data_gt['pt3d_valid_1'],
                                                         data_gt['oflow'], mode='nearest') * \
                                             data_gt['oflow_valid']

                oflow_valid_and_noc = (~data_gt['oflow_occ']) * data_gt['oflow_valid']
                data_gt['depth_f0_1'] = o4warp.warp(data_gt['depth_1'], data_gt['oflow'],
                                                    mode='nearest') * oflow_valid_and_noc
                data_gt['depth_valid_f0_1'] = o4warp.warp(data_gt['depth_valid_1'], data_gt['oflow'],
                                                          mode='nearest') * oflow_valid_and_noc
                data_gt['depth_inbounds_f0_1'] = o4warp.warp(data_gt['depth_inbounds_1'], data_gt['oflow'],
                                                             mode='nearest') * oflow_valid_and_noc
                data_gt['disp_f0_1'] = o4warp.warp(data_gt['disp_1'], data_gt['oflow'],
                                                   mode='nearest') * oflow_valid_and_noc
                data_gt['disp_valid_f0_1'] = o4warp.warp(data_gt['disp_valid_1'], data_gt['oflow'],
                                                         mode='nearest') * oflow_valid_and_noc
                data_gt['disp_occ_f0_1'] = ~(
                        (~o4warp.warp(data_gt['disp_occ_1'], data_gt['oflow'], mode='nearest')) * oflow_valid_and_noc)
                data_gt['pt3d_f0_1'] = o4warp.warp(data_gt['pt3d_1'], data_gt['oflow'],
                                                   mode='nearest') * oflow_valid_and_noc
            else:
                data_gt['depth_f0_1'] = data_pred_sflow['depth_f0_1']
                data_gt['disp_f0_1'] = data_pred_sflow['disp_f0_1']
                data_gt['pt3d_f0_1'] = data_pred_sflow['pt3d_f0_1']

                data_gt['depth_valid_f0_1'] = data_pred_sflow['depth_valid_f0_1']
                data_gt['depth_inbounds_f0_1'] = data_pred_sflow['depth_inbounds_f0_1']
                data_gt['disp_valid_f0_1'] = data_pred_sflow['disp_valid_f0_1']
                data_gt['disp_occ_f0_1'] = data_pred_sflow['disp_occ_f0_1']
                data_gt['pt3d_valid_f0_1'] = data_pred_sflow['pt3d_valid_f0_1']

        data_gt['pt3d_valid_0'] = data_gt['depth_valid_0'] * data_gt['depth_inbounds_0']
        data_gt['pt3d_pair_valid'] = data_gt['pt3d_valid_0'] * data_gt['pt3d_valid_f0_1']

        # add se3 information / seg for gt
        if "objs_labels" in data_gt.keys():
            # objs_labels, objs_masks
            # pts: 3 x H x W
            # masks in : K x H x W
            #a = b
            data_gt['objs_params'] = {}
            data_gt['objs_params']['se3'] = {}

            data_gt['objs_params']['se3']['se3'] = o4geo_se3_fit_3d_3d.fit_se3_to_corresp_3d_3d_and_masks(masks_in=data_gt['objs_masks'],
                                                                                                          pts1=data_gt['pt3d_0'][0],
                                                                                                          pts2=data_gt['pt3d_f0_1'][0],
                                                                                                          oflow=data_gt['oflow'][0],
                                                                                                          method='cpu-epnp',
                                                                                                          proj_mat=data_in['projection_matrix'][0],
                                                                                                          corresp_count_max=1000)[None]

            if "flyingthings3d" in args.data_dataset_tags or "sintel" in args.data_dataset_tags:
                #data_gt['objs_labels_orig'] = o4cluster.onehot_2_label(data_gt['objs_masks'][None].clone())
                gt_objs_se3_connected = o4geo_se3.se3_mats_similar(data_gt['objs_params']['se3']['se3'][0], rpe_transl_thresh=0.02, rpe_angle_thresh=1.0, angle_unit='deg')

                objs_masks_limited = o4masks.random_subset_for_max(masks=data_gt['objs_masks'], N_max=1000)
                objs_K = len(objs_masks_limited)
                objs_pts = [data_gt['pt3d_0'][0][:, objs_masks_limited[id]] for id in range(objs_K)]
                objs_range = torch.stack(
                    [(objs_pts[id][:, :, None] - objs_pts[id][:, None, :]).norm(dim=0).max() for id in range(objs_K)])
                objs_dists = torch.stack(
                    [(objs_pts[id1][:, :, None] - objs_pts[id2][:, None, :]).norm(dim=0).min() for id1 in range(objs_K) for id2 in
                     range(objs_K)]).reshape(objs_K, objs_K)
                objs_range_avg = (objs_range[:, None] + objs_range[None, :]) / 2.0
                gt_objs_geo_connected = objs_dists < args.sflow2se3_model_euclidean_nn_connect_global_dist_range_ratio_max * objs_range_avg

                gt_objs_label = o4cluster_hierarch.agglomerative(1.0 - 1.0 * gt_objs_se3_connected)# * gt_objs_geo_connected)

                gt_objs_ids_clustered = o4cluster.label_2_onehot(gt_objs_label, negative_handling="ignore")
                K_cluster = len(gt_objs_ids_clustered)
                if K_cluster > 0:
                    gt_objs_masks_clustered = []
                    for k in range(K_cluster):
                        gt_objs_masks_clustered.append((
                                    data_gt['objs_masks'][None] * gt_objs_ids_clustered[k, :, None, None]).sum(dim=1).bool())
                    gt_objs_masks_clustered = torch.cat(gt_objs_masks_clustered, dim=0)
                else:
                    gt_objs_masks_clustered = torch.zeros_like(data_gt['objs_masks'][0:0])

                # gt_objs_masks_clustered = (data_gt['objs_masks'][None] * gt_objs_ids_clustered[:, :, None, None]).sum(dim=1).bool()
                data_gt['objs_masks'] = torch.cat((gt_objs_masks_clustered, data_gt['objs_masks'][(gt_objs_label==-1)]), dim=0)
                data_gt['objs_params']['se3']['se3'] = o4geo_se3_fit_3d_3d.fit_se3_to_corresp_3d_3d_and_masks(masks_in=data_gt['objs_masks'],
                                                                                                              pts1=data_gt['pt3d_0'][0],
                                                                                                              pts2=data_gt['pt3d_f0_1'][0],
                                                                                                              oflow=data_gt['oflow'][0],
                                                                                                              method='cpu-epnp',
                                                                                                              proj_mat=data_in['projection_matrix'][0],
                                                                                                              corresp_count_max=1000)[None]

                data_gt['objs_label'] = o4cluster.onehot_2_label(data_gt['objs_masks'][None])

        #####       COMPLETION GROUND TRUTH      #####
        #####                 END                #####

        metrics_batch["sflow_duration"] = torch.Tensor([(time.time() - time_start)])


        if args.eval_visualize_pred_sflow:
            #visual_dir = os.path.join(logger.run_dir, data_in['seq_tag'][0])
            o4visual.visualize_pts3d(data_pred_sflow['pt3d_0'][0], img=data_pred_sflow['rgb_l_01'][0, :3], change_viewport=True, visualize_rot_x=False)
            o4visual.visualize_data(data_pred_sflow, height=args.eval_visualize_height_max, width=args.eval_visualize_width_max)

        #####       APPROACH SE3 SCENE FLOW      #####
        #####                START               #####
        metrics_batch["sflow_to_se3_start"] = torch.DoubleTensor([time.time()])

        try:
            if args.sflow2se3_approach == 'raft3d':
                #net_raft3d = se3_raft3d.RAFT3D_Wrapper(args)
                data_pred_se3.update(net_raft3d.forward(data_pred_sflow, args))

            elif args.sflow2se3_approach == 'rigidmask':
                data_pred_se3.update(net_rigidmask.forward(data_pred_sflow))
            else:

                if "objs_labels" in data_gt.keys():
                    object_masks = o4cluster.label_2_onehot(data_gt['objs_masks'])[0]
                    #object_masks = o4mask_rearr.label2unique2onehot(data_gt['objs_labels'])[0]
                    gt_mask_rgb = o4visual.mask2rgb(object_masks, img=data_in['rgb_l_01'][0, :3])

                else:
                    gt_mask_rgb = None

                """
                (
                    labels_objs,
                    masks_objs,
                    models_params,
                    pts3d_1,
                    pts3d_1_ftf,
                ) = o4ret_sflow2se3.se3retrieval(
                    data_pred_sflow,
                    args=args,
                    gt_mask_rgb=gt_mask_rgb,
                    timings=metrics_batch,
                )
                """

                from tensor_operations.retrieval.sflow2se3.proposal_selection import sflow2se3

                (
                    labels_objs,
                    masks_objs,
                    models_params,
                    pts3d_1,
                    pts3d_1_ftf,
                ) = sflow2se3(data=data_pred_sflow, args=args, logger=logger)


                data_pred_se3['pt3d_0'] = pts3d_1
                data_pred_se3['pt3d_f0_1'] = pts3d_1_ftf
                data_pred_se3['objs_labels'] = labels_objs
                data_pred_se3['objs_masks'] = masks_objs
                data_pred_se3['objs_params'] = models_params
        except RuntimeError as e:
            print(e)
            from time import sleep
            sleep(1)
            raise e
        print("INFO :: eval :: setted sflow2se3")
        print("INFO :: eval :: set sflow2se3 meta")

        print("INFO :: eval :: set sflow2se3 -> oflow")

        if 'oflow' not in data_pred_se3.keys():
            data_pred_se3['oflow'] = o4pinhole.pt3d_2_oflow(data_pred_se3['pt3d_f0_1'], data_in['projection_matrix']) * data_pred_sflow['disp_valid_0'] * data_pred_sflow['depth_valid_0']
        data_pred_se3['depth_0'] = o4pinhole.pt3d_2_depth(data_pred_se3['pt3d_0'])
        data_pred_se3['disp_0'] = o4pinhole.pt3d_2_disp(data_pred_se3['pt3d_0'], proj_mats=data_in['projection_matrix'], baseline=data_in['baseline'])
        data_pred_se3['depth_f0_1'] = o4pinhole.pt3d_2_depth(data_pred_se3['pt3d_f0_1'])
        data_pred_se3['disp_f0_1'] = o4pinhole.pt3d_2_disp(
            data_pred_se3['pt3d_f0_1'], data_in['projection_matrix'], data_in['baseline']
        )

        metrics_batch["sflow_to_se3_duration"] = torch.DoubleTensor([time.time()]) - metrics_batch["sflow_to_se3_start"]

        if args.sflow2se3_approach == 'raft3d':
            metrics_batch["rgb_to_se3_duration"] = metrics_batch["sflow_to_se3_duration"] + metrics_batch["aggreg_warped_sflow_duration"]
        elif args.sflow2se3_approach == 'rigidmask':
            metrics_batch["rgb_to_se3_duration"] = metrics_batch["sflow_to_se3_duration"] + metrics_batch["aggreg_depth_duration"]
        else:
            metrics_batch["rgb_to_se3_duration"] = metrics_batch["sflow_to_se3_duration"] + metrics_batch["aggreg_warped_sflow_duration"]

        #####       APPROACH SE3 SCENE FLOW      #####
        #####                 END                #####


        #####       SEQUENTIAL EXTENSION SE3     #####
        #####                START               #####
        print("INFO :: eval :: set se3 sequenial")
        metrics_batch["se3_seq_ext_start"] = torch.DoubleTensor([time.time()])
        if 'objs_params' in data_pred_se3.keys():
            data_pred_se3['ego_se3'] = torch.linalg.inv(data_pred_se3['objs_params']['se3']['se3'][:, 0, :, :])

            #data_pred_se3['objs_params']['geo']['pt3d_1'] =
            #data_pred_se3['objs_params']['geo']['pt3d_1'] =
        else:
            B = data_pred_se3['pt3d_0'].shape[0]
            data_pred_se3['ego_se3'] = torch.eye(4, dtype = data_pred_se3['pt3d_0'].dtype, device=data_pred_se3['pt3d_0'].device).repeat(1, 1, 1)

        if 'seq_ego_poses_0' not in data_pred_se3.keys() or data_in['seq_el_id'] == 0:
            data_pred_se3['seq_ego_poses_0'] = []
            data_pred_se3['seq_ego_poses_1'] = []
            dtype = data_pred_se3['ego_se3'].dtype
            device = data_pred_se3['ego_se3'].device
            B = data_pred_se3['ego_se3'].shape[0]
            data_pred_se3['seq_ego_poses_0'].append(torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1))
            #if 'objs_params' in data_pred_se3.keys():
            #    data_pred_se3['seq_objs_params'] = data_pred_se3['objs_params']
        else:
            data_pred_se3['seq_ego_poses_0'].append(data_pred_se3['seq_ego_poses_1'][-1])
            #if 'objs_params' in data_pred_se3.keys():
            #    data_pred_se3['seq_objs_params'] = data_pred_se3['objs_params']
        data_pred_se3['seq_ego_poses_1'].append(torch.matmul(data_pred_se3['seq_ego_poses_0'][-1], data_pred_se3['ego_se3']))

        metrics_batch["se3_seq_ext_duration"] = torch.DoubleTensor([time.time()]) - metrics_batch["se3_seq_ext_start"]
        #####       SEQUENTIAL EXTENSION SE3     #####
        #####                 END                #####


        metrics_batch["se3_duration"] = torch.Tensor([(time.time() - time_start)])
        metrics_batch["se3_objects_count"] = torch.Tensor([(data_pred_se3['objs_masks'].shape[0])])
        metrics_batch["se3_gpu_memory_reserved"] = torch.Tensor([torch.cuda.max_memory_reserved(0) / 1e+09])
        metrics_batch["sflow_std_disp"] = args.sflow2se3_model_se3_likelihood_disp_abs_std
        metrics_batch["sflow_std_oflow_x"] = args.sflow2se3_model_se3_likelihood_oflow_abs_std[0:1]
        metrics_batch["sflow_std_oflow_y"] = args.sflow2se3_model_se3_likelihood_oflow_abs_std[1:2]


        #####    EXTEND 3D POINTS CALCULATION    #####
        #####                START               #####
        print("INFO :: eval :: objs center pred se3")
        if 'objs_params' in data_pred_se3.keys():
            if 'geo' in data_pred_se3['objs_params']:
                data_pred_se3['objs_center_3d_0'] = (data_pred_se3['objs_params']['geo']['pts_assign'][0, :, None] *
                                                     data_pred_se3['objs_params']['geo']['pts']).flatten(2).sum(dim=2) / data_pred_se3['objs_params']['geo']['pts_assign'][0, :, None].flatten(2).sum(dim=2)
            else:
                data_pred_se3['objs_center_3d_0'] = (data_pred_se3['objs_masks'][:, None] * data_pred_se3[
                    'pt3d_0']).flatten(2).sum(dim=2) / data_pred_se3['objs_masks'][:, None].flatten(2).sum(dim=2)
            data_pred_se3['objs_center_3d_1'] = o4geo_se3_transf.pts3d_transform(
                data_pred_se3['objs_center_3d_0'][:, :, None, None],
                data_pred_se3['objs_params']['se3']['se3'][0, :])[:, :, 0, 0]
            _, _, H, W = data_pred_se3['pt3d_0'].shape
            py = int(0.9 * H)
            px = int(0.5 * W)
            print("INFO :: eval :: objs center pred se3 pt3d_0")
            if data_pred_sflow['pt3d_valid_0'][0, 0, py, px]:
                data_pred_se3['objs_center_3d_0'][0] = data_pred_se3['pt3d_0'][0, :, py, px]
            else:
                data_pred_se3['objs_center_3d_0'][0] = o4pinhole.depth_2_pt3d(
                    torch.ones(1, 1, 1, 1).type(data_pred_se3['pt3d_0'].dtype).to(data_pred_se3['pt3d_0'].device),
                    reproj_mats=data_in['reprojection_matrix'])[0, :, 0, 0]

            print("INFO :: eval :: objs center pred se3 pt3d_1")
            # data_pred_se3['objs_center_3d_0'][0] = torch.Tensor([0., 0.1, 0.]).type(data_pred_se3['objs_center_3d_0'].dtype).to(data_pred_se3['objs_center_3d_0'].device)
            data_pred_se3['objs_center_3d_1'][0] = o4geo_se3_transf.pts3d_transform(
                data_pred_se3['objs_center_3d_0'][:1, :, None, None],
                torch.linalg.inv(data_pred_se3['objs_params']['se3']['se3'])[0, :1])[0, :, 0, 0]

            print("INFO :: eval :: objs center pred se3 pxl2d_0")
            data_pred_se3['objs_center_2d_0'] = \
                o4pinhole.pt3d_2_pxl2d(data_pred_se3['objs_center_3d_0'][None], data_in['projection_matrix'])[0]
            data_pred_se3['objs_center_2d_1'] = \
                o4pinhole.pt3d_2_pxl2d(data_pred_se3['objs_center_3d_1'][None], data_in['projection_matrix'])[0]

        #####    EXTEND 3D POINTS CALCULATION    #####
        #####                 END                #####

        #####            VISUALIZATION           #####
        #####                START               #####
        print("INFO :: eval :: start visualization")
        ###### EVALUATION
        if not args.eval_live:
            if args.eval_visualize_pred_se3:
                o4visual.visualize_data(data_pred_se3, height=args.eval_visualize_height_max, width=args.eval_visualize_width_max)

            print("INFO :: eval :: calc metrics eval_data se3")

            metrics_batch_se3, outlier_pred_se3 = o4eval.eval_data(data_pred_se3, data_gt, visual_dir=os.path.join(logger.run_dir, data_in['seq_tag'][0]))
            print("INFO :: eval :: calculated metrics eval_data se3")


            print("INFO :: eval :: objs center gt")

            if 'ego_se3' in data_gt.keys() and "objs_center_3d_0" in data_pred_se3.keys():
                data_gt['objs_center_3d_0'] = data_pred_se3['objs_center_3d_0'][:1]
                data_gt['objs_center_3d_1'] = o4geo_se3_transf.pts3d_transform(
                    data_gt['objs_center_3d_0'][:1, :, None, None], data_gt['ego_se3'])[:, :, 0, 0]

                data_gt['objs_center_2d_0'] = \
                o4pinhole.pt3d_2_pxl2d(data_gt['objs_center_3d_0'][None], data_in['projection_matrix'])[0]
                data_gt['objs_center_2d_1'] = \
                o4pinhole.pt3d_2_pxl2d(data_gt['objs_center_3d_1'][None], data_in['projection_matrix'])[0]

            if 'objs_params' in data_gt.keys():
                data_gt['objs_center_3d_0'] = (data_gt['objs_masks'][:, None] * data_gt['pt3d_0']).flatten(2).sum(
                    dim=2) / data_gt['objs_masks'][:, None].flatten(2).sum(dim=2)
                data_gt['objs_center_3d_1'] = o4geo_se3_transf.pts3d_transform(
                    data_gt['objs_center_3d_0'][:, :, None, None], data_gt['objs_params']['se3']['se3'][0, :])[:, :, 0,
                                              0]
                _, _, H, W = data_gt['pt3d_0'].shape
                py = int(0.9 * H)
                px = int(0.5 * W)
                print("INFO :: eval :: objs center pred se3 pt3d_0")

                if "objs_center_3d_0" in data_pred_se3.keys():
                    data_gt['objs_center_3d_0'][0] = data_pred_se3['objs_center_3d_0'][0]
                else:
                    if data_gt['pt3d_valid_0'][0, 0, py, px]:
                       data_gt['objs_center_3d_0'][0] = data_gt['pt3d_0'][0, :, py, px]
                    else:
                       data_gt['objs_center_3d_0'][0] = o4pinhole.depth_2_pt3d(
                           torch.ones(1, 1, 1, 1).type(data_gt['pt3d_0'].dtype).to(data_gt['pt3d_0'].device),
                           reproj_mats=data_in['reprojection_matrix'])[0, :, 0, 0]

                print("INFO :: eval :: objs center pred se3 pt3d_1")
                # data_gt['objs_center_3d_0'][0] = torch.Tensor([0., 0.1, 0.]).type(data_gt['objs_center_3d_0'].dtype).to(data_gt['objs_center_3d_0'].device)
                data_gt['objs_center_3d_1'][0] = o4geo_se3_transf.pts3d_transform(
                    data_gt['objs_center_3d_0'][:1, :, None, None],
                    torch.linalg.inv(data_gt['objs_params']['se3']['se3'])[0, :1])[0, :, 0, 0]

                print("INFO :: eval :: objs center pred se3 pxl2d_0")
                data_gt['objs_center_2d_0'] = \
                o4pinhole.pt3d_2_pxl2d(data_gt['objs_center_3d_0'][None], data_in['projection_matrix'])[0]
                data_gt['objs_center_2d_1'] = \
                    o4pinhole.pt3d_2_pxl2d(data_gt['objs_center_3d_1'][None], data_in['projection_matrix'])[0]

            for key, val in metrics_batch_se3.items():
                metrics_batch['se3_' + key] = val

            print("INFO :: eval :: calc metrics eval_data sflow")
            metrics_batch_sflow, outlier_pred_sflow = o4eval.eval_data(data_pred_sflow, data_gt)
            print("INFO :: eval :: calculated metrics eval_data sflow")
            for key, val in metrics_batch_sflow.items():
                metrics_batch['sflow_' + key] = val

            for key, val in metrics_batch.items():
                if key not in metrics:
                    metrics[key] = []
                    metrics[key].append(val)
                else:
                    metrics[key].append(val)

            print("INFO :: eval :: log metrics")

            metrics_batch_avg = logger.metrics_2_avg(metrics_batch)
            table_batch = metrics_2_table(metrics_batch_avg, dataset=args.data_dataset_tags, approach=args.sflow2se3_approach, datetime_now=args.datetime)
            logger.log_table(table_batch)
            logger.log_metrics(metrics_batch_avg, step=i)

            if data_in['seq_el_id'] + 1 == data_in['seq_len']:
                metrics_seq_avg = {}
                for key, val in metrics.items():
                    try:
                        metrics_seq_avg[key] = torch.stack(metrics[key])[-data_in['seq_len']:].mean().item()
                    except TypeError as e:
                        print("key", key, "val", val)
                        raise e
                table_seq = metrics_2_table(metrics_seq_avg, dataset=args.data_dataset_tags + [data_in['seq_tag'][0]],
                            approach=args.sflow2se3_approach, datetime_now=args.datetime)
                logger.log_table(table_seq, table_key="metrics_seq_not_sync", step=i)

                metrics_seq_avg_prefixed = {}
                for key, val in metrics_seq_avg.items():
                    metrics_seq_avg_prefixed['seq_' + key] = val
                logger.log_metrics(metrics_seq_avg_prefixed, step=i)

            for visualization_key in args.eval_visualization_keys:
                if visualization_key == "kitti":
                    if "kitti" in args.data_dataset_tags:
                        from m4_io import kitti as io_kitti
                        dir_disp_0 = os.path.join(logger.results_dir, logger.run_dir, "KITTI", "disp_0")
                        dir_disp_1 = os.path.join(logger.results_dir, logger.run_dir, "KITTI", "disp_1")
                        dir_oflow = os.path.join(logger.results_dir, logger.run_dir, "KITTI", "flow")
                        if not os.path.exists(dir_disp_0):
                            os.makedirs(dir_disp_0)
                        if not os.path.exists(dir_disp_1):
                            os.makedirs(dir_disp_1)
                        if not os.path.exists(dir_oflow):
                            os.makedirs(dir_oflow)
                        io_kitti.write_disp(data_pred_se3['disp_0'][0], fname=os.path.join(dir_disp_0, data_in['seq_tag'][0] + "_10.png"))
                        io_kitti.write_disp(data_pred_se3['disp_f0_1'][0], fname=os.path.join(dir_disp_1, data_in['seq_tag'][0] + "_10.png"))
                        io_kitti.write_oflow(data_pred_se3['oflow'][0], fname=os.path.join(dir_oflow, data_in['seq_tag'][0] + "_10.png"))
                else:
                    visual_comp_objs_labels = get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key=visualization_key)
                    logger.log_image(visual_comp_objs_labels, visualization_key + '/' + data_in['seq_tag'][0] + '_' + str(data_in['seq_el_id'][0].item()))

                    if args.eval_visualize_result:
                        o4visual.visualize_img(visual_comp_objs_labels, height=args.eval_visualize_height_max, width=args.eval_visualize_width_max)
                # o4visual.visualize_data_comparison(data_pred_sflow, outlier_pred_sflow, data_pred_se3, outlier_pred_se3, data_gt, data_in)
                # o4vis.visualize_img(torch.cat((o4vis.flow2rgb(gt_flows_fwd[0]), o4vis.flow2rgb(pred_flows_fwd[0])), dim=1), height=900)
                # o4vis.visualize_img(torch.cat((o4vis.mask2rgb(o4mask_rearr.label2unique2onehot(labels_objs)[0]), o4vis.mask2rgb(o4mask_rearr.label2unique2onehot(gt_labels_objs)[0])), dim=2))
        else:
            outlier_pred_se3 = None
            outlier_pred_sflow = None
        #####          EVALUATION - BATCH        #####
        #####                 END                #####


        #####            REMOTE ANSWER           #####
        #####                START               #####
        if "remote_rgbd" in args.data_dataset_tags:
                #if data_in["socket"].poll(10) == zmq.POLLIN:
                #print("INFO :: received", data_in["socket"].recv())
                rgb = get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_contours_se3')
                rgb_cv = o4visual.tensor_to_cv_img(rgb)
                import cv2
                if args.eval_remote_frame_encode:
                    #encoding_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                    _, rgb_enc = cv2.imencode('.png', rgb_cv)#, encoding_params)
                else:
                    rgb_enc = rgb_cv
                data_in["socket"].send(rgb_enc.tobytes())
        #####            REMOTE ANSWER           #####
        #####                 END                #####


    #####          EVALUATION - TOTAL        #####
    #####                START               #####
    if not args.eval_live:
        metrics_avg = logger.metrics_2_avg(metrics)
        table_total = metrics_2_table(metrics_avg, dataset=args.data_dataset_tags, approach=args.sflow2se3_approach, datetime_now=args.datetime)
        logger.log_table(table_total, table_key="metrics_not_sync", remove_old=False)
    #####          EVALUATION - TOTAL        #####
    #####                 END                #####

def get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key):
    if key == 'objs_labels_gt_orig':
        object_masks = o4mask_rearr.label2unique2onehot(data_gt['objs_labels_orig'])[0]
        vis_objs_labels_gt = o4visual.mask2rgb(
            object_masks,
            img=data_in['rgb_l_01'][0, :3],
            #colors=o4visual.get_colors(K, device=object_masks.device)
        )
        return vis_objs_labels_gt

    if key == 'objs_labels_gt':
        if 'objs_labels' in data_gt.keys():
            object_masks = o4mask_rearr.label2unique2onehot(data_gt['objs_labels'])[0]
            #K = len(object_masks)
            K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            vis_objs_labels_gt = o4visual.mask2rgb(
                object_masks,
                img=data_in['rgb_l_01'][0, :3],
                colors = o4visual.get_colors(K, K_rel=K_rel, device=object_masks.device)
            )

        else:
            K = 1
            K_rel = 1
            vis_objs_labels_gt = data_in['rgb_l_01'][0, :3]

        if 'objs_center_2d_0' in data_gt.keys():
            #draw_pixels
            vis_objs_labels_gt = o4visual.draw_pixels(vis_objs_labels_gt, data_gt['objs_center_2d_0'][:1])
            vis_objs_labels_gt = o4visual.draw_arrows_in_rgb(vis_objs_labels_gt, data_gt['objs_center_2d_0'],
                                                             data_gt['objs_center_2d_1'], colors=None, thickness=5)
            vis_objs_labels_gt = o4visual.draw_arrows_in_rgb(vis_objs_labels_gt, data_gt['objs_center_2d_0'],
                                                             data_gt['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)

        #vis_objs_labels_gt = o4visual.draw_text_in_rgb(vis_objs_labels_gt, 'objects labels: gt')
        return vis_objs_labels_gt

    if key == 'objs_labels_se3':
        if 'objs_masks' in data_gt.keys():
            K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
        else:
            K = len(data_pred_se3['objs_masks'])
            K_rel = K
        vis_objs_labels_se3 = o4visual.mask2rgb(data_pred_se3['objs_masks'], img=data_in['rgb_l_01'][0, :3], colors=o4visual.get_colors(K, K_rel=K_rel, device=data_pred_se3['objs_masks'].device))
        if 'objs_center_2d_0' in data_pred_se3.keys():
            vis_objs_labels_se3 = o4visual.draw_pixels(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'][:1])
            vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=None, thickness=5)
            vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
        return vis_objs_labels_se3

    if key == 'sflow_objs_labels_contours_se3':

        sflow_rgb = get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_se3')
        if 'objs_masks' in data_pred_se3.keys():
            if 'objs_masks' in data_gt.keys():
                K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
                K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            else:
                K = len(data_pred_se3['objs_masks'])
                K_rel = K

            vis_objs_labels_se3 = o4visual.mask2rgb(data_pred_se3['objs_masks'], img=sflow_rgb, colors=o4visual.get_colors(K, K_rel=K_rel, device=data_pred_se3['objs_masks'].device), contours=True)
            if 'objs_center_2d_0' in data_pred_se3.keys():
                vis_objs_labels_se3 = o4visual.draw_pixels(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'][:1])
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=None, thickness=5)
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
            return vis_objs_labels_se3
        else:
            return sflow_rgb

    if key == 'sflow_objs_labels_contours_gt':

        sflow_rgb = get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_gt')

        if 'objs_masks' in data_gt.keys():
            if 'objs_masks' in data_pred_se3.keys():
                K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
                K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            else:
                K = len(data_gt['objs_masks'])
                K_rel = K

            vis_objs_labels_se3 = o4visual.mask2rgb(data_gt['objs_masks'], img=sflow_rgb, colors=o4visual.get_colors(K, K_rel=K_rel, device=data_pred_se3['objs_masks'].device), contours=True)
            if 'objs_center_2d_0' in data_gt.keys():
                vis_objs_labels_se3 = o4visual.draw_pixels(vis_objs_labels_se3, data_gt['objs_center_2d_0'][:1])
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=None, thickness=5)
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
            return vis_objs_labels_se3
        else:
            if 'objs_masks' in data_pred_se3.keys():
                K = len(data_pred_se3['objs_masks'])
            else:
                K = 1
            K_rel = 1
            if 'objs_center_2d_0' in data_gt.keys():
                sflow_rgb = o4visual.draw_pixels(sflow_rgb, data_gt['objs_center_2d_0'][:1])
                sflow_rgb = o4visual.draw_arrows_in_rgb(sflow_rgb, data_gt['objs_center_2d_0'],
                                                        data_gt['objs_center_2d_1'], colors=None, thickness=5)
                sflow_rgb = o4visual.draw_arrows_in_rgb(sflow_rgb, data_gt['objs_center_2d_0'],
                                                        data_gt['objs_center_2d_1'],
                                                        colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
            return sflow_rgb

    if key == 'sflow_objs_labels_contours_gt_vs_se3':
        vis = torch.cat([get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_objs_labels_contours_gt'),
                         get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_objs_labels_contours_se3')],
        dim=2)
        return vis

    if key == 'objs_labels_contours_se3':
        if 'objs_masks' in data_pred_se3.keys():
            if 'objs_masks' in data_gt.keys():
                K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
                K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            else:
                K = len(data_pred_se3['objs_masks'])
                K_rel = K
            vis_objs_labels_se3 = o4visual.mask2rgb(data_pred_se3['objs_masks'], img=data_in['rgb_l_01'][0, :3], colors=o4visual.get_colors(K, K_rel=K_rel, device=data_pred_se3['objs_masks'].device), contours=True)
            if 'objs_center_2d_0' in data_pred_se3.keys():
                vis_objs_labels_se3 = o4visual.draw_pixels(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'][:1])
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=None, thickness=5)
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_pred_se3['objs_center_2d_0'], data_pred_se3['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
            return vis_objs_labels_se3
        else:
            img_white = torch.ones_like(data_in['rgb_l_01'][0, :3])
            return img_white

    if key == 'objs_labels_contours_gt':
        if 'objs_masks' in data_gt.keys():
            if 'objs_masks' in data_pred_se3.keys():
                K = max(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
                K_rel = min(len(data_pred_se3['objs_masks']), len(data_gt['objs_masks']))
            else:
                K = len(data_gt['objs_masks'])
                K_rel = K
            vis_objs_labels_se3 = o4visual.mask2rgb(data_gt['objs_masks'], img=data_in['rgb_l_01'][0, :3], colors=o4visual.get_colors(K, K_rel=K_rel, device=data_pred_se3['objs_masks'].device), contours=True)
            if 'objs_center_2d_0' in data_gt.keys():
                vis_objs_labels_se3 = o4visual.draw_pixels(vis_objs_labels_se3, data_gt['objs_center_2d_0'][:1])
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=None, thickness=5)
                vis_objs_labels_se3 = o4visual.draw_arrows_in_rgb(vis_objs_labels_se3, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)
            return vis_objs_labels_se3
        else:
            img_white = torch.ones_like(data_in['rgb_l_01'][0, :3])
            if 'objs_masks' in data_pred_se3.keys():
                K = len(data_pred_se3['objs_masks'])
            else:
                K = 1
            K_rel = 1
            if 'objs_center_2d_0' in data_gt.keys():
                img_white = o4visual.draw_pixels(img_white, data_gt['objs_center_2d_0'][:1])
                img_white = o4visual.draw_arrows_in_rgb(img_white, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=None, thickness=5)
                img_white = o4visual.draw_arrows_in_rgb(img_white, data_gt['objs_center_2d_0'], data_gt['objs_center_2d_1'], colors=o4visual.get_colors(K, K_rel=K_rel), thickness=2)

            return img_white

    if key == 'objs_labels_contours_gt_vs_se3':
        vis = torch.cat([get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_contours_gt'),
                         get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_contours_se3')],
        dim=2)

        return vis


    if key == 'objs_labels_se3_title':
        vis_objs_labels_se3 = get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_se3')

        #vis_objs_labels_se3 = o4visual.draw_text_in_rgb(vis_objs_labels_se3, 'objects labels: se3')

        return vis_objs_labels_se3

    if key == 'objs_labels_gt_vs_se3':
        vis_objs_labels = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_gt'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_se3'),
            ),
            dim=2,
        )
        return vis_objs_labels

    if key == 'objs_labels_orig_vs_gt':
        vis_objs_labels = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_gt_orig'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_gt'),
            ),
            dim=2,
        )
        return vis_objs_labels

    elif key == 'sflow_sflow':
        sflow = data_pred_sflow['pt3d_f0_1'] - data_pred_sflow['pt3d_0']

        if data_gt['pt3d_pair_valid'].sum() > 0:
            gt_sflow_max = float((data_gt['pt3d_f0_1'] - data_gt['pt3d_0']).norm(dim=1, keepdim=True)[
                                     data_gt['pt3d_pair_valid']].quantile(0.90))
        else:
            gt_sflow_max = 1.0
        vis_sflow_sflow = o4visual.sflow2rgb(sflow[0], reprojection_matrix=data_pred_sflow["reprojection_matrix"][0], sflow_vis_max=gt_sflow_max)
        #vis_sflow_sflow = o4visual.draw_text_in_rgb(vis_sflow_sflow, 'sflow: sflow')

        return vis_sflow_sflow

    elif key == 'sflow_se3':
        sflow = data_pred_se3['pt3d_f0_1'] - data_pred_se3['pt3d_0']
        #gt_sflow_max = float((data_gt['pt3d_f0_1'] - data_gt['pt3d_0']).norm(dim=1, keepdim=True)[data_gt['pt3d_pair_valid']].median())
        if data_gt['pt3d_pair_valid'].sum() > 0:
            gt_sflow_max = float((data_gt['pt3d_f0_1'] - data_gt['pt3d_0']).norm(dim=1, keepdim=True)[data_gt['pt3d_pair_valid']].quantile(0.90))
        else:
            gt_sflow_max = 1.0
        vis_sflow_se3 = o4visual.sflow2rgb(sflow[0], reprojection_matrix=data_pred_sflow["reprojection_matrix"][0], sflow_vis_max=gt_sflow_max)
        #vis_sflow_sflow = o4visual.draw_text_in_rgb(vis_sflow_sflow, 'sflow: sflow')

        return vis_sflow_se3

    elif key == 'sflow_gt':
        sflow = data_gt['pt3d_f0_1'] - data_gt['pt3d_0']
        if data_gt['pt3d_pair_valid'].sum() > 0:
            gt_sflow_max = float((data_gt['pt3d_f0_1'] - data_gt['pt3d_0']).norm(dim=1, keepdim=True)[data_gt['pt3d_pair_valid']].quantile(0.90))
        else:
            gt_sflow_max = 1.0
        vis_sflow_se3 = o4visual.sflow2rgb(sflow[0], reprojection_matrix=data_pred_sflow["reprojection_matrix"][0], sflow_vis_max=gt_sflow_max)
        #vis_sflow_sflow = o4visual.draw_text_in_rgb(vis_sflow_sflow, 'sflow: sflow')

        return vis_sflow_se3

    elif key == 'dro_se3' and 'objs_params' in data_pred_se3.keys() and 'geo' in data_pred_se3['objs_params']:
        #  and 'geo' in data_pred_se3['objs_params'].keys()

        if 'objs_masks' in data_gt.keys():
            K_gt =len(data_gt['objs_masks'])
        else:
            K_gt = 1
        K = len(data_pred_se3['objs_params']['geo']['pts_assign'][0])
        objs_pts = [data_pred_se3['objs_params']['geo']['pts'][0, :, data_pred_se3['objs_params']['geo']['pts_assign'][0, k]] for k in range(K)]
        #o4visual.visualize_pts3d(objs_pts, change_viewport=True)
        K = max(len(data_pred_se3['objs_masks']), K_gt)
        K_rel = min(len(data_pred_se3['objs_masks']), K_gt)
        _, H, W = data_pred_se3['objs_masks'].shape
        #intrinsics = data_in["projection_matrix"][0]
        #extrinsics = data_in["eval_camera_extrinsics"]
        #visual_settings = data_in["visual_settings"]

        """
        o4visual.visualize_pts3d(objs_pts, change_viewport=True, extrinsics=data_in["visual_settings"].extrinsics,
                                 intrinsics=data_in["visual_settings"].intrinsics, return_img=False,
                                 colors=o4visual.get_colors(K, K_rel=K_rel),
                                 H=data_in["visual_settings"].H, W=data_in["visual_settings"].W,
                                 radius=data_in["visual_settings"].pt3d_radius,
                                 se3s=data_pred_se3['objs_params']['se3']['se3'][0]
                                 )
        """

        vis_dro_se3 = o4visual.visualize_pts3d(objs_pts, change_viewport=True, extrinsics=data_in["visual_settings"].extrinsics,
                                               intrinsics=data_in["visual_settings"].intrinsics, return_img=True,
                                               colors=o4visual.get_colors(K, K_rel=K_rel),
                                               H=data_in["visual_settings"].H, W=data_in["visual_settings"].W,
                                               radius=data_in["visual_settings"].pt3d_radius,
                                               se3s=data_pred_se3['objs_params']['se3']['se3'][0])


        #objs_params = data_pred_se3['objs_params']
        #vis_sflow_se3 = o4visual.sflow2rgb(sflow[0], reprojection_matrix=data_pred_sflow["reprojection_matrix"][0])
        # vis_sflow_sflow = o4visual.draw_text_in_rgb(vis_sflow_sflow, 'sflow: sflow')

        return vis_dro_se3

    elif key == 'rgb_to_sflow_overview':
        vis_rgb_to_sflow_overview = torch.cat(
            (
                data_in['rgb_l_01'][0, :3],
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_0_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_f0_1_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='sflow_sflow'),
                #get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                #                           outlier_pred_se3, key='dro_se3'),
                #get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                #                           outlier_pred_se3, key='objs_labels_se3'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                            outlier_pred_se3, key='sflow_objs_labels_contours_se3'),
                #get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                #                            outlier_pred_se3, key='sflow_gt'),
            ),
            dim=1,
        )

        return vis_rgb_to_sflow_overview

    elif key == 'oflow_sflow':
        vis_oflow_sflow = o4visual.flow2rgb(data_pred_sflow['oflow'][0], draw_arrows=False)
        #vis_oflow_sflow = o4visual.draw_text_in_rgb(vis_oflow_sflow, 'oflow: sflow')

        return vis_oflow_sflow

    elif key == 'oflow_bwd_sflow':
        vis_oflow_sflow = o4visual.flow2rgb(data_pred_sflow['oflow_bwd'][0], draw_arrows=False)
        #vis_oflow_sflow = o4visual.draw_text_in_rgb(vis_oflow_sflow, 'oflow backward: sflow')

        return vis_oflow_sflow

    elif key == 'disp_0_gt':
        vis_disp_gt = o4visual.disp2rgb(data_gt['disp_0'][0], draw_arrows=True)
        #vis_disp_gt = o4visual.draw_text_in_rgb(vis_disp_gt, 'disp 0: gt')
        return vis_disp_gt

    elif key == 'disp_0_se3':
        vis_disp_se3 = o4visual.disp2rgb(data_pred_se3['disp_0'][0])
        #vis_disp_se3 = o4visual.draw_text_in_rgb(vis_disp_se3, 'disp 0: se3')
        return vis_disp_se3

    elif key == 'disp_f0_1_se3':
        vis_disp_se3 = o4visual.disp2rgb(data_pred_se3['disp_f0_1'][0], draw_arrows=True)
        #vis_disp_se3 = o4visual.draw_text_in_rgb(vis_disp_se3, 'disp f0 1: se3')
        return vis_disp_se3

    elif key == 'disp_0_sflow':
        vis_disp_sflow = o4visual.disp2rgb(data_pred_sflow['disp_0'][0], draw_arrows=True)
        #vis_disp_sflow = o4visual.draw_text_in_rgb(vis_disp_sflow, 'disp 0: sflow')
        return vis_disp_sflow

    elif key == 'disp_1_sflow':
        vis_disp_sflow = o4visual.disp2rgb(data_pred_sflow['disp_1'][0], draw_arrows=True)
        #vis_disp_sflow = o4visual.draw_text_in_rgb(vis_disp_sflow, 'disp 0: sflow')
        return vis_disp_sflow

    elif key == 'disp_f0_1_sflow':
        vis_disp_sflow = o4visual.disp2rgb(data_pred_sflow['disp_f0_1'][0])
        #vis_disp_sflow = o4visual.draw_text_in_rgb(vis_disp_sflow, 'disp f0 1: sflow')
        return vis_disp_sflow

    elif key == 'oflow_se3':
        vis_oflow_se3 = o4visual.flow2rgb(data_pred_se3['oflow'][0], draw_arrows=True)
        #vis_oflow_se3 = o4visual.draw_text_in_rgb(vis_oflow_se3, 'oflow: se3')

        return vis_oflow_se3


    elif key == 'oflow_gt':
        if 'oflow' in data_gt.keys():
            vis_oflow_gt = o4visual.flow2rgb(data_gt['oflow'][0], draw_arrows=False)
        else:
            vis_oflow_gt = data_in['rgb_l_01'][0, :3]
        #vis_oflow_gt = o4visual.draw_text_in_rgb(vis_oflow_gt, 'oflow: gt')

        return vis_oflow_gt

    elif key =='oflow_sflow_vs_se3':
        vis_oflow_sflow_vs_se3 = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_se3')
            ),
            dim=2,
        )
        return vis_oflow_sflow_vs_se3

    elif key =='disp_0_gt_vs_sflow':
        vis_oflow_sflow_vs_se3 = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_0_gt'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_0_sflow')
            ),
            dim=2,
        )
        return vis_oflow_sflow_vs_se3

    elif key =='oflow_gt_vs_se3':
        vis_oflow = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_gt'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_se3')
            ),
            dim=2,
        )
        return vis_oflow

    elif key == "oflow_fwd_bwd_sflow":
        vis_oflow = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_bwd_sflow')
            ),
            dim=2,
        )
        return vis_oflow
    elif key =='oflow_gt_vs_sflow':
        vis_oflow = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_gt'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='oflow_sflow')
            ),
            dim=2,
        )
        return vis_oflow

    elif key == 'disp_0_outlier_sflow':
        vis_disp_0_outlier_sflow = o4visual.mask2rgb((outlier_pred_sflow['disp_0'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_disp_0_outlier_sflow = o4visual.draw_text_in_rgb(vis_disp_0_outlier_sflow, 'disp1 outlier: sflow')
        return vis_disp_0_outlier_sflow

    elif key == 'disp_0_outlier_se3':
        vis_disp_0_outlier_se3 = o4visual.mask2rgb((outlier_pred_se3['disp_0'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_disp_0_outlier_se3 = o4visual.draw_text_in_rgb(vis_disp_0_outlier_se3, 'disp1 outlier: se3')

        return vis_disp_0_outlier_se3

    elif key=='disp_0_outlier':
        vis_disp_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_0_outlier_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_0_outlier_se3'),
            ),
            dim=2,
        )
        return vis_disp_outlier

    elif key=='oflow_outlier_pred_sflow':
        #data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0]
        vis = o4visual.mask2rgb(data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0])
        return vis

    elif key=='oflow_outlier_pred':
        #data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0]
        vis = torch.cat((o4visual.mask2rgb(data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0]),
                         o4visual.mask2rgb(data_pred_sflow['oflow_fwdbwd_dev_inlier_prob'][0])), dim=2)
        return vis

    elif key == 'disp_f0_1_outlier_sflow':
        vis_disp_f0_1_outlier_sflow = o4visual.mask2rgb((outlier_pred_sflow['disp_f0_1'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_disp_f0_1_outlier_sflow = o4visual.draw_text_in_rgb(vis_disp_f0_1_outlier_sflow, 'disp2 outlier: sflow')
        return vis_disp_f0_1_outlier_sflow

    elif key == 'disp_f0_1_outlier_se3':
        vis_disp_f0_1_outlier_se3 = o4visual.mask2rgb((outlier_pred_se3['disp_f0_1'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_disp_f0_1_outlier_se3 = o4visual.draw_text_in_rgb(vis_disp_f0_1_outlier_se3, 'disp2 outlier: se3')

        return vis_disp_f0_1_outlier_se3

    elif key=='disp_f0_1_outlier':
        vis_disp_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_f0_1_outlier_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                                           outlier_pred_se3, key='disp_f0_1_outlier_se3'),
            ),
            dim=2,
        )

        return vis_disp_outlier

    elif key=='disp_outlier':
        vis_disp_outlier = torch.cat(
            [
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='disp_0_outlier'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='disp_f0_1_outlier'),
                #get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow,
                #                           outlier_pred_se3, key='oflow_outlier_pred'),
            ],
            dim=1,
        )
        return vis_disp_outlier

    elif key == 'oflow_outlier_sflow':
        vis_oflow_outlier_sflow = o4visual.mask2rgb((outlier_pred_sflow['oflow'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_oflow_outlier_sflow = o4visual.draw_text_in_rgb(vis_oflow_outlier_sflow, 'oflow outlier: sflow')
        return vis_oflow_outlier_sflow

    elif key == 'oflow_outlier_se3':
        vis_oflow_outlier_se3 = o4visual.mask2rgb((outlier_pred_se3['oflow'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_oflow_outlier_se3 = o4visual.draw_text_in_rgb(vis_oflow_outlier_se3, 'oflow outlier: se3')

        return vis_oflow_outlier_se3

    elif key=='oflow_outlier':
        vis_oflow_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='oflow_outlier_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='oflow_outlier_se3'),
            ),
            dim=2,
        )

        return vis_oflow_outlier


    elif key == 'sflow_outlier_sflow':
        vis_sflow_outlier_sflow = o4visual.mask2rgb((outlier_pred_sflow['sflow'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_sflow_outlier_sflow = o4visual.draw_text_in_rgb(vis_sflow_outlier_sflow, 'sflow outlier: sflow')
        return vis_sflow_outlier_sflow

    elif key == 'sflow_outlier_se3':
        vis_sflow_outlier_se3 = o4visual.mask2rgb((outlier_pred_se3['sflow'])[0], img=data_in['rgb_l_01'][0, :3])
        #vis_sflow_outlier_se3 = o4visual.draw_text_in_rgb(vis_sflow_outlier_se3, 'sflow outlier: se3')

        return vis_sflow_outlier_se3

    elif key=='sflow_outlier':
        vis_sflow_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_outlier_sflow'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_outlier_se3'),
            ),
            dim=2,
        )
        return vis_sflow_outlier

    elif key=='outlier':
        vis_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='disp_outlier'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='oflow_outlier'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='sflow_outlier')
            ),
            dim=1,
        )
        return vis_outlier

    elif key=='objs_labels_gt_vs_se3+outlier':
        vis_outlier = torch.cat(
            (
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='objs_labels_gt_vs_se3'),
                get_visual_data_comparison(data_in, data_gt, data_pred_sflow, data_pred_se3, outlier_pred_sflow, outlier_pred_se3, key='outlier')
            ),
            dim=1,
        )
        return vis_outlier
    else:
        img_white = torch.ones_like(data_in['rgb_l_01'][0, :3])
        return img_white

def metrics_2_table(metrics_in, dataset='-', approach='-', datetime_now=None):
    metrics = {}
    for key, val in metrics_in.items():
        if isinstance(val, torch.Tensor):
            val = val.item()

        if 'duration' in key:
            metrics[key] = round(val, 4)
        elif 'ego_se3' in key:
            metrics[key] = round(val, 3)
        else:
            metrics[key] = round(val, 2)
        # wandb.run.summary[key] = metrics[key]

    req_keys = ["se3_disp_0_outlier_perc",
                "se3_disp_f0_1_outlier_perc",
                "se3_oflow_outlier_perc",
                "se3_sflow_outlier_perc",
                "se3_seg_acc",
                "sflow_disp_0_outlier_perc",
                "sflow_disp_f0_1_outlier_perc",
                "sflow_oflow_outlier_perc",
                "sflow_sflow_outlier_perc",
                "sflow_depth_0_abs",
                "sflow_depth_0_rel",
                "sflow_depth_0_inlier_perc",
                "se3_depth_0_abs",
                "se3_depth_0_rel",
                "se3_depth_0_inlier_perc",
                "sflow_depth_f0_1_abs",
                "sflow_depth_f0_1_rel",
                "sflow_depth_f0_1_inlier_perc",
                "se3_depth_f0_1_abs",
                "se3_depth_f0_1_rel",
                "se3_depth_f0_1_inlier_perc",
                "se3_ego_se3_rpe_dist",
                "se3_ego_se3_rpe_angle",
                "se3_ego_se3_ate",
                "sflow_disp_0_epe",
                "sflow_disp_f0_1_epe",
                "sflow_oflow_epe",
                "sflow_sflow_epe",
                "se3_disp_0_epe",
                "se3_disp_f0_1_epe",
                "se3_oflow_epe",
                "se3_sflow_epe",
                "se3_objects_count",
                "aggreg_warped_sflow_duration",
                "aggreg_depth_duration",
                "aggreg_oflow_duration",
                "sflow_to_se3_duration",
                "se3_seq_ext_duration",
                "approach_duration",
                "rgb_to_se3_duration",
                "downsampling_duration",
                "retrieval_dro_duration",
                "assign_pxl_to_dro_duration",
                "deduct_rigid_object_sflow_duration",
                "retrieval_dro_extract_duration",
                "retrieval_dro_refine_duration",
                "retrieval_dro_extract_se3_duration",
                "retrieval_dro_extract_geo_duration",
                "retrieval_dro_extract_se3_cluster_duration",
                "retrieval_dro_extract_se3_fit_duration",
                "retrieval_dro_extract_se3_select_duration",
                "se3_gpu_memory_reserved",
                "sflow_std_disp",
                "sflow_std_oflow_x",
                "sflow_std_oflow_y",
                ]

    for key in req_keys:
        if key not in metrics:
            metrics[key] = -1.

    if datetime_now is None:
        datetime_now = datetime.now()
    date_string = datetime_now.strftime("%d/%m/%Y %H:%M:%S")
    columns = ["date", "dataset", "approach", "output", "disp1 [%]", "disp2 [%]", "depth1-abs [m]", "depth1-rel [%]",
               "depth1-in [%]",
               "depth2-abs [m]", "depth2-rel [%]", "depth2-in [%]", "oflow [%]", "sflow [%]", "seg acc [%]",
               "ego-se3-dist [m/s]", "ego-se3-angle [deg/s]", "ego-se3-ate [m]", "objects [#]", "disp1 epe [pxl]", "disp2 epe [pxl]", "oflow epe [pxl]", "sflow epe [m]",
               "duration-total [s]", "duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]",
               "duration-downsample [s]", "duration-retrieve-objects [s]", "duration-assign-objects [s]", "duration-deduct-sflow [s]",
               "duration-retrieve-objects-extract [s]", "duration-retrieve-objects-refine [s]", "duration-retrieve-objects-extract-se3 [s]", "duration-retrieve-objects-extract-geo [s]",
               "duration-retrieve-objects-extract-se3-cluster [s]", "duration-retrieve-objects-extract-se3-fit [s]", "duration-retrieve-objects-extract-se3-select [s]", "gpu-memory [GB]",
               "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]",
               ]
    data = []

    data.append([
        date_string,
        dataset,
        "wsflow",
        "sflow",
        metrics["sflow_disp_0_outlier_perc"],
        metrics["sflow_disp_f0_1_outlier_perc"],
        metrics["sflow_depth_0_abs"],
        metrics["sflow_depth_0_rel"],
        metrics["sflow_depth_0_inlier_perc"],
        metrics["sflow_depth_f0_1_abs"],
        metrics["sflow_depth_f0_1_rel"],
        metrics["sflow_depth_f0_1_inlier_perc"],
        metrics["sflow_oflow_outlier_perc"],
        metrics["sflow_sflow_outlier_perc"],
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        metrics["sflow_disp_0_epe"],
        metrics["sflow_disp_f0_1_epe"],
        metrics["sflow_oflow_epe"],
        metrics["sflow_sflow_epe"],
        metrics["aggreg_warped_sflow_duration"],
        metrics["aggreg_depth_duration"],
        metrics["aggreg_oflow_duration"],
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.,
        metrics["sflow_std_disp"],
        metrics["sflow_std_oflow_x"],
        metrics["sflow_std_oflow_y"],
    ])

    data.append([
        date_string,
        dataset,
        approach,
        "se3",
        metrics["se3_disp_0_outlier_perc"],
        metrics["se3_disp_f0_1_outlier_perc"],
        metrics["se3_depth_0_abs"],
        metrics["se3_depth_0_rel"],
        metrics["se3_depth_0_inlier_perc"],
        metrics["se3_depth_f0_1_abs"],
        metrics["se3_depth_f0_1_rel"],
        metrics["se3_depth_f0_1_inlier_perc"],
        metrics["se3_oflow_outlier_perc"],
        metrics["se3_sflow_outlier_perc"],
        metrics["se3_seg_acc"],
        metrics["se3_ego_se3_rpe_dist"],
        metrics["se3_ego_se3_rpe_angle"],
        metrics["se3_ego_se3_ate"],
        metrics["se3_objects_count"],
        metrics["se3_disp_0_epe"],
        metrics["se3_disp_f0_1_epe"],
        metrics["se3_oflow_epe"],
        metrics["se3_sflow_epe"],
        metrics["rgb_to_se3_duration"],
        metrics["aggreg_depth_duration"],
        metrics["aggreg_oflow_duration"],
        metrics["sflow_to_se3_duration"],
        metrics["downsampling_duration"],
        metrics["retrieval_dro_duration"],
        metrics["assign_pxl_to_dro_duration"],
        metrics["deduct_rigid_object_sflow_duration"],
        metrics["retrieval_dro_extract_duration"],
        metrics["retrieval_dro_refine_duration"],
        metrics["retrieval_dro_extract_se3_duration"],
        metrics["retrieval_dro_extract_geo_duration"],
        metrics["retrieval_dro_extract_se3_cluster_duration"],
        metrics["retrieval_dro_extract_se3_fit_duration"],
        metrics["retrieval_dro_extract_se3_select_duration"],
        metrics["se3_gpu_memory_reserved"],
        metrics["sflow_std_disp"],
        metrics["sflow_std_oflow_x"],
        metrics["sflow_std_oflow_y"],
    ])

    table = [columns] + data
    return table


if __name__ == "__main__":
    main()
