import os
import torch
import numpy as np
import m4_io.sintel as io_sintel

from tensor_operations.clustering import elemental as o4cluster
from datasets.seq_dataset import SeqDataset
import PIL
class SintelDataset(SeqDataset):

    def __init__(self, *args, **kwargs):
        super(SintelDataset, self).__init__(*args, **kwargs)

    def read_data(self, fpath, ftype, dkey, offset=0):
        print('read ', fpath)
        dkey = dkey + '_' + str(offset)
        print('key ', dkey)
        data = {}
        if ftype == 'rgb':
            data[dkey] = np.array(PIL.Image.open(fpath)).astype(self.dtype_np) / 255.
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'mask_bool':
            data[dkey] = np.array(PIL.Image.open(fpath)).astype(bool)
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'oflow':
            u, v = io_sintel.flow_read(fpath)
            data[dkey] = np.stack((u, v), axis=-1).astype(self.dtype_np)
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'disp':
            data[dkey] = io_sintel.disparity_read(fpath).astype(self.dtype_np)
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'depth':
            data[dkey] = io_sintel.depth_read(fpath).astype(self.dtype_np)
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'label':
            data[dkey] = io_sintel.segmentation_read(fpath).astype(int)
            data[dkey] = self.transform_to_tensor(data[dkey])
            #from tensor_operations.visual import _2d as o4visual2d
            #masks = o4cluster.label_2_onehot(data[dkey])[0]
            #masks = torch.cat((torch.sum(masks[:46], dim=0, keepdim=True).bool(), masks[46:]), dim=0)
            #o4visual2d.visualize_img(o4visual2d.mask2rgb(masks), duration=1)
            objs_masks = o4cluster.label_2_unique_2_onehot(data[dkey])[0]
            objs_masks_smaller_than_25 = (objs_masks.flatten(1).sum(dim=1) < 25)
            objs_masks = torch.cat((torch.sum(objs_masks[objs_masks_smaller_than_25], dim=0, keepdim=True), objs_masks[~objs_masks_smaller_than_25]), dim=0)
            objs_labels = o4cluster.onehot_2_label(objs_masks[None])[0] - 1
            data[dkey] = objs_labels
        elif ftype == 'camera_data':
            # K intrinsics
            # T extrinsics (R, t)
            K, T = io_sintel.cam_read(fpath)
            K = K.astype(self.dtype_np)
            T = T.astype(self.dtype_np)
            T = self.transform_to_tensor(T)[0]
            T = torch.cat((T, torch.zeros_like(T[:1, :4])), dim=0)
            T[3, 3] = 1.0
            # saved world-to-camera transofrmation: p_cam = T p_world
            T = torch.linalg.inv(T)
            # retrieve camera-to-world transformation: p_world = T p_cam

            #T_switch_axis = torch.Tensor([[1, 0, 0, 0],
            #                              [0, -1, 0, 0],
            #                              [0, 0, -1, 0],
            #                              [0, 0, 0, 1]]).type(self.dtype).to(T.device)
            #T = T_switch_axis @ T @ T_switch_axis

            data['ego_pose_' + str(offset)] = T
            # K: 3x3
            projection_key = 'projection_matrix_' + str(offset)
            reprojection_key = 'reprojection_matrix_' + str(offset)
            data[projection_key] = K[:2, :3]
            # 2 x 3
            # 3D-2D Projection:
            # u = (fx*x + cx * z) / z
            # v = (fy*y + cy * y) / z
            # shift on plane: delta_x = (fx * bx) / z
            #                 delta_y = (fy * by) / z
            # uv = (P * xyz) / z
            # P = [ fx   0  cx]
            #     [ 0   fy  cy]

            data[reprojection_key] = np.array(
                [[1 / K[0, 0], 0.0, -K[0, 2] / K[0, 0]], [0.0, 1 / K[1, 1], -K[1, 2] /K[1, 1]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            # 3 x 3
            # 2D-3D Re-Projection:
            # x = (u/fx - cx/fx) * z
            # y = (v/fy - cy/fy) * z
            # z = z
            # xyz = (RP * uv1) * z
            # RP = [ 1/fx     0  -cx/fx ]
            #      [    0  1/fy  -cy/fy ]
            #      [    0      0      1 ]

            data[reprojection_key] = self.transform_to_tensor(data[reprojection_key])[0]
            data[projection_key] = self.transform_to_tensor(data[projection_key])[0]
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

    data_load_dirs = {'rgb_l' : os.path.join(args.setup_dataset_dir, args.data_dataset_rgb_left_subdir),
                     'rgb_r' : os.path.join(args.setup_dataset_dir, args.data_dataset_rgb_right_subdir),
                     'oflow' : os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_subdir),
                     'oflow_occ' : os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_occ_subdir),
                     'oflow_invalid': os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_invalid_subdir),
                     'camera_data' : os.path.join(args.setup_dataset_dir, args.data_dataset_camera_subdir),
                     'disp' : os.path.join(args.setup_dataset_dir, args.data_dataset_disp_subdir),
                     'disp_occ': os.path.join(args.setup_dataset_dir, args.data_dataset_disp_occ_subdir),
                     'disp_oof': os.path.join(args.setup_dataset_dir, args.data_dataset_disp_oof_subdir),
                     'objs_labels' : os.path.join(args.setup_dataset_dir, "MPI-Sintel-segmentation/training/segmentation"),
                     #'depth' : os.path.join(args.setup_dataset_dir, args.data_dataset_depth_subdir),
                     }

    data_load_offsets = {
        'rgb_l' : [0, 1],
        'rgb_r': [0, 1],
        'disp': [0, 1],
        'disp_occ': [0, 1],
        'disp_oof': [0, 1],
        'camera_data' : [0, 1],
        #'depth': [0, 1]
    }

    data_load_types = {'rgb_l' : 'rgb',
                      'rgb_r' : 'rgb',
                      'oflow' : 'oflow',
                      'oflow_occ' : 'mask_bool',
                      'oflow_invalid' : 'mask_bool',
                      'camera_data' : 'camera_data',
                      'disp' : 'disp',
                      'disp_occ' : 'mask_bool',
                      'disp_oof' : 'mask_bool',
                      'objs_labels' : 'label',
                      #'depth' : 'depth',
                       }

    data_fix = {'baseline': args.data_baseline}

    # defaults: camera_transformation -> identiyty
    # perhaps missing:
    #   baseline,
    #   disp2,
    #   camera_transformation
    #   'label' : os.path.join(args.setup_dataset_dir, args.data_dataset_label_subdir)

    dataset = SintelDataset(
        width=args.data_res_width,
        height=args.data_res_height,
        dev=args.setup_dataloader_device,
        data_load_dirs=data_load_dirs,
        data_load_offsets=data_load_offsets,
        data_load_types=data_load_types,
        data_fix=data_fix,
        data_load_root=args.setup_dataset_dir,
        meta_use=args.data_meta_use,
        meta_recalc=args.data_meta_recalc,
        seqs_dirs_filter_tags=args.data_seqs_dirs_filter_tags,
        index_shift=args.data_dataset_index_shift
    )

    #max_num_imgs=args.data_dataset_max_num_imgs,
    #index_shift=args.data_dataset_index_shift,
    return dataset

