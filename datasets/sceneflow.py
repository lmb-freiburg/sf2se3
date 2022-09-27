import os
import cv2
import torch
import numpy as np
import m4_io.brox as io_brox
from datasets.seq_dataset import SeqDataset
import PIL
class SceneFlowDataset(SeqDataset):

    def __init__(self, *args, **kwargs):
        super(SceneFlowDataset, self).__init__(*args, **kwargs)

    def read_data(self, fpath, ftype, dkey, offset=0):
        print('read ', fpath)
        dkey = dkey + '_' + str(offset)
        print('key ', dkey)
        data = {}
        if ftype == 'rgb':
            # TODO: check if / 255. is required
            data[dkey] = np.array(PIL.Image.open(fpath)).astype(self.dtype_np) / 255.
            data[dkey] = data[dkey][:, :, :3]
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'mask_bool':
            data[dkey] = np.array(PIL.Image.open(fpath)).astype(bool)
            data[dkey] = self.transform_to_tensor(data[dkey])
        elif ftype == 'oflow':
            flow = io_brox.read(fpath)
            # H x W x 3 : dtype=np.uint16
            # numpy.ndarray: HxWx3
            # flow_valid = torch.from_numpy(flow[:, :, 2].astype(np.bool)).to(device)
            # torch.bool: HxW
            flow_uv = flow[:, :, :2].copy()
            # TODO: consider flipping the numpy array to regain right order
            # torch.float32: 2xHxW
            data[dkey] = flow_uv
            data[dkey] = self.transform_to_tensor(data[dkey])

        elif ftype == 'disp':
            # TODO: check if cv2.imread(disp_fn, cv2.IMREAD_UNCHANGED) reads as np.uint16
            disp = io_brox.read(fpath)
            # H x W : dtype: float32
            disp = disp.copy()

            data[dkey] = disp
            data[dkey] = self.transform_to_tensor(data[dkey])

            # TODO: consider flipping the numpy array to regain right order
            data[dkey] = torch.abs(data[dkey])

        elif ftype == 'label':
            objs_labels = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(
                np.uint8
            )

            # shape: 1 x H x W range: [0, num_objects_max], type: torch.uint8
            data[dkey] = objs_labels
            data[dkey] = self.transform_to_tensor(data[dkey])

        elif ftype == 'camera_data':
            pass
            '''
            fp_transf = os.path.join(
                args.setup_dataset_dir, args.data_dataset_transfs_fpath_rel
            )
            frames_ids, transfs_left, transfs_right = io_brox.readCameraData(
                fp_transf, self.dtype
            )

            self.gt_transfs = (
                torch.stack(transfs_left).to(self.device).type(self.dtype)
            )
            '''
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
                     'oflow_l_01' : os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_subdir),
                     #'oflow_occ_l_01' : os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_occ_subdir),
                     'disp_l' : os.path.join(args.setup_dataset_dir, args.data_dataset_disp_subdir),
                     #'disp_occ_l': os.path.join(args.setup_dataset_dir, args.data_dataset_disp_occ_subdir),
                     'objs_labels_l': os.path.join(args.setup_dataset_dir, args.data_dataset_label_subdir)
                     }

    data_load_offsets = {
        'rgb_l' : [0, 1],
        'rgb_r': [0, 1],
        'disp_l': [0, 1],
        #'disp_occ_l': [0, 1]
    }

    data_load_types = {'rgb_l' : 'rgb',
                       'rgb_r' : 'rgb',
                       'oflow_l_01' : 'oflow',
                       #'oflow_occ_l_01' : 'mask_bool',
                       'disp_l' : 'disp',
                       #'disp_occ_l' : 'mask_bool',
                       'objs_labels_l' : 'label'
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

    dataset = SceneFlowDataset(
        width=args.data_res_width,
        height=args.data_res_height,
        dev=args.setup_dataloader_device,
        directory_structure=args.data_dataset_directory_structure,
        data_load_dirs=data_load_dirs,
        data_load_offsets=data_load_offsets,
        data_load_types=data_load_types,
        data_fix=data_fix,
        index_shift=args.data_dataset_index_shift,
    )

    #max_num_imgs=args.data_dataset_max_num_imgs,
    #index_shift=args.data_dataset_index_shift,
    return dataset

