import torch
import numpy as np
import datasets.flyingthings3d
import datasets.kitti
import datasets.sintel
import datasets.bonn_rgbd
import datasets.sceneflow
import datasets.custom_stereo
import datasets.custom_rgbd
import zmq
import m4_io.m4 as m4_io
import time
import cv2

def args2dataset(args):
    if "flyingthings3d" in args.data_dataset_tags:
        #dataloader = datasets.brox.dataloader_from_args(args)
        dataset = datasets.flyingthings3d.dataset_from_args(args)
    elif "sceneflow" in args.data_dataset_tags:
        dataset = datasets.sceneflow.dataset_from_args(args)
    elif "kitti" in args.data_dataset_tags:
        dataset = datasets.kitti.dataset_val_from_args(args)
    elif "sintel" in args.data_dataset_tags:
        dataset = datasets.sintel.dataset_from_args(args)
    elif "bonn_rgbd" in args.data_dataset_tags:
        dataset = datasets.bonn_rgbd.dataset_from_args(args)
    elif "tum_rgbd" in args.data_dataset_tags:
        dataset = datasets.bonn_rgbd.dataset_from_args(args)
    elif "custom_stereo" in args.data_dataset_tags:
        dataset = datasets.custom_stereo.dataset_from_args(args)
    elif "custom_rgbd" in args.data_dataset_tags:
        dataset = datasets.custom_rgbd.dataset_from_args(args)
    else:
        print("unknown dataset tags: ", args.data_dataset_tags)
        print("using flyingthings3d...")
        dataset = datasets.brox.dataloader_from_args(args)

    return dataset

def args2dataloader(args):
    if "remote_rgbd" in args.data_dataset_tags:
        dataloader = RemoteDataloader(args)
    else:
        dataset = args2dataset(args)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.setup_dataloader_num_workers,
            pin_memory=args.setup_dataloader_pin_memory,
        )
    # prefetch_factor=2,
    # pin_memory
    # num_workers

    return dataloader


class RemoteDataloader:
    def __init__(self, args):
        print("INFO :: setting up remote connection")
        self.args = args
        self.port_rep = "2308"
        self.context = zmq.Context()
        self.socket_rep = self.context.socket(zmq.REP)
        self.socket_rep.bind("tcp://*:%s" % self.port_rep)

        self.dtype = torch.float32
        self.dtype_np = np.float32
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = args.setup_dataloader_device
        self.el_id = 0

        projection_matrix = np.array(
            [[args.data_cam_fx, 0.0, args.data_cam_cx], [0.0, args.data_cam_fy, args.data_cam_cy]], dtype=self.dtype_np)
        projection_matrix = projection_matrix[None,]
        reprojection_matrix = np.array(
            [[1 / args.data_cam_fx, 0.0, -args.data_cam_cx / args.data_cam_fx],
             [0.0, 1 / args.data_cam_fy, -args.data_cam_cy / args.data_cam_fy], [0.0, 0.0, 1.0]],
            dtype=self.dtype_np,
        )
        reprojection_matrix = reprojection_matrix[None,]

        baseline = np.array([args.data_baseline]).astype(self.dtype_np)
        baseline = baseline[None,]

        self.H = args.data_res_height
        self.W = args.data_res_width
        self.data_fix = {'cam_k1': args.data_cam_d0,
                        'cam_k2': args.data_cam_d1,
                        'cam_k3': args.data_cam_d4,
                        'cam_p1': args.data_cam_d2,
                        'cam_p2': args.data_cam_d3,
                        'baseline': baseline,
                        'projection_matrix': projection_matrix,
                        'reprojection_matrix': reprojection_matrix}

        rgbd_enc = self.socket_rep.recv_multipart()
        self.socket_rep.send(rgbd_enc[0])

        if self.args.eval_remote_frame_encode:
            rgb_enc = np.frombuffer(rgbd_enc[0], np.uint8)
            depth_enc = np.frombuffer(rgbd_enc[1], np.uint8)
            depth_dec = cv2.imdecode(depth_enc, flags=-1)
            rgb_dec = cv2.imdecode(rgb_enc, flags=-1)
        else:
            rgb_enc = np.frombuffer(rgbd_enc[0], np.uint8)
            depth_enc = np.frombuffer(rgbd_enc[1], np.uint16)
            rgb_dec = rgb_enc.reshape(args.eval_remote_frame_height, args.eval_remote_frame_width, 3)
            depth_dec = depth_enc.reshape(args.eval_remote_frame_height, args.eval_remote_frame_width)

        rgb_dec = cv2.cvtColor(rgb_dec, cv2.COLOR_BGR2RGB)

        print("INFO :: received data from remote")

        self.rgb_0 = rgb_dec[None, :, :, :3].transpose(0, 3, 1, 2).astype(self.dtype_np) / 255.
        self.depth_0 = depth_dec[None, :, :, None].transpose(0, 3, 1, 2).astype(self.dtype_np) * args.data_cam_depth_scale


    def __iter__(self):
        return self

    def __next__(self):

        #rgbd = m4_io.recv_array(self.socket)
        rgbd_enc = self.socket_rep.recv_multipart()
        #self.socket_sub.send(b'OK')
        if self.args.eval_remote_frame_encode:
            rgb_enc = np.frombuffer(rgbd_enc[0], np.uint8)
            depth_enc = np.frombuffer(rgbd_enc[1], np.uint8)
            depth_dec = cv2.imdecode(depth_enc, flags=-1)
            rgb_dec = cv2.imdecode(rgb_enc, flags=-1)
        else:
            rgb_enc = np.frombuffer(rgbd_enc[0], np.uint8)
            depth_enc = np.frombuffer(rgbd_enc[1], np.uint16)
            rgb_dec = rgb_enc.reshape(self.args.eval_remote_frame_height, self.args.eval_remote_frame_width, 3)
            depth_dec = depth_enc.reshape(self.args.eval_remote_frame_height, self.args.eval_remote_frame_width)

        rgb_dec = cv2.cvtColor(rgb_dec, cv2.COLOR_BGR2RGB)
        self.rgb_1 = rgb_dec[None, :, :, :3].transpose(0, 3, 1, 2).astype(self.dtype_np) / 255.
        self.depth_1 = depth_dec[None, :, :, None].transpose(0, 3, 1, 2).astype(self.dtype_np) * self.args.data_cam_depth_scale


        data_in = {}
        data_in["rgb_0"] = self.rgb_0
        data_in["rgb_1"] = self.rgb_1
        data_in["depth_0"] = self.depth_0
        data_in["depth_1"] = self.depth_1

        data_in["depth_valid_0"] = ~(self.depth_0 < 0.1)
        data_in["depth_valid_1"] = ~(self.depth_1 < 0.1)

        for key, val in self.data_fix.items():
            data_in[key] = val

        for key, val in data_in.items():
            if type(val) == np.ndarray:
                data_in[key] = torch.from_numpy(data_in[key])
                data_in[key] = data_in[key].to(self.device)

        data_in["rgb_l_01"] = torch.cat((data_in["rgb_0"], data_in["rgb_1"]), dim=1)
        data_in["seq_tag"] = "live"
        data_in["seq_el_id"] = np.array([self.el_id])
        data_in["seq_len"] = np.array([1000])

        self.el_id += 1
        self.rgb_0 = self.rgb_1.copy()
        self.depth_0 = self.depth_1.copy()

        data_in["socket"] = self.socket_rep

        return data_in