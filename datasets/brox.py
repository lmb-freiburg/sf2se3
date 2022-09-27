from torch.utils.data import Dataset as PytorchDataset
import os
import torch
from torchvision import transforms
import numpy as np
import PIL
import cv2
import m4_io.brox as io_brox


class BroxDataset(PytorchDataset):
    def __init__(
        self,
        imgs_dir=None,
        return_left_and_right=False,
        return_flow_fwd=False,
        return_flow_bwd=False,
        flows_dir=None,
        return_flow_occ=False,
        flows_occ_dir=None,
        return_disp1=False,
        return_disp2=False,
        disps_dir=None,
        return_disp_occ=False,
        disps_occ_dir=None,
        return_mask_objects=False,
        masks_objects_dir=None,
        return_transf=False,
        fp_transf=None,
        return_projection_matrices=False,
        return_baselines=False,
        preload=False,
        dev=None,
        max_num_imgs=None,
        index_shift=None,
        width=640,
        height=640,
    ):

        self.dtype = torch.float32

        if dev == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = dev

        self.width = width
        self.height = height

        self.max_num_imgs = max_num_imgs
        self.index_shift = index_shift

        self.imgs_dir = imgs_dir
        self.imgs_seq_dirs = sorted(os.listdir(self.imgs_dir))

        self.imgs_fns = []
        self.imgs_fps = []
        self.imgs_seq_ids = []

        for seq_dir in self.imgs_seq_dirs:
            imgs_seq_fns = sorted(
                os.listdir(os.path.join(self.imgs_dir, seq_dir, "left"))
            )
            self.imgs_seq_ids = self.imgs_seq_ids + [int(seq_dir)] * len(imgs_seq_fns)
            self.imgs_fns = self.imgs_fns + imgs_seq_fns
            imgs_seq_fps = [
                os.path.join(self.imgs_dir, seq_dir, "left", img_fn)
                for img_fn in imgs_seq_fns
            ]
            self.imgs_fps += imgs_seq_fps

        self.imgs_fns = self.imgs_fns[: self.max_num_imgs]

        self.imgs_seq_ids = np.array(self.imgs_seq_ids)
        self.imgs_el_ids = np.array(
            [int(img_fn.split(".")[0]) for img_fn in self.imgs_fns]
        )

        self.seq_ids_unique, self.imgs_seq_ids = np.unique(
            self.imgs_seq_ids, return_inverse=True
        )
        self.seq_ids_unique = np.unique(self.imgs_seq_ids)
        self.el_ids_unique, self.imgs_el_ids = np.unique(
            self.imgs_el_ids, return_inverse=True
        )
        self.el_ids_unique = np.unique(self.imgs_el_ids)
        self.num_seq = len(self.seq_ids_unique)

        self.return_left_and_right = return_left_and_right

        self.imgs_seq_lengths = np.array(
            [
                np.sum(np.array(self.imgs_seq_ids) == seq_id)
                for seq_id in self.seq_ids_unique
            ]
        )
        self.imgs_seq_lengths_acc = np.add.accumulate(
            np.insert(self.imgs_seq_lengths, 0, values=0)
        )

        self.imgspairs_seq_lengths = self.imgs_seq_lengths - 1
        self.imgspairs_seq_lengths_acc = np.add.accumulate(
            np.insert(self.imgspairs_seq_lengths, 0, values=0)
        )

        self.imgpairs_seq_ids = np.delete(
            self.imgs_seq_ids, self.imgs_seq_lengths_acc[:-1]
        )
        self.imgpairs_el_ids = np.delete(
            self.imgs_el_ids, self.imgs_seq_lengths_acc[1:] - 1
        )

        self.num_imgpairs = len(self.imgpairs_seq_ids)

        self.return_flow_fwd = return_flow_fwd
        self.return_flow_bwd = return_flow_bwd
        self.return_flow_occ = return_flow_occ
        if self.return_flow_fwd:
            if flows_dir is None:
                print("error: missing flow directory")
                self.return_flow_fwd = False
                self.return_flow_bwd = False
            else:
                self.max_num_flows = self.num_imgpairs + 1

                self.flows_fwd_fns = []
                self.flows_fwd_fps = []
                self.flows_dir = flows_dir
                for seq_dir in self.imgs_seq_dirs:
                    seq_flows_fns = sorted(
                        os.listdir(
                            os.path.join(self.flows_dir, seq_dir, "left", "into_future")
                        )
                    )
                    self.flows_fwd_fns += seq_flows_fns
                    seq_flows_fps = [
                        os.path.join(
                            self.flows_dir, seq_dir, "left", "into_future", flow_fn
                        )
                        for flow_fn in seq_flows_fns
                    ]
                    self.flows_fwd_fps += seq_flows_fps

                self.flows_fwd_fns = self.flows_fwd_fns[: self.max_num_flows]
                self.flows_fwd_fps = self.flows_fwd_fps[: self.max_num_flows]

            if self.return_flow_occ:
                self.flows_fwd_occ_fps = [
                    fp.replace(flows_dir, flows_occ_dir).replace(".flo", ".png")
                    for fp in self.flows_fwd_fps
                ]

        if self.return_flow_bwd:
            if flows_dir is None:
                print("error: missing flow directory")
                self.return_flow_fwd = False
                self.return_flow_bwd = False
            else:
                self.max_num_flows = self.num_imgpairs + 1

                self.flows_bwd_fns = []
                self.flows_bwd_fps = []
                self.flows_dir = flows_dir
                for seq_dir in self.imgs_seq_dirs:
                    seq_flows_fns = sorted(
                        os.listdir(
                            os.path.join(self.flows_dir, seq_dir, "left", "into_past")
                        )
                    )
                    self.flows_bwd_fns += seq_flows_fns
                    seq_flows_fps = [
                        os.path.join(
                            self.flows_dir, seq_dir, "left", "into_past", flow_fn
                        )
                        for flow_fn in seq_flows_fns
                    ]
                    self.flows_bwd_fps += seq_flows_fps

                self.flows_bwd_fns = self.flows_bwd_fns[: self.max_num_flows]
                self.flows_bwd_fps = self.flows_bwd_fps[: self.max_num_flows]

            if self.return_flow_occ:
                self.flows_bwd_occ_fps = [
                    fp.replace(flows_dir, flows_occ_dir).replace(".flo", ".png")
                    for fp in self.flows_bwd_fps
                ]

        self.return_disp1 = return_disp1
        self.return_disp2 = return_disp2
        if self.return_disp1 or self.return_disp2:
            if disps_dir is None:
                print("error: missing disparity directory")
                self.return_disp1 = False
                self.return_disp2 = False
            else:
                self.disps_dir = disps_dir
                self.disps_fns = []
                self.disps_fps = []
                for seq_dir in self.imgs_seq_dirs:
                    seq_disps_fns = sorted(
                        os.listdir(os.path.join(self.disps_dir, seq_dir, "left"))
                    )
                    self.disps_fns += seq_disps_fns

                    seq_disps_fps = [
                        os.path.join(self.disps_dir, seq_dir, "left", disp_fn)
                        for disp_fn in seq_disps_fns
                    ]
                    self.disps_fps += seq_disps_fps

                self.disps_fns = self.disps_fns[: self.num_imgpairs + 1]
                self.disps_fps = self.disps_fps[: self.num_imgpairs + 1]

        self.return_disp_occ = return_disp_occ
        if self.return_disp_occ:
            self.disps_occ_fps = [
                fp.replace(disps_dir, disps_occ_dir).replace(".pfm", ".png")
                for fp in self.disps_fps
            ]

        self.return_mask_objects = return_mask_objects
        # TODO: implement masks

        if self.return_mask_objects:
            if masks_objects_dir is None:
                print("error: missing mask objects directory")
            else:
                self.masks_objects_dir = masks_objects_dir
                self.masks_objects_fns = []
                self.masks_objects_fps = []

                for seq_dir in self.imgs_seq_dirs:
                    seq_masks_objects_fns = sorted(
                        os.listdir(
                            os.path.join(self.masks_objects_dir, seq_dir, "left")
                        )
                    )
                    self.masks_objects_fns += seq_masks_objects_fns

                    seq_masks_objects_fps = [
                        os.path.join(
                            self.masks_objects_dir, seq_dir, "left", mask_objects_fn
                        )
                        for mask_objects_fn in seq_masks_objects_fns
                    ]
                    self.masks_objects_fps += seq_masks_objects_fps

                self.masks_objects_fns = self.masks_objects_fns[: self.num_imgpairs]
                self.masks_objects_fps = self.masks_objects_fps[: self.num_imgpairs]

        self.return_transf = return_transf
        if self.return_transf:
            if fp_transf is None:
                print("error: missing transformations filepath")
                self.return_transf = False
            else:
                frames_ids, transfs_left, transfs_right = io_brox.readCameraData(
                    fp_transf, self.dtype
                )

                self.gt_transfs = (
                    torch.stack(transfs_left).to(self.device).type(self.dtype)
                )
                print("read", self.gt_transfs.shape[0], "transformations")

        self.return_projection_matrices = return_projection_matrices

        if self.return_projection_matrices:

            fx = 1050.0
            fy = 1050.0
            cx = 479.5
            cy = 269.5

            reprojection_matrix, projection_matrix = self.get_re_and_proj_matrix(
                fx, fy, cx, cy, device=self.device
            )
            self.projection_matrix = projection_matrix
            self.reprojection_matrix = reprojection_matrix

        self.return_baselines = return_baselines
        if self.return_baselines:
            self.baseline = torch.from_numpy(np.array([1.0], dtype=np.float32)).to(self.device)

        self.preload = preload

        self.transform = transforms.Compose([transforms.ToTensor()])

        if self.preload:
            self.imgs_left = []
            self.imgs_right = []
            self.flows_fwd = []
            self.flows_bwd = []
            self.disps = []
            self.disps_occ = []
            self.masks_objects = []

            for i, img_fn in enumerate(self.imgs_fns):
                img_fp = self.imgs_fps[i]
                self.imgs_left.append(self.read_rgb(img_fp, device="cpu"))

                if self.return_left_and_right:
                    img_fp = self.imgs_fps[i].replace("image_02", "image_03")
                    self.imgs_right.append(self.read_rgb(img_fp, device="cpu"))

            if self.return_flow_fwd:
                for i, flow_fn in enumerate(self.flows_fwd_fns):
                    flow_fp = self.flows_fwd_fps[i]
                    self.flows_fwd.append(self.read_flow(flow_fp, device="cpu"))

            if self.return_flow_bwd:
                for i, flow_fn in enumerate(self.flows_bwd_fns):
                    flow_fp = self.flows_bwd_fps[i]
                    self.flows_bwd.append(self.read_flow(flow_fp, device="cpu"))

            if self.return_disp1 or self.return_disp2:
                for i, disp_fn in enumerate(self.disps_fns):
                    disp_fp = self.disps_fps[i]
                    disp = self.read_disp(disp_fp, device="cpu")
                    self.disps.append(disp)

                    if self.return_disp_occ:
                        disp_occ_fp = self.disps_occ_fps[i]
                        disp_occ = self.read_occlusion(disp_occ_fp, device="cpu")
                        self.disps_occ.append(disp_occ)

            if self.return_mask_objects:
                for i, mask_objects_fp in enumerate(self.masks_objects_fps):
                    mask_objects = self.read_mask_objects(mask_objects_fp, device="cpu")
                    self.masks_objects.append(mask_objects)

        print("self.num_seq", self.num_seq)
        print("self.num_imgpairs", self.num_imgpairs)

    def __len__(self):

        return self.num_imgpairs

    def __getitem__(self, imgpair_id):
        if self.index_shift is not None:
            imgpair_id += self.index_shift
            if imgpair_id >= self.__len__():
                imgpair_id = self.__len__() - 1

        imgpair_seq_id = self.imgpairs_seq_ids[imgpair_id]
        imgpair_el_id = self.imgpairs_el_ids[imgpair_id]

        # if imgpair_el_id > 10:
        #    return self.__getitem__(torch.randint(size=(1,), low=0, high=self.__len__()))

        img1_id = self.imgs_seq_lengths_acc[imgpair_seq_id] + imgpair_el_id
        img2_id = self.imgs_seq_lengths_acc[imgpair_seq_id] + imgpair_el_id + 1

        return_list = {}

        if self.preload:
            img_left1 = self.imgs_left[img1_id].to(self.device)
            img_left2 = self.imgs_left[img2_id].to(self.device)

        else:
            img1_fn = self.imgs_fps[img1_id]
            img2_fn = self.imgs_fps[img2_id]
            img_left1 = self.read_rgb(img1_fn, device=self.device)
            img_left2 = self.read_rgb(img2_fn, device=self.device)

        #imgpair_left = torch.cat((img_left1, img_left2), dim=0)
        #return_list += [imgpair_left]

        return_list['rgb_l_0'] = img_left1
        return_list['rgb_l_1'] = img_left2

        if self.return_left_and_right:
            if self.preload:
                img_right1 = self.imgs_right[img1_id].to(self.device)
                img_right2 = self.imgs_right[img2_id].to(self.device)
            else:
                img1_fn = self.imgs_fps[img1_id].replace("left", "right")
                img2_fn = self.imgs_fps[img2_id].replace("left", "right")
                img_right1 = self.read_rgb(img1_fn, device=self.device)
                img_right2 = self.read_rgb(img2_fn, device=self.device)

            #imgpair_right = torch.cat((img_right1, img_right2))
            #return_list += [imgpair_right]
            return_list['rgb_r_0'] = img_right1
            return_list['rgb_r_1'] = img_right2

        if self.return_flow_fwd:
            flow_id = imgpair_id
            if self.preload:

                flow_uv, flow_valid = self.flows_fwd[flow_id]
                flow_uv = flow_uv.to(self.device)
                flow_valid = flow_valid.to(self.device)
            else:

                flow_fp = self.flows_fwd_fps[flow_id]
                flow_uv = self.read_flow(flow_fp, device=self.device)

            #return_list += [flow_uv]  # , flow_valid]
            return_list['oflow_01'] = flow_uv

            if self.return_flow_occ:
                if self.preload:
                    flow_occ = self.flows_fwd_occ[flow_id]
                    flow_occ = flow_occ.to(self.device)
                else:
                    flow_occ_fp = self.flows_fwd_occ_fps[flow_id]
                    flow_occ = self.read_occlusion(flow_occ_fp, device=self.device)

                #return_list += [flow_occ]
                return_list['oflow_occ_01'] = flow_occ

        if self.return_flow_bwd:
            flow_id = imgpair_id  # + 1
            if self.preload:
                flow_uv, flow_valid = self.flows_bwd[flow_id]
                flow_uv = flow_uv.to(self.device)
                flow_valid = flow_valid.to(self.device)
            else:
                flow_fp = self.flows_bwd_fps[
                    flow_id
                ]  # .replace("into_future", "into_past")
                flow_uv = self.read_flow(flow_fp, device=self.device)

            #return_list += [flow_uv]  # , flow_valid]
            return_list['oflow_10'] = flow_uv

            if self.return_flow_occ:
                if self.preload:
                    flow_occ = self.flows_bwd_occ[flow_id]
                    flow_occ = flow_occ.to(self.device)
                else:
                    flow_occ_fp = self.flows_bwd_occ_fps[flow_id]
                    flow_occ = self.read_occlusion(flow_occ_fp, device=self.device)

                #return_list += [flow_occ]
                return_list['oflow_occ_10'] = flow_uv

        if self.return_disp1:
            disp_id = imgpair_id

            if self.preload:
                disp = self.disps[disp_id]
                disp = disp.to(self.device)
            else:
                disp_fp = self.disps_fps[disp_id]
                disp = self.read_disp(disp_fp, device=self.device)

            #return_list += [disp]
            return_list['disp_0'] = disp

            if self.return_disp_occ:
                if self.preload:
                    disp_occ = self.disps_occ[disp_id]
                    disp_occ = disp_occ.to(self.device)
                else:
                    disp_occ_fp = self.disps_occ_fps[disp_id]
                    disp_occ = self.read_occlusion(disp_occ_fp, device=self.device)

                #return_list += [disp_occ]
                return_list['disp_occ_0'] = disp_occ

        if self.return_disp2:
            disp_id = imgpair_id + 1

            if self.preload:
                disp = self.disps[disp_id]
                disp = disp.to(self.device)
            else:
                disp_fp = self.disps_fps[disp_id]
                disp = self.read_disp(disp_fp, device=self.device)

            #return_list += [disp]
            return_list['disp_1'] = disp

            if self.return_disp_occ:
                if self.preload:
                    disp_occ = self.disps_occ[disp_id]
                    disp_occ = disp_occ.to(self.device)
                else:
                    disp_occ_fp = self.disps_occ_fps[disp_id]
                    disp_occ = self.read_occlusion(disp_occ_fp, device=self.device)

                #return_list += [disp_occ]
                return_list['disp_occ_1'] = disp_occ
        if self.return_mask_objects:
            mask_objects_id = imgpair_id

            if self.preload:
                mask_objects = self.masks_objects[mask_objects_id]
                mask_objects = mask_objects.to(self.device)
            else:
                mask_objects_fp = self.masks_objects_fps[mask_objects_id]
                mask_objects = self.read_mask_objects(
                    mask_objects_fp, device=self.device
                )
            #return_list += [mask_objects]
            return_list['objs_labels'] = mask_objects

        if self.return_transf:
            transf_id = imgpair_seq_id
            transf = self.gt_transfs[transf_id]

            #return_list += [transf]
            return_list['ego_se3'] = transf

        if self.return_projection_matrices:
            projection_matrix = self.projection_matrix
            reprojection_matrix = self.reprojection_matrix

            #return_list += [projection_matrix, reprojection_matrix]
            return_list['projection_matrix'] = projection_matrix
            return_list['reprojection_matrix'] = reprojection_matrix

        if self.return_baselines:
            #return_list += [self.baseline]
            return_list['baseline'] = self.baseline
        return return_list

    def read_flow(self, flow_fn, device):

        flow = io_brox.read(flow_fn)

        # H x W x 3 : dtype=np.uint16
        # numpy.ndarray: HxWx3

        # flow_valid = torch.from_numpy(flow[:, :, 2].astype(np.bool)).to(device)
        # torch.bool: HxW

        flow_uv = torch.from_numpy(flow[:, :, :2].copy()).to(device).permute(2, 0, 1)
        # TODO: consider flipping the numpy array to regain right order

        # torch.float32: 2xHxW

        return flow_uv  # , flow_valid

    def read_rgb(self, img_fn, device):
        img = PIL.Image.open(img_fn)
        img = self.transform(img).to(device)

        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )[0]

        return img

    def read_disp(self, disp_fn, device):
        # TODO: check if cv2.imread(disp_fn, cv2.IMREAD_UNCHANGED) reads as np.uint16
        disp = io_brox.read(disp_fn)
        # H x W : dtype: float32
        disp = torch.from_numpy(disp.copy()).to(device)
        # TODO: consider flipping the numpy array to regain right order

        disp = disp.unsqueeze(0)
        disp = torch.abs(disp)

        return disp

    def read_occlusion(self, occ_fp, device):
        occ = PIL.Image.open(occ_fp)
        occ = self.transform(occ).to(device)

        occ = (
            torch.nn.functional.interpolate(
                occ.unsqueeze(0),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )[0]
            == 1.0
        )

        # occ = io_brox.read(occ_fp)
        # H x W : dtype: float32
        # occ = torch.from_numpy(occ.copy()).to(device)[None,]
        return ~occ

    def read_mask_objects(self, mask_objects_fn, device):
        mask_objects = cv2.imread(mask_objects_fn, cv2.IMREAD_UNCHANGED).astype(
            np.uint8
        )

        mask_objects = torch.from_numpy(mask_objects).to(device)
        mask_objects = mask_objects.unsqueeze(0)
        # shape: 1 x H x W range: [0, num_objects_max], type: torch.uint8
        return mask_objects

    def get_re_and_proj_matrix(self, fx, fy, cx, cy, device):
        """Read in a calibration file and parse into a dictionary."""

        # 3D-2D Projection:
        # u = (fx*x + cx * z) / z
        # v = (fy*y + cy * y) / z
        # shift on plane: delta_x = (fx * bx) / z
        #                 delta_y = (fy * by) / z
        # uv = (P * xyz) / z
        # P = [ fx   0  cx]
        #     [ 0   fy  cy]
        projection_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy]], dtype=np.float32)

        # 2D-3D Re-Projection:
        # x = (u/fx - cx/fx) * z
        # y = (v/fy - cy/fy) * z
        # z = z
        # xyz = (RP * uv1) * z
        # RP = [ 1/fx     0  -cx/fx ]
        #      [    0  1/fy  -cy/fy ]
        #      [    0      0      1 ]
        reprojection_matrix = np.array(
            [[1 / fx, 0.0, -cx / fx], [0.0, 1 / fy, -cy / fy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        projection_matrix = torch.from_numpy(projection_matrix).to(device)
        reprojection_matrix = torch.from_numpy(reprojection_matrix).to(device)

        return reprojection_matrix, projection_matrix


def dataset_from_args(args):
    args.setup_dataset_dir = os.path.join(
        args.setup_datasets_dir, args.data_dataset_subdir
    )
    dataset = BroxDataset(
        imgs_dir=os.path.join(args.setup_dataset_dir, args.data_dataset_rgb_subdir),
        return_left_and_right=True,
        flows_dir=os.path.join(args.setup_dataset_dir, args.data_dataset_oflow_subdir),
        flows_occ_dir=os.path.join(
            args.setup_dataset_dir, args.data_dataset_oflow_occ_subdir
        ),
        return_flow_fwd=True,
        return_flow_bwd=False,
        return_flow_occ=True,
        fp_transf=os.path.join(
            args.setup_dataset_dir, args.data_dataset_transfs_fpath_rel
        ),
        return_transf=False,  # True -> False
        return_projection_matrices=True,
        return_baselines=True,
        disps_dir=os.path.join(args.setup_dataset_dir, args.data_dataset_disp_subdir),
        return_disp1=True,
        return_disp2=True,
        return_disp_occ=True,
        disps_occ_dir=os.path.join(
            args.setup_dataset_dir, args.data_dataset_disp_occ_subdir
        ),
        preload=False,
        width=args.data_res_width,
        height=args.data_res_height,
        dev=args.setup_dataloader_device,
        return_mask_objects=True,
        masks_objects_dir=os.path.join(
            args.setup_dataset_dir, args.data_dataset_label_subdir
        ),
    )
    return dataset


def dataloader_from_args(args):

    dataset = dataset_from_args(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    return dataloader
