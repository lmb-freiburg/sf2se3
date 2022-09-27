import collections
from torch.utils.data import Dataset as PytorchDataset
import os
import torch
from torchvision import transforms
import numpy as np
import PIL
import cv2
import m4_io.m4 as io_m4
import tensor_operations.mask as o4mask
import tensor_operations.geometric as o4geo
import tensor_operations.vision.visualization as o4visual


class KittiDataset(PytorchDataset):
    def __init__(
        self,
        raw_dataset=False,
        fp_imgs_filenames=None,
        raw_dir=None,
        imgs_left_dir=None,
        return_left_and_right=False,
        imgs_right_dir=None,
        return_flow_occ=False,
        return_flow_noc=False,
        flows_noc_dir=None,
        flows_occ_dir=None,
        return_disp=False,
        disps0_dir=None,
        disps1_dir=None,
        return_mask_objects=False,
        masks_objects_dir=None,
        return_transf=False,
        fp_transf=None,
        return_projection_matrices=False,
        return_baselines=False,
        calibs_dir=None,
        preload=False,
        dev=None,
        max_num_imgs=None,
        index_shift=0,
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

        self.raw_dataset = raw_dataset
        if self.raw_dataset == True:
            self.raw_dir = raw_dir
            self.imgs_left_dir = self.raw_dir
            self.imgs_right_dir = self.raw_dir

            self.calibs_filenames = []

            self.fp_imgs_filenames = fp_imgs_filenames
            self.imgs_seq_ids = []
            self.imgs_el_ids = []
            self.imgs_filenames = []
            self.num_seq = 0

            last_drive_el_id = -2
            seq_el_id = 0
            with open(fp_imgs_filenames, "r") as f:
                for line in f.readlines():

                    dir, fn = line.replace("\n", "").split(" ", 1)

                    fp = os.path.join(
                        dir.split("_d")[0], dir, "image_02", "data", fn + ".jpg"
                    )
                    self.imgs_filenames.append(fp)

                    if last_drive_el_id + 1 != int(fn):
                        # new sequence:
                        self.calibs_filenames.append(
                            "calib_cam_to_cam_" + dir.split("_d")[0] + ".txt"
                        )
                        self.num_seq += 1
                        seq_el_id = 0

                    self.imgs_seq_ids.append(self.num_seq - 1)
                    self.imgs_el_ids.append(seq_el_id)

                    last_drive_el_id = int(fn)
                    seq_el_id = seq_el_id + 1

            # adding last image of each sequence (which is not written in file from self-mono-sf)
            for i in range(len(self.imgs_seq_ids), 0, -1):
                if (
                    i == len(self.imgs_seq_ids)
                    or self.imgs_seq_ids[i - 1] != self.imgs_seq_ids[i]
                ):
                    self.imgs_seq_ids.insert(i, self.imgs_seq_ids[i - 1])
                    self.imgs_el_ids.insert(i, self.imgs_el_ids[i - 1] + 1)
                    drive_id_i = (
                        self.imgs_filenames[i - 1]
                        .split("\\")[-1]
                        .split("/")[-1]
                        .split(".")[0]
                    )
                    drive_id_ip1 = str(int(drive_id_i) + 1).zfill(10)
                    self.imgs_filenames.insert(
                        i, self.imgs_filenames[i - 1].replace(drive_id_i, drive_id_ip1)
                    )
            self.imgs_filenames = self.imgs_filenames[: self.max_num_imgs]
            self.imgs_seq_ids = self.imgs_seq_ids[: self.max_num_imgs]
            self.imgs_el_ids = self.imgs_el_ids[: self.max_num_imgs]

        else:
            self.imgs_left_dir = imgs_left_dir
            self.imgs_right_dir = imgs_right_dir
            self.imgs_filenames = sorted(os.listdir(self.imgs_left_dir))
            self.imgs_filenames = self.imgs_filenames[: self.max_num_imgs]

            self.imgs_seq_ids = np.array(
                [int(img_fn.split("_")[0]) for img_fn in self.imgs_filenames]
            )
            self.imgs_el_ids = np.array(
                [
                    int(img_fn.split("_")[1].split(".")[0])
                    for img_fn in self.imgs_filenames
                ]
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

        self.return_flow_occ = return_flow_occ
        self.return_flow_noc = return_flow_noc
        if self.return_flow_noc:
            if flows_noc_dir is None:
                print("error: missing noc-flow directory")
                self.return_flow_noc = False
            else:
                self.max_num_flows = self.num_imgpairs
                self.flows_noc_dir = flows_noc_dir
                self.flows_noc_filenames = sorted(os.listdir(self.flows_noc_dir))
                self.flows_noc_filenames = self.flows_noc_filenames[
                    : self.max_num_flows
                ]

        if self.return_flow_occ:
            if flows_occ_dir is None:
                print("error: missing occ-flow directory")
                self.return_flow_occ = False
            else:
                self.max_num_flows = self.num_imgpairs
                self.flows_occ_dir = flows_occ_dir
                self.flows_occ_filenames = sorted(os.listdir(self.flows_occ_dir))
                self.flows_occ_filenames = self.flows_occ_filenames[
                    : self.max_num_flows
                ]
            # self.flows_seq_ids = np.array([int(flow_fn.split('_')[0]) for flow_fn in self.flows_noc_filenames])
            # self.flows_el_ids = np.array([int(flow_fn.split('_')[1].split('.')[0]) for flow_fn in self.flows_noc_filenames])

        self.return_disp = return_disp
        if self.return_disp:
            if disps0_dir is None or disps1_dir is None:
                print("error: missing disparity directory")
                self.return_disp = False
            else:
                self.disps0_dir = disps0_dir
                self.disps0_filenames = sorted(os.listdir(self.disps0_dir))
                self.disps0_filenames = self.disps0_filenames[: self.num_imgpairs]

                self.disps1_dir = disps1_dir
                self.disps1_filenames = sorted(os.listdir(self.disps1_dir))
                self.disps1_filenames = self.disps1_filenames[: self.num_imgpairs]

        self.return_mask_objects = return_mask_objects
        if self.return_mask_objects:
            if masks_objects_dir is None:
                print("error: missing mask objects directory")
            else:
                self.masks_objects_dir = masks_objects_dir
                self.masks_objects_filenames = sorted(
                    os.listdir(self.masks_objects_dir)
                )
                self.masks_objects_filenames = self.masks_objects_filenames[
                    : self.num_imgpairs
                ]

        self.return_transf = return_transf
        if self.return_transf:
            if fp_transf is None:
                print("error: missing transformations filepath")
                self.return_transf = False
            else:
                self.gt_transfs = (
                    io_m4.read_nptxt_as_torch(fp_transf)
                    .reshape(-1, 4, 4)
                    .to(self.device)
                    .type(self.dtype)
                )
                print("read", self.gt_transfs.shape[0], "transformations")

        self.return_projection_matrices = return_projection_matrices
        self.return_baselines = return_baselines
        if self.return_projection_matrices or self.return_baselines:
            if calibs_dir is None:
                print("error: missing calibration directory")
                self.return_projection_matrices = False
                self.return_baselines = False

            else:
                self.calibs_dir = calibs_dir
                if self.raw_dataset == False:
                    self.calibs_filenames = sorted(os.listdir(self.calibs_dir))

            self.projection_matrices = []
            self.reprojection_matrices = []
            self.baselines = []
            for calibs_fname in self.calibs_filenames:
                projection_matrix, reprojection_matrix, baseline = self.read_calib(
                    os.path.join(self.calibs_dir, calibs_fname), device=self.device
                )
                self.projection_matrices.append(projection_matrix)
                self.reprojection_matrices.append(reprojection_matrix)
                self.baselines.append(baseline)

        self.preload = preload

        self.transform = transforms.Compose([transforms.ToTensor()])

        if self.preload:
            self.imgs_left = []
            self.imgs_right = []
            self.flows_noc = []
            self.flows_occ = []
            self.disps0 = []
            self.disps0_mask = []
            self.disps1 = []
            self.disps1_mask = []
            self.masks_objects = []

            for i, img_fn in enumerate(self.imgs_filenames):
                img_fp = os.path.join(self.imgs_left_dir, img_fn)
                self.imgs_left.append(self.read_rgb(img_fp, device="cpu"))

                if self.return_left_and_right:
                    img_fp = os.path.join(
                        self.imgs_right_dir, img_fn.replace("image_02", "image_03")
                    )
                    self.imgs_right.append(self.read_rgb(img_fp, device="cpu"))

            if self.return_flow_noc:
                for i, flow_noc_fn in enumerate(self.flows_noc_filenames):
                    flow_noc_fp = os.path.join(self.flows_noc_dir, flow_noc_fn)
                    self.flows_noc.append(self.read_flow(flow_noc_fp, device="cpu"))

            if self.return_flow_occ:
                for i, flow_occ_fn in enumerate(self.flows_occ_filenames):
                    flow_occ_fp = os.path.join(self.flows_occ_dir, flow_occ_fn)
                    self.flows_occ.append(self.read_flow(flow_occ_fp, device="cpu"))

            if self.return_disp:
                for i, disp_fn in enumerate(self.disps0_filenames):
                    disp_fp = os.path.join(self.disps0_dir, disp_fn)
                    disp, disp_mask = self.read_disp(disp_fp, device="cpu")
                    self.disps0.append(disp)
                    self.disps0_mask.append(disp_mask)

                for i, disp_fn in enumerate(self.disps1_filenames):
                    disp_fp = os.path.join(self.disps1_dir, disp_fn)
                    disp, disp_mask = self.read_disp(disp_fp, device="cpu")
                    self.disps1.append(disp)
                    self.disps1_mask.append(disp_mask)

            if self.return_mask_objects:
                for i, mask_objects_fn in enumerate(self.masks_objects_filenames):
                    mask_objects_fp = os.path.join(
                        self.masks_objects_dir, mask_objects_fn
                    )
                    mask_objects = self.read_mask_objects(mask_objects_fp, device="cpu", valid=self.disps0_mask[-1])
                    self.masks_objects.append(mask_objects)

        print("self.num_seq", self.num_seq)
        print("self.num_imgpairs", self.num_imgpairs)

    def __len__(self):

        return self.num_imgpairs

    def __getitem__(self, imgpair_id):
        imgpair_id += self.index_shift
        imgpair_id %= self.__len__()
        return self.get_item_without_shift(imgpair_id)

    def get_item_without_shift(self, imgpair_id):
        imgpair_seq_id = self.imgpairs_seq_ids[imgpair_id]
        imgpair_el_id = self.imgpairs_el_ids[imgpair_id]

        # if imgpair_el_id > 10:
        #    return self.__getitem__(torch.randint(size=(1,), low=0, high=self.__len__()))

        img1_id = self.imgs_seq_lengths_acc[imgpair_seq_id] + imgpair_el_id
        img2_id = self.imgs_seq_lengths_acc[imgpair_seq_id] + imgpair_el_id + 1

        #return_list = []
        return_list = {}

        return_list['seq_el_id'] = imgpair_el_id
        return_list['seq_len'] = self.imgspairs_seq_lengths[imgpair_seq_id]
        return_list['seq_tag'] = f'{imgpair_seq_id:06d}'

        if self.preload:
            img_left1 = self.imgs_left[img1_id].to(self.device)
            img_left2 = self.imgs_left[img2_id].to(self.device)

        else:

            img1_fn = os.path.join(self.imgs_left_dir, self.imgs_filenames[img1_id])
            img2_fn = os.path.join(self.imgs_left_dir, self.imgs_filenames[img2_id])
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
                img1_fn = os.path.join(
                    self.imgs_right_dir,
                    self.imgs_filenames[img1_id].replace("image_02", "image_03"),
                )
                img2_fn = os.path.join(
                    self.imgs_right_dir,
                    self.imgs_filenames[img2_id].replace("image_02", "image_03"),
                )
                img_right1 = self.read_rgb(img1_fn, device=self.device)
                img_right2 = self.read_rgb(img2_fn, device=self.device)

            #imgpair_right = torch.cat((img_right1, img_right2))
            #return_list += [imgpair_right]
            return_list['rgb_r_0'] = img_right1
            return_list['rgb_r_1'] = img_right2

        if self.return_flow_noc:
            flow_id = imgpair_id
            if self.preload:
                flow_noc_uv, flow_noc_valid = self.flows_noc[flow_id]
                flow_noc_uv = flow_noc_uv.to(self.device)
                flow_noc_valid = flow_noc_valid.to(self.device)
            else:
                flow_noc_fp = os.path.join(
                    self.flows_noc_dir, self.flows_noc_filenames[flow_id]
                )
                flow_noc_uv, flow_noc_valid = self.read_flow(
                    flow_noc_fp, device=self.device
                )
            #return_list += [flow_noc_uv, flow_noc_valid]
            return_list['oflow_noc_uv'] = flow_noc_uv
            return_list['oflow_noc_valid'] = flow_noc_valid

        if self.return_flow_occ:
            flow_id = imgpair_id
            if self.preload:
                flow_occ_uv, flow_occ_valid = self.flows_occ[flow_id]
                flow_occ_uv = flow_occ_uv.to(self.device)
                flow_occ_valid = flow_occ_valid.to(self.device)
            else:
                flow_occ_fp = os.path.join(
                    self.flows_occ_dir, self.flows_occ_filenames[flow_id]
                )
                flow_occ_uv, flow_occ_valid = self.read_flow(
                    flow_occ_fp, device=self.device
                )
            #return_list += [flow_occ_uv, flow_occ_valid]
            return_list['oflow'] = flow_occ_uv
            return_list['oflow_valid'] = flow_occ_valid

        if self.return_disp:
            disp_id = imgpair_id

            if self.preload:
                disp0 = self.disps0[disp_id]
                disp0 = disp0.to(self.device)
                disp0_mask = self.disps0_mask[disp_id]
                disp0_mask = disp0_mask.to(self.device)

                disp1 = self.disps1[disp_id]
                disp1 = disp1.to(self.device)
                disp1_mask = self.disps1_mask[disp_id]
                disp1_mask = disp1_mask.to(self.device)
            else:
                disp_fp = os.path.join(self.disps0_dir, self.disps0_filenames[disp_id])
                disp0, disp0_mask = self.read_disp(disp_fp, device=self.device)

                disp_fp = os.path.join(self.disps1_dir, self.disps1_filenames[disp_id])
                disp1, disp1_mask = self.read_disp(disp_fp, device=self.device)

            #return_list += [disp0, disp0_mask, disp1, disp1_mask]
            return_list['disp_0'] = disp0
            return_list['disp_valid_0'] = disp0_mask
            return_list['disp_f0_1'] = disp1
            return_list['disp_valid_f0_1'] = disp1_mask

        if self.return_mask_objects:
            mask_objects_id = imgpair_id

            if self.preload:
                mask_objects = self.masks_objects[mask_objects_id]
                mask_objects = mask_objects.to(self.device)
            else:
                mask_objects_fp = os.path.join(
                    self.masks_objects_dir,
                    self.masks_objects_filenames[mask_objects_id],
                )
                mask_objects = self.read_mask_objects(
                    mask_objects_fp, device=self.device, valid=disp0_mask
                )
            #return_list += [mask_objects]

            return_list['objs_labels'] = mask_objects

        if self.return_transf:
            transf_id = imgpair_seq_id
            transf = self.gt_transfs[transf_id]

            #return_list += [transf]
            return_list['ego_se3'] = torch.linalg.inv(transf)

        if self.return_projection_matrices:

            calib_id = imgpair_seq_id

            projection_matrix = self.projection_matrices[calib_id]
            reprojection_matrix = self.reprojection_matrices[calib_id]

            #return_list += [projection_matrix, reprojection_matrix]
            return_list['projection_matrix'] = projection_matrix
            return_list['reprojection_matrix'] = reprojection_matrix

        if self.return_baselines:
            calib_id = imgpair_seq_id

            baseline = self.baselines[calib_id]

            #return_list += [baseline]
            return_list['baseline'] = baseline
        return return_list

    def read_flow(self, flow_fn, device):

        flow = cv2.imread(flow_fn, cv2.IMREAD_UNCHANGED)
        # H x W x 3 : dtype=np.uint16
        flow = flow[:, :, ::-1]
        # numpy.ndarray: HxWx3

        flow_valid = torch.from_numpy(flow[:, :, 2].astype(bool)).to(device)[
            None,
        ]
        # torch.bool: HxW

        flow_uv = (
            torch.from_numpy(flow[:, :, :2].astype(np.int32))
            .to(device)
            .permute(2, 0, 1)
            - 2 ** 15
        ) / 64.0
        # torch.float32: 2xHxW

        # flow_inside = o4mask.oflow_2_mask_inside(flow_uv[None,])[0]
        # o4mask.pxl2d_2_mask_inside()

        # flow_uv = o4visual.resize(flow_uv, H_out=self.height, W_out=self.width, mode='nearest', vals_rescale=True)
        # flow_valid = o4visual.resize(flow_valid, H_out=self.height, W_out=self.width, mode='nearest')

        return flow_uv, flow_valid

    def read_rgb(self, img_fn, device):
        print(img_fn)
        img = PIL.Image.open(img_fn)

        img = self.transform(img).to(device)

        #img = torch.nn.functional.interpolate(
        #    img.unsqueeze(0),
        #    size=(self.height, self.width),
        #    mode="bilinear",
        #    align_corners=True,
        #)[0]

        return img

    def read_disp(self, disp_fn, device):
        # TODO: check if cv2.imread(disp_fn, cv2.IMREAD_UNCHANGED) reads as np.uint16
        disp = cv2.imread(disp_fn, cv2.IMREAD_ANYDEPTH)
        # H x W : dtype=np.uint16 note: maximum > 256
        disp = torch.from_numpy(disp.astype(np.float32)).to(device) / 256.0

        disp = disp.unsqueeze(0)

        # disp = o4visual.resize(disp, H_out=self.height, W_out=self.width, mode='nearest', vals_rescale=True)

        disp_mask = disp > 0.0

        return disp, disp_mask

    def read_mask_objects(self, mask_objects_fn, device, valid=None):
        mask_objects = cv2.imread(mask_objects_fn, cv2.IMREAD_UNCHANGED).astype(
            np.uint8
        )

        mask_objects = torch.from_numpy(mask_objects).to(device)
        mask_objects = mask_objects.unsqueeze(0)
        # shape: 1 x H x W range: [0, num_objects_max], type: torch.uint8

        # mask_objects = o4visual.resize(mask_objects, H_out=self.height, W_out=self.width, mode='nearest')
        # (self.height, self.width),

        # o4visual.visualize_img(mask_objects == 1)
        # o4visual.visualize_img(flow_occ_valid)
        # o4visual.visualize_img((mask_objects == 0) * valid)
        mask_objects = mask_objects.int()
        if valid is not None:
            mask_objects[~valid] = -1

        return mask_objects

    def read_calib(self, calib_fp, device):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(calib_fp, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # indices 0, 1, 2, 3  = left-gray, right-gray, left-rgb, right-rgb
        # note: left-rgb, right-rgb have same width, height, fx, fy, cx, cy

        # width = data['S_rect_02'][0]
        # height = data['S_rect_02'][1]
        #sx = self.width / data["S_rect_02"][0]
        #sy = self.height / data["S_rect_02"][1]
        sx = 1.0
        sy = 1.0

        fx = data["P_rect_02"][0] * sx
        fy = data["P_rect_02"][5] * sy

        cx = data["P_rect_02"][2] * sx
        cy = data["P_rect_02"][6] * sy

        baseline = torch.from_numpy(np.array([data["P_rect_02"][3] / data["P_rect_02"][0] -  data["P_rect_03"][3] / data["P_rect_03"][0]],
                                    dtype=np.float32)).to(device)
        #baseline = torch.from_numpy(
        #    np.array([data["T_02"][0] - data["T_03"][0]], dtype=np.float32)
        #).to(device)

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

        return projection_matrix, reprojection_matrix, baseline


class DataloaderIterator(collections.abc.Iterator):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __next__(self):
        try:
            batch = self.iterator.next()
        except:
            print("end of iter reached - restat")
            self.iterator = iter(self.dataloader)
            batch = next(self)
            # print('batch', batch)
        return batch

    def __len__(self):
        return len(self.iterator)


def dataset_val_from_args(args):

    args.setup_dataset_dir = os.path.join(
        args.setup_datasets_dir, args.data_dataset_subdir
    )
    if "train" in args.data_dataset_tags:
        val_dataset = KittiDataset(
            imgs_left_dir=os.path.join(args.setup_dataset_dir, "image_2"),
            imgs_right_dir=os.path.join(args.setup_dataset_dir, "image_3"),
            return_left_and_right=True,
            flows_noc_dir=os.path.join(args.setup_dataset_dir, "flow_noc"),
            flows_occ_dir=os.path.join(args.setup_dataset_dir, "flow_occ"),
            return_flow_occ=True,
            return_flow_noc=False,
            fp_transf=os.path.join(
                ".", "datasets", "kitti_raw_meta/gt_transf/gt_transfs.txt"
            ),
            return_transf=True,
            calibs_dir=os.path.join(
                args.setup_dataset_dir,
                "data_scene_flow_calib/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            return_baselines=True,
            disps0_dir=os.path.join(args.setup_dataset_dir, "disp_occ_0"),
            disps1_dir=os.path.join(args.setup_dataset_dir, "disp_occ_1"),
            return_disp=True,
            masks_objects_dir=os.path.join(args.setup_dataset_dir, "obj_map"),
            return_mask_objects=True,
            preload=False,
            max_num_imgs=args.data_dataset_max_num_imgs,
            width=args.data_res_width,
            height=args.data_res_height,
            dev=args.setup_dataloader_device,
            index_shift=args.data_dataset_index_shift,
        )
    else:
        val_dataset = KittiDataset(
            imgs_left_dir=os.path.join(args.setup_dataset_dir, "image_2"),
            imgs_right_dir=os.path.join(args.setup_dataset_dir, "image_3"),
            return_left_and_right=True,
            #flows_noc_dir=os.path.join(args.setup_dataset_dir, "flow_noc"),
            #flows_occ_dir=os.path.join(args.setup_dataset_dir, "flow_occ"),
            #return_flow_occ=True,
            #return_flow_noc=False,
            #fp_transf=os.path.join(
            #    ".", "datasets", "kitti_raw_meta/gt_transf/gt_transfs.txt"
            #),
            #return_transf=True,
            calibs_dir=os.path.join(
                args.setup_dataset_dir,
                "data_scene_flow_calib/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            return_baselines=True,
            #disps0_dir=os.path.join(args.setup_dataset_dir, "disp_occ_0"),
            #disps1_dir=os.path.join(args.setup_dataset_dir, "disp_occ_1"),
            #return_disp=True,
            #masks_objects_dir=os.path.join(args.setup_dataset_dir, "obj_map"),
            #return_mask_objects=True,
            preload=False,
            max_num_imgs=args.data_dataset_max_num_imgs,
            width=args.data_res_width,
            height=args.data_res_height,
            dev=args.setup_dataloader_device,
            index_shift=args.data_dataset_index_shift,
        )

    return val_dataset
