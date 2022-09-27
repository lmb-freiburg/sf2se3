import torch
import numpy as np
import os
from tensor_operations import transforms3d
from usflow import options
from datasets.datasets import KittiDataset
from util import my_io, helpers

parser = options.setup_comon_options()

preliminary_args = [
    "-s",
    "config/config_setup_0.yaml",
    "-c",
    "config/config_coach_def_usceneflow.yaml",
]
args = parser.parse_args(preliminary_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader_device = device
dataloader_pin_memory = False

if args.dataloader_num_workers > 0:
    dataloader_device = "cpu"
    dataloader_pin_memory = True

val_dataset = KittiDataset(
    imgs_left_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/image_2"),
    flows_noc_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/flow_noc"),
    flows_occ_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/flow_occ"),
    return_flow=True,
    calibs_dir=os.path.join(
        args.datasets_dir,
        "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
    ),
    return_projection_matrices=True,
    disps0_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/disp_noc_0"),
    disps1_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/disp_noc_1"),
    return_disp=True,
    masks_objects_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/obj_map"),
    return_mask_objects=True,
    fp_transf="kitti_raw_meta/gt_transf/gt_transfs.txt",
    return_transf=True,
    preload=False,
    max_num_imgs=args.val_dataset_max_num_imgs,
    dev=dataloader_device,
    width=args.arch_res_width,
    height=args.arch_res_height,
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.dataloader_num_workers,
    pin_memory=dataloader_pin_memory,
)

val_epoch_length = len(val_dataloader)

gt_transfs_calc = torch.zeros(size=(val_epoch_length, 4, 4))
for batch_id, (
    imgpairs_left_fwd,
    gt_flow_noc_uv_fwd,
    gt_flow_noc_valid_fwd,
    gt_flow_occ_uv_fwd,
    gt_flow_occ_valid_fwd,
    gt_disps_left_fwd,
    gt_disps_masks_left_fwd,
    gt_disps_left_bwd,
    gt_disps_masks_left_bwd,
    gt_masks_objects_left_fwd,
    gt_transfs_fwd,
    proj_mats_left_fwd,
    reproj_mats_left_fwd,
) in enumerate(val_dataloader):
    imgpairs_left_fwd = imgpairs_left_fwd.to(device)
    gt_flow_noc_uv_fwd = gt_flow_noc_uv_fwd.to(device)
    gt_flow_noc_valid_fwd = gt_flow_noc_valid_fwd.to(device)
    gt_flow_occ_uv_fwd = gt_flow_occ_uv_fwd.to(device)
    gt_flow_occ_valid_fwd = gt_flow_occ_valid_fwd.to(device)
    gt_disps_left_fwd = gt_disps_left_fwd.to(device)
    gt_disps_masks_left_fwd = gt_disps_masks_left_fwd.to(device)
    gt_disps_left_bwd = gt_disps_left_bwd.to(device)
    gt_disps_masks_left_bwd = gt_disps_masks_left_bwd.to(device)
    gt_masks_objects_left_fwd = gt_masks_objects_left_fwd.to(device)
    gt_transfs_fwd = gt_transfs_fwd.to(device)
    proj_mats_left_fwd = proj_mats_left_fwd.to(device)
    reproj_mats_left_fwd = reproj_mats_left_fwd.to(device)

    _, _, gt_H, gt_W = gt_flow_occ_uv_fwd.shape
    _, _, H, W = imgpairs_left_fwd.shape
    sx = gt_W / W
    sy = gt_H / H
    # print('sx, sy', sx, sy)
    proj_mats_left_fwd, reproj_mats_left_fwd = helpers.rescale_intrinsics(
        proj_mats_left_fwd, reproj_mats_left_fwd, sx, sy
    )

    gt_pts3d_left1 = helpers.disp2xyz(
        gt_disps_left_fwd, proj_mats_left_fwd, reproj_mats_left_fwd
    )
    gt_pts3d_left2 = helpers.disp2xyz(
        gt_disps_left_bwd, proj_mats_left_fwd, reproj_mats_left_fwd, gt_flow_occ_uv_fwd
    )

    own_mask = torch.zeros_like(gt_disps_masks_left_fwd)
    _, _, H, W = own_mask.shape
    gt_mask_pts3d_valid = (
        gt_disps_masks_left_fwd
        * gt_disps_masks_left_bwd
        * (gt_masks_objects_left_fwd == 0)
        * gt_flow_noc_valid_fwd
    )

    transf_calc = transforms3d.calc_transform_between_pointclouds(
        gt_pts3d_left1[0, :, gt_mask_pts3d_valid[0, 0]],
        gt_pts3d_left2[0, :, gt_mask_pts3d_valid[0, 0]],
    )

    gt_transfs_calc[batch_id] = transf_calc

    gt_transf_loaded = gt_transfs_fwd[0]
    dist, angle = transforms3d.dist_angle_transfs(
        transf_calc.unsqueeze(0), gt_transf_loaded.unsqueeze(0)
    )
    _, angle1 = transforms3d.dist_angle_transfs(
        transf_calc.unsqueeze(0),
        torch.eye(4, dtype=transf_calc.dtype, device=transf_calc.device).unsqueeze(0),
    )
    print(
        batch_id,
        "angle",
        (angle.item() / np.pi) * 180,
        (angle1.item() / np.pi) * 180,
        "deg",
        "dist",
        dist.item(),
    )

my_io.save_torch_as_nptxt(
    gt_transfs_calc.flatten(1), "kitti_raw_meta/gt_transf/gt_transfs.txt"
)
