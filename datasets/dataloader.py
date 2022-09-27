import torch
import numpy as np
import os

from datasets.kitti import KittiDataset


def create_train_dataloader(args):
    print("args.train_dataset_name", args.train_dataset_name)
    if args.train_dataset_name == "kitti-multiview":
        train_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow_multiview/testing/image_2"
            ),
            imgs_right_dir=os.path.join(
                args.datasets_dir, "KITTI_flow_multiview/testing/image_3"
            ),
            return_left_and_right=True,
            calibs_dir=os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/testing/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            max_num_imgs=args.train_dataset_max_num_imgs,
            width=args.arch_res_width,
            height=args.arch_res_height,
            dev=args.dataloader_device,
            index_shift=args.train_dataset_index_shift,
        )

    elif args.train_dataset_name.startswith("kitti-raw-"):
        train_dataset = KittiDataset(
            raw_dataset=True,
            fp_imgs_filenames=os.path.join(
                args.repro_dir,
                "datasets",
                "kitti_raw_meta",
                "lists_imgpair_filenames",
                args.train_dataset_name[10:].replace("-", "_"),
                "train_files.txt",
            ),
            raw_dir=os.path.join(args.datasets_dir, "KITTI_complete"),
            return_left_and_right=True,
            calibs_dir=os.path.join(
                args.repro_dir, "datasets", "kitti_raw_meta", "cam_intrinsics"
            ),
            return_projection_matrices=True,
            preload=False,
            max_num_imgs=args.train_dataset_max_num_imgs,
            width=args.arch_res_width,
            height=args.arch_res_height,
            dev=args.dataloader_device,
            index_shift=args.train_dataset_index_shift,
        )
    elif args.train_dataset_name == "kitti-val":
        train_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            imgs_right_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_3"
            ),
            return_left_and_right=True,
            calibs_dir=os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            max_num_imgs=args.train_dataset_max_num_imgs,
            width=args.arch_res_width,
            height=args.arch_res_height,
            dev=args.dataloader_device,
            index_shift=args.train_dataset_index_shift,
        )
    else:
        print("error: unknown training dataset name")
        return 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(np.ceil(args.train_batch_size / torch.cuda.device_count())),
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    return train_dataloader


def create_present_dataloader(args):

    if not args.present_dataset_val:
        present_dataset = KittiDataset(
            raw_dataset=True,
            fp_imgs_filenames=os.path.join(
                args.repro_dir,
                "datasets",
                "kitti_raw_meta",
                "lists_imgpair_filenames",
                args.present_dataset_name,
                args.present_fname_imgs_filenames,
            ),
            raw_dir=os.path.join(args.datasets_dir, "KITTI_complete"),
            return_left_and_right=True,
            calibs_dir=os.path.join(
                args.repro_dir, "datasets", "kitti_raw_meta", "cam_intrinsics"
            ),
            return_projection_matrices=True,
            preload=False,
            width=args.arch_res_width,
            height=args.arch_res_height,
            dev=args.dataloader_device,
        )
    else:
        present_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow_occ=True,
            fp_transf=os.path.join(
                args.repro_dir, "datasets/kitti_raw_meta/gt_transf/gt_transfs.txt"
            ),
            return_transf=True,
            calibs_dir=os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            disps0_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/disp_occ_0"
            ),
            disps1_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/disp_occ_1"
            ),
            return_disp=True,
            preload=False,
            max_num_imgs=args.val_dataset_max_num_imgs,
            width=args.arch_res_width,
            height=args.arch_res_height,
            dev=args.dataloader_device,
            index_shift=args.val_dataset_index_shift,
        )

    present_dataloader = torch.utils.data.DataLoader(
        present_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    return present_dataloader


def create_val_dataloader(args):
    # noc: flow only for non-occluded
    # occ: flow for all pixels
    val_dataset = KittiDataset(
        imgs_left_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/image_2"),
        flows_noc_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/flow_noc"),
        flows_occ_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/flow_occ"),
        return_flow_occ=True,
        fp_transf=os.path.join(
            args.repro_dir, "datasets", "kitti_raw_meta/gt_transf/gt_transfs.txt"
        ),
        return_transf=True,
        calibs_dir=os.path.join(
            args.datasets_dir,
            "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
        ),
        return_projection_matrices=True,
        disps0_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/disp_occ_0"),
        disps1_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/disp_occ_1"),
        return_disp=True,
        preload=False,
        max_num_imgs=args.val_dataset_max_num_imgs,
        width=args.arch_res_width,
        height=args.arch_res_height,
        dev=args.dataloader_device,
        index_shift=args.val_dataset_index_shift,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    return val_dataloader
