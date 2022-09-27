import os

print("cwd", os.getcwd())
from datasets.brox import BroxDataset
import datasets.brox
import torch
import tensor_operations.vision.vision as o4vis
import tensor_operations.geometric as o4geo
import options.parser
import sys
import cv2
import numpy as np

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def test_object_index():
    fpath = "/media/driveD/test/TEST/A/0000/left/0006.pfm"
    objs_labels = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(
        np.uint8
    )
    print(np.unique(objs_labels))
    objs_labels[objs_labels <= 201] = 0

    import torchvision
    transform_to_tensor = torchvision.transforms.ToTensor()
    objs_labels = transform_to_tensor(objs_labels)
    from tensor_operations.mask import rearrange as o4mask_rearr
    objs_labels = o4mask_rearr.label2unique(objs_labels)
    from tensor_operations.clustering import elemental as o4cluster
    objs_masks = o4cluster.label_2_onehot(objs_labels[None,])[0]
    from tensor_operations.visual import _2d as o4visual2d
    o4visual2d.visualize_img(o4visual2d.mask2rgb(objs_masks))
    # shape: 1 x H x W range: [0, num_objects_max], type: torch.uint8
    #data[dkey] = objs_labels
    #data[dkey] = self.transform_to_tensor(data[dkey])
    #data[dkey] = o4mask_rearr.label2unique(data[dkey])

def test_flyingthings3d():
    # sys.argv['s'] = 'configs/setup/tower_2080ti.yaml'

    sys.argv.append("-s")
    # sys.argv.append('configs/setup/tower_2080ti.yaml')
    sys.argv.append("configs/setup/cluster_cs.yaml")
    sys.argv.append("-d")
    sys.argv.append("configs/data/flyingthings3d.yaml")

    print("argv", sys.argv)
    # sys.argv = sys.argv[1:]
    args = options.parser.get_args()
    # -s configs/setup/tower_2080ti.yaml -d configs/data/flyingthings3d.yaml
    # -s configs/setup/cluster_cs.yaml -d configs/data/flyingthings3d.yaml
    # args["val_dataset_max_num_imgs"] = None
    # args["arch_res_width"] = 960
    # args["arch_res_height"] = 540
    # args["dataloader_device"] = "cpu"
    # args = Bunch(args)

    dataloader = datasets.brox.dataloader_from_args(args)

    for (i, batch) in enumerate(dataloader):
        (
            imgpairs_left_fwd,
            imgpairs_right_fwd,
            gt_flows_fwd,
            gt_masks_flows_fwd_valid,
            gt_flows_bwd,
            gt_masks_flows_bwd_valid,
            gt_disps1,
            gt_masks_disps1_valid,
            gt_disps2,
            gt_masks_disps2_valid,
            gt_labels_objs,
            # gt_se3s_l1,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) = batch

        # MASK OBJECTS
        # o4vis.visualize_img(
        #    o4vis.mask2rgb(o4vis.label2unique2onehot(gt_masks_objects)[0])
        # )

        ## IMAGES
        img1s_left_right = torch.cat(
            (imgpairs_left_fwd[0, :3], imgpairs_right_fwd[0, :3]), dim=2
        )
        img2s_left_right = torch.cat(
            (imgpairs_left_fwd[0, 3:], imgpairs_right_fwd[0, 3:]), dim=2
        )
        imgs_left_right = torch.cat((img1s_left_right, img2s_left_right), dim=1)
        o4vis.visualize_img(imgs_left_right, height=720)

        ## FLOW
        # o4vis.visualize_img(o4vis.flow2rgb(gt_flow_uv_fwd[0], draw_arrows=False))

        ## DISP
        # o4vis.visualize_img(o4vis.disp2rgb(gt_disps_left_fwd[0], vmax=None))

        ## POINTS 3D
        # pts3d = o4geo.disp_2_pt3d(
        #    gt_disps_left_fwd,
        #    proj_mats=proj_mats_left_fwd,
        #    reproj_mats=reproj_mats_left_fwd,
        #    baseline=1.0,
        # )
        # o4vis.visualize_pts3d(pts3d[0], imgpairs_left_fwd[0, :3])


if __name__ == "__main__":
    test_flyingthings3d()
