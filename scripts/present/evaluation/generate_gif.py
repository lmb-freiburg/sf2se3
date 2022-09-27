import os
import cv2
from datetime import datetime
import glob
import imageio

import numpy as np

global roi
# objs_labels_contours_gt_vs_se3 sflow_objs_labels_contours_gt_vs_se3
# ids: FT3D: 0-2, KITTI: 3-5, Sintel: 6-8, FR2: 9-11, FR3: 12-14, Bonn: 15-17
# 'rgbd_dataset_freiburg3_sitting_halfsphere*.png'
# '0014_1.png'
args = {
    "save_fname": "drosf_rigidmask_ft3d_A_0014_4.gif",
    "run_imgs_sub_dir": "imgs/objs_labels_contours_gt_vs_se3/A", # A / B / C,
    "img_glob_pattern": '000*.png',
    "fps": 5,
    "runs_selected_ids": [2], # 3, 1]
    "runs_texts": ["SF2SE3",] #"RigidMask"]
}

"""
args = {
    "save_fname": "drosf_rigidmask_bonn_rgbd_balloon_tracking.gif",
    "run_imgs_sub_dir": "imgs/objs_labels_contours_gt_vs_se3", # A / B / C,
    "img_glob_pattern": 'rgbd_bonn_balloon_tracking*.png',
    "fps": 10,
    "runs_selected_ids": [17, 15, ], # 3, 1]
    "runs_texts": ["SF2SE3", "RigidMask"]
}

args = {
    "save_fname": "drosf_rigidmask_bonn_rgbd_crowd.gif",
    "run_imgs_sub_dir": "imgs/objs_labels_contours_gt_vs_se3", # A / B / C,
    "img_glob_pattern": 'rgbd_bonn_crowd_26*.png',
    "fps": 1,
    "runs_selected_ids": [17, 15, ], # 3, 1]
    "runs_texts": ["SF2SE3", "RigidMask"]
}

"""
#img_glob_pattern = 'rgbd_dataset_freiburg3_sitting_halfsphere*.png'
#img_glob_pattern = '000*.png'

def main():
    # ls -1 /media/driveD/sflow2rigid3d/results_cs | sed -e "s/.*/prefix&suffix/"
    # ls -1 /media/driveD/sflow2rigid3d/results_cs | sed -e 's/.*/"&",/'
    # find /media/driveD/sflow2rigid3d/results_cs -maxdepth 1 -mtime -6 -type d -printf '%P\n' | sort -n | sed -e 's/.*/"&",/'
    results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    target_dir = "/media/driveD/sflow2rigid3d/evaluation"
    save_fpath = os.path.join(target_dir, args["save_fname"])
    run_imgs_sub_dir = args["run_imgs_sub_dir"]
    img_glob_pattern = args["img_glob_pattern"]
    fps = args["fps"]
    runs_selected_ids = args["runs_selected_ids"] # [3, 3, ] # 3, 1]
    runs_texts = args["runs_texts"] #["DROSF", "RigidMask"]
    # Sintel
    # runs_selected_ids = [11,]# runs_selected[11], runs_selected[9]]
    # TUM RGBD FR2
    # runs_selected_ids = [15,]# runs_selected[15], runs_selected[13]]
    # TUM RGBD FR3
    # runs_selected_ids = [19,]# runs_selected[19], runs_selected[17]]
    # Bonn RGBD
    # runs_selected_ids = [23,]#, runs_selected[23], runs_selected[21]]
    # state of the art comparison
    runs_selected = [
        #"flyingthings3d_dispnet_test_classic__30_01_2022__19_54_02",
        "flyingthings3d_dispnet_test_classic__18_02_2022__14_30_22",
        "flyingthings3d_dispnet_test_rigidmask__27_01_2022__14_36_05",
        "flyingthings3d_dispnet_test_raft3d__27_01_2022__14_40_16",
        # "flyingthings3d_dispnet_test_classic__30_01_2022__19_54_02",
        # "flyingthings3d_dispnet_test_classic__31_01_2022__09_19_02",
        #"flyingthings3d_dispnet_test_classic__31_01_2022__17_56_14",
        "flyingthings3d_dispnet_test_classic__18_02_2022__14_30_22",
        "kitti_train_classic__29_01_2022__23_19_39",
        "kitti_train_rigidmask__27_01_2022__14_12_52",
        "kitti_train_raft3d__27_01_2022__14_26_36",
        # "kitti_train_classic__29_01_2022__23_19_39",
        # "kitti_train_classic__31_01_2022__09_04_55",
        "kitti_train_classic__31_01_2022__17_41_56",
        "sintel_train_classic__30_01_2022__01_20_08",
        "sintel_train_rigidmask__28_01_2022__04_45_05",
        "sintel_train_raft3d__27_01_2022__20_48_00",
        # "sintel_train_classic__30_01_2022__01_20_08",
        # "sintel_train_classic__31_01_2022__17_19_14",
        "sintel_train_classic__01_02_2022__01_53_49",
        "tum_rgbd_fr2_classic__30_01_2022__00_21_23",
        "tum_rgbd_fr2_rigidmask__28_01_2022__01_59_24",
        "tum_rgbd_fr2_raft3d__27_01_2022__19_26_42",
        # "tum_rgbd_fr2_classic__30_01_2022__00_21_23",
        "tum_rgbd_fr2_classic__31_01_2022__23_40_18",
        #"tum_rgbd_fr3_classic__30_01_2022__19_54_44",
        "tum_rgbd_fr3_classic__17_02_2022__19_52_30",
        "tum_rgbd_fr3_rigidmask__31_01_2022__08_58_10",
        "tum_rgbd_fr3_raft3d__27_01_2022__19_53_38",
        # "tum_rgbd_fr3_classic__30_01_2022__19_54_44",
        #"tum_rgbd_fr3_classic__01_02_2022__00_20_10",
        "tum_rgbd_fr3_classic__17_02_2022__19_52_30",
        "bonn_rgbd_classic__30_01_2022__00_02_45",
        "bonn_rgbd_rigidmask__28_01_2022__01_31_45",
        "bonn_rgbd_raft3d__27_01_2022__19_15_00",
        # "bonn_rgbd_classic__30_01_2022__00_02_45",
        # "bonn_rgbd_classic__31_01_2022__14_54_36",
        "bonn_rgbd_classic__31_01_2022__23_20_47",
    ]

    runs_selected = [
        "flyingthings3d_dispnet_test_rigidmask__21_02_2022__22_38_01",
        "flyingthings3d_dispnet_test_raft3d__21_02_2022__17_19_06",
        "flyingthings3d_dispnet_test_classic__22_02_2022__09_56_31",
        "kitti_train_rigidmask__21_02_2022__16_41_19",
        "kitti_train_raft3d__21_02_2022__16_25_33",
        "kitti_train_classic__21_02_2022__17_04_19",
        "sintel_train_rigidmask__22_02_2022__01_18_08",
        "sintel_train_raft3d__21_02_2022__23_18_47",
        "sintel_train_classic__22_02_2022__03_54_29",
        "tum_rgbd_fr2_rigidmask__21_02_2022__17_26_52",
        "tum_rgbd_fr2_raft3d__21_02_2022__16_58_11",
        "tum_rgbd_fr2_classic__21_02_2022__18_24_30",
        "tum_rgbd_fr3_rigidmask__21_02_2022__20_01_48",
        "tum_rgbd_fr3_raft3d__21_02_2022__19_03_41",
        "tum_rgbd_fr3_classic__21_02_2022__21_55_10",
        "bonn_rgbd_rigidmask__22_02_2022__09_30_55",
        "bonn_rgbd_raft3d__22_02_2022__09_18_42",
        "bonn_rgbd_classic__22_02_2022__09_59_00",
    ]

    runs_selected = [runs_selected[id] for id in runs_selected_ids]

    first_imgs_fpaths = []
    for i, run_selected in enumerate(runs_selected):
        source_dir = os.path.join(results_dir, run_selected, run_imgs_sub_dir)
        imgs_fnames = os.listdir(source_dir)
        imgs_fnames = glob.glob(os.path.join(source_dir, img_glob_pattern))
        imgs_fnames = [img_fname[len(source_dir)+1:] for img_fname in imgs_fnames]
        imgs_fnames.sort(key=sort_key)
        #imgs_fnames_ids_sorted = [img_fname.split("_")[-1] for img_fname in imgs_fnames]
        #imgs_fnames = imgs_fnames[:10]
        first_imgs_fpaths.append(os.path.join(source_dir, imgs_fnames[0]))

    if "runs_rois_crop" in args.keys():
        runs_rois_crop = args["runs_rois_crop"]
    else:
        # select for crop
        print("select roi for crop...")
        runs_rois_crop = select_roi_for_multiple_imgs(first_imgs_fpaths)
    print("rois crop", runs_rois_crop)

    if "runs_rois_text" in args.keys():
        runs_rois_text = args["runs_rois_text"]
    else:
        # select for text
        print("select roi for text...")
        runs_rois_text = select_roi_for_multiple_imgs(first_imgs_fpaths, runs_rois_crop)
    print("rois text", runs_rois_text)

    if "rois_frames" in args.keys():
        rois_frames = args["rois_frames"]
    else:
        # select for frames
        print("select roi for frames...")
        rois_frames = select_rois(first_imgs_fpaths[0], runs_rois_crop[0])
    print("rois frames", rois_frames)

    imgs = []
    for i, img_fname in enumerate(imgs_fnames):
        print("writing image:", i, ":", len(imgs_fnames))
        for r, run_selected in enumerate(runs_selected):
            source_dir = os.path.join(results_dir, run_selected, run_imgs_sub_dir)
            roi = runs_rois_crop[r]
            img_crop = cv2.imread(os.path.join(source_dir, img_fname))
            img_crop = img_crop[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            font =  cv2.FONT_HERSHEY_SIMPLEX
            fontColor = (255, 255, 255)
            lineType = 2 #10 * int(text_height)
            lwidth = int(runs_rois_text[r][3] / 30)
            img_crop = cv2.putText(img_crop, runs_texts[r],
                                (runs_rois_text[r][0], runs_rois_text[r][1] + int(runs_rois_text[r][3] )),
                                font,
                                runs_rois_text[r][3] / 70.,
                                fontColor,
                                lwidth,
                                lineType)
            for roi_frame in rois_frames:
             img_crop = draw_roi(img_crop, roi_frame, lwidth=lwidth)

            if r == 0:
                img = img_crop
            else:
                img = np.concatenate((img, img_crop), axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    imageio.mimsave(save_fpath, imgs, fps=fps)

def sort_key(fname):
    return "_".join(fname.split("_")[:-1]) + "_" + str(int(fname.split("_")[-1].split(".")[0])).zfill(10)

def select_rois(img_fpath, roi_crop=None):
    img = cv2.imread(img_fpath)
    if roi_crop is not None:
        img = img[
                    int(roi_crop[1]):int(roi_crop[1] + roi_crop[3]),
                    int(roi_crop[0]):int(roi_crop[0] + roi_crop[2])
                    ]
    rois = cv2.selectROIs("image", img)
    rois = [rois[i] for i in range(len(rois))]
    for i in range(len(rois)):
        rois[i] = [rois[i][j] for j in range(len(rois[i]))]

    return rois

def select_roi_for_multiple_imgs(imgs_fpaths, rois_crops=None):

    rois = []
    for i, img_fpath in enumerate(imgs_fpaths):
        first_img = cv2.imread(img_fpath)
        if rois_crops is not None:
            first_img = first_img[
                        int(rois_crops[i][1]):int(rois_crops[i][1] + rois_crops[i][3]),
                        int(rois_crops[i][0]):int(rois_crops[i][0] + rois_crops[i][2])
                        ]
        if i == 0:
            global roi
            roi = cv2.selectROI("image", first_img)
        else:
            print(roi)
            param = {"image" : first_img,}
            cv2.imshow("image", draw_roi(first_img, roi))
            cv2.setMouseCallback("image", click_and_move, param)
            cv2.waitKey(0)
            print(roi)
        rois.append(roi)

    return rois

def click_and_move(event, x, y, flags, param):
    global ref_pt, roi
    image = param["image"]
    # roi: x, y, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dx = x - ref_pt[0]
        dy = y - ref_pt[1]
        roi = (x, y, roi[2], roi[3])
        cv2.imshow("image", draw_roi(image, roi))

def draw_roi(image, roi, lwidth=10):
    image = cv2.rectangle(image.copy(), (roi[0] - lwidth, roi[1] - lwidth),
                          (roi[0] + roi[2] + lwidth, roi[1] + roi[3] + lwidth), color=(0, 0, 0),
                          thickness=lwidth)
    image = cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]),
                          color=(255, 255, 255), thickness=lwidth)

    return image


if __name__ == "__main__":
    main()
