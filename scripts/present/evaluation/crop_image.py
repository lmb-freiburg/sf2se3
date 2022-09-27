
import cv2
import os


args = {
    "fname_in" : "tum_fr3_walking_halfsphere_3.png",
    "fname_out" : "tum_fr3_walking_halfsphere_3_cropped.png",
    "runs_rois_crop" : (25, 85, 605, 395),
}


args = {
    "fname_in" : "ft3d_A_0000_0.png",
    "fname_out" : "ft3d_A_0000_0_cropped.png",
    "runs_rois_crop" : (10, 11, 942, 523),
}


args = {
    "fname_in" : "kitti_test_43.png",
    "fname_out" : "kitti_test_43_cropped.png",
    "runs_rois_crop": (17, 20, 1194, 341),
}

source_dir = "/media/driveD/sflow2rigid3d/evaluation/orig_imgs"
roi = args["runs_rois_crop"]
img_crop = cv2.imread(os.path.join(source_dir, args["fname_in"]))
img_crop = img_crop[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2]), :]
cv2.imwrite(filename=os.path.join(source_dir, args["fname_out"]), img=img_crop)