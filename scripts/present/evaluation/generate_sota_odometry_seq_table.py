from scripts.present.tools.generate import generate

def main():

    generate_type = "table"  # table plot_bars

    results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    csv_fname = "metrics_seq_not_sync.csv" # "metrics.csv"
    # results_dir = "/media/driveD/sflow2rigid3d/results"

    #metrics_selected = ["disp1 [%]", "disp2 [%]", "oflow [%]", "sflow [%]"]
    metrics_selected = ["ego-se3-dist [m/s]", "ego-se3-angle [deg/s]", "ego-se3-ate [m]"]
    #metrics_selected = ["seg acc [%]", "objects [#]"]
    # metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]"]

    # params_selected = ["sflow2se3-downscale-factor"]
    params_selected = []

    # state of the art comparison
    runs_selected = [
        "tum_rgbd_fr2_raft3d__31_12_2021__00_16_54",
        "tum_rgbd_fr2_rigidmask__31_12_2021__00_40_45",
        "tum_rgbd_fr2_classic__31_12_2021__01_28_38",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "tum_rgbd_fr3_raft3d__31_12_2021__02_00_39",
        "tum_rgbd_fr3_rigidmask__31_12_2021__02_46_43",
        "tum_rgbd_fr3_classic__31_12_2021__04_19_02",
        "bonn_rgbd_raft3d__30_12_2021__23_28_20",
        "bonn_rgbd_rigidmask__30_12_2021__23_39_26",
        "bonn_rgbd_classic__31_12_2021__00_01_51",
        "bonn_rgbd_raft3d__30_12_2021__23_28_20",
        "bonn_rgbd_rigidmask__30_12_2021__23_39_26",
        "bonn_rgbd_classic__31_12_2021__00_01_51"
                     ]

    runs_selected = [
        "flyingthings3d_dispnet_test_classic__22_02_2022__09_56_31",
        "flyingthings3d_dispnet_test_rigidmask__21_02_2022__22_38_01",
        "flyingthings3d_dispnet_test_raft3d__21_02_2022__17_19_06",
        "flyingthings3d_dispnet_test_classic__22_02_2022__09_56_31",
        "kitti_train_classic__21_02_2022__17_04_19",
        "kitti_train_rigidmask__21_02_2022__16_41_19",
        "kitti_train_raft3d__21_02_2022__16_25_33",
        #"kitti_train_classic__21_02_2022__17_04_19",
        "kitti_train_classic__21_02_2022__17_04_19",
        "sintel_train_classic__22_02_2022__03_54_29",
        "sintel_train_rigidmask__22_02_2022__01_18_08",
        "sintel_train_raft3d__21_02_2022__23_18_47",
        "sintel_train_classic__22_02_2022__03_54_29",
        "tum_rgbd_fr2_classic__21_02_2022__18_24_30",
        "tum_rgbd_fr2_rigidmask__21_02_2022__17_26_52",
        "tum_rgbd_fr2_raft3d__21_02_2022__16_58_11",
        "tum_rgbd_fr2_classic__21_02_2022__18_24_30",
        "tum_rgbd_fr3_classic__21_02_2022__21_55_10",
        "tum_rgbd_fr3_rigidmask__21_02_2022__20_01_48",
        "tum_rgbd_fr3_raft3d__21_02_2022__19_03_41",
        "tum_rgbd_fr3_classic__21_02_2022__21_55_10",
        "bonn_rgbd_classic__22_02_2022__09_59_00",
        "bonn_rgbd_rigidmask__22_02_2022__09_30_55",
        "bonn_rgbd_raft3d__22_02_2022__09_18_42",
        "bonn_rgbd_classic__22_02_2022__09_59_00",
        #"kitti_test_classic__22_02_2022__19_49_48",
        #"kitti_test_rigidmask__22_02_2022__19_27_04",
        #"kitti_test_raft3d__22_02_2022__19_13_00",
        #"kitti_test_classic__22_02_2022__19_49_48",
    ]

    runs_selected_reorder = []
    for i in [3, 4, 5]:
        rep_dataset = [0, 0, 0, 1, 8, 2]
        for k in range(rep_dataset[i]):
            runs_selected_reorder.append(runs_selected[i * 4 + 2])
            runs_selected_reorder.append(runs_selected[i * 4 + 1])
            runs_selected_reorder.append(runs_selected[i * 4 + 3])
    runs_selected = runs_selected_reorder

    runs_selected_approach = [
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic",
        "raft3d",
        "rigidmask",
        "classic"
        ]

    runs_selected_seq_tag = ["rgbd_dataset_freiburg2_desk_with_person",
                             "rgbd_dataset_freiburg2_desk_with_person",
                             "rgbd_dataset_freiburg2_desk_with_person",
                             "rgbd_dataset_freiburg3_sitting_static",
                             "rgbd_dataset_freiburg3_sitting_static",
                             "rgbd_dataset_freiburg3_sitting_static",
                             "rgbd_dataset_freiburg3_sitting_xyz",
                             "rgbd_dataset_freiburg3_sitting_xyz",
                             "rgbd_dataset_freiburg3_sitting_xyz",
                             "rgbd_dataset_freiburg3_sitting_rpy",
                             "rgbd_dataset_freiburg3_sitting_rpy",
                             "rgbd_dataset_freiburg3_sitting_rpy",
                             "rgbd_dataset_freiburg3_sitting_halfsphere",
                             "rgbd_dataset_freiburg3_sitting_halfsphere",
                             "rgbd_dataset_freiburg3_sitting_halfsphere",
                             "rgbd_dataset_freiburg3_walking_static",
                             "rgbd_dataset_freiburg3_walking_static",
                             "rgbd_dataset_freiburg3_walking_static",
                             "rgbd_dataset_freiburg3_walking_xyz",
                             "rgbd_dataset_freiburg3_walking_xyz",
                             "rgbd_dataset_freiburg3_walking_xyz",
                             "rgbd_dataset_freiburg3_walking_rpy",
                             "rgbd_dataset_freiburg3_walking_rpy",
                             "rgbd_dataset_freiburg3_walking_rpy",
                             "rgbd_dataset_freiburg3_walking_halfsphere",
                             "rgbd_dataset_freiburg3_walking_halfsphere",
                             "rgbd_dataset_freiburg3_walking_halfsphere",
                             "rgbd_bonn_balloon_tracking",
                             "rgbd_bonn_balloon_tracking",
                             "rgbd_bonn_balloon_tracking",
                             "rgbd_bonn_crowd",
                             "rgbd_bonn_crowd",
                             "rgbd_bonn_crowd"
                              ]

    approaches_map = {
        "raft3d": "Static",
        "rigidmask": "RigidMask",
        "classic": "DROSF (ours.)",
        "wsflow": "Warped Scene Flow"
    }

    generate(generate_type, results_dir, metrics_selected, runs_selected, runs_selected_approach, params_selected,
             approaches_map, csv_fname, runs_selected_seq_tag, decimals=3)

if __name__ == "__main__":
    main()
