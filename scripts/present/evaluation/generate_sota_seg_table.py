from scripts.present.tools.generate import generate

def main():

    generate_type = "table"  # table plot_bars

    results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    # results_dir = "/media/driveD/sflow2rigid3d/results"
    metrics_fname = "metrics_sync.csv"

    #metrics_selected = ["disp1 [%]", "disp2 [%]", "oflow [%]", "sflow [%]"]
    #metrics_selected = ["disp1 epe [pxl]", "disp2 epe [pxl]", "oflow epe [pxl]", "sflow epe [m]"]
    metrics_selected = ["seg acc [%]", "objects [#]"]
    # metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]"]

    # params_selected = ["sflow2se3-downscale-factor"]
    params_selected = []

    runs_selected = [
        "flyingthings3d_dispnet_test_classic__30_01_2022__19_54_02",
        "flyingthings3d_dispnet_test_rigidmask__27_01_2022__14_36_05",
        "flyingthings3d_dispnet_test_raft3d__27_01_2022__14_40_16",
        #"flyingthings3d_dispnet_test_classic__30_01_2022__19_54_02",
        #"flyingthings3d_dispnet_test_classic__31_01_2022__09_19_02",
        "flyingthings3d_dispnet_test_classic__31_01_2022__17_56_14",
        "kitti_train_classic__29_01_2022__23_19_39",
        "kitti_train_rigidmask__27_01_2022__14_12_52",
        "kitti_train_raft3d__27_01_2022__14_26_36",
        #"kitti_train_classic__29_01_2022__23_19_39",
        #"kitti_train_classic__31_01_2022__09_04_55",
        "kitti_train_classic__31_01_2022__17_41_56",
        "sintel_train_classic__30_01_2022__01_20_08",
        "sintel_train_rigidmask__28_01_2022__04_45_05",
        "sintel_train_raft3d__27_01_2022__20_48_00",
        #"sintel_train_classic__30_01_2022__01_20_08",
        #"sintel_train_classic__31_01_2022__17_19_14",
        "sintel_train_classic__01_02_2022__01_53_49",
        "tum_rgbd_fr2_classic__30_01_2022__00_21_23",
        "tum_rgbd_fr2_rigidmask__28_01_2022__01_59_24",
        "tum_rgbd_fr2_raft3d__27_01_2022__19_26_42",
        #"tum_rgbd_fr2_classic__30_01_2022__00_21_23",
        "tum_rgbd_fr2_classic__31_01_2022__23_40_18",
        "tum_rgbd_fr3_classic__30_01_2022__19_54_44",
        "tum_rgbd_fr3_rigidmask__31_01_2022__08_58_10",
        "tum_rgbd_fr3_raft3d__27_01_2022__19_53_38",
        #"tum_rgbd_fr3_classic__30_01_2022__19_54_44",
        "tum_rgbd_fr3_classic__01_02_2022__00_20_10",
        "bonn_rgbd_classic__30_01_2022__00_02_45",
        "bonn_rgbd_rigidmask__28_01_2022__01_31_45",
        "bonn_rgbd_raft3d__27_01_2022__19_15_00",
        #"bonn_rgbd_classic__30_01_2022__00_02_45",
        #"bonn_rgbd_classic__31_01_2022__14_54_36",
        "bonn_rgbd_classic__31_01_2022__23_20_47",
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
    for i in [0, 1, 2, 3]:
        runs_selected_reorder.append(runs_selected[i*4+1])
        runs_selected_reorder.append(runs_selected[i*4+3])
    runs_selected = runs_selected_reorder
    #runs_selected = [runs_selected[i] for i in [1, 3, 5, 7, 9, 11, ]]

    runs_selected_approach = ["rigidmask", "classic"] * (len(runs_selected) //2)


    generate(generate_type, results_dir, metrics_selected, runs_selected, runs_selected_approach, params_selected, csv_fname=metrics_fname)

if __name__ == "__main__":
    main()
