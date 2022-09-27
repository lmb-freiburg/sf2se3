

from scripts.present.tools.generate import generate

def main():

    generate_type = "table"  # table plot_bars

    #results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    metrics_fname = "metrics_sync.csv"

    #metrics_selected = ["disp1 [%]", "disp2 [%]", "oflow [%]", "sflow [%]"]
    #metrics_selected = ["disp1 epe [pxl]", "disp2 epe [pxl]", "oflow epe [pxl]", "sflow epe [m]"]
    #metrics_selected = ["seg acc [%]", "objects [#]"]
    metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]", "duration-total [s]"]
    #metrics_selected = ["duration-total [s]"]
    # duration-downsample [s]
    # duration-retrieve-objects [s]
    # duration-assign-objects [s]
    # duration-deduct-sflow [s]
    # duration-retrieve-objects-extract [s]
    # duration-retrieve-objects-refine [s]
    # duration-retrieve-objects-extract-se3 [s]
    # duration-retrieve-objects-extract-geo [s]
    # duration-retrieve-objects-extract-se3-cluster [s]
    # duration-retrieve-objects-extract-se3-fit [s]
    # duration-retrieve-objects-extract-se3-select [s]
    # params_selected = ["sflow2se3-downscale-factor"]
    params_selected = []

    # state of the art comparison
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

    runs_selected = [runs_selected[i] for i in range(len(runs_selected)) if i % 4 != 0]

    runs_selected_approach = ["rigidmask",
                              "raft3d",
                              "classic"] * (len(runs_selected) //3 )

    generate(generate_type, results_dir, metrics_selected, runs_selected, runs_selected_approach, params_selected, decimals=3, csv_fname=metrics_fname)

if __name__ == "__main__":
    main()
