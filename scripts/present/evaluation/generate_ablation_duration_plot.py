

from scripts.present.tools.generate import generate

def main():

    generate_type = "plot_bars"  # table plot_bars

    #results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    results_dir = "/media/driveD/sflow2rigid3d/results"

    #metrics_selected = ["disp1 [%]", "disp2 [%]", "oflow [%]", "sflow [%]"]
    #metrics_selected = ["disp1 epe [pxl]", "disp2 epe [pxl]", "oflow epe [pxl]", "sflow epe [m]"]
    #metrics_selected = ["seg acc [%]", "objects [#]"]
    #metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]"]

    metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-downsample [s]", "duration-retrieve-objects [s]", "duration-assign-objects [s]", "duration-deduct-sflow [s]"]
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
    runs_selected = ["flyingthings3d_dispnet_test_classic__30_12_2021__16_05_03",]
    """
    ,
                     "kitti_train_classic__28_12_2021__10_45_06",
                     "kitti_train_rigidmask__28_12_2021__10_22_47",
                     "kitti_train_raft3d__28_12_2021__10_07_39",
                     "kitti_train_classic__28_12_2021__10_45_06",
                     "sintel_train_classic__28_12_2021__17_35_35",
                     "sintel_train_rigidmask__28_12_2021__15_01_04",
                     "sintel_train_raft3d__28_12_2021__13_04_46",
                     "sintel_train_classic__28_12_2021__17_35_35",
                     "tum_rgbd_fr2_classic__28_12_2021__12_10_09",
                     "tum_rgbd_fr2_rigidmask__28_12_2021__11_21_22",
                     "tum_rgbd_fr2_raft3d__28_12_2021__10_56_47",
                     "tum_rgbd_fr2_classic__28_12_2021__12_10_09",
                     "tum_rgbd_fr3_classic__28_12_2021__12_57_49",
                     "tum_rgbd_fr3_rigidmask__28_12_2021__12_47_22",
                     "tum_rgbd_fr3_raft3d__28_12_2021__12_41_47",
                     "tum_rgbd_fr3_classic__28_12_2021__12_57_49",
                     "bonn_rgbd_classic__28_12_2021__10_41_49",
                     "bonn_rgbd_rigidmask__28_12_2021__10_19_32",
                     "bonn_rgbd_raft3d__28_12_2021__10_08_09",
                     "bonn_rgbd_classic__28_12_2021__10_41_49"]
    """

    runs_selected_approach = ["classic",]
    """
    ,
                              "wsflow",
                              "rigidmask",
                              "raft3d",
                              "classic",
                              "wsflow",
                              "rigidmask",
                              "raft3d",
                              "classic",
                              "wsflow",
                              "rigidmask",
                              "raft3d",
                              "classic",
                              "wsflow",
                              "rigidmask",
                              "raft3d",
                              "classic",
                              "wsflow",
                              "rigidmask",
                              "raft3d",
                              "classic",
                              ]
    """
    generate(generate_type, results_dir, metrics_selected, runs_selected, runs_selected_approach, params_selected)

if __name__ == "__main__":
    main()
