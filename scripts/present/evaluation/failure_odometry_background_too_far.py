from scripts.present.tools.generate import generate

def main():

    generate_type = "table"  # table plot_bars

    results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    # results_dir = "/media/driveD/sflow2rigid3d/results"
    metrics_fname = "metrics_sync.csv"


    metrics_selected = ["disp1 [%]", "disp2 [%]", "oflow [%]", "sflow [%]"]
    #metrics_selected = ["disp1 epe [pxl]", "disp2 epe [pxl]", "oflow epe [pxl]", "sflow epe [m]"]
    #metrics_selected = ["seg acc [%]", "objects [#]"]
    # metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-warped-sflow-to-rigid-sflow [s]"]

    # params_selected = ["sflow2se3-downscale-factor"]
    params_selected = []

    # state of the art comparison
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
    # FT3D
    #runs_selected = [runs_selected[3], runs_selected[1]]
    # Sintel
    runs_selected = [runs_selected[11], runs_selected[11], runs_selected[9]]
    # Bonn RGBD
    #runs_selected = [runs_selected[23], runs_selected[23], runs_selected[21]]

    # ls -1 /media/driveD/sflow2rigid3d/results_cs | sed -e "s/.*/prefix&suffix/"
    # ls -1 /media/driveD/sflow2rigid3d/results_cs | sed -e 's/.*/"&",/'
    # find /media/driveD/sflow2rigid3d/results_cs -maxdepth 1 -mtime -6 -type d -printf '%P\n' | sort -n | sed -e 's/.*/"&",/'

    runs_selected_approach = ["classic",
                              "classic",
                              "rigidmask",]

    #image_fpath_rel = "imgs/outlier/A/0147_2.png"
    image_fpath_rel = "imgs/objs_labels_gt_vs_se3/mountain_1_21.png"
    #image_fpath_rel = "imgs/objs_labels_gt_vs_se3/rgbd_bonn_crowd_104.png"
    image_text = ["", "approach", "approach"]  # approach ablation
    generate_type = "images"  # images table plot_bars
    #save_fpath = "/media/driveD/sflow2rigid3d/evaluation/hyp_oflow_sflow_pos_effect_ft3d_A_0147_2.png"
    save_fpath = "/media/driveD/sflow2rigid3d/evaluation/failure_odometry_background_sintel_mountain_1_21.png"
    #save_fpath = "/media/driveD/sflow2rigid3d/evaluation/failure_odometry_foreground_bonn_crowd_104.png"


    plot_markersize = 8
    plot_tick_fontsize = 12
    plot_label_fontsize = 14
    plot_markertype = "--x"
    plot_figuresize = (8, 4) #(8.5, 4)
    adjust_right = 0.75

    plot_first_value = "first-param"  # "run", "first-param"
    bars_second_values = "add"  # "add", "parallel"
    bars_second_label = "duration [s]"

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath, image_text=image_text, image_fpath_rel=image_fpath_rel
    )

if __name__ == "__main__":
    main()
