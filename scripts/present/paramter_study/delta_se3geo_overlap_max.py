from scripts.present.tools.generate import generate

def main():

    generate_type = "plot_bars"  # images table plot_bars

    #results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    results_dir = "/media/driveD/sflow2rigid3d/results"
    metrics_fname = "metrics_not_sync.csv"
    metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-downsample [s]", "duration-retrieve-objects [s]", "duration-assign-objects [s]", "duration-deduct-sflow [s]"]
    metrics_selected = ["seg acc [%]", "objects [#]", "sflow [%]"]
    #metrics_selected = ["objects [#]"]
    #metrics_selected = ["sflow [%]"]
    #metrics_selected = ["seg acc [%]"]
    metrics_selected = ["sflow [%]", "objects [#]"]#, "seg acc [%]"] #, "ego-se3-dist [m/s]", "ego-se3-angle [deg/s]"]#, "ego-se3-ate [m]"]#, "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]"] #, "gpu-memory [GB]", "duration-total [s]"]

    #metrics_selected = ["seg acc [%]", "objects [#]"]
    #metrics_selected = ["gpu-memory [GB]", "duration-total [s]"]

    # "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]"

    params_selected = ["sflow2se3-se3geofilter-prob-same-mask-max"]
    plot_markersize = 8
    plot_tick_fontsize = 12
    plot_label_fontsize = 14
    plot_markertype = "--x"
    plot_figuresize = (8, 4) #(8.5, 4)
    adjust_right = 0.75

    plot_first_value = "first-param"  # "run", "first-param"
    bars_second_values = "add"  # "add", "parallel"
    bars_second_label = "duration [s]"

    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/delta_se3geo_overlap_max.svg"

    # state of the art comparison
    runs_selected = [
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_19_07",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_25_42",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_32_20",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_38_55",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_45_06",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_51_11",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_57_11",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_03_10",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_09_01",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_14_57",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_43_37",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_49_41",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_55_41",
        "flyingthings3d_dispnet_test_classic__26_01_2022__22_01_49",
        "flyingthings3d_dispnet_test_classic__26_01_2022__22_07_57",

    ]

    runs_selected_approach = ["classic"] * len(runs_selected)

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath, adjust_right=adjust_right
    )


    """
    image_fpath_rel = "imgs/objs_labels_gt_vs_se3/A/0002_3.png"
    image_text = "ablation"  # approach ablation
    generate_type = "images"  # images table plot_bars
    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/delta_se3geo_overlap_max_A0002_3.png"

    runs_selected = [
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_19_07",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_25_42",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_32_20",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_38_55",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_45_06",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_51_11",
        "flyingthings3d_dispnet_test_classic__26_01_2022__20_57_11",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_03_10",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_09_01",
        "flyingthings3d_dispnet_test_classic__26_01_2022__21_14_57",
    ]
    runs_selected_approach = ["classic"] * len(runs_selected)

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath, image_text=image_text, image_fpath_rel=image_fpath_rel
    )
    """


if __name__ == "__main__":
    main()
