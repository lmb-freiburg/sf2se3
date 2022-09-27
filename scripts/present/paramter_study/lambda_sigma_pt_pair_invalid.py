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
    metrics_selected = ["sflow [%]"]#, "objects [#]"]#, "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]"] #, "gpu-memory [GB]", "duration-total [s]"]

    #metrics_selected = ["seg acc [%]", "objects [#]"]
    #metrics_selected = ["gpu-memory [GB]", "duration-total [s]"]

    # "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]"

    params_selected = ["sflow2se3-model-se3-std-factor-occ"]
    plot_markersize = 8
    plot_tick_fontsize = 12
    plot_label_fontsize = 14
    plot_markertype = "--x"
    plot_figuresize = (8, 4) #(8.5, 4)
    adjust_right = 0.65

    plot_first_value = "first-param"  # "run", "first-param"
    bars_second_values = "add"  # "add", "parallel"
    bars_second_label = "duration [s]"

    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/lambda_sigma_pt_pair_invalid.svg"

    # state of the art comparison
    runs_selected = [
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_11_16",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_17_00",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_22_43",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_28_26",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_34_10",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_39_58",
    ]

    runs_selected_approach = ["classic"] * len(runs_selected)

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath, adjust_right=adjust_right
    )

    """
    image_fpath_rel = "imgs/objs_labels_gt_vs_se3/B/0001_6.png"
    image_text = "ablation"  # approach ablation
    generate_type = "images"  # images table plot_bars
    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/sigma_min_B0001_6.png"


    runs_selected = [
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_11_16",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_17_00",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_22_43",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_28_26",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_34_10",
        "flyingthings3d_dispnet_test_classic__26_01_2022__09_39_58",
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
