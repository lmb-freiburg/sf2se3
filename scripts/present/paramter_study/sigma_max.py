from scripts.present.tools.generate import generate

def main():

    generate_type = "plot_bars"  # table plot_bars

    #results_dir = "/media/driveD/sflow2rigid3d/results_cs"
    results_dir = "/media/driveD/sflow2rigid3d/results"
    metrics_fname = "metrics_not_sync.csv"
    metrics_selected = ["duration-aggregate-depth [s]", "duration-aggregate-oflow [s]", "duration-downsample [s]", "duration-retrieve-objects [s]", "duration-assign-objects [s]", "duration-deduct-sflow [s]"]
    metrics_selected = ["seg acc [%]", "objects [#]", "sflow [%]"]
    #metrics_selected = ["objects [#]"]
    #metrics_selected = ["sflow [%]"]
    #metrics_selected = ["seg acc [%]"]
    metrics_selected = ["sflow [%]", "objects [#]"] #, "gpu-memory [GB]", "duration-total [s]"]
    metrics_selected = ["sflow [%]", "objects [#]", "std disp [pxl]", "std oflow x [pxl]", "std oflow y [pxl]"] #, "gpu-memory [GB]", "duration-total [s]"]

    #metrics_selected = ["seg acc [%]", "objects [#]"]
    #metrics_selected = ["gpu-memory [GB]", "duration-total [s]"]
    params_selected = ["sflow2se3-oflow-disp-std-abs-max"]
    plot_markersize = 8
    plot_tick_fontsize = 12
    plot_label_fontsize = 14
    plot_markertype = "--x"
    plot_figuresize = (10., 4)
    adjust_right = 0.65

    plot_first_value = "first-param"  # "run", "first-param"
    bars_second_values = "add"  # "add", "parallel"
    bars_second_label = "duration [s]"

    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/sigma_max.svg"

    # state of the art comparison
    runs_selected = [
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_30_58",
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_36_40",
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_42_23",
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_48_04",
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_53_42",
        "flyingthings3d_dispnet_test_classic__26_01_2022__07_59_26",
        "flyingthings3d_dispnet_test_classic__26_01_2022__08_05_10",
        "flyingthings3d_dispnet_test_classic__26_01_2022__08_10_54",
        "flyingthings3d_dispnet_test_classic__26_01_2022__08_16_38",
        "flyingthings3d_dispnet_test_classic__26_01_2022__08_22_19",
        "flyingthings3d_dispnet_test_classic__26_01_2022__08_27_57",
    ]

    runs_selected_approach = ["classic"] * len(runs_selected)

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath, adjust_right= adjust_right
    )



if __name__ == "__main__":
    main()
