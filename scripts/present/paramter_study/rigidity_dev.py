

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
    metrics_selected = ["sflow [%]", "objects [#]"]
    #metrics_selected = ["seg acc [%]", "objects [#]"]
    params_selected = ["sflow2se3-rigid-dist-dev-max"]
    plot_markersize = 8
    plot_tick_fontsize = 12
    plot_label_fontsize = 14
    plot_markertype = "--x"
    plot_figuresize = (7, 4)

    plot_first_value = "first-param"  # "run", "first-param"
    bars_second_values = "add"  # "add", "parallel"
    bars_second_label = "duration [s]"

    save_fpath = "/media/driveD/sflow2rigid3d/parameter_study/rigidity_dev.svg"

    # state of the art comparison
    runs_selected = [
        "flyingthings3d_dispnet_test_classic__25_01_2022__08_46_35",
        "flyingthings3d_dispnet_test_classic__25_01_2022__08_52_38",
        "flyingthings3d_dispnet_test_classic__25_01_2022__08_58_32",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_04_24",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_10_14",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_16_05",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_21_35",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_27_19",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_33_03",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_38_50",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_44_37",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_50_21",
        "flyingthings3d_dispnet_test_classic__25_01_2022__09_56_09",
        "flyingthings3d_dispnet_test_classic__25_01_2022__10_01_59",
    ]

    runs_selected_approach = ["classic"] * len(runs_selected)

    generate(generate_type, results_dir, metrics_selected, runs_selected,
             runs_selected_approach, params_selected, csv_fname=metrics_fname, decimals=4, plot_first_value=plot_first_value,
             bars_second_values=bars_second_values, bars_second_label=bars_second_label,
             plot_figuresize=plot_figuresize, plot_markersize=plot_markersize, plot_tick_fontsize=plot_tick_fontsize, plot_label_fontsize=plot_label_fontsize, plot_markertype=plot_markertype,
             save_fpath=save_fpath
    )



if __name__ == "__main__":
    main()
