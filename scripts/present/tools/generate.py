import csv

from tensor_operations import string as o4string
from tabulate import tabulate
import os
import configargparse
import matplotlib.pyplot as plt
import numpy as np

def generate(generate_type, results_dir,  metrics_selected, runs_selected, runs_selected_approach, params_selected,
             approaches_map=None, csv_fname="metrics.csv", runs_selected_seq_tag=None, decimals=2, plot_first_value = "run",
             bars_second_values = "add", bars_second_label = "duration [s]",
             plot_figuresize=(6, 6), plot_markersize=12, plot_tick_fontsize = 12, plot_label_fontsize = 14, plot_markertype = "-",
             save_fpath="test.svg", image_text=None, image_fpath_rel=None, adjust_right=0.75):

    #plot_first_value = "run"  # "run", "first-param"
    #bars_second_values = "add"  # "add", "parallel"
    #image_fpath_rel = "imgs/objs_labels_gt_vs_se3/B/0000_6.png"
    #image_text = "ablation"  # approach ablation
    add_approach = True
    add_dataset = True
    params_map = {
        "sflow2se3-downscale-factor" : r"$\lambda_{downscale}$ $\left[ \frac{\%}{100}  \right]$",
        "sflow2se3-rigid-dist-dev-max" : r"$\delta_{rigid-dev-max}$ [m]",
        "sflow2se3-oflow-disp-std-abs-min" : r"$\sigma_{min}$ [pxl]",
        "sflow2se3-oflow-disp-std-abs-max": r"$\sigma_{max}$ [pxl]",
        "sflow2se3-model-se3-std-factor-occ": r"$\lambda_{sigma-pt-pair-invalid}$ [pxl]",
        "sflow2se3-model-euclidean-nn-uv-dev-rel-to-width-std": r"$\sigma_{geo-2D}$ [pxl]",
        "sflow2se3-model-euclidean-nn-rel-depth-dev-std": r"$\sigma_{geo-depth-rel}$ [pxl]",
        "sflow2se3-model-euclidean-nn-connect-global-dist-range-ratio-max": r"$\delta_{geo-global-connect}$ $\left[ \frac{\%}{100}  \right]$",
        "sflow2se3-se3filter-prob-gain-min": r"$\delta_{contribute-min}$ [pxl]",
        "sflow2se3-se3filter-prob-same-mask-max": r"$\delta_{overlap-max}$ $\left[ \frac{\%}{100}  \right]$",
        "sflow2se3-se3geofilter-prob-same-mask-max": r"$\delta_{overlap-max}$ $\left[ \frac{\%}{100}  \right]$",
        "sflow2se3-rigid-clustering-repetitions": r"$\rho_{extraction-cycles}$ [#]",
        "sflow2se3-refinements-per-extraction": r"$\rho_{refinement-cycles}$ [#]",
        "sflow2se3-extraction-refinement-cycles": r"$\rho_{extraction-refinement-cycles}$ [#]",
    }


    metrics_map = {
        "disp1 [%]" : "D1 Out. [%]",
        "disp2 [%]" : "D2 Out. [%]",
        "oflow [%]" : "OF Out. [%]",
        "sflow [%]" : "SF Out. [%]",
        "disp1 epe [pxl]" : "D1 EPE [pxl]",
        "disp2 epe [pxl]" : "D2 EPE [pxl]",
        "oflow epe [pxl]" : "OF EPE [pxl]",
        "sflow epe [m]" : "SF EPE [m]",
        "seg acc [%]" : "Segmentation Acc. [%]",
        "objects [#]" : "Objects Count [#]",
        "ego-se3-dist [m/s]" : "RPE transl. [m/s]",
        "ego-se3-angle [deg/s]" : "RPE rot. [deg/s]",
        "ego-se3-ate [m]" : "ATE [m]",
        "duration-aggregate-depth [s]" : "depth",
        "duration-aggregate-oflow [s]" : "optical flow",
        "duration-warped-sflow-to-rigid-sflow [s]" : "scene flow",
        "duration-downsample [s]" : "downsampling",
        "duration-retrieve-objects [s]" : "retrieval of objects",
        "duration-assign-objects [s]" : "assignment of pixels to objects",
        "duration-deduct-sflow [s]" : "deduction of rigid object scene flow",
        "duration-retrieve-objects-extract [s]" : "objects extraction",
        "duration-retrieve-objects-refine [s]" : "objects refinement",
        "duration-retrieve-objects-extract-se3 [s]" : "objects extraction se3",
        "duration-retrieve-objects-extract-geo [s]" : "objects extraction geo",
        "duration-retrieve-objects-extract-se3-cluster [s]" : "objects extraction se3 cluster",
        "duration-retrieve-objects-extract-se3-fit [s]" : "objects extraction se3 fit",
        "duration-retrieve-objects-extract-se3-select [s]" : "objects extraction se3 select",
        "gpu-memory [GB]" : "GPU Memory Usage [GB]",
        "duration-total [s]": "Duration [s]",
        "std disp [pxl]": r"$\sigma_{disp}$ [pxl]",
        "std oflow x [pxl]": r"$\sigma_{oflow-x}$ [pxl]",
        "std oflow y [pxl]": r"$\sigma_{oflow-y}$ [pxl]",
    }

    runs_selected_count = len(runs_selected)

    #configs_selected = []

    approaches = ["raft3d", "rigidmask", "classic", "wsflow"]

    if approaches_map is None:
        approaches_map = {
            "raft3d" : "RAFT-3D",
            "rigidmask" : "RigidMask",
            "classic" : "SF2SE3 (ours.)",
            "wsflow" : "Warped Scene Flow"
        }

    dataset_map = {
        "kitti_train" : "KITTI - train",
        "flyingthings3d_dispnet_test": "FT3D - test",
        "sintel_train" : "Sintel - train",
        "tum_rgbd_fr2" : "TUM FR2",
        "tum_rgbd_fr3" : "TUM FR3",
        "bonn_rgbd": "Bonn"
    }

    seq_tag_map = {
        "rgbd_dataset_freiburg2_desk_with_person" : "desk-with-person",
        "rgbd_dataset_freiburg3_sitting_static" : "sitting-static",
        "rgbd_dataset_freiburg3_sitting_rpy" : "sitting-rpy",
        "rgbd_dataset_freiburg3_sitting_xyz" : "sitting-xyz",
        "rgbd_dataset_freiburg3_sitting_halfsphere" : "sitting-halfsphere",
        "rgbd_dataset_freiburg3_walking_static" : "walking-static",
        "rgbd_dataset_freiburg3_walking_rpy" : "walking-rpy",
        "rgbd_dataset_freiburg3_walking_xyz" : "walking-xyz",
        "rgbd_dataset_freiburg3_walking_halfsphere" : "walking-halfsphere",
        "rgbd_bonn_balloon_tracking" : "balloon-tracking",
        "rgbd_bonn_crowd" : "crowd"
    }

    datasets = ["kitti_train", "flyingthings3d_dispnet_test", "sintel_train", "tum_rgbd_fr1", "tum_rgbd_fr2", "tum_rgbd_fr3", "bonn_rgbd"]

    columns_selected_ids = [None for column_selected in metrics_selected]
    runs_selected_dataset = [ o4string.remove_suffixes(rs.split("__")[0], approaches)[:-1] for rs in runs_selected]
    #runs_selected_approach = [ cs.split("__")[0].split("_")[-1] for cs in runs_selected]
    runs_selected_datetime = [ "__".join(rs.split("__")[1:3]) for rs in runs_selected]

    if runs_selected_seq_tag is None:
        results_from_csv = {runs_selected_datetime[i] + runs_selected_approach[i]: [] for i in
                            range(runs_selected_count)}
    else:
        results_from_csv = {runs_selected_datetime[i] + runs_selected_approach[i] + runs_selected_seq_tag[i]: [] for i in
                            range(runs_selected_count)}
    csv_fpath = os.path.join(results_dir, csv_fname)

    if len(params_selected) != 0:
        runs_selected_dicts_args = []
        for run_selected in runs_selected:
            runs_selected_dicts_args.append([])
            import glob
            config_fpaths = glob.glob(os.path.join(results_dir, run_selected, 'configs', 'ablation' +  '*.yaml'))
            if len(config_fpaths) == 0:
                config_fpaths = glob.glob(os.path.join(results_dir, run_selected, 'configs', '*', '*', 'ablation' +  '*.yaml'))

            config_fpath = config_fpaths[0]
            #config_fpath = os.path.join(results_dir, run_selected, 'configs/ablation*.yaml')
            with open(config_fpath) as f:
                parser = configargparse.ConfigparserConfigFileParser() #ConfigFileParser()
                args = parser.parse(f) #
                runs_selected_dicts_args[-1] = args
        print(runs_selected_dicts_args)

    with open(csv_fpath, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = None
        for row in csv_reader:
            if headers is None:
                headers = row
                for header_id in range(len(headers)):
                    for column_id in range(len(metrics_selected)):
                        if headers[header_id] == metrics_selected[column_id]:
                            columns_selected_ids[column_id] = header_id
                #print(' '.join(headers))
            else:
                date = row[0].split(" ")[0].replace("/", "_")
                time = row[0].split(" ")[1].replace(":", "_")
                datetime = date + "__" + time
                dataset = row[1].replace("', '", "_").replace("['", "").replace("']", "")
                approach = row[2]
                dataset_tags = row[1].replace("['", "").replace("']", "").split("', '")
                if runs_selected_seq_tag is not None:
                    seq_tag = dataset_tags[-1]
                else:
                    seq_tag = ""
                #print('datetime', date + " " + time)
                #print('dataset', dataset)
                #print('approach', approach)
                if datetime+approach+seq_tag in results_from_csv.keys():
                    print(datetime+approach+seq_tag)
                    results_from_csv[datetime+approach+seq_tag] = [("{:." + str(decimals) + "f}").format(float(row[column_id])) for column_id in columns_selected_ids]

        if runs_selected_seq_tag is None:
            runs_metrics = [results_from_csv[runs_selected_datetime[i]+runs_selected_approach[i]] for i in range(runs_selected_count)]
        else:
            runs_metrics = [results_from_csv[runs_selected_datetime[i]+runs_selected_approach[i]+runs_selected_seq_tag[i]] for i in range(runs_selected_count)]

        runs_params = [[runs_selected_dicts_args[i][param] for param in params_selected] for i in range(runs_selected_count)]
        headers_metrics = [metrics_map[metric] for metric in metrics_selected]
        headers_params = [params_map[param] for param in params_selected]

        rows = []
        headers = headers_params + headers_metrics
        rows.append(headers)
        for i in range(runs_selected_count):
            row_values = runs_params[i] + runs_metrics[i]
            rows.append(row_values)

        if add_dataset:
            for i, row in enumerate(rows):
                if i == 0:
                    row.insert(0, "Dataset")
                else:
                    if runs_selected_seq_tag is None:
                        row.insert(0, dataset_map[runs_selected_dataset[i-1]])
                    else:
                        row.insert(0, dataset_map[runs_selected_dataset[i - 1]] + " : " + seq_tag_map[runs_selected_seq_tag[i - 1]])

        if add_approach:
            for i, row in enumerate(rows):
                if i == 0:
                    row.insert(0, "Method")
                else:
                    row.insert(0, approaches_map[runs_selected_approach[i-1]])

        if generate_type == "table":
            #floatfmt="#.2g"
            print(tabulate(rows, headers='firstrow', tablefmt ='latex', disable_numparse=True))
        elif generate_type == "plot_bars":
            #import seaborn as sns
            if plot_first_value == "run":
                import pandas as pd
                data = np.array([np.array([float(run_metric) for run_metric in run_metrics]) for run_metrics in runs_metrics])

                approaches_written = [approaches_map[runs_selected_approach[i]] + " \n " + dataset_map[runs_selected_dataset[i]] for i in range(runs_selected_count)]

                data = pd.DataFrame(data, columns=headers_metrics, index=approaches_written)
                data = data.reindex(index=data.index[::-1])
                data.plot.barh(figsize=(8, 8), stacked=bars_second_values == "add")

                #plt.yticks(plt.yticks()[0] * 2, plt.yticks()[1])
                plt.xlabel(bars_second_label)
                plt.tight_layout()
                #bbox_inches = "tight"
                plt.savefig(save_fpath)
                plt.show()
            else:
                data_2 = np.array(
                    [np.array([float(run_metric) for run_metric in run_metrics]) for run_metrics in runs_metrics])
                data_1 = np.array([float(run_params[0]) for run_params in runs_params])

                ids_sorted = np.argsort(data_1)
                data_1 = data_1[ids_sorted]
                data_2 = data_2[ids_sorted]
                metrics_count = data_2.shape[1]

                np.sort(data_1)
                """
                plt.figure(figsize=plot_figuresize)
                for m in range(metrics_count):
                    plt.plot(data_1, data_2[:, m], plot_markertype, label=headers_metrics[m], markersize=plot_markersize)
                    plt.xlabel(headers_params[0], fontsize=plot_label_fontsize)
                    plt.xticks(fontsize=plot_tick_fontsize)
                    plt.yticks(fontsize=plot_tick_fontsize)
                    plt.legend(fontsize=plot_label_fontsize)
                    plt.tight_layout()

                #plot_figuresize=(6, 6), plot_markersize=12, plot_tick_fontsize = 12, plot_label_fontsize = 14, plot_markertype = "-"
                plt.savefig(save_fpath)
                plt.show()
                #runs_params
                """

                fig, ax = plt.subplots(figsize=plot_figuresize)
                fig.subplots_adjust(right=adjust_right) # 0.75
                tkw = dict(labelsize=plot_tick_fontsize)#, size=4.0 width=1.5)
                axes = [ax]

                p1, = ax.plot(data_1, data_2[:, 0], "C"+str(0)+plot_markertype, label=headers_metrics[0], markersize=plot_markersize)
                ax.set_xlabel(headers_params[0], fontsize=plot_label_fontsize)
                ax.set_ylabel(headers_metrics[0], fontsize=plot_label_fontsize)
                plots = [p1]
                axes[-1].yaxis.label.set_color(plots[-1].get_color())
                axes[-1].tick_params(axis='y', colors=plots[-1].get_color(), **tkw)

                for m in range(1, metrics_count):
                    axes.append(ax.twinx())

                    # Offset the right spine of twin2.  The ticks and label have already been
                    # placed on the right by twinx above.
                    axes[-1].spines.right.set_position(("axes", 1 + .2 * max(0, len(axes)-2)))

                    pn, = axes[-1].plot(data_1, data_2[:, m], "C"+str(m)+plot_markertype, label=headers_metrics[m], markersize=plot_markersize)
                    plots.append(pn)
                    #ax.set_xlim(0, 2)
                    #twin1.set_ylim(0, 4)

                    axes[-1].set_ylabel(headers_metrics[m], fontsize=plot_label_fontsize)
                    axes[-1].yaxis.label.set_color(plots[-1].get_color())
                    axes[-1].tick_params(axis='y', colors=plots[-1].get_color(), **tkw)

                ax.tick_params(axis='x', **tkw)
                ax.legend(handles=plots, fontsize=plot_label_fontsize)
                plt.tight_layout()
                plt.savefig(save_fpath)
                plt.show()


        elif generate_type == "images":

            if isinstance(image_fpath_rel, list):
                image_fpath = os.path.join(results_dir, runs_selected[0], image_fpath_rel[0])
            else:
                image_fpath = os.path.join(results_dir, runs_selected[0], image_fpath_rel)
            images = []

            import cv2
            #from tensor_operations.visual import _2d as o4visual2d


            image = cv2.imread(image_fpath)

            if image_text == "ablation":
                data_1 = np.array([float(run_params[0]) for run_params in runs_params])
                ids_sorted = np.argsort(data_1)
            else:
                ids_sorted = np.arange(runs_selected_count)[1:]

            H, W, C = image.shape

            #r = cv2.selectROI("win", image, cv2.WINDOW_NORMAL)
            #gt_image = cv2.rectangle(gt_image, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), color=(0, 0, 0), thickness=5)

            offset_H = 0
            text_first = "ground truth"
            if isinstance(image_text, list):

                if image_text[0] == "disparity 1":
                    image = image[:, :W // 2, :]
                    text_first = image_text[0]
                    offset_H = int(H // 4 * 3)
                    image = image[: H // 4]
                elif image_text[0] == "optical flow":
                    image = image[:, :W // 2, :]
                    text_first = image_text[0]
                    offset_H = int(H // 4 * 3)
                    image = image[H // 4  * 2: H // 4 * 3]
                elif image_text[0] == "scene flow":
                    image = image[:, W // 2 : , :]
                    text_first = approaches_map[runs_selected_approach[0]]
                    offset_H = int(H // 4 * 3)
                    image = image[H // 4  * 3: H // 4 * 4]
                else:
                    text_first = image_text[0]
                    image = image[:, :W // 2, :]
            else:
                image = image[:, :W // 2, :]
            r = cv2.selectROI("win", image, cv2.WINDOW_NORMAL)
            image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            cropped_H, cropped_W, _ = image.shape

            rect_pos = cv2.selectROI("win", image)
            while rect_pos[0] is not 0:
                lwidth = int(cropped_W / 150.)
                image = cv2.rectangle(image, (rect_pos[0]-lwidth, rect_pos[1]-lwidth), (rect_pos[0]+rect_pos[2]+lwidth, rect_pos[1]+ rect_pos[3]+lwidth), color=(0, 0, 0), thickness=lwidth)
                image = cv2.rectangle(image, (rect_pos[0], rect_pos[1]), (rect_pos[0]+rect_pos[2], rect_pos[1]+rect_pos[3]), color=(255, 255, 255), thickness=lwidth)
                rect_pos = cv2.selectROI("win", image)
                print(rect_pos)

            text_pos = cv2.selectROI("win", image)
            text_height = text_pos[3] / 60.
            # r: x, y, dx, dy

            font =  cv2.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (text_pos[0], text_pos[1] + int(text_pos[3] / 2.))  # left, top
            fontScale = text_height
            fontColor = (255, 255, 255)
            lineType =  2 #10 * int(text_height)
            text_lwidth = int(lwidth / 2.)
            image = cv2.putText(image, text_first,
                        topLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        text_lwidth,
                        lineType)



            images.append(image)
            for i in ids_sorted:
                if isinstance(image_fpath_rel, list):
                    offset_H = 0
                    image_fpath = os.path.join(results_dir, runs_selected[i], image_fpath_rel[i])
                else:
                    image_fpath = os.path.join(results_dir, runs_selected[i], image_fpath_rel)
                image = cv2.imread(image_fpath)
                if image is not None:
                    image = image[:, W // 2:, :]
                    # image = cv2.rectangle(image, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), color=(0, 0, 0), thickness=5)
                    image = image[offset_H + int(r[1]):offset_H + int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                    if image_text is None:
                        text = ""
                    elif isinstance(image_text, list):
                        text = image_text[i]
                        if text == "approach":
                            text = approaches_map[runs_selected_approach[i]]
                        elif text == "ablation":
                            print(i)
                            text = str(data_1[i])
                        else:
                            text = image_text
                    elif image_text == "approach":
                        text = approaches_map[runs_selected_approach[i]]
                    elif image_text == "ablation":
                        print(i)
                        text = str(data_1[i])
                    else:
                        text = image_text

                    image = cv2.rectangle(image, (rect_pos[0] - lwidth, rect_pos[1] - lwidth),
                                          (rect_pos[0] + rect_pos[2] + lwidth, rect_pos[1] + rect_pos[3] + lwidth),
                                          color=(0, 0, 0), thickness=lwidth)
                    image = cv2.rectangle(image, (rect_pos[0], rect_pos[1]),
                                          (rect_pos[0] + rect_pos[2], rect_pos[1] + rect_pos[3]), color=(255, 255, 255),
                                          thickness=lwidth)

                    image = cv2.putText(image, text,
                                        topLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        text_lwidth,
                                        lineType)

                    images.append(image)

            cv2.imwrite(filename=save_fpath, img=np.concatenate(images, axis=1))
            #cv2.imshow("img", np.concatenate(images, axis=1))
            #cv2.waitKey(0)