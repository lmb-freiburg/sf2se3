import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



def parser_add_args(parser):

    parser.add_argument(
        "--data-dataset-tags",
        default="flyingthings3d-train",
        type=str,
        help="(default: flyingthings3d-train)",
    )

    parser.add_argument(
        "--data-meta-use",
        default=False,
        type=str2bool,
        help="data-meta-use (default: True)",
    )

    parser.add_argument(
        "--data-meta-recalc",
        default=False,
        type=str2bool,
        help="data-meta-recalc (default: False",
    )

    parser.add_argument(
        "--data-baseline",
        default=0.54,
        type=float,
        help="data-baseline (default: 0.54)",
    )

    parser.add_argument(
        "--data-res-width", default=640, type=int, help="data-res-width (default: 640)"
    )
    parser.add_argument(
        "--data-res-height",
        default=640,
        type=int,
        help="data-res-height (default: 640)",
    )

    parser.add_argument(
        "--data-cam-fx",
        default=0.,
        type=float,
        help="data-cam-fx (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-fy",
        default=0.,
        type=float,
        help="data-cam-fy (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-cx",
        default=0.,
        type=float,
        help="data-cam-cx (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-cy",
        default=0.,
        type=float,
        help="data-cam-cy (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-d0",
        default=0.,
        type=float,
        help="data-cam-d0 (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-d1",
        default=0.,
        type=float,
        help="data-cam-d1 (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-d2",
        default=0.,
        type=float,
        help="data-cam-d2 (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-d3",
        default=0.,
        type=float,
        help="data-cam-d3 (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-d4",
        default=0.,
        type=float,
        help="data-cam-d4 (default: 0.)",
    )

    parser.add_argument(
        "--data-cam-depth-scale",
        default=1.,
        type=float,
        help="data-cam-depth-scale (default: 1.)",
    )

    parser.add_argument(
        "--data-dataset-max-num-imgs",
        default=None,
        type=int,
        help="data-dataset-max-num-imgs (default: None)",
    )

    parser.add_argument(
        "--data-dataset-index-shift",
        default=0,
        type=int,
        help="data-dataset-index-shift (default: 0)",
    )

    parser.add_argument(
        "--data-dataset-directory-structure",
        default="DATA-SEQ",
        required=False,
        type=str,
        help="directory structure of dataset DATA-SEQ or SEQ-DATA",
    )

    parser.add_argument(
        "--data-dataset-directory-structure-seq-depth",
        default=1,
        type=int,
        help="data-dataset-directory-structure-seq-depth (default: 1)",
    )

    parser.add_argument(
        "--data-dataset-subdir",
        default="Brox_FlyingThings3D",
        required=False,
        type=str,
        help="subdir of dataset in datasets",
    )

    parser.add_argument(
        "--data-dataset-transfs-fpath-rel",
        default="camera_data.txt",
        required=False,
        type=str,
        help="rel fpath to camera transfs file",
    )

    parser.add_argument(
        "--data-dataset-label-subdir",
        default="object_index",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-oflow-subdir",
        default="optical_flow",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-oflow-occ-subdir",
        default="optical_flow_occlusions",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-oflow-invalid-subdir",
        default="optical_flow_invalid",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-rgb-subdir",
        default="RGB_cleanpass",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-rgb-left-subdir",
        default="clean_left",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-rgb-right-subdir",
        default="clean_right",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-disp-subdir",
        default="disparity",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-depth-subdir",
        default="depth",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-disp-occ-subdir",
        default="disparity_occlusions",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-dataset-disp-oof-subdir",
        default="disparity_outofframe",
        required=False,
        type=str,
        help="subdir",
    )


    parser.add_argument(
        "--data-dataset-camera-subdir",
        default="camera",
        required=False,
        type=str,
        help="subdir",
    )

    parser.add_argument(
        "--data-seqs-dirs-filter-tags",
        default=[],
        required=False,
        action='append',
        type=str,
        help="tags to filter sequences directories (default: [])"
    )

    parser.add_argument(
        "--data-fps",
        default=None,
        required=False,
        type=float,
        help="data-fps (default: None)"
    )

    parser.add_argument(
        "--data-timestamp-inter-data-max-diff",
        default=0.02,
        required=False,
        type=float,
        help="timestamp-inter-data-max-diff (default: 0.02)"
    )

    return parser
